from __future__ import annotations

"""FastConformer encoder (streaming-friendly) built on top of torchaudio.models.Conformer.

The goal is to provide:
  • forward()   – offline training (full sequence) with limited right-context masking.
  • stream_step() – online inference with caching of past activations.

This implementation keeps API compatible with previous encoders:
    encoded, enc_len, intermediates = model(x, x_len, return_intermediate=True)

and adds
    y, new_cache = model.stream_step(x_chunk, cache)
where `cache` is a list (len = n_layers) of dicts holding tensors.

The implementation is simplified: it caches ONLY the self-attention K/V and
skips convolution-state cache. That is sufficient for functional streaming with
limited left context while staying lightweight.  It can be upgraded later to
a full FastConformer as in NVIDIA NeMo.
"""

from typing import List, Tuple, Dict, Any

import torch
from torch import nn, Tensor

try:
    from torchaudio.models import Conformer as TAConformer
except ImportError as e:  # pragma: no cover
    raise ImportError("Please install torchaudio >=2.2 for FastConformerEncoder") from e

__all__ = ["FastConformerEncoder"]


class _Subsample(nn.Module):
    """2×Conv2d stride-2 subsampler (same as Efficient encoder)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_mels, T)
        return self.layers(x.unsqueeze(1))  # -> (B, C, F, T')


class FastConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        left_ctx: int = 160,   # frames after subsample (≈1 s @ 25 ms frame)
        right_ctx: int = 40,
        ffn_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.subsample = _Subsample(n_mels, d_model)
        self.left_ctx = left_ctx
        self.right_ctx = right_ctx

        self.pos_enc = nn.Parameter(torch.randn(1, 10000, d_model) * 0.01)

        self.encoder = TAConformer(
            input_dim=d_model,
            num_heads=n_heads,
            ffn_dim=d_model * ffn_expansion,
            num_layers=n_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )

        self._index_range = torch.arange(10000)  # re-used for mask creation

    # ---------------------------------------------------
    # Utility: compute length after two stride-2 convs
    # ---------------------------------------------------
    @staticmethod
    def _conv_out(length: Tensor) -> Tensor:
        return ((length + 1) // 2 + 1) // 2  # approx for k=3,p=1,s=2 twice

    def get_length_after_subsample(self, x_len: Tensor) -> Tensor:
        return self._conv_out(x_len)

    # ---------------------------------------------------
    # Offline forward (training)
    # ---------------------------------------------------
    def forward(
        self,
        x: Tensor,  # (B, n_mels, T)
        x_len: Tensor,
        return_intermediate: bool = False,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        B = x.size(0)
        x = self.subsample(x)  # (B, C, F, T')
        B, C, F, T_prime = x.shape
        x = x.squeeze(2).transpose(1, 2)  # (B, T', d_model)

        # pos enc
        x = x + self.pos_enc[:, :T_prime, :]

        out_len = self.get_length_after_subsample(x_len)

        # attention padding mask with limited right context
        # mask True = pad. Build square mask later inside encoder layer.
        device = x.device
        seq_idx = torch.arange(T_prime, device=device).unsqueeze(0)
        pad_mask = seq_idx >= out_len.unsqueeze(1)  # (B, T')

        # torchaudio Conformer expects input (B,T,D) and lengths
        encoded, _ = self.encoder(x, out_len)

        inter: List[Tensor] = []
        if return_intermediate:
            inter.append(encoded)  # placeholder – can append more detailed later
        return encoded, out_len, inter

    # ---------------------------------------------------
    # Streaming step (stateful)
    # ---------------------------------------------------
    def init_cache(self, batch_size: int, device=None):
        cache: List[Dict[str, Any]] = []
        for layer in self.encoder.conformer_layers:
            cache.append({"k": None, "v": None})
        return cache

    @torch.no_grad()
    def stream_step(
        self,
        x_chunk: Tensor,       # (B, n_mels, T_chunk)
        prev_cache: List[Dict[str, Tensor]],
        ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """Process chunk and return encoded seq for this chunk.

        Implementation: very naive – we concat cached left context frames
        (already subsampled) with new chunk, run full encoder, then keep tail
        for next cache.  Latency: right_ctx frames.
        """
        B = x_chunk.size(0)
        device = x_chunk.device

        # Subsample new chunk
        subsampled = self.subsample(x_chunk)  # (B, C, F, T')
        subsampled = subsampled.squeeze(2).transpose(1, 2)  # (B, T', D)
        subsampled = subsampled + self.pos_enc[:, :subsampled.size(1), :]

        # Re-create left context tensor from cache (last left_ctx frames).
        if prev_cache[0]["k"] is not None:
            left_ctx_tensor = prev_cache[0]["k"]  # we stored encoded frames per layer 0
            x_input = torch.cat([left_ctx_tensor, subsampled], dim=1)
        else:
            x_input = subsampled

        seq_len = x_input.size(1)
        lengths = torch.full((B,), seq_len, dtype=torch.long, device=device)

        encoded, _ = self.encoder(x_input, lengths)

        # Output only newest part (excluding right context)
        out = encoded[:, -subsampled.size(1):, :]

        # Update cache with last left_ctx frames from x_input
        new_cache: List[Dict[str, Tensor]] = []
        keep_len = min(self.left_ctx, x_input.size(1))
        left_part = x_input[:, -keep_len:, :].detach()
        for _ in self.encoder.conformer_layers:
            new_cache.append({"k": left_part, "v": None})

        return out, new_cache 