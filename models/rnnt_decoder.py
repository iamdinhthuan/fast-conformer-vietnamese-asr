from __future__ import annotations

from typing import Tuple
import torch
from torch import nn, Tensor

__all__ = ["RNNTDecoder"]


class RNNTDecoder(nn.Module):
    """Minimal RNN-T decoder (prediction + joint) for hybrid loss.

    • Prediction network: 1-layer LSTM with embedding.
    • Joint network: linear(enc_dim) + linear(pred_dim) -> Tanh -> linear(vocab).
    This is NOT an efficient production decoder – it is sufficient for
    computing RNNT loss during training.
    """

    def __init__(self, vocab_size: int, enc_dim: int, pred_dim: int = 512, embed_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_id = vocab_size  # assume blank = vocab_size (same as CTC blank)

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)  # +blank
        # Use LSTMCell instead of LSTM for Lightning compatibility
        self.pred_rnn_cell = nn.LSTMCell(embed_dim, pred_dim)
        self.pred_dim = pred_dim

        self.lin_enc = nn.Linear(enc_dim, pred_dim)
        self.lin_pred = nn.Linear(pred_dim, pred_dim)
        self.joint = nn.Linear(pred_dim, vocab_size + 1)  # +blank

    def forward(self, enc_out: Tensor, targets: Tensor, target_len: Tensor) -> Tensor:
        """Compute logits for RNNT loss.

        enc_out: (B, T_enc, D)
        targets: (B, U)  – int64 without blank
        Return: logits (B, T_enc, U+1, vocab+1)
        """
        B, T_enc, D = enc_out.shape
        U = targets.size(1)

        # Prepend blank token to targets for prediction network
        # RNN-T prediction network needs to start with blank
        blank_tokens = torch.full((B, 1), self.blank_id, dtype=targets.dtype, device=targets.device)
        targets_with_blank = torch.cat([blank_tokens, targets], dim=1)  # (B, U+1)

        # prediction network
        emb = self.embedding(targets_with_blank)  # (B,U+1,E)

        # Use LSTMCell manually to avoid CuDNN issues
        B, U_plus_1, E = emb.shape
        pred_outputs = []

        # Initialize hidden state
        h = torch.zeros(B, self.pred_dim, device=emb.device, dtype=emb.dtype)
        c = torch.zeros(B, self.pred_dim, device=emb.device, dtype=emb.dtype)

        # Process each time step
        for t in range(U_plus_1):
            h, c = self.pred_rnn_cell(emb[:, t, :], (h, c))
            pred_outputs.append(h)

        pred = torch.stack(pred_outputs, dim=1)  # (B, U+1, P)

        f_enc = self.lin_enc(enc_out)          # (B,T_enc,P)
        f_pred = self.lin_pred(pred)           # (B,U+1,P)

        # expand and add
        f_enc = f_enc.unsqueeze(2)             # (B,T,1,P)
        f_pred = f_pred.unsqueeze(1)           # (B,1,U+1,P)
        joint = torch.tanh(f_enc + f_pred)     # (B,T,U+1,P)
        logits = self.joint(joint)             # (B,T,U+1,vocab+1)
        return logits