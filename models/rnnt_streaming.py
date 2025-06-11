from __future__ import annotations

from typing import List, Tuple, Optional
import torch
from torch import Tensor

from .rnnt_decoder import RNNTDecoder

__all__ = ["StreamingGreedyRNNT"]


class StreamingGreedyRNNT:
    """Stateful greedy streaming decoder for an *already-trained* :class:`RNNTDecoder`.

    The implementation follows the standard TIMIT greedy algorithm used in RNNT: for each
    encoder time-step we repeatedly invoke the prediction network until a blank token is
    emitted, then advance to the next encoder frame.

    Notes
    -----
    • Supports *only* greedy decoding but is fully streaming: internal predictor state is
      preserved between successive audio chunks.
    • Designed for small to medium vocabularies (≤4k). For larger vocabularies consider
      beam-search with pruning.
    • This helper does **not** own the parameters – it simply holds a reference to a
      frozen :class:`RNNTDecoder` instance.
    """

    def __init__(self, rnnt: RNNTDecoder, device: torch.device | str = "cpu") -> None:
        self.rnnt = rnnt.eval()  # prediction & joint network
        self.device = torch.device(device)
        self.blank_id = rnnt.blank_id

        # predictor recurrent state (h, c) – initialised lazily
        self._hidden: Optional[Tuple[Tensor, Tensor]] = None
        # last emitted non-blank token (starts with blank)
        self._prev_token: Tensor = torch.tensor([self.blank_id], dtype=torch.long)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def reset(self):
        """Clear internal predictor state – call between utterances."""
        self._hidden = None
        self._prev_token = torch.tensor([self.blank_id], dtype=torch.long)

    @torch.no_grad()
    def infer(self, enc_out: Tensor) -> List[int]:
        """Greedy-decode *one* encoder chunk.

        Parameters
        ----------
        enc_out : Tensor
            Encoder outputs of shape ``(B, T, D)`` where ``B == 1``.

        Returns
        -------
        List[int]
            Sequence of emitted token IDs for this chunk.
        """
        assert enc_out.dim() == 3 and enc_out.size(0) == 1, "enc_out must be (1, T, D)"
        emitted: List[int] = []

        # Remove batch dimension for convenience → (T, D)
        for enc_t in enc_out.squeeze(0):
            finished = False
            while not finished:
                # Ensure token on same device as model
                if self._prev_token.device != self.rnnt.embedding.weight.device:
                    self._prev_token = self._prev_token.to(self.rnnt.embedding.weight.device)
                pred_embed = self.rnnt.embedding(self._prev_token).unsqueeze(0)  # (1,1,E)
                # Use LSTMCell for streaming
                if self._hidden is None:
                    h = torch.zeros(1, self.rnnt.pred_dim, device=pred_embed.device, dtype=pred_embed.dtype)
                    c = torch.zeros(1, self.rnnt.pred_dim, device=pred_embed.device, dtype=pred_embed.dtype)
                    self._hidden = (h, c)

                h, c = self.rnnt.pred_rnn_cell(pred_embed.squeeze(1), self._hidden)
                self._hidden = (h, c)
                pred_out = h.unsqueeze(1)  # (1,1,P)

                f_enc = self.rnnt.lin_enc(enc_t.unsqueeze(0).unsqueeze(0))  # (1,1,P)
                f_pred = self.rnnt.lin_pred(pred_out)                      # (1,1,P)
                joint = torch.tanh(f_enc + f_pred)                         # (1,1,P)
                logits = self.rnnt.joint(joint)                            # (1,1,V+1)

                next_token = int(logits.argmax(dim=-1))

                if next_token == self.blank_id:
                    # Emit blank → move to next encoder frame
                    finished = True
                    # Important: predictor *state* is carried forward but previous token
                    # resets to blank so that the next prediction starts with blank.
                    self._prev_token = torch.tensor([self.blank_id], device=self.rnnt.embedding.weight.device)
                else:
                    emitted.append(next_token)
                    self._prev_token = torch.tensor([next_token], device=self.rnnt.embedding.weight.device)
                    # stay on same encoder frame (finished remains False)

        return emitted
