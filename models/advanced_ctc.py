import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class BeamHypothesis:
    """Improved beam search hypothesis with language model support"""
    sequence: List[int]
    score: float
    last_token: Optional[int] = None
    
    def __lt__(self, other):
        return self.score < other.score


class AdvancedCTCDecoder(nn.Module):
    """Advanced CTC decoder with optimized beam search and prefix beam search"""
    
    def __init__(self, vocab_size: int, blank_token: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_token = blank_token
        
    def greedy_decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
        """Optimized greedy decoding"""
        batch_size, max_len, _ = log_probs.shape
        predictions = log_probs.argmax(dim=-1)  # (B, T)
        
        decoded_sequences = []
        for batch_idx in range(batch_size):
            seq_len = input_lengths[batch_idx].item()
            seq_preds = predictions[batch_idx, :seq_len]
            
            # Optimized CTC collapse
            decoded_ids = []
            prev_token = None
            
            for pred in seq_preds:
                pred_item = pred.item()
                if pred_item != self.blank_token and pred_item != prev_token:
                    decoded_ids.append(pred_item)
                prev_token = pred_item
                
            decoded_sequences.append(decoded_ids)
            
        return decoded_sequences
    
    def prefix_beam_search(self, 
                          log_probs: torch.Tensor, 
                          input_lengths: torch.Tensor,
                          beam_size: int = 5,
                          alpha: float = 0.3) -> List[List[int]]:
        """Prefix beam search - more accurate than standard beam search for CTC"""
        batch_size = log_probs.shape[0]
        decoded_sequences = []
        
        for batch_idx in range(batch_size):
            seq_len = input_lengths[batch_idx].item()
            seq_log_probs = log_probs[batch_idx, :seq_len, :]  # (T, V)
            
            # Initialize prefix beam
            # Each prefix is represented as (prefix, prob_blank, prob_non_blank)
            prefixes = {(): (0.0, float('-inf'))}  # (prefix): (p_b, p_nb)
            
            for t in range(seq_len):
                new_prefixes = {}
                
                for prefix, (p_b, p_nb) in prefixes.items():
                    # Current prefix probability
                    p_prefix = torch.logsumexp(torch.tensor([p_b, p_nb]), dim=0).item()
                    
                    # Blank token
                    blank_score = seq_log_probs[t, self.blank_token].item()
                    new_p_b = torch.logsumexp(torch.tensor([p_b, p_nb]), dim=0).item() + blank_score
                    
                    if prefix not in new_prefixes:
                        new_prefixes[prefix] = (new_p_b, float('-inf'))
                    else:
                        current_p_b, current_p_nb = new_prefixes[prefix]
                        new_prefixes[prefix] = (torch.logsumexp(torch.tensor([current_p_b, new_p_b]), dim=0).item(), current_p_nb)
                    
                    # Non-blank tokens
                    for token in range(self.vocab_size):
                        if token == self.blank_token:
                            continue
                            
                        token_score = seq_log_probs[t, token].item()
                        new_prefix = prefix + (token,)
                        
                        if len(prefix) > 0 and prefix[-1] == token:
                            # Repeat token - only extend from blank
                            new_p_nb = p_b + token_score
                        else:
                            # New token
                            new_p_nb = torch.logsumexp(torch.tensor([p_b, p_nb]), dim=0).item() + token_score
                        
                        if new_prefix not in new_prefixes:
                            new_prefixes[new_prefix] = (float('-inf'), new_p_nb)
                        else:
                            current_p_b, current_p_nb = new_prefixes[new_prefix]
                            new_prefixes[new_prefix] = (current_p_b, torch.logsumexp(torch.tensor([current_p_nb, new_p_nb]), dim=0).item())
                
                # Prune to beam size
                prefix_scores = []
                for prefix, (p_b, p_nb) in new_prefixes.items():
                    score = torch.logsumexp(torch.tensor([p_b, p_nb]), dim=0).item()
                    # Length normalization
                    normalized_score = score / (len(prefix) + 1) ** alpha
                    prefix_scores.append((normalized_score, prefix, (p_b, p_nb)))
                
                prefix_scores.sort(reverse=True)
                prefixes = {prefix: probs for _, prefix, probs in prefix_scores[:beam_size]}
            
            # Get best sequence
            best_score = float('-inf')
            best_sequence = []
            
            for prefix, (p_b, p_nb) in prefixes.items():
                score = torch.logsumexp(torch.tensor([p_b, p_nb]), dim=0).item()
                normalized_score = score / (len(prefix) + 1) ** alpha
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_sequence = list(prefix)
            
            decoded_sequences.append(best_sequence)
            
        return decoded_sequences


class AdvancedCTCHead(nn.Module):
    """Improved CTC head with better initialization and regularization"""
    
    def __init__(self, input_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, vocab_size + 1)  # +1 for blank
        
        # Better initialization
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) encoder outputs
        Returns:
            logits: (B, T, V+1) CTC logits
        """
        x = self.layer_norm(x)
        x = self.dropout(x)
        logits = self.projection(x)
        return logits


class CTCLossWithLabelSmoothing(nn.Module):
    """CTC Loss with label smoothing for better generalization"""
    
    def __init__(self, blank_token: int, label_smoothing: float = 0.1, zero_infinity: bool = True):
        super().__init__()
        self.blank_token = blank_token
        self.label_smoothing = label_smoothing
        self.zero_infinity = zero_infinity
        
    def forward(self, 
                log_probs: torch.Tensor,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probs: (T, B, V+1) log probabilities
            targets: (B, S) target sequences
            input_lengths: (B,) input lengths
            target_lengths: (B,) target lengths
        """
        ctc_loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank_token,
            reduction='mean',
            zero_infinity=self.zero_infinity
        )
        
        if self.label_smoothing > 0:
            # Apply label smoothing
            smooth_loss = -log_probs.mean()
            ctc_loss = (1 - self.label_smoothing) * ctc_loss + self.label_smoothing * smooth_loss
            
        return ctc_loss 