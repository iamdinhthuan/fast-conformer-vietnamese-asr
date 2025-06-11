from __future__ import annotations

"""RNNT-only training module for FastConformer encoder.

This file replaces the previous hybrid CTC+RNNT training (train.py).
It keeps the same API expected by run.py but drops all CTC logic to
focus purely on RNNT loss.
"""

from typing import Any, Dict, List, Optional
import time

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from jiwer import wer
from loguru import logger
import sentencepiece as spm

import torchaudio

from config import ExperimentConfig
from models.fast_conformer import FastConformerEncoder
from models.rnnt_decoder import RNNTDecoder
from models.rnnt_streaming import StreamingGreedyRNNT


# -----------------------------------------------------------------------------
# Helper: callbacks
# -----------------------------------------------------------------------------

def create_advanced_callbacks(config: ExperimentConfig):
    """Create advanced callbacks for training"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoint_dir,
        filename="{epoch:02d}-{val_wer_epoch:.3f}",
        monitor="val_wer_epoch",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_wer_epoch",
        min_delta=0.001,
        patience=config.training.early_stopping_patience,
        verbose=True,
        mode="min"
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks


# -----------------------------------------------------------------------------
# LightningModule
# -----------------------------------------------------------------------------

class StreamingRNNT(pl.LightningModule):
    """FastConformer + RNNT training without any CTC components."""

    def __init__(
        self,
        config: ExperimentConfig,
        learning_rate: float | None = None,
        min_learning_rate: float | None = None,
        warmup_steps: int | None = None,
        total_steps: int | None = None,
        gradient_clip_val: float | None = None,
        accumulate_grad_batches: int | None = None,
    ) -> None:
        super().__init__()

        # Optimize for Tensor Cores
        torch.set_float32_matmul_precision('medium')
        self.config = config

        # Override config with explicit parameters if provided
        self.learning_rate = learning_rate or config.training.learning_rate
        self.min_learning_rate = min_learning_rate or config.training.min_learning_rate
        self.warmup_steps = warmup_steps or config.training.warmup_steps
        self.total_steps = total_steps or config.training.total_steps
        self.gradient_clip_val = gradient_clip_val or config.training.gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches or config.training.accumulate_grad_batches

        self.save_hyperparameters()

        # --------------- architecture ---------------
        self.encoder = FastConformerEncoder(
            n_mels=config.audio.n_mels,
            d_model=config.model.n_state,
            n_heads=config.model.n_head,
            n_layers=config.model.n_layer,
            dropout=config.model.dropout,
            ffn_expansion=config.model.ffn_expansion,
            left_ctx=config.model.left_ctx,
            right_ctx=config.model.right_ctx,
        )

        self.rnnt_decoder = RNNTDecoder(
            vocab_size=config.model.vocab_size,
            enc_dim=config.model.n_state,
        )

        self.rnnt_loss_fn = torchaudio.transforms.RNNTLoss(
            blank=config.model.rnnt_blank
        )

        # Greedy decoder helper for WER evaluation
        self.greedy_streamer = StreamingGreedyRNNT(self.rnnt_decoder, device=self.device)

        # Tokenizer
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=config.model.tokenizer_model_path
        )

        # Metric buffers
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.step_start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, x_len: torch.Tensor):
        """Encode audio; returns encoder output and lengths."""
        enc_out, enc_len, _ = self.encoder(x, x_len, return_intermediate=False)
        return enc_out, enc_len

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def on_train_start(self):
        self.step_start_time = time.time()

    def training_step(self, batch, batch_idx: int):
        if batch_idx == 0:
            start_time = time.time()
            print(f"[🔄] Starting first training batch...")

        x, x_len, y, y_len = batch
        
        enc_out, enc_len = self.forward(x, x_len)
        
        if batch_idx == 0:
            print(f"[🔄] Encoder output shape: {enc_out.shape}, encoder lengths: {enc_len.shape}")

        logits = self.rnnt_decoder(enc_out, y, y_len)

        if batch_idx == 0:
            print(f"[🔄] RNNT decoder output shape: {logits.shape}")
            print(f"[🔄] Encoder output shape: {enc_out.shape}, enc_len: {enc_len.shape}")
            print(f"[🔄] Targets shape: {y.shape}, y_len: {y_len.shape}")
            print(f"[🔄] enc_len values: {enc_len[:5]}")
            print(f"[🔄] y_len values: {y_len[:5]}")
            print(f"[⏱️] First batch completed in {time.time() - start_time:.2f} seconds")

        loss = self.rnnt_loss_fn(
            logits,
            y.to(torch.int32),
            enc_len.to(torch.int32),
            y_len.to(torch.int32),
        )
        
        if batch_idx == 0:
            print(f"[✅] First loss calculated: {loss.item():.4f}")
            print(f"[🚀] Training loop is running - wait for progress bar to update")

        # Periodic WER logging
        if batch_idx % 2000 == 0:
            predictions = self._greedy_decode(enc_out, enc_len)
            targets = self._decode_targets(y, y_len)
            train_wer = self._compute_wer(predictions, targets, "TRAIN", batch_idx)
            self.log("train_wer", train_wer, prog_bar=True, on_step=True, on_epoch=False)

        # Logging
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False
        )

        if batch_idx % 100 == 0:
            step_time = time.time() - self.step_start_time
            self.log("step_time", step_time, on_step=True, on_epoch=False)
            self.step_start_time = time.time()

        return loss

    def validation_step(self, batch, batch_idx: int):
        x, x_len, y, y_len = batch
        enc_out, enc_len = self.forward(x, x_len)
        logits = self.rnnt_decoder(enc_out, y, y_len)
        val_loss = self.rnnt_loss_fn(
            logits,
            y.to(torch.int32),
            enc_len.to(torch.int32),
            y_len.to(torch.int32),
        )

        predictions = self._greedy_decode(enc_out, enc_len)
        targets = self._decode_targets(y, y_len)
        val_wer = self._compute_wer(predictions, targets, "VAL", batch_idx)

        self.validation_step_outputs.append(
            {"val_loss": val_loss, "val_wer": val_wer, "batch_size": x.size(0)}
        )

        return {"val_loss": val_loss, "val_wer": val_wer}

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # Compute weighted averages
        total_samples = sum(out["batch_size"] for out in self.validation_step_outputs)
        avg_loss = sum(
            out["val_loss"] * out["batch_size"] for out in self.validation_step_outputs
        ) / total_samples
        avg_wer = sum(
            out["val_wer"] * out["batch_size"] for out in self.validation_step_outputs
        ) / total_samples

        self.log("val_loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        self.log("val_wer_epoch", avg_wer, prog_bar=True, on_epoch=True)

        logger.info(f"Validation - Loss: {avg_loss:.4f}, WER: {avg_wer:.4f}")
        self.validation_step_outputs.clear()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01,
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=self.learning_rate / self.min_learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _greedy_decode(self, enc_out: torch.Tensor, enc_len: torch.Tensor) -> List[str]:
        """Greedy decode batch using StreamingGreedyRNNT in offline mode."""
        predictions: List[str] = []
        for b in range(enc_out.size(0)):
            self.greedy_streamer.reset()
            tokens = self.greedy_streamer.infer(enc_out[b : b + 1])
            predictions.append(self.tokenizer.decode(tokens))
        return predictions

    def _decode_targets(self, y: torch.Tensor, y_len: torch.Tensor) -> List[str]:
        targets: List[str] = []
        for i in range(y.size(0)):
            ids = y[i, : y_len[i]].cpu().tolist()
            targets.append(self.tokenizer.decode(ids))
        return targets

    def _compute_wer(self, predictions: List[str], targets: List[str], prefix: str, batch_idx: int) -> float:
        """Compute WER and log examples."""
        if not predictions or not targets:
            return 1.0

        wer_score = wer(targets, predictions)

        # Log examples occasionally
        if batch_idx % 1000 == 0:
            logger.info(f"{prefix} Example:")
            logger.info(f"  Target: {targets[0]}")
            logger.info(f"  Prediction: {predictions[0]}")
            logger.info(f"  WER: {wer_score:.4f}")

        return wer_score
