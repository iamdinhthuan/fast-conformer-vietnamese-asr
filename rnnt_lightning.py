from __future__ import annotations

"""RNNT-only training module for FastConformer encoder.

This file replaces the previous hybrid CTC+RNNT training (train.py).
It keeps the same API expected by run.py but drops all CTC logic to
focus purely on RNNT loss.
"""

from typing import Any, Dict, List, Optional
import time
import random
import os
from pathlib import Path

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
import librosa

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
        # Keep RNN decoder in train mode during training
        self.rnnt_decoder.train()

        self.rnnt_loss_fn = torchaudio.transforms.RNNTLoss(
            blank=config.model.rnnt_blank
        )

        # Fallback CTC loss for debugging
        self.ctc_loss_fn = torch.nn.CTCLoss(blank=config.model.rnnt_blank)

        # Greedy decoder helper for WER evaluation
        self.greedy_streamer = StreamingGreedyRNNT(self.rnnt_decoder, device=self.device)

        # Tokenizer
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=config.model.tokenizer_model_path
        )

        # Metric buffers
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.step_start_time: Optional[float] = None

    def train(self, mode: bool = True):
        """Override train to ensure RNN decoder stays in correct mode"""
        super().train(mode)
        # Always keep RNN decoder in train mode during training
        if mode:
            self.rnnt_decoder.train()
        return self

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

        # Add timeout and error handling for RNN-T loss
        try:
            if batch_idx == 0:
                print(f"[🔄] Computing RNN-T loss...")
                start_loss_time = time.time()

            loss = self.rnnt_loss_fn(
                logits,
                y.to(torch.int32),
                enc_len.to(torch.int32),
                y_len.to(torch.int32),
            )

            if batch_idx == 0:
                loss_time = time.time() - start_loss_time
                print(f"[⏱️] RNN-T loss computed in {loss_time:.2f}s")

        except Exception as e:
            print(f"[❌] RNN-T loss failed: {e}")
            print(f"[🔄] Switching to CTC fallback...")

            # Use CTC loss as fallback
            # Project logits to (B, T, V) by taking mean over U dimension
            ctc_logits = logits.mean(dim=2)  # (B, T, V)
            ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T, B, V)

            loss = self.ctc_loss_fn(
                ctc_log_probs,
                y,
                enc_len,
                y_len
            )
            print(f"[✅] CTC fallback loss: {loss.item():.4f}")
        
        if batch_idx == 0:
            print(f"[✅] First loss calculated: {loss.item():.4f}")
            print(f"[🔄] About to return loss for backward pass...")

        # Skip expensive WER computation for first few batches
        if batch_idx % 2000 == 0 and batch_idx > 0 and hasattr(self, 'trainer') and self.trainer is not None:
            predictions = self._greedy_decode(enc_out, enc_len)
            targets = self._decode_targets(y, y_len)
            train_wer = self._compute_wer(predictions, targets, "TRAIN", batch_idx)
            self.log("train_wer", train_wer, prog_bar=True, on_step=True, on_epoch=False)

        # Safe logging - only if trainer is available
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
            if hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
                self.log(
                    "learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False
                )

            if batch_idx % 100 == 0:
                step_time = time.time() - self.step_start_time if self.step_start_time else 0
                self.log("step_time", step_time, on_step=True, on_epoch=False)
                self.step_start_time = time.time()
        else:
            if batch_idx == 0:
                print(f"[⚠️] Trainer not available - skipping logging")

        if batch_idx == 0:
            print(f"[🔄] Returning loss {loss.item():.4f} - backward pass will start...")

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer step"""
        if self.global_step == 0:
            print(f"[🔄] About to perform optimizer step {self.global_step}")

    def on_after_backward(self):
        """Called after backward pass"""
        if self.global_step == 0:
            print(f"[✅] Backward pass completed for step {self.global_step}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after training batch ends"""
        if batch_idx == 0:
            print(f"[✅] Training batch {batch_idx} completed!")

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

        # Predict random validation samples
        self._predict_random_validation_samples()

        # Predict custom directory if specified
        if self.config.training.val_predict_dir and os.path.exists(self.config.training.val_predict_dir):
            self._predict_custom_directory()

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

    def _predict_random_validation_samples(self):
        """Predict random samples from validation dataset"""
        try:
            if not hasattr(self.trainer, 'val_dataloaders') or not self.trainer.val_dataloaders:
                return

            val_dataloader = self.trainer.val_dataloaders
            if hasattr(val_dataloader, '__iter__'):
                val_dataset = val_dataloader.dataset
            else:
                return

            # Get random samples
            num_samples = min(self.config.training.val_predict_samples, len(val_dataset))
            random_indices = random.sample(range(len(val_dataset)), num_samples)

            logger.info(f"🎯 Predicting {num_samples} random validation samples...")

            self.eval()
            with torch.no_grad():
                for i, idx in enumerate(random_indices):
                    try:
                        # Get sample (dataset returns melspec, transcript_tensor)
                        sample = val_dataset[idx]
                        x, y = sample

                        # Compute lengths manually (like collate function does)
                        x_len = x.shape[-1]  # mel spectrogram length
                        y_len = len(y)       # transcript length

                        # Add batch dimension
                        x = x.unsqueeze(0).to(self.device)
                        x_len = torch.tensor([x_len]).to(self.device)
                        y = y.unsqueeze(0).to(self.device)
                        y_len = torch.tensor([y_len]).to(self.device)

                        # Forward pass
                        enc_out, enc_len = self.forward(x, x_len)

                        # Decode prediction
                        predictions = self._greedy_decode(enc_out, enc_len)
                        targets = self._decode_targets(y, y_len)

                        # Compute WER for this sample
                        sample_wer = wer([targets[0]], [predictions[0]]) if targets[0] and predictions[0] else 1.0

                        logger.info(f"📝 Random Sample {i+1}/{num_samples} (idx={idx}):")
                        logger.info(f"   Target: '{targets[0]}'")
                        logger.info(f"   Prediction: '{predictions[0]}'")
                        logger.info(f"   WER: {sample_wer:.4f}")

                    except Exception as e:
                        logger.warning(f"Failed to predict sample {idx}: {e}")

            self.train()

        except Exception as e:
            logger.warning(f"Failed to predict random validation samples: {e}")

    def _predict_custom_directory(self):
        """Predict all audio files in custom directory"""
        try:
            predict_dir = Path(self.config.training.val_predict_dir)
            if not predict_dir.exists():
                logger.warning(f"Prediction directory does not exist: {predict_dir}")
                return

            # Find audio files
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(predict_dir.glob(f"*{ext}"))
                audio_files.extend(predict_dir.glob(f"**/*{ext}"))

            if not audio_files:
                logger.warning(f"No audio files found in {predict_dir}")
                return

            logger.info(f"🎯 Predicting {len(audio_files)} files from {predict_dir}...")

            self.eval()
            with torch.no_grad():
                for i, audio_file in enumerate(audio_files[:10]):  # Limit to 10 files
                    try:
                        # Load audio
                        audio, sr = librosa.load(str(audio_file), sr=self.config.audio.sample_rate)

                        # Convert to mel spectrogram
                        mel = self._audio_to_mel(torch.from_numpy(audio))

                        # Add batch dimension
                        x = mel.unsqueeze(0).to(self.device)
                        x_len = torch.tensor([mel.shape[1]]).to(self.device)

                        # Forward pass
                        enc_out, enc_len = self.forward(x, x_len)

                        # Decode prediction
                        predictions = self._greedy_decode(enc_out, enc_len)

                        logger.info(f"📁 File {i+1}/{min(len(audio_files), 10)}: {audio_file.name}")
                        logger.info(f"   Prediction: '{predictions[0]}'")
                        logger.info(f"   Duration: {len(audio)/sr:.2f}s")

                    except Exception as e:
                        logger.warning(f"Failed to predict {audio_file}: {e}")

            self.train()

        except Exception as e:
            logger.warning(f"Failed to predict custom directory: {e}")

    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram"""
        # Simple mel spectrogram conversion
        # You might want to use the same preprocessing as in your dataset
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.audio.sample_rate,
            n_mels=self.config.audio.n_mels,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
        )

        mel = mel_transform(audio)
        mel = torch.log(mel + 1e-8)  # Log mel
        return mel
