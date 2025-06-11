# DEPRECATED: This file has been replaced by rnnt_lightning.py
# This file contains the old hybrid CTC+RNN-T training implementation
# For RNN-T only training, use rnnt_lightning.py instead

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import sentencepiece as spm
from jiwer import wer
from loguru import logger
import wandb
from typing import Optional, Dict, Any
import time

from models.fast_conformer import FastConformerEncoder
from models.advanced_ctc import AdvancedCTCHead, AdvancedCTCDecoder, CTCLossWithLabelSmoothing
from models.rnnt_decoder import RNNTDecoder
import torchaudio

# All parameters now come from config

from utils.dataset import AudioDataset, create_collate_fn
from utils.scheduler import WarmupLR


class StreamingCTC(pl.LightningModule):
    """CTC model with advanced features"""
    
    def __init__(self, 
                 config,
                 learning_rate: float = None,
                 min_learning_rate: float = None,
                 warmup_steps: int = None,
                 total_steps: int = None,
                 label_smoothing: float = None,
                 gradient_clip_val: float = None,
                 accumulate_grad_batches: int = None,
                 use_advanced_decoder: bool = True,
                 dropout: float = None):
        super().__init__()
        
        # Store config
        self.config = config
        
        # Use config values with optional overrides
        self.learning_rate = learning_rate or config.training.learning_rate
        self.min_learning_rate = min_learning_rate or config.training.min_learning_rate
        self.warmup_steps = warmup_steps or config.training.warmup_steps
        self.total_steps = total_steps or config.training.total_steps
        self.gradient_clip_val = gradient_clip_val or config.training.gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches or config.training.accumulate_grad_batches
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model components
        self._init_encoder()
        self._init_ctc_components(dropout or config.model.dropout, label_smoothing or config.model.label_smoothing)
        self._init_tokenizer()
        
        # RNNT decoder for hybrid
        self.rnnt_decoder = RNNTDecoder(
            vocab_size=self.config.model.vocab_size,
            enc_dim=self.config.model.n_state,
        )
        self.rnnt_loss_fn = torchaudio.transforms.RNNTLoss(blank=self.config.model.rnnt_blank)
        
        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Timing
        self.step_start_time = None
        
    def _init_encoder(self):
        """Initialize Conformer encoder – pretrained weights are ignored for architectural mismatch"""
        self.encoder = FastConformerEncoder(
            n_mels=self.config.audio.n_mels,
            d_model=self.config.model.n_state,
            n_heads=self.config.model.n_head,
            n_layers=self.config.model.n_layer,
            dropout=self.config.model.dropout,
            ffn_expansion=self.config.model.ffn_expansion,
            left_ctx=self.config.model.left_ctx,
            right_ctx=self.config.model.right_ctx,
        )
        logger.info("🏗️ Fast Conformer encoder initialized")
        
    def _init_ctc_components(self, dropout: float, label_smoothing: float):
        """Initialize CTC head, decoder and loss"""
        self.ctc_head = AdvancedCTCHead(self.config.model.n_state, self.config.model.vocab_size, dropout)
        self.ctc_decoder = AdvancedCTCDecoder(self.config.model.vocab_size, self.config.model.rnnt_blank)
        self.ctc_loss_fn = CTCLossWithLabelSmoothing(
            blank_token=self.config.model.rnnt_blank,
            label_smoothing=label_smoothing
        )
        
    def _init_tokenizer(self):
        """Initialize tokenizer"""
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.config.model.tokenizer_model_path)
        
    def forward(self, x: torch.Tensor, x_len: torch.Tensor, return_intermediate: bool = False):
        """Forward pass returning logits (and optional intermediates)"""
        enc_out, enc_len, intermediates = self.encoder(x, x_len, return_intermediate=return_intermediate)
        logits = self.ctc_head(enc_out)
        if return_intermediate:
            return logits, enc_len, intermediates, enc_out
        return logits, enc_len
        
    def advanced_decoding(self, x: torch.Tensor, x_len: torch.Tensor, use_beam_search: bool = False) -> list:
        """Advanced decoding with multiple strategies"""
        with torch.no_grad():
            logits, enc_len = self.forward(x, x_len, return_intermediate=False)
            log_probs = F.log_softmax(logits, dim=-1)
            
            if use_beam_search:
                decoded_ids_batch = self.ctc_decoder.prefix_beam_search(
                    log_probs, enc_len, beam_size=5, alpha=0.3
                )
            else:
                decoded_ids_batch = self.ctc_decoder.greedy_decode(log_probs, enc_len)
            
            # Decode to text
            decoded_texts = []
            for decoded_ids in decoded_ids_batch:
                try:
                    text = self.tokenizer.decode(decoded_ids)
                    decoded_texts.append(text)
                except Exception as e:
                    logger.warning(f"Decoding error: {e}")
                    decoded_texts.append("")
                    
            return decoded_texts
    
    def compute_wer_and_log_examples(self, predictions: list, targets: list, stage: str, batch_idx: int):
        """Compute WER and log examples"""
        try:
            wer_score = wer(targets, predictions)
            
            # Log examples
            if batch_idx % 1000 == 0 and len(predictions) > 0:
                for i, (pred, true) in enumerate(zip(predictions[:3], targets[:3])):
                    logger.info(f"[{stage}] Example {i+1}:")
                    logger.info(f"  Pred: '{pred}'")
                    logger.info(f"  True: '{true}'")
                    
            return wer_score
        except Exception as e:
            logger.warning(f"WER computation error: {e}")
            return 1.0
    
    def training_step(self, batch, batch_idx):
        """Training step with better logging"""
        if self.step_start_time is None:
            self.step_start_time = time.time()
            
        x, x_len, y, y_len = batch
        
        # Forward pass
        logits, enc_len, intermediates, enc_out = self.forward(x, x_len, return_intermediate=True)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # CTC loss (T, B, V+1) format required
        log_probs_ctc = log_probs.transpose(0, 1)
        
        main_loss = self.ctc_loss_fn(log_probs_ctc, y, enc_len, y_len)
        
        # RNNT loss (no label smoothing)
        rnnt_logits = self.rnnt_decoder(enc_out, y, y_len)
        rnnt_loss = self.rnnt_loss_fn(rnnt_logits, y, enc_len, y_len)
        
        # Auxiliary CTC loss from intermediate representations
        aux_losses = []
        for inter in intermediates:
            aux_logits = self.ctc_head(inter)  # reuse head weights
            aux_log_probs = F.log_softmax(aux_logits, dim=-1).transpose(0, 1)
            aux_losses.append(self.ctc_loss_fn(aux_log_probs, y, enc_len, y_len))

        aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=self.device)

        # hybrid total loss
        loss = (
            self.config.training.lambda_ctc * main_loss
            + (1 - self.config.training.lambda_ctc) * rnnt_loss
            + self.config.training.aux_loss_weight * aux_loss
        )
        
        # Periodic evaluation and logging
        if batch_idx % 2000 == 0:
            # Decode predictions for WER calculation
            predictions = self.advanced_decoding(x, x_len, use_beam_search=False)
            targets = []
            for i, y_i in enumerate(y):
                y_i = y_i.cpu().numpy().astype(int).tolist()
                y_i = y_i[:y_len[i]]
                targets.append(self.tokenizer.decode_ids(y_i))
            
            train_wer = self.compute_wer_and_log_examples(predictions, targets, "TRAIN", batch_idx)
            
            # Log metrics
            self.log("train_wer", train_wer, prog_bar=True, on_step=True, on_epoch=False)
            
        # Always log loss and learning rate
        self.log("train_loss", main_loss, prog_bar=True, on_step=True, on_epoch=False)
        if aux_losses:
            self.log("aux_loss", aux_loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        self.log("rnnt_loss", rnnt_loss, prog_bar=False, on_step=True, on_epoch=False)
        
        # Log training speed
        if batch_idx % 100 == 0 and self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.log("step_time", step_time, on_step=True, on_epoch=False)
            self.step_start_time = time.time()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, x_len, y, y_len = batch
        
        # Compute validation loss
        logits, enc_len = self.forward(x, x_len, return_intermediate=False)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_ctc = log_probs.transpose(0, 1)
        
        val_loss = self.ctc_loss_fn(log_probs_ctc, y, enc_len, y_len)
        
        # Decode predictions (use greedy for faster validation)
        predictions = self.advanced_decoding(x, x_len, use_beam_search=False)  # Use greedy for speed
        targets = []
        for i, y_i in enumerate(y):
            y_i = y_i.cpu().numpy().astype(int).tolist()
            y_i = y_i[:y_len[i]]
            targets.append(self.tokenizer.decode_ids(y_i))
        
        # Compute WER
        val_wer = self.compute_wer_and_log_examples(predictions, targets, "VAL", batch_idx)
        
        # Store for epoch-end aggregation
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'val_wer': val_wer,
            'batch_size': x.size(0)
        })
        
        return {'val_loss': val_loss, 'val_wer': val_wer}
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics"""
        if not self.validation_step_outputs:
            return
            
        # Weighted average by batch size
        total_samples = sum(output['batch_size'] for output in self.validation_step_outputs)
        avg_loss = sum(output['val_loss'] * output['batch_size'] for output in self.validation_step_outputs) / total_samples
        avg_wer = sum(output['val_wer'] * output['batch_size'] for output in self.validation_step_outputs) / total_samples
        
        self.log("val_loss_epoch", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_wer_epoch", avg_wer, prog_bar=True, sync_dist=True)
        
        # Clear for next epoch
        self.validation_step_outputs.clear()
        
    def configure_optimizers(self):
        """Configure optimizers with advanced scheduling"""
        # Optimizer with better defaults
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-6
        )
        
        # One cycle learning rate scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
            pct_start=self.warmup_steps / self.total_steps,
            anneal_strategy='cos',
            final_div_factor=self.learning_rate / self.min_learning_rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    @rank_zero_only
    def on_train_epoch_end(self):
        """Optionally save checkpoint at epoch end if enabled in config"""
        if not self.config.training.save_epoch_checkpoint:
            return  # Skip epoch checkpoints to save disk space

        if hasattr(self.trainer, 'checkpoint_callback'):
            checkpoint_path = f"{self.config.paths.checkpoint_dir}/ctc_epoch_{self.current_epoch}.ckpt"
            self.trainer.save_checkpoint(checkpoint_path, weights_only=True)
            logger.info(f"💾 Epoch checkpoint saved: {checkpoint_path}")

    def on_train_end(self):
        """Save fp16 weights for efficient storage"""
        from pathlib import Path
        fp16_path = Path(self.config.paths.checkpoint_dir) / "final_model_fp16.ckpt"
        half_state = {k: v.half() for k, v in self.state_dict().items()}
        torch.save(half_state, fp16_path)
        logger.info(f"💾 FP16 weights saved to {fp16_path}")


def create_advanced_callbacks(config):
    """Create advanced callbacks for training"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_wer_epoch',
        dirpath=config.paths.checkpoint_dir,
        filename='ctc-step{step}-wer{val_wer_epoch:.4f}',
        save_top_k=1,            # keep only the best model (lowest WER)
        mode='min',
        save_weights_only=True,
        every_n_train_steps=config.training.checkpoint_every_n_steps,
        save_on_train_epoch_end=False,   # rely on step-based saving only
        save_last=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_wer_epoch',
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def main(config=None):
    """Main training function"""
    from config import ExperimentConfig, get_config
    
    # Load config
    config = config or get_config()
    
    # Persist the exact config used for this run
    from pathlib import Path
    config_path = Path(config.paths.checkpoint_dir) / "config.json"
    try:
        config.save(str(config_path))
        logger.info(f"📝 Experiment config saved to {config_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save config to {config_path}: {e}")
    
    # Set up logging
    logger.info("🚀 Starting CTC training...")
    
    # Dataset setup using config with auto train/val split
    from utils.dataset import create_dataset
    
    train_dataset = create_dataset(
        config,
        mode='train',
        augment=True,
        enable_caching=False,
        adaptive_augmentation=True
    )
    
    val_dataset = create_dataset(
        config,
        mode='val',
        augment=False,
        enable_caching=False,
        adaptive_augmentation=False
    )
    
    # Create collate function with config
    collate_fn = create_collate_fn(config)
    
    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded to avoid pickle issues
        persistent_workers=False,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,  # Single-threaded to avoid pickle issues
        persistent_workers=False,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model initialization
    model = StreamingCTC(config)
    
    # Callbacks
    callbacks = create_advanced_callbacks(config)
    
    # Logger setup
    tb_logger = TensorBoardLogger(config.paths.log_dir, name="ctc")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=config.training.precision,
        strategy="auto",
        callbacks=callbacks,
        logger=tb_logger,
        num_sanity_val_steps=config.training.num_sanity_val_steps,
        check_val_every_n_epoch=None,
        val_check_interval=config.training.val_check_interval,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=config.training.log_every_n_steps,
        enable_progress_bar=config.training.enable_progress_bar,
        enable_model_summary=True
    )
    
    # Start training
    logger.info("🎯 Starting training...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    logger.info("✅ Training completed!")


if __name__ == "__main__":
    main() 