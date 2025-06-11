#!/usr/bin/env python3
"""
Improved version of run.py with all enhancements
Use this instead of the original run.py for better performance and features
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from loguru import logger

# Import components
from config import ExperimentConfig, get_config
from train import StreamingCTC, create_advanced_callbacks
from utils.dataset import create_dataset, create_collate_fn


def setup_logging(log_dir: str):
    """Setup comprehensive logging"""
    log_path = Path(log_dir) / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path / "training.log",
        rotation="50 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.info("🚀 Starting Whisper ASR training")


def create_data_loaders(config: ExperimentConfig):
    """Create optimized data loaders"""
    logger.info("📊 Creating datasets with auto train/val split...")
    
    # Training dataset
    train_dataset = create_dataset(
        config,
        mode='train',
        augment=True,
        enable_caching=False,  # Disable caching to avoid pickle issues
        adaptive_augmentation=True
    )
    
    # Validation dataset (auto-split from same metadata file)
    val_dataset = create_dataset(
        config,
        mode='val',
        augment=False,  # No augmentation for validation
        enable_caching=False,  # Disable caching to avoid pickle issues
        adaptive_augmentation=False
    )
    
    logger.info(f"📊 Dataset split completed: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create optimized data loaders
    # Use single-threaded loading to avoid pickle issues with AudioCache on Windows
    num_workers = 0  # Force single-threaded to avoid pickle issues
    
    # Create collate function with config
    collate_fn = create_collate_fn(config)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=16,
        persistent_workers=True,  # Disable for single-threaded
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # For stable training
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=16,
        persistent_workers=True,  # Disable for single-threaded
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader


def create_model(config: ExperimentConfig):
    """Create model"""
    logger.info("🏗️ Initializing model...")
    
    model = StreamingCTC(config)
    
    logger.info("✅ Model created successfully")
    return model


def create_trainer(config: ExperimentConfig):
    """Create optimized trainer"""
    logger.info("⚙️ Setting up advanced trainer...")
    
    # Create callbacks
    callbacks = create_advanced_callbacks(config)
    
    # Setup loggers
    from pytorch_lightning.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(
        config.paths.log_dir,
        name="ctc",
        version=f"v{config.version}"
    )
    
    # Optional: WandB logger
    loggers = [tb_logger]
    if config.paths.wandb_project:
        try:
            from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project=config.paths.wandb_project,
                name=f"{config.name}_v{config.version}",
                config=config.to_dict()
            )
            loggers.append(wandb_logger)
            logger.info("📊 WandB logging enabled")
        except ImportError:
            logger.warning("⚠️ WandB not available, skipping...")
    
    # Create trainer with optimizations
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        devices=1,
        accelerator="auto",
        precision=config.training.precision,
        strategy="auto",
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=config.training.num_sanity_val_steps,
        check_val_every_n_epoch=None,
        val_check_interval=config.training.val_check_interval,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        log_every_n_steps=config.training.log_every_n_steps,
        enable_progress_bar=config.training.enable_progress_bar,
        enable_model_summary=True,
        deterministic=config.deterministic
    )
    
    logger.info("✅ Trainer configured successfully")
    return trainer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Whisper ASR Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="Test setup without training")
    parser.add_argument("--fast-dev-run", action="store_true", help="Fast development run")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    # Override config parameters
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--max-epochs", type=int, help="Override max epochs")
    parser.add_argument("--devices", type=int, help="Number of devices")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Apply command line overrides
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
    
    # Setup logging
    setup_logging(config.paths.log_dir)
    
    # Persist the exact config for this run
    config_path = Path(config.paths.checkpoint_dir) / "config.json"
    try:
        config.save(str(config_path))
        logger.info(f"📝 Experiment config saved to {config_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save config to {config_path}: {e}")
    
    # Log configuration
    logger.info("📋 Configuration:")
    logger.info(f"  Model: {config.model.n_state}d-{config.model.n_head}h-{config.model.n_layer}l")
    logger.info(f"  Training: LR={config.training.learning_rate}, Batch={config.training.batch_size}")
    logger.info(f"  Data: {config.data.metadata_file} (split {config.data.train_val_split:.0%}:{100-config.data.train_val_split*100:.0f}%)")
    
    # Set seed for reproducibility
    if config.seed:
        pl.seed_everything(config.seed, workers=True)
        logger.info(f"🎲 Seed set to {config.seed}")
    
    try:
        # Create components
        train_dataloader, val_dataloader = create_data_loaders(config)
        model = create_model(config)
        trainer = create_trainer(config)
        
        # Fast development run
        if args.fast_dev_run:
            trainer.fast_dev_run = True
            logger.info("🏃‍♂️ Fast development run enabled")
        
        # Enable profiling
        if args.profile:
            trainer.profiler = "simple"
            logger.info("📊 Profiling enabled")
        
        # Test setup
        if args.test:
            logger.info("🧪 Testing setup...")
            
            # Test data loading
            batch = next(iter(train_dataloader))
            logger.info(f"✅ Data loading works: batch shapes {[x.shape for x in batch]}")
            
            # Test model forward pass
            model.eval()
            with torch.no_grad():
                loss = model.training_step(batch, 0)
                logger.info(f"✅ Model forward pass works: loss = {loss:.4f}")
            
            logger.info("🎉 Setup test completed successfully!")
            return
        
        # Start training
        logger.info("🚀 Starting training...")
        logger.info(f"📊 Monitoring: tensorboard --logdir {config.paths.log_dir}")
        
        # Fit model
        trainer.fit(
            model, 
            train_dataloader, 
            val_dataloader,
            ckpt_path=args.resume
        )
        
        logger.info("🎉 Training completed successfully!")
        
        # Save final model
        final_model_path = Path(config.paths.checkpoint_dir) / "final_model.ckpt"
        trainer.save_checkpoint(final_model_path, weights_only=True)
        logger.info(f"💾 Final model saved to {final_model_path}")
        
        # additionally save fp16 version (~50% size)
        half_state = {k: v.half() for k, v in model.state_dict().items()}
        torch.save(half_state, Path(config.paths.checkpoint_dir) / "final_model_fp16.ckpt")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    finally:
        logger.info("🏁 Training session ended")


if __name__ == "__main__":
    main() 