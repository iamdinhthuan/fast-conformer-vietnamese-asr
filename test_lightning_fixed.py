#!/usr/bin/env python3
"""
Test Lightning with RNN mode fixes
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from config import get_config
from rnnt_lightning import StreamingRNNT

def test_lightning_fixed():
    """Test Lightning with RNN mode fixes"""
    
    print("🔧 Testing Lightning with RNN mode fixes...")
    
    # Create config with small vocab
    config = get_config("config.json")
    config.model.vocab_size = 128
    config.model.rnnt_blank = 128
    
    # Create tiny dataset
    n_samples = 10
    seq_len = 25
    target_len = 3
    
    x_data = torch.randn(n_samples, config.audio.n_mels, seq_len * 4)
    x_len_data = torch.full((n_samples,), seq_len * 4)
    y_data = torch.randint(0, config.model.vocab_size, (n_samples, target_len))
    y_len_data = torch.full((n_samples,), target_len)
    
    dataset = TensorDataset(x_data, x_len_data, y_data, y_len_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Create model
    model = StreamingRNNT(config)
    
    # Explicitly set training mode
    model.train()
    model.rnnt_decoder.train()
    
    print(f"Model training mode: {model.training}")
    print(f"RNN decoder training mode: {model.rnnt_decoder.training}")
    print(f"Pred RNN training mode: {model.rnnt_decoder.pred_rnn.training}")
    
    # Create trainer with minimal setup
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=5,
        devices=1,
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
    )
    
    try:
        print("Starting Lightning training...")
        trainer.fit(model, dataloader)
        print("✅ Lightning training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Lightning training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_lightning_fixed()
