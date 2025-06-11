#!/usr/bin/env python3
"""
Simple test without Lightning - pure PyTorch training loop
"""

import torch
import time
from config import get_config
from rnnt_lightning import StreamingRNNT

def simple_training_loop():
    """Simple PyTorch training loop without Lightning"""
    
    print("🚀 Simple PyTorch training loop test...")
    
    # Load config
    config = get_config("config.json")
    config.model.vocab_size = 128
    config.model.rnnt_blank = 128
    
    # Create model
    model = StreamingRNNT(config)
    model = model.cuda() if torch.cuda.is_available() else model
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create tiny batch
    batch_size = 2
    seq_len = 25
    target_len = 3
    
    x = torch.randn(batch_size, config.audio.n_mels, seq_len * 4).cuda()
    x_len = torch.tensor([seq_len * 4, seq_len * 3]).cuda()
    y = torch.randint(0, config.model.vocab_size, (batch_size, target_len)).cuda()
    y_len = torch.tensor([target_len, target_len - 1]).cuda()
    
    batch = (x, x_len, y, y_len)
    
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    
    # Training loop
    for step in range(5):
        print(f"\n📍 Step {step + 1}/5")
        
        start_time = time.time()
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get encoder output
        enc_out, enc_len = model.forward(x, x_len)
        
        # Get RNN-T logits
        logits = model.rnnt_decoder(enc_out, y, y_len)
        
        # Compute loss
        loss = model.rnnt_loss_fn(
            logits,
            y.to(torch.int32),
            enc_len.to(torch.int32),
            y_len.to(torch.int32),
        )
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        step_time = time.time() - start_time
        
        print(f"  ✅ Loss: {loss.item():.4f}, Time: {step_time:.2f}s")
        
        # Check if loss is decreasing
        if step == 0:
            first_loss = loss.item()
        elif step == 4:
            final_loss = loss.item()
            if final_loss < first_loss:
                print(f"  📈 Loss decreased: {first_loss:.4f} → {final_loss:.4f}")
            else:
                print(f"  📊 Loss: {first_loss:.4f} → {final_loss:.4f}")
    
    print("\n🎉 Simple training loop completed successfully!")
    return True

def test_with_lightning():
    """Test with Lightning but fixed logging"""
    
    print("\n🔍 Testing with Lightning (fixed logging)...")
    
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create config
    config = get_config("config.json")
    config.model.vocab_size = 128
    config.model.rnnt_blank = 128
    
    # Create tiny dataset
    n_samples = 5
    seq_len = 25
    target_len = 3
    
    x_data = torch.randn(n_samples, config.audio.n_mels, seq_len * 4)
    x_len_data = torch.full((n_samples,), seq_len * 4)
    y_data = torch.randint(0, config.model.vocab_size, (n_samples, target_len))
    y_len_data = torch.full((n_samples,), target_len)
    
    dataset = TensorDataset(x_data, x_len_data, y_data, y_len_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create model
    model = StreamingRNNT(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=3,
        devices=1,
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    
    try:
        print("Starting Lightning training...")
        start_time = time.time()
        
        trainer.fit(model, dataloader)
        
        total_time = time.time() - start_time
        print(f"✅ Lightning training completed in {total_time:.2f}s!")
        return True
        
    except Exception as e:
        print(f"❌ Lightning training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing RNN-T training (fixed logging)...")
    
    # Test simple PyTorch loop
    simple_ok = simple_training_loop()
    
    if simple_ok:
        # Test Lightning
        lightning_ok = test_with_lightning()
        
        if lightning_ok:
            print("\n🎉 Both tests passed! Training should work now.")
            print("\nTry running:")
            print("python run.py --config config.json --fast-dev-run")
        else:
            print("\n⚠️ Lightning still has issues, but pure PyTorch works")
    else:
        print("\n❌ Basic training failed")
