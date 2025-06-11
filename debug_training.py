#!/usr/bin/env python3
"""
Debug training step by step with timeout
"""

import torch
import time
import signal
from config import get_config
from rnnt_lightning import StreamingRNNT

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def debug_training_step():
    """Debug each part of training step"""
    
    print("🔍 Debugging training step by step...")
    
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
    batch_size = 1
    seq_len = 25
    target_len = 3
    
    x = torch.randn(batch_size, config.audio.n_mels, seq_len * 4).cuda()
    x_len = torch.tensor([seq_len * 4]).cuda()
    y = torch.randint(0, config.model.vocab_size, (batch_size, target_len)).cuda()
    y_len = torch.tensor([target_len]).cuda()
    
    batch = (x, x_len, y, y_len)
    
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    
    # Step 1: Forward pass
    print("\n1️⃣ Testing forward pass...")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        start_time = time.time()
        loss = model.training_step(batch, 0)
        forward_time = time.time() - start_time
        
        signal.alarm(0)  # Cancel timeout
        print(f"✅ Forward pass: {forward_time:.2f}s, Loss: {loss.item():.4f}")
        
    except TimeoutError:
        print("❌ Forward pass timed out!")
        return False
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Step 2: Backward pass
    print("\n2️⃣ Testing backward pass...")
    try:
        signal.alarm(10)  # 10 second timeout
        
        start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time = time.time() - start_time
        
        signal.alarm(0)
        print(f"✅ Backward pass: {backward_time:.2f}s")
        
    except TimeoutError:
        print("❌ Backward pass timed out!")
        return False
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    # Step 3: Optimizer step
    print("\n3️⃣ Testing optimizer step...")
    try:
        signal.alarm(10)  # 10 second timeout
        
        start_time = time.time()
        optimizer.step()
        optimizer_time = time.time() - start_time
        
        signal.alarm(0)
        print(f"✅ Optimizer step: {optimizer_time:.2f}s")
        
    except TimeoutError:
        print("❌ Optimizer step timed out!")
        return False
    except Exception as e:
        print(f"❌ Optimizer step failed: {e}")
        return False
    
    # Step 4: Multiple iterations
    print("\n4️⃣ Testing multiple iterations...")
    try:
        for i in range(3):
            print(f"  Iteration {i+1}...")
            
            signal.alarm(15)  # 15 second timeout per iteration
            
            start_time = time.time()
            
            optimizer.zero_grad()
            loss = model.training_step(batch, i+1)
            loss.backward()
            optimizer.step()
            
            iteration_time = time.time() - start_time
            signal.alarm(0)
            
            print(f"    ✅ Completed in {iteration_time:.2f}s, Loss: {loss.item():.4f}")
            
    except TimeoutError:
        print(f"❌ Iteration {i+1} timed out!")
        return False
    except Exception as e:
        print(f"❌ Iteration {i+1} failed: {e}")
        return False
    
    print("\n🎉 All training steps completed successfully!")
    return True

def test_lightning_trainer():
    """Test with Lightning trainer but minimal setup"""
    
    print("\n🔍 Testing with Lightning trainer...")
    
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create minimal dataset
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create model
    model = StreamingRNNT(config)
    
    # Create trainer with minimal setup
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=3,  # Only 3 steps
        devices=1,
        accelerator="auto",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    try:
        print("Starting Lightning training...")
        trainer.fit(model, dataloader)
        print("✅ Lightning training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Lightning training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Debugging RNN-T training issues...")
    
    # Test manual training step
    manual_ok = debug_training_step()
    
    if manual_ok:
        # Test Lightning trainer
        test_lightning_trainer()
    else:
        print("\n❌ Manual training failed - Lightning will likely fail too")
        
    print("\n💡 If training is still slow:")
    print("1. The issue might be in Lightning's internal loops")
    print("2. Try reducing model size further")
    print("3. Check GPU memory usage: nvidia-smi")
    print("4. Consider using CPU for debugging: CUDA_VISIBLE_DEVICES=''")
