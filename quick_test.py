#!/usr/bin/env python3
"""
Quick test script with minimal batch size for RNN-T
"""

import torch
import time
from config import get_config
from rnnt_lightning import StreamingRNNT

def quick_test():
    """Quick test with small batch"""
    
    print("🚀 Quick RNN-T test with minimal batch size...")
    
    # Load config and override batch size
    config = get_config("config.json")
    
    # Create model
    model = StreamingRNNT(config)
    model = model.cuda() if torch.cuda.is_available() else model
    
    # Create dummy batch (very small)
    batch_size = 2
    seq_len = 100  # Short sequence
    target_len = 10  # Short target
    
    # Dummy data
    x = torch.randn(batch_size, config.audio.n_mels, seq_len * 4).cuda()  # Raw audio features
    x_len = torch.tensor([seq_len * 4, seq_len * 3]).cuda()
    y = torch.randint(0, config.model.vocab_size, (batch_size, target_len)).cuda()
    y_len = torch.tensor([target_len, target_len - 2]).cuda()
    
    batch = (x, x_len, y, y_len)
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  x_len: {x_len.shape}")
    print(f"  y: {y.shape}")
    print(f"  y_len: {y_len.shape}")
    
    # Test forward pass
    start_time = time.time()
    
    try:
        with torch.no_grad():
            loss = model.training_step(batch, 0)
            
        forward_time = time.time() - start_time
        
        print(f"✅ Forward pass successful!")
        print(f"⏱️ Time: {forward_time:.2f}s")
        print(f"📊 Loss: {loss:.4f}")
        
        # Estimate time for full batch
        full_batch_time = forward_time * (64 / batch_size)
        print(f"📈 Estimated time for batch_size=64: {full_batch_time:.2f}s")
        
        if full_batch_time > 30:
            print("⚠️ Full batch would be very slow. Recommend batch_size=8 or smaller.")
        else:
            print("✅ Full batch should be reasonable.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    quick_test()
