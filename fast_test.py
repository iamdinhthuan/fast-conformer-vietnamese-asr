#!/usr/bin/env python3
"""
Ultra-fast test with minimal vocab size
"""

import torch
import time
from config import get_config
from rnnt_lightning import StreamingRNNT

def ultra_fast_test():
    """Ultra fast test with tiny vocab"""
    
    print("⚡ Ultra-fast RNN-T test with minimal vocab...")
    
    # Load config
    config = get_config("config.json")
    
    # Override with tiny vocab for testing
    config.model.vocab_size = 128  # Very small vocab
    config.model.rnnt_blank = 128
    
    print(f"Using vocab_size: {config.model.vocab_size}")
    
    # Create model
    model = StreamingRNNT(config)
    model = model.cuda() if torch.cuda.is_available() else model
    
    # Create tiny batch
    batch_size = 2
    seq_len = 50  # Very short sequence
    target_len = 5  # Very short target
    
    # Dummy data
    x = torch.randn(batch_size, config.audio.n_mels, seq_len * 4).cuda()
    x_len = torch.tensor([seq_len * 4, seq_len * 3]).cuda()
    y = torch.randint(0, config.model.vocab_size, (batch_size, target_len)).cuda()
    y_len = torch.tensor([target_len, target_len - 1]).cuda()
    
    batch = (x, x_len, y, y_len)
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")
    print(f"  Expected RNN-T tensor size: ~{batch_size * seq_len * (target_len + 1) * (config.model.vocab_size + 1):,}")
    
    # Test forward pass
    start_time = time.time()
    
    try:
        with torch.no_grad():
            loss = model.training_step(batch, 0)
            
        forward_time = time.time() - start_time
        
        print(f"✅ Forward pass successful!")
        print(f"⏱️ Time: {forward_time:.2f}s")
        print(f"📊 Loss: {loss:.4f}")
        
        # Test multiple iterations
        print("\n🔄 Testing multiple iterations...")
        times = []
        for i in range(3):
            start = time.time()
            with torch.no_grad():
                loss = model.training_step(batch, i)
            times.append(time.time() - start)
            print(f"  Iteration {i+1}: {times[-1]:.2f}s, Loss: {loss:.4f}")
        
        avg_time = sum(times) / len(times)
        print(f"📈 Average time: {avg_time:.2f}s")
        
        # Estimate for larger vocab
        large_vocab_factor = (14500 + 1) / (config.model.vocab_size + 1)
        estimated_time = avg_time * large_vocab_factor
        print(f"📊 Estimated time with vocab_size=14500: {estimated_time:.2f}s")
        
        if estimated_time > 10:
            print("⚠️ Large vocab will be very slow!")
            print("💡 Consider using smaller vocab or gradient checkpointing")
        else:
            print("✅ Large vocab should be manageable")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def benchmark_vocab_sizes():
    """Benchmark different vocab sizes"""
    
    print("\n📊 Benchmarking different vocab sizes...")
    
    vocab_sizes = [128, 512, 1024, 2048]
    
    for vocab_size in vocab_sizes:
        print(f"\n🧪 Testing vocab_size = {vocab_size}")
        
        config = get_config("config.json")
        config.model.vocab_size = vocab_size
        config.model.rnnt_blank = vocab_size
        
        model = StreamingRNNT(config)
        model = model.cuda() if torch.cuda.is_available() else model
        
        # Small batch
        batch_size = 1
        seq_len = 25
        target_len = 3
        
        x = torch.randn(batch_size, config.audio.n_mels, seq_len * 4).cuda()
        x_len = torch.tensor([seq_len * 4]).cuda()
        y = torch.randint(0, vocab_size, (batch_size, target_len)).cuda()
        y_len = torch.tensor([target_len]).cuda()
        
        batch = (x, x_len, y, y_len)
        
        try:
            start_time = time.time()
            with torch.no_grad():
                loss = model.training_step(batch, 0)
            elapsed = time.time() - start_time
            
            print(f"  ✅ Time: {elapsed:.2f}s, Loss: {loss:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

if __name__ == "__main__":
    ultra_fast_test()
    benchmark_vocab_sizes()
