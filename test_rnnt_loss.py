#!/usr/bin/env python3
"""
Test different RNN-T loss implementations
"""

import torch
import time
import torchaudio

def test_rnnt_loss_basic():
    """Test basic RNN-T loss with minimal data"""
    
    print("🧪 Testing RNN-T loss with minimal data...")
    
    # Minimal dimensions
    B, T, U, V = 1, 10, 3, 50  # Very small
    
    # Create test data
    logits = torch.randn(B, T, U, V, requires_grad=True).cuda()
    targets = torch.randint(0, V-1, (B, U-1)).cuda()  # U-1 because no blank in targets
    logit_lengths = torch.tensor([T]).cuda()
    target_lengths = torch.tensor([U-1]).cuda()
    
    print(f"Shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  logit_lengths: {logit_lengths}")
    print(f"  target_lengths: {target_lengths}")
    
    # Test RNN-T loss
    rnnt_loss = torchaudio.transforms.RNNTLoss(blank=V-1)
    
    try:
        print("Computing RNN-T loss...")
        start_time = time.time()
        
        loss = rnnt_loss(
            logits,
            targets.to(torch.int32),
            logit_lengths.to(torch.int32),
            target_lengths.to(torch.int32)
        )
        
        elapsed = time.time() - start_time
        print(f"✅ RNN-T loss: {loss.item():.4f} (computed in {elapsed:.3f}s)")
        
        # Test backward
        print("Testing backward pass...")
        start_time = time.time()
        loss.backward()
        elapsed = time.time() - start_time
        print(f"✅ Backward pass completed in {elapsed:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ RNN-T loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rnnt_loss_scaling():
    """Test RNN-T loss with different sizes"""
    
    print("\n📊 Testing RNN-T loss scaling...")
    
    sizes = [
        (1, 5, 2, 10),    # Tiny
        (1, 10, 3, 50),   # Small
        (1, 20, 5, 100),  # Medium
        (2, 50, 6, 129),  # Our actual size
    ]
    
    for B, T, U, V in sizes:
        print(f"\n🧪 Testing size: B={B}, T={T}, U={U}, V={V}")
        
        logits = torch.randn(B, T, U, V).cuda()
        targets = torch.randint(0, V-1, (B, U-1)).cuda()
        logit_lengths = torch.full((B,), T).cuda()
        target_lengths = torch.full((B,), U-1).cuda()
        
        rnnt_loss = torchaudio.transforms.RNNTLoss(blank=V-1)
        
        try:
            start_time = time.time()
            loss = rnnt_loss(
                logits,
                targets.to(torch.int32),
                logit_lengths.to(torch.int32),
                target_lengths.to(torch.int32)
            )
            elapsed = time.time() - start_time
            
            print(f"  ✅ Loss: {loss.item():.4f}, Time: {elapsed:.3f}s")
            
            if elapsed > 5.0:
                print(f"  ⚠️ Very slow! ({elapsed:.1f}s)")
                break
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            break

def test_alternative_loss():
    """Test alternative loss for comparison"""
    
    print("\n🔄 Testing alternative loss (CTC-like)...")
    
    B, T, V = 2, 50, 129
    
    # Simulate encoder output
    logits = torch.randn(B, T, V).cuda()
    targets = torch.randint(0, V-1, (B, 6)).cuda()
    input_lengths = torch.tensor([T, T-10]).cuda()
    target_lengths = torch.tensor([6, 5]).cuda()
    
    # CTC loss for comparison
    ctc_loss = torch.nn.CTCLoss(blank=V-1)
    
    try:
        start_time = time.time()
        loss = ctc_loss(
            torch.log_softmax(logits, dim=-1).transpose(0, 1),  # (T, B, V)
            targets,
            input_lengths,
            target_lengths
        )
        elapsed = time.time() - start_time
        
        print(f"✅ CTC loss: {loss.item():.4f}, Time: {elapsed:.3f}s")
        print("💡 CTC loss is much faster - consider as fallback")
        
    except Exception as e:
        print(f"❌ CTC loss failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing RNN-T loss implementations...")
    
    # Test basic functionality
    basic_ok = test_rnnt_loss_basic()
    
    if basic_ok:
        # Test scaling
        test_rnnt_loss_scaling()
    
    # Test alternative
    test_alternative_loss()
    
    print("\n💡 If RNN-T loss is consistently slow/stuck:")
    print("1. Use smaller vocab size (< 1000)")
    print("2. Use shorter sequences")
    print("3. Consider CTC loss as fallback")
    print("4. Check torchaudio version compatibility")
