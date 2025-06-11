#!/usr/bin/env python3
"""
Final test with LSTMCell implementation
"""

import torch
import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from config import get_config
from rnnt_lightning import StreamingRNNT

def test_lstmcell_implementation():
    """Test LSTMCell implementation"""
    
    print("🔧 Testing LSTMCell implementation...")
    
    # Create config with small vocab
    config = get_config("config.json")
    config.model.vocab_size = 128
    config.model.rnnt_blank = 128
    
    # Create model
    model = StreamingRNNT(config)
    model = model.cuda() if torch.cuda.is_available() else model
    model.train()
    
    # Create tiny batch
    batch_size = 2
    seq_len = 25
    target_len = 3
    
    x = torch.randn(batch_size, config.audio.n_mels, seq_len * 4)
    x_len = torch.tensor([seq_len * 4, seq_len * 3])
    y = torch.randint(0, config.model.vocab_size, (batch_size, target_len))
    y_len = torch.tensor([target_len, target_len - 1])
    
    if torch.cuda.is_available():
        x, x_len, y, y_len = x.cuda(), x_len.cuda(), y.cuda(), y_len.cuda()
    
    batch = (x, x_len, y, y_len)
    
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    print(f"Model has LSTMCell: {hasattr(model.rnnt_decoder, 'pred_rnn_cell')}")
    
    # Test forward pass
    try:
        start_time = time.time()
        
        # Test encoder
        enc_out, enc_len = model.forward(x, x_len)
        print(f"✅ Encoder: {enc_out.shape}")
        
        # Test RNN-T decoder
        logits = model.rnnt_decoder(enc_out, y, y_len)
        print(f"✅ RNN-T decoder: {logits.shape}")
        
        # Test loss
        loss = model.rnnt_loss_fn(
            logits,
            y.to(torch.int32),
            enc_len.to(torch.int32),
            y_len.to(torch.int32),
        )
        print(f"✅ Loss: {loss.item():.4f}")
        
        # Test backward
        loss.backward()
        print(f"✅ Backward pass completed")
        
        elapsed = time.time() - start_time
        print(f"⏱️ Total time: {elapsed:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lightning_final():
    """Final Lightning test"""
    
    print("\n🚀 Final Lightning test...")
    
    # Create config
    config = get_config("config.json")
    config.model.vocab_size = 128
    config.model.rnnt_blank = 128
    
    # Create tiny dataset
    n_samples = 8
    seq_len = 20
    target_len = 3
    
    x_data = torch.randn(n_samples, config.audio.n_mels, seq_len * 4)
    x_len_data = torch.full((n_samples,), seq_len * 4)
    y_data = torch.randint(0, config.model.vocab_size, (n_samples, target_len))
    y_len_data = torch.full((n_samples,), target_len)
    
    dataset = TensorDataset(x_data, x_len_data, y_data, y_len_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
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
        gradient_clip_val=1.0,
        precision="16-mixed" if torch.cuda.is_available() else "32",
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
    print("🧪 Final RNN-T test with LSTMCell...")
    
    # Test basic functionality
    basic_ok = test_lstmcell_implementation()
    
    if basic_ok:
        # Test Lightning
        lightning_ok = test_lightning_final()
        
        if lightning_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("\n✅ RNN-T model is ready for training!")
            print("\nRun full training with:")
            print("python run.py --config config.json --fast-dev-run")
            print("python run.py --config config.json")
        else:
            print("\n⚠️ Lightning test failed, but basic functionality works")
    else:
        print("\n❌ Basic test failed")
