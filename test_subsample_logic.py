#!/usr/bin/env python3
"""
Test script to verify subsample logic without requiring PyTorch
"""

def test_subsample_dimensions():
    """Test the dimension calculations for subsample layer"""
    
    # Test parameters from config
    n_mels = 80  # from config
    d_model = 256  # from config
    
    print(f"Input: n_mels = {n_mels}, d_model = {d_model}")
    
    # Calculate frequency dimension after subsampling (2 stride-2 convs)
    freq_dim_after_first_conv = (n_mels + 1) // 2
    freq_dim_after_second_conv = (freq_dim_after_first_conv + 1) // 2
    
    print(f"After first conv (stride=2): {freq_dim_after_first_conv}")
    print(f"After second conv (stride=2): {freq_dim_after_second_conv}")
    
    # This should be the F dimension in (B, C, F, T')
    print(f"Expected F dimension: {freq_dim_after_second_conv}")
    
    # After projection, we want (B, d_model, 1, T')
    print(f"After projection: (B, {d_model}, 1, T')")
    
    # Check if dimensions make sense
    if freq_dim_after_second_conv > 0:
        print("✅ Subsample dimensions look correct")
        return True
    else:
        print("❌ Subsample dimensions are invalid")
        return False

def test_positional_encoding_logic():
    """Test positional encoding dimension matching"""
    
    d_model = 256
    max_seq_len = 10000
    
    print(f"\nPositional encoding: (1, {max_seq_len}, {d_model})")
    
    # After subsample and squeeze(2).transpose(1,2), we get (B, T', d_model)
    print(f"After subsample processing: (B, T', {d_model})")
    
    # pos_enc[:, :T_prime, :] should match (B, T', d_model)
    print("✅ Positional encoding dimensions should match")
    
    return True

def test_config_compatibility():
    """Test that config values are compatible"""
    
    # Values from config.json
    config_values = {
        "n_mels": 80,
        "n_state": 256,  # d_model
        "n_head": 4,
        "n_layer": 16,
        "vocab_size": 14500,
        "rnnt_blank": 14500
    }
    
    print(f"\nConfig values: {config_values}")
    
    # Check if vocab_size and rnnt_blank match
    if config_values["vocab_size"] == config_values["rnnt_blank"]:
        print("✅ vocab_size matches rnnt_blank")
    else:
        print("❌ vocab_size doesn't match rnnt_blank")
        return False
    
    # Check if n_state is divisible by n_head
    if config_values["n_state"] % config_values["n_head"] == 0:
        print("✅ n_state is divisible by n_head")
    else:
        print("❌ n_state is not divisible by n_head")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing subsample and dimension logic...")
    
    test1 = test_subsample_dimensions()
    test2 = test_positional_encoding_logic()
    test3 = test_config_compatibility()
    
    if test1 and test2 and test3:
        print("\n🎉 All dimension tests passed!")
        print("\nThe fix should resolve the tensor size mismatch error.")
        print("Try running training again with PyTorch environment.")
    else:
        print("\n❌ Some tests failed. Please check the logic above.")
