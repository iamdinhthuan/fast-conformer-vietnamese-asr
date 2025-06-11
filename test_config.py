#!/usr/bin/env python3
"""
Test script to verify config loading works correctly after RNN-T migration
"""

import json
from pathlib import Path

def test_config_loading():
    """Test that config.json can be loaded without errors"""
    try:
        # Test JSON parsing
        with open('config.json', 'r') as f:
            config_dict = json.load(f)
        
        print("✅ JSON parsing successful")
        
        # Check required fields
        required_sections = ['model', 'training', 'inference', 'paths']
        for section in required_sections:
            if section not in config_dict:
                print(f"❌ Missing section: {section}")
                return False
            print(f"✅ Found section: {section}")
        
        # Check model section
        model_config = config_dict['model']
        required_model_fields = ['vocab_size', 'rnnt_blank', 'n_state', 'n_head', 'n_layer']
        for field in required_model_fields:
            if field not in model_config:
                print(f"❌ Missing model field: {field}")
                return False
            print(f"✅ Found model field: {field} = {model_config[field]}")
        
        # Check that deprecated fields are removed
        deprecated_fields = ['ctc_blank', 'label_smoothing']
        for field in deprecated_fields:
            if field in model_config:
                print(f"⚠️ Found deprecated field in model: {field}")
            else:
                print(f"✅ Deprecated field removed: {field}")
        
        # Check training section
        training_config = config_dict['training']
        if 'early_stopping_patience' in training_config:
            print(f"✅ Found early_stopping_patience: {training_config['early_stopping_patience']}")
        
        # Check inference section
        inference_config = config_dict['inference']
        required_inference_fields = ['use_streaming', 'chunk_size_ms', 'overlap_ms']
        for field in required_inference_fields:
            if field not in inference_config:
                print(f"❌ Missing inference field: {field}")
                return False
            print(f"✅ Found inference field: {field} = {inference_config[field]}")
        
        print("\n🎉 All config tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        'rnnt_lightning.py',
        'models/rnnt_streaming.py',
        'models/rnnt_decoder.py',
        'models/fast_conformer.py',
        'config.py',
        'run.py',
        'streaming_inference.py',
        'inference.py'
    ]
    
    print("\n📁 Checking file structure...")
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("🧪 Testing RNN-T migration...")
    
    config_ok = test_config_loading()
    files_ok = test_file_structure()
    
    if config_ok and files_ok:
        print("\n🎉 Migration test successful! Ready to train RNN-T model.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start training: python run.py --config config.json")
        print("3. For streaming inference: python streaming_inference.py --audio audio.wav --checkpoint model.ckpt")
    else:
        print("\n❌ Migration test failed. Please check the errors above.")
