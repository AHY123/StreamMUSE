#!/usr/bin/env python3
"""
Test script to verify tokenization in discriminative model matches main model.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.discriminative_model import DiscriminativeRewardModel, DURATION_TEMPLATES

def test_tokenization():
    """Test that tokenization produces valid token ranges"""
    
    print("Testing discriminative model tokenization...")
    print(f"Duration templates: {DURATION_TEMPLATES}")
    print(f"Number of duration templates: {len(DURATION_TEMPLATES)}")
    
    # Create model
    model = DiscriminativeRewardModel(hidden_size=512, num_layers=6, num_heads=8, vocab_size=3205)
    
    # Create test data [batch_size=2, seq_len=4, 12] 
    # Format: [program, pitch, duration_index] for 4 notes per frame
    batch_size, seq_len = 2, 4
    
    test_data = torch.zeros(batch_size, seq_len, 12, dtype=torch.long)
    
    # Fill with valid test values
    for b in range(batch_size):
        for s in range(seq_len):
            for note in range(4):  # 4 notes per frame
                idx = note * 3
                test_data[b, s, idx] = 1      # program (0=melody, 1=acc)  
                test_data[b, s, idx+1] = 60 + note  # pitch (60-63)
                test_data[b, s, idx+2] = note % len(DURATION_TEMPLATES)  # duration_index (0-23)
    
    # Test with various pitch shifts
    for pitch_shift in [0, 3, -2, 6]:
        print(f"\nTesting with pitch_shift = {pitch_shift}")
        
        pitch_shift_tensor = torch.tensor([pitch_shift, pitch_shift], dtype=torch.long)
        
        # Preprocess
        processed = model.preprocess(test_data, pitch_shift_tensor)
        print(f"  Input shape: {test_data.shape}")
        print(f"  Processed shape: {processed.shape}")
        
        # Check token ranges
        min_token = processed.min().item()
        max_token = processed.max().item()
        print(f"  Token range: [{min_token}, {max_token}]")
        
        # Verify all tokens are in valid range
        valid_tokens = (processed >= 0) & (processed < model.vocab_size)
        if valid_tokens.all():
            print(f"  ✓ All tokens in valid range [0, {model.vocab_size-1}]")
        else:
            invalid_count = (~valid_tokens).sum().item()
            print(f"  ✗ {invalid_count} tokens out of range!")
            
        # Calculate expected max token for verification
        max_pitch = 127
        max_duration_idx = len(DURATION_TEMPLATES) - 1  # 23
        expected_max = max_pitch + max_duration_idx * 128 + 2 + abs(pitch_shift)
        print(f"  Expected max token: {expected_max}")
    
    print(f"\nVocabulary size: {model.vocab_size}")
    print(f"Special tokens: SOS={model.SOS_TOKEN}, EOS={model.EOS_TOKEN}, PAD={model.PAD_TOKEN}")
    
    # Test with pad and eos tokens
    print(f"\nTesting special token handling...")
    special_data = test_data.clone()
    special_data[0, 0, 1] = 255  # pitch=255 -> should become PAD
    special_data[0, 1, 0] = 254  # program=254 -> should become EOS
    
    processed_special = model.preprocess(special_data, torch.zeros(batch_size, dtype=torch.long))
    
    # Check if special tokens are applied correctly
    if (processed_special[0, 0, 1] == model.PAD_TOKEN).item():
        print("  ✓ PAD token correctly applied")
    else:
        print("  ✗ PAD token not applied correctly")
        
    if (processed_special[0, 1, 0] == model.EOS_TOKEN).item():
        print("  ✓ EOS token correctly applied") 
    else:
        print("  ✗ EOS token not applied correctly")

if __name__ == "__main__":
    test_tokenization()