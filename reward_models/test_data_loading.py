#!/usr/bin/env python3
"""
Quick test of data loading for discriminative model.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.data_loader import DiscriminativeDataset

def test_data_loading():
    """Test basic data loading functionality"""
    
    print("Testing data loading...")
    
    melody_path = "data/reward_training_mel_cp4.pt"
    acc_path = "data/reward_training_acc_cp4.pt"
    target_length = 32
    
    try:
        # Create dataset
        dataset = DiscriminativeDataset(melody_path, acc_path, target_length)
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test first few samples
        for i in range(min(3, len(dataset))):
            print(f"\nTesting sample {i}:")
            sample = dataset[i]
            
            sequence = sample['sequence']
            label = sample['label']
            pitch_shift = sample['pitch_shift']
            
            print(f"  Sequence shape: {sequence.shape}")
            print(f"  Label: {label.item()}")
            print(f"  Pitch shift: {pitch_shift}")
            print(f"  Sequence token range: [{sequence.min().item()}, {sequence.max().item()}]")
            
            # Check for invalid tokens (should be program, pitch, duration_index format)
            reshaped = sequence.view(-1, 3)  # [total_notes, 3]
            programs = reshaped[:, 0]
            pitches = reshaped[:, 1] 
            durations = reshaped[:, 2]
            
            print(f"  Program range: [{programs.min().item()}, {programs.max().item()}]")
            print(f"  Pitch range: [{pitches.min().item()}, {pitches.max().item()}]")
            print(f"  Duration range: [{durations.min().item()}, {durations.max().item()}]")
            
            if i >= 2:  # Limit to first 3 samples
                break
        
        print("\n✓ Data loading test completed successfully!")
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()