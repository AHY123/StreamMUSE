#!/usr/bin/env python3
"""
Minimal test of discriminative model training.
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.discriminative_model import DiscriminativeRewardModel
from reward_models.data_loader import DiscriminativeDataset

def test_forward_pass():
    """Test forward pass and loss calculation"""
    
    print("Testing discriminative model forward pass...")
    
    # Create model
    model = DiscriminativeRewardModel(hidden_size=256, num_layers=2, num_heads=4, vocab_size=3205)
    model.eval()  # Set to eval to avoid training mode issues
    
    # Create dataset
    dataset = DiscriminativeDataset(
        melody_path="data/reward_training_mel_cp4.pt",
        acc_path="data/reward_training_acc_cp4.pt", 
        target_length=32
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Test single sample
    try:
        sample = dataset[0]
        sequence = sample['sequence'].unsqueeze(0)  # Add batch dimension
        label = sample['label'].unsqueeze(0) 
        pitch_shift = sample['pitch_shift']
        
        print(f"Sample shapes: sequence={sequence.shape}, label={label.shape}, pitch_shift={pitch_shift.shape}")
        
        # Forward pass
        with torch.no_grad():
            logits = model(sequence, pitch_shift)
            print(f"Output logits: {logits}")
            print(f"Output shape: {logits.shape}")
            
            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(logits, label)
            print(f"Loss: {loss.item()}")
            
        print("✓ Forward pass successful!")
        
        # Test batch processing
        print("\nTesting batch processing...")
        batch_size = 2
        batch_sequences = torch.stack([dataset[i]['sequence'] for i in range(batch_size)])
        batch_labels = torch.stack([dataset[i]['label'] for i in range(batch_size)])
        batch_pitch_shifts = torch.stack([dataset[i]['pitch_shift'] for i in range(batch_size)])
        
        print(f"Batch shapes: sequences={batch_sequences.shape}, labels={batch_labels.shape}, pitch_shifts={batch_pitch_shifts.shape}")
        
        with torch.no_grad():
            batch_logits = model(batch_sequences, batch_pitch_shifts.squeeze(-1))
            batch_loss = F.binary_cross_entropy_with_logits(batch_logits, batch_labels)
            
            print(f"Batch logits: {batch_logits}")
            print(f"Batch loss: {batch_loss.item()}")
            
        print("✓ Batch processing successful!")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forward_pass()