#!/usr/bin/env python3
"""
Phase 3: Test model overfitting capability on 5 known good samples.
If the model can't memorize 5 samples perfectly, there's a fundamental architecture bug.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reward_models.discriminative_model import DiscriminativeRewardModel
from dataset_with_segment_info import DiscriminativeDatasetWithSegmentInfo


def debug_overfit_mini():
    """
    Test if the model can perfectly memorize 5 known good samples.
    This is the critical test to verify the model architecture works.
    """
    print("🧠 PHASE 3: MODEL OVERFITTING TEST")
    print("=" * 60)
    print("Testing if model can memorize 5 samples perfectly...")
    print("If this fails, there's a fundamental model architecture bug.")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = Path("reward_models/debug_data")
    
    # Create small model for faster testing
    model = DiscriminativeRewardModel(
        hidden_size=128,  # Smaller for faster convergence
        num_layers=2,     # Fewer layers for easier overfitting
        num_heads=4       # Fewer heads
    ).to(device)
    
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    
    # Create dataset with our 5 samples
    target_length = 64  # Short segments for faster testing
    
    dataset = DiscriminativeDatasetWithSegmentInfo(
        melody_path=str(data_dir / "debug_mel.pt"),
        acc_path=str(data_dir / "debug_acc.pt"),
        target_length=target_length
    )
    
    print(f"Dataset: {len(dataset)} total samples from {len(dataset.valid_indices)} songs")
    
    # Fix a small subset of samples for overfitting
    fixed_samples = []
    fixed_labels = []
    fixed_sample_info = []
    
    # Get exactly 10 samples (5 real + 5 fake)
    num_fixed_samples = 10
    
    print(f"\n📋 CREATING FIXED DATASET ({num_fixed_samples} samples):")
    
    for i in range(num_fixed_samples):
        sample = dataset[i]
        sequences = sample['sequence'].to(device)
        label = sample['label'].to(device)
        
        # Store detailed info for analysis
        sample_info = {
            'idx': i,
            'label': label.item(),
            'label_str': "REAL" if label.item() == 1.0 else "FAKE",
            'melody_info': sample['melody_info'],
            'acc_info': sample['acc_info'],
            'is_real': sample['is_real']
        }
        
        fixed_samples.append(sequences)
        fixed_labels.append(label)
        fixed_sample_info.append(sample_info)
        
        print(f"  Sample {i}: {sample_info['label_str']} "
              f"(mel: song {sample_info['melody_info']['song_idx']}, "
              f"acc: song {sample_info['acc_info']['song_idx']})")
    
    # Stack into batch tensors
    fixed_sequences = torch.stack(fixed_samples)  # [10, 2*target_length, 12]
    fixed_labels = torch.stack(fixed_labels)      # [10]
    
    print(f"\nFixed dataset shapes:")
    print(f"  Sequences: {fixed_sequences.shape}")
    print(f"  Labels: {fixed_labels.shape}")
    print(f"  Labels: {fixed_labels.tolist()}")
    
    # Count real vs fake
    real_count = (fixed_labels == 1.0).sum().item()
    fake_count = (fixed_labels == 0.0).sum().item()
    print(f"  Distribution: {real_count} real, {fake_count} fake")
    
    # Test model forward pass
    print(f"\n🔬 TESTING MODEL FORWARD PASS:")
    try:
        with torch.no_grad():
            # Create dummy pitch shifts (all zeros since we disabled them)
            pitch_shifts = torch.zeros(fixed_sequences.shape[0], device=device)
            
            logits = model(fixed_sequences, pitch_shifts)
            probs = torch.sigmoid(logits)
            
            print(f"✅ Forward pass successful!")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print(f"  Probs range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup optimizer (simple Adam, high learning rate for overfitting)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    print(f"\n🏋️ OVERFITTING TRAINING:")
    print("Goal: 100% accuracy and loss < 0.01 within 200 steps")
    print("-" * 60)
    
    # Training loop
    for step in range(200):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on fixed samples
        pitch_shifts = torch.zeros(fixed_sequences.shape[0], device=device)
        logits = model(fixed_sequences, pitch_shifts)
        loss = F.binary_cross_entropy_with_logits(logits, fixed_labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            correct = (predictions == fixed_labels).sum().item()
            accuracy = 100.0 * correct / len(fixed_labels)
        
        # Log progress
        if step % 20 == 0 or accuracy == 100.0:
            print(f"Step {step:3d}: Loss={loss.item():.4f}, Accuracy={accuracy:5.1f}%, "
                  f"GradNorm={total_grad_norm:.3f}")
            
            if step % 40 == 0:  # Show detailed predictions occasionally
                print(f"  Predictions: {predictions.cpu().numpy()}")
                print(f"  Labels:      {fixed_labels.cpu().numpy()}")
                print(f"  Probs:       {probs.cpu().numpy()}")
        
        # Early stopping conditions
        if accuracy == 100.0 and loss.item() < 0.01:
            print(f"\n🎉 PERFECT OVERFITTING ACHIEVED at step {step}!")
            print(f"   Final loss: {loss.item():.6f}")
            print(f"   Final accuracy: {accuracy:.1f}%")
            break
            
        if step == 100 and accuracy < 70:
            print(f"\n⚠️  WARNING: After 100 steps, accuracy only {accuracy:.1f}%")
            print("   Model may have fundamental issues")
            
        if step == 199:
            print(f"\n❌ FAILED TO OVERFIT after 200 steps!")
            print(f"   Final accuracy: {accuracy:.1f}%")
            print(f"   Final loss: {loss.item():.4f}")
    
    # Final detailed analysis
    print(f"\n" + "=" * 60)
    print("FINAL OVERFITTING ANALYSIS")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        pitch_shifts = torch.zeros(fixed_sequences.shape[0], device=device)
        final_logits = model(fixed_sequences, pitch_shifts)
        final_probs = torch.sigmoid(final_logits)
        final_preds = (final_probs > 0.5).float()
        final_accuracy = (final_preds == fixed_labels).sum().item() / len(fixed_labels)
        
        print("Sample-by-sample results:")
        print("Idx | Type | Label | Logit  | Prob  | Pred | Correct | Songs")
        print("-" * 70)
        
        for i in range(len(fixed_samples)):
            info = fixed_sample_info[i]
            label = fixed_labels[i].item()
            logit = final_logits[i].item()
            prob = final_probs[i].item()
            pred = final_preds[i].item()
            correct = "✅" if pred == label else "❌"
            
            mel_song = info['melody_info']['song_idx']
            acc_song = info['acc_info']['song_idx']
            songs_str = f"mel:{mel_song}, acc:{acc_song}"
            
            print(f"{i:3d} | {info['label_str']:4s} | {label:5.1f} | {logit:6.3f} | "
                  f"{prob:.3f} | {pred:4.0f} | {correct:7s} | {songs_str}")
    
    # Interpretation
    print(f"\n🎯 OVERFITTING TEST RESULTS:")
    print("=" * 40)
    
    if final_accuracy == 1.0:
        print("✅ SUCCESS: Model can perfectly memorize small dataset")
        print("   → Model architecture is working correctly")
        print("   → Original training issues were likely data-related")
        print("   → Ready to proceed with full training")
    elif final_accuracy > 0.8:
        print("⚠️  PARTIAL SUCCESS: Model can mostly memorize dataset")
        print("   → Model architecture mostly works")
        print("   → May need hyperparameter tuning")
        print("   → Could proceed with caution")
    else:
        print("❌ FAILURE: Model cannot memorize small dataset")
        print("   → Fundamental model architecture bug")
        print("   → Need to debug model internals before proceeding")
        print("   → Check gradient flow, loss computation, data preprocessing")
    
    return final_accuracy


def debug_model_internals():
    """
    If overfitting fails, debug model internals step by step.
    """
    print(f"\n🔍 DEBUGGING MODEL INTERNALS")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create minimal model
    model = DiscriminativeRewardModel(
        hidden_size=64,   # Very small
        num_layers=1,     # Single layer
        num_heads=2       # Minimal heads
    ).to(device)
    
    # Create tiny synthetic data
    batch_size = 2
    seq_len = 32
    
    # Create simple synthetic data that should be easy to memorize
    sequences = torch.randint(0, 100, (batch_size, seq_len, 12), device=device).float()
    labels = torch.tensor([1.0, 0.0], device=device)  # Simple binary labels
    pitch_shifts = torch.zeros(batch_size, device=device)
    
    print(f"Synthetic data:")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels: {labels}")
    
    # Test forward pass step by step
    try:
        print(f"\nTesting forward pass components:")
        
        # Test preprocessing
        model.eval()
        with torch.no_grad():
            x_processed = model.preprocess(sequences, pitch_shifts)
            print(f"  Preprocessing: {sequences.shape} → {x_processed.shape}")
            print(f"  Token range: [{x_processed.min():.0f}, {x_processed.max():.0f}]")
            
            # Test local encoding
            h, _ = model.local_encode(x_processed, None)
            print(f"  Local encoding: {x_processed.shape} → {h.shape}")
            
            # Test global encoding preparation
            batch_size, seq_len = x_processed.shape[:2]
            h = h.view(batch_size, seq_len, -1)
            sos = model.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
            h = torch.cat([sos, h[:, :-1]], dim=1)
            print(f"  Global prep: → {h.shape}")
            
            # Test global encoding
            h_global = model.global_encoder(h, attention_mask=model.buffered_future_mask(h), interleave_pos=True)[0]
            print(f"  Global encoding: → {h_global.shape}")
            
            # Test classification
            sequence_representation = h_global[:, -1]
            logits = model.classifier(sequence_representation)
            print(f"  Classification: → {logits.shape}")
            print(f"  Final logits: {logits}")
            
        print(f"✅ All forward pass components working")
        
    except Exception as e:
        print(f"❌ Forward pass component failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    accuracy = debug_overfit_mini()
    
    # If overfitting failed, run additional debugging
    if accuracy < 0.8:
        debug_model_internals()