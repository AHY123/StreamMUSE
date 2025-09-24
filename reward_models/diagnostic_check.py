#!/usr/bin/env python3
"""
Diagnostic script for discriminative reward model.
Checks gradient flow, label distribution, model behavior, and potential issues.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.discriminative_model import DiscriminativeRewardModel
from reward_models.data_loader import create_dataloaders


def check_label_distribution(data_loader, num_batches=10):
    """Check if we're getting balanced real/fake labels"""
    print("=" * 60)
    print("LABEL DISTRIBUTION CHECK")
    print("=" * 60)
    
    all_labels = []
    all_pitch_shifts = []
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
            
        labels = batch['labels']
        pitch_shifts = batch['pitch_shifts'].squeeze(-1)
        
        all_labels.extend(labels.tolist())
        all_pitch_shifts.extend(pitch_shifts.tolist())
    
    all_labels = np.array(all_labels)
    all_pitch_shifts = np.array(all_pitch_shifts)
    
    # Label statistics
    real_count = np.sum(all_labels == 1.0)
    fake_count = np.sum(all_labels == 0.0)
    total = len(all_labels)
    
    print(f"Total samples: {total}")
    print(f"Real pairs (label=1): {real_count} ({100*real_count/total:.1f}%)")
    print(f"Fake pairs (label=0): {fake_count} ({100*fake_count/total:.1f}%)")
    print(f"Expected: ~50% each for balanced dataset")
    
    if abs(real_count - fake_count) > total * 0.1:  # More than 10% imbalance
        print("⚠️  WARNING: Significant label imbalance detected!")
    else:
        print("✅ Label distribution looks balanced")
    
    # Pitch shift statistics
    print(f"\nPitch shift range: [{np.min(all_pitch_shifts)}, {np.max(all_pitch_shifts)}]")
    print(f"Pitch shift mean: {np.mean(all_pitch_shifts):.2f}, std: {np.std(all_pitch_shifts):.2f}")
    
    # Check for extreme pitch shifts
    extreme_shifts = np.abs(all_pitch_shifts) > 24  # More than 2 octaves
    if np.any(extreme_shifts):
        print(f"⚠️  WARNING: {np.sum(extreme_shifts)} extreme pitch shifts (>24 semitones)")
        print(f"   Extreme values: {all_pitch_shifts[extreme_shifts][:10]}...")  # Show first 10
    else:
        print("✅ Pitch shifts within reasonable range")


def check_gradient_flow(model, data_loader, device, num_batches=5):
    """Check if gradients are flowing through the model properly"""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW CHECK")
    print("=" * 60)
    
    model.train()
    model.to(device)
    
    # Track gradients for each layer
    layer_gradients = {}
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
            
        sequences = batch['sequences'].to(device)
        labels = batch['labels'].to(device)
        pitch_shifts = batch['pitch_shifts'].to(device).squeeze(-1)
        
        model.zero_grad()
        
        # Forward pass
        logits = model(sequences, pitch_shifts)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Collect gradient norms for each named parameter
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if name not in layer_gradients:
                    layer_gradients[name] = []
                layer_gradients[name].append(grad_norm)
            else:
                print(f"⚠️  WARNING: No gradient for {name}")
        
        print(f"Batch {batch_idx}: loss = {loss.item():.4f}")
    
    # Analyze gradient statistics
    print("\nGradient norm statistics (mean ± std):")
    print("-" * 60)
    
    critical_layers = []
    
    for name, grad_norms in layer_gradients.items():
        mean_grad = np.mean(grad_norms)
        std_grad = np.std(grad_norms)
        
        # Identify problematic layers
        if mean_grad < 1e-6:
            status = "❌ VANISHING"
            critical_layers.append(name)
        elif mean_grad > 1e2:
            status = "❌ EXPLODING" 
            critical_layers.append(name)
        elif mean_grad < 1e-4:
            status = "⚠️  Very small"
        elif mean_grad > 1e1:
            status = "⚠️  Large"
        else:
            status = "✅ Normal"
        
        print(f"{name:40s}: {mean_grad:.2e} ± {std_grad:.2e} {status}")
    
    if critical_layers:
        print(f"\n❌ CRITICAL ISSUES detected in {len(critical_layers)} layers:")
        for layer in critical_layers[:5]:  # Show first 5
            print(f"  - {layer}")
    else:
        print("\n✅ Gradient flow looks healthy across all layers")


def check_model_outputs(model, data_loader, device, num_batches=3):
    """Check model output distributions and behavior"""
    print("\n" + "=" * 60)
    print("MODEL OUTPUT CHECK")
    print("=" * 60)
    
    model.eval()
    model.to(device)
    
    all_logits = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            pitch_shifts = batch['pitch_shifts'].to(device).squeeze(-1)
            
            logits = model(sequences, pitch_shifts)
            probs = torch.sigmoid(logits)
            
            all_logits.extend(logits.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            print(f"Batch {batch_idx}:")
            print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print(f"  Probs range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
            print(f"  Labels: {labels.cpu().tolist()}")
    
    all_logits = np.array(all_logits)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"\nOverall statistics:")
    print(f"Logits  - Mean: {np.mean(all_logits):.3f}, Std: {np.std(all_logits):.3f}")
    print(f"Probs   - Mean: {np.mean(all_probs):.3f}, Std: {np.std(all_probs):.3f}")
    
    # Check for saturation
    saturated_high = np.sum(all_probs > 0.99)
    saturated_low = np.sum(all_probs < 0.01)
    total_outputs = len(all_probs)
    
    print(f"\nSaturation check:")
    print(f"High saturation (>0.99): {saturated_high}/{total_outputs} ({100*saturated_high/total_outputs:.1f}%)")
    print(f"Low saturation (<0.01): {saturated_low}/{total_outputs} ({100*saturated_low/total_outputs:.1f}%)")
    
    if saturated_high > total_outputs * 0.8 or saturated_low > total_outputs * 0.8:
        print("❌ WARNING: Model outputs are heavily saturated - likely not learning")
    elif saturated_high + saturated_low > total_outputs * 0.5:
        print("⚠️  Model outputs show some saturation")
    else:
        print("✅ Model outputs have good dynamic range")
    
    # Check prediction accuracy
    predictions = (all_probs > 0.5).astype(float)
    accuracy = np.mean(predictions == all_labels)
    print(f"\nCurrent accuracy: {100*accuracy:.1f}%")
    
    if accuracy < 0.45 or accuracy > 0.55:
        if accuracy < 0.45:
            print("❌ Accuracy significantly below chance (50%) - possible label flip issue")
        else:
            print("❌ Accuracy too high for early training - possible data leakage")
    else:
        print("✅ Accuracy near chance level (expected for early training)")


def check_data_preprocessing(model, data_loader, device):
    """Check preprocessing and tokenization"""
    print("\n" + "=" * 60) 
    print("DATA PREPROCESSING CHECK")
    print("=" * 60)
    
    model.eval()
    model.to(device)
    
    batch = next(iter(data_loader))
    sequences = batch['sequences'].to(device)[:2]  # Take 2 samples
    pitch_shifts = batch['pitch_shifts'].to(device)[:2].squeeze(-1)
    
    print(f"Input sequences shape: {sequences.shape}")
    print(f"Input range: [{sequences.min().item()}, {sequences.max().item()}]")
    print(f"Pitch shifts: {pitch_shifts.tolist()}")
    
    # Test preprocessing
    with torch.no_grad():
        processed = model.preprocess(sequences, pitch_shifts)
        
    print(f"Processed tokens shape: {processed.shape}")
    print(f"Processed range: [{processed.min().item()}, {processed.max().item()}]")
    print(f"Vocab size: {model.vocab_size}")
    
    # Check for out-of-bounds tokens
    oob_tokens = (processed < 0) | (processed >= model.vocab_size)
    if oob_tokens.any():
        print(f"❌ {oob_tokens.sum().item()} out-of-bounds tokens found!")
        print(f"   Min token: {processed.min().item()}")
        print(f"   Max token: {processed.max().item()}")
    else:
        print("✅ All tokens within valid range")
    
    # Check special tokens
    pad_count = (processed == model.PAD_TOKEN).sum().item()
    eos_count = (processed == model.EOS_TOKEN).sum().item()
    print(f"Special tokens - PAD: {pad_count}, EOS: {eos_count}")


def main():
    print("🔧 DISCRIMINATIVE REWARD MODEL DIAGNOSTICS")
    print("=" * 60)
    
    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DiscriminativeRewardModel(
        hidden_size=512,
        num_layers=6,
        num_heads=8
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloader (small batch for diagnostics)
    try:
        train_loader, _ = create_dataloaders(
            melody_path="data/reward_training_mel_cp4.pt",
            acc_path="data/reward_training_acc_cp4.pt", 
            batch_size=8,
            target_length=128,  # Smaller for faster diagnostics
            num_workers=0
        )
        print(f"Created dataloader with {len(train_loader)} batches")
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        return
    
    # Run diagnostics
    try:
        check_label_distribution(train_loader)
        check_data_preprocessing(model, train_loader, device)
        check_model_outputs(model, train_loader, device)
        check_gradient_flow(model, train_loader, device)
        
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print("✅ Diagnostics completed successfully")
        print("Review the output above for any WARNING or CRITICAL issues")
        print("If gradient flow and data look good, try training the model")
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()