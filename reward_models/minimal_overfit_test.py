#!/usr/bin/env python3
"""
Minimal overfitting test - can the model memorize just 10 samples?
If this fails, there's a fundamental bug in the setup.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.discriminative_model import DiscriminativeRewardModel
from reward_models.data_loader import create_dataloaders


def minimal_overfit_test():
    print("🧪 MINIMAL OVERFITTING TEST")
    print("=" * 60)
    print("Testing if model can memorize 10 samples perfectly")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model 
    model = DiscriminativeRewardModel(
        hidden_size=128,  # Much smaller for this test
        num_layers=2,
        num_heads=4
    ).to(device)
    
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    
    # Get just a few samples
    train_loader, _ = create_dataloaders(
        melody_path="data/reward_training_mel_cp4.pt",
        acc_path="data/reward_training_acc_cp4.pt",
        batch_size=10,  # Small batch
        target_length=64,  # Very short sequences
        num_workers=0
    )
    
    # Get exactly 10 samples and fix them
    batch = next(iter(train_loader))
    fixed_sequences = batch['sequences'][:10].to(device)
    fixed_labels = batch['labels'][:10].to(device)
    fixed_pitch_shifts = batch['pitch_shifts'][:10].to(device).squeeze(-1)
    
    print(f"Fixed dataset: {fixed_sequences.shape}")
    print(f"Labels: {fixed_labels.tolist()}")
    print(f"Pitch shifts: {fixed_pitch_shifts.tolist()}")
    
    # Simple optimizer (no fancy scheduling)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    print(f"\nTraining to memorize these 10 samples...")
    print("If working correctly, loss should drop to near 0 and accuracy to 100%")
    print("-" * 60)
    
    # Train until perfect memorization
    for step in range(200):  # Should be plenty to memorize 10 samples
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on the same 10 samples every time
        logits = model(fixed_sequences, fixed_pitch_shifts)
        loss = F.binary_cross_entropy_with_logits(logits, fixed_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Check accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            predictions = probs > 0.5
            correct = (predictions == (fixed_labels > 0.5)).sum().item()
            accuracy = 100.0 * correct / len(fixed_labels)
        
        if step % 20 == 0:
            print(f"Step {step:3d}: Loss={loss.item():.4f}, Accuracy={accuracy:.1f}%")
            print(f"          Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print(f"          Probs range:  [{probs.min().item():.3f}, {probs.max().item():.3f}]")
        
        # Early stopping if perfect
        if accuracy == 100.0 and loss.item() < 0.01:
            print(f"\n🎉 PERFECT MEMORIZATION achieved at step {step}!")
            break
        
        # Warning if not improving after many steps
        if step == 100 and accuracy < 80:
            print(f"\n⚠️  WARNING: After 100 steps, accuracy only {accuracy:.1f}%")
            print("    This suggests a fundamental issue with the model/data")
        
        if step == 199:
            print(f"\n❌ FAILED to memorize after 200 steps!")
            print(f"   Final accuracy: {accuracy:.1f}%, Final loss: {loss.item():.4f}")
            print("   This indicates a serious bug in the model or data pipeline")
    
    # Final detailed analysis
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        final_logits = model(fixed_sequences, fixed_pitch_shifts)
        final_probs = torch.sigmoid(final_logits)
        final_preds = final_probs > 0.5
        
        print("Sample-by-sample results:")
        print("Idx | Label | Logit  | Prob  | Pred | Correct")
        print("-" * 45)
        for i in range(10):
            label = fixed_labels[i].item()
            logit = final_logits[i].item()
            prob = final_probs[i].item()
            pred = final_preds[i].item()
            correct = "✅" if (pred > 0.5) == (label > 0.5) else "❌"
            print(f"{i:3d} | {label:5.1f} | {logit:6.3f} | {prob:.3f} | {pred:4.0f} | {correct}")
    
    # Gradient check on final step
    model.train()
    optimizer.zero_grad()
    final_logits = model(fixed_sequences, fixed_pitch_shifts)
    final_loss = F.binary_cross_entropy_with_logits(final_logits, fixed_labels)
    final_loss.backward()
    
    print(f"\nGradient check (final step):")
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            if 'classifier' in name:  # Focus on the final classifier
                print(f"  {name}: {grad_norm:.6f}")
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    
    if total_grad_norm < 1e-6:
        print("  ❌ Gradients are too small - vanishing gradient problem")
    elif total_grad_norm > 100:
        print("  ❌ Gradients are too large - exploding gradient problem")
    else:
        print("  ✅ Gradient magnitudes look reasonable")


if __name__ == "__main__":
    minimal_overfit_test()