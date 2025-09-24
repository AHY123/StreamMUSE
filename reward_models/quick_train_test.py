#!/usr/bin/env python3
"""
Quick training test to verify the discriminative model can learn.
Runs a few epochs with limited data to check if accuracy/loss improve.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.discriminative_model import DiscriminativeRewardModel
from reward_models.data_loader import create_dataloaders


def quick_train_test():
    print("🚀 QUICK TRAINING TEST")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model (smaller for quick test)
    model = DiscriminativeRewardModel(
        hidden_size=256,  # Smaller for faster training
        num_layers=3,
        num_heads=8
    ).to(device)
    
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    
    # Create dataloaders (small batch for quick test)
    train_loader, val_loader = create_dataloaders(
        melody_path="data/reward_training_mel_cp4.pt",
        acc_path="data/reward_training_acc_cp4.pt",
        batch_size=16,
        target_length=128,  # Shorter sequences for speed
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup optimizer and scheduler
    epochs = 5  # Just a few epochs to test
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=2e-4, total_steps=total_steps, pct_start=0.1)
    
    print(f"Training for {epochs} epochs ({total_steps} total steps)...")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 20:  # Limit to first 20 batches per epoch for speed
                break
                
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            pitch_shifts = batch['pitch_shifts'].to(device).squeeze(-1)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(sequences, pitch_shifts)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Statistics
            epoch_loss += loss.item()
            predictions = torch.sigmoid(logits) > 0.5
            correct += (predictions == (labels > 0.5)).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                batch_acc = 100. * correct / total if total > 0 else 0
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={batch_acc:.1f}%, LR={lr:.6f}")
        
        # Epoch summary
        avg_loss = epoch_loss / min(20, len(train_loader))
        accuracy = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")
        
        # Quick validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 5:  # Just a few validation batches
                    break
                    
                sequences = batch['sequences'].to(device)
                labels = batch['labels'].to(device)
                pitch_shifts = batch['pitch_shifts'].to(device).squeeze(-1)
                
                logits = model(sequences, pitch_shifts)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                
                val_loss += loss.item()
                predictions = torch.sigmoid(logits) > 0.5
                val_correct += (predictions == (labels > 0.5)).sum().item()
                val_total += labels.size(0)
        
        val_avg_loss = val_loss / min(5, len(val_loader))
        val_accuracy = 100. * val_correct / val_total
        
        print(f"         Validation: Loss={val_avg_loss:.4f}, Accuracy={val_accuracy:.1f}%")
        print("-" * 60)
    
    print("\n🏁 TRAINING TEST COMPLETE")
    print("=" * 60)
    print("Check the results above:")
    print("✅ GOOD: Loss decreasing, accuracy improving over epochs")
    print("❌ BAD: Loss stuck, accuracy staying around 50% or getting worse")
    print("⚠️  CONCERNING: Accuracy consistently below 45% (possible label flip)")


if __name__ == "__main__":
    quick_train_test()