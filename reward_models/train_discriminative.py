#!/usr/bin/env python3
"""
Simple training script for discriminative reward model.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import time
from pathlib import Path

from discriminative_model import DiscriminativeRewardModel
from data_loader import create_dataloaders


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        sequences = batch['sequences'].to(device)  # [batch_size, 2*seq_len, 12]
        labels = batch['labels'].to(device)        # [batch_size]
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(sequences)  # [batch_size]
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.sigmoid(logits) > 0.5
        correct += (predictions == (labels > 0.5)).sum().item()
        total += labels.size(0)
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(sequences)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.sigmoid(logits) > 0.5
            correct += (predictions == (labels > 0.5)).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Discriminative Reward Model')
    parser.add_argument('--melody_path', required=True, help='Path to melody .pt file')
    parser.add_argument('--acc_path', required=True, help='Path to accompaniment .pt file')
    parser.add_argument('--output_dir', default='./checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--target_length', type=int, default=384, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = DiscriminativeRewardModel(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        melody_path=args.melody_path,
        acc_path=args.acc_path,
        batch_size=args.batch_size,
        target_length=args.target_length,
        num_workers=args.num_workers
    )
    
    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print("Starting training...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s, LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 60)
        
        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'args': args
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved new best model with val_acc: {val_acc:.2f}%")
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'args': args
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest_model.pt'))
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()