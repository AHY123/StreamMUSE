#!/usr/bin/env python3
"""
Simple training script for discriminative reward model.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import os
import time
import logging
from pathlib import Path
import wandb
from datetime import datetime

from discriminative_model import DiscriminativeRewardModel
from data_loader import create_dataloaders


def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        sequences = batch['sequences'].to(device)  # [batch_size, 2*seq_len, 12]
        labels = batch['labels'].to(device)        # [batch_size]
        
        optimizer.zero_grad()
        
        # Forward pass with pitch shift
        pitch_shifts = batch['pitch_shifts'].to(device)  # Already [batch_size] from our updated data_loader
        logits = model(sequences, pitch_shifts)  # [batch_size]
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # OneCycleLR steps per batch
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.sigmoid(logits) > 0.5
        correct += (predictions == (labels > 0.5)).sum().item()
        total += labels.size(0)
        
        # Log training progress
        if batch_idx % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            batch_acc = 100. * correct / total if total > 0 else 0
            
            # Log to console and file
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                       f'Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}')
            
            # Log to wandb
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_accuracy': batch_acc,
                'train/learning_rate': current_lr,
                'train/epoch': epoch,
                'train/batch': batch_idx
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Log epoch summary
    logger.info(f'Epoch {epoch} Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def validate(model, val_loader, device, logger):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            pitch_shifts = batch['pitch_shifts'].to(device)  # Already [batch_size] from our updated data_loader
            
            logits = model(sequences, pitch_shifts)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.sigmoid(logits) > 0.5
            correct += (predictions == (labels > 0.5)).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Log validation summary
    logger.info(f'Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
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
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--quality_filter_mode', default='resample', choices=['resample', 'strict'], 
                       help='Quality filtering mode: resample (same samples, retry) or strict (fewer high-quality samples)')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train/validation split ratio')
    
    # Logging and wandb arguments
    parser.add_argument('--wandb_project', default='discriminative-reward-model', help='Wandb project name')
    parser.add_argument('--wandb_entity', default=None, help='Wandb entity/team name')
    parser.add_argument('--wandb_name', default=None, help='Wandb run name')
    parser.add_argument('--wandb_tags', nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Setup wandb
    if not args.no_wandb:
        wandb_name = args.wandb_name or f"discriminative-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_name,
            tags=args.wandb_tags,
            config=vars(args)
        )
        logger.info(f"Initialized wandb with project: {args.wandb_project}, run: {wandb_name}")
    else:
        logger.info("Wandb logging disabled")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = DiscriminativeRewardModel(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)
    
    model_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_params:,}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    logger.info(f"Using quality filter mode: {args.quality_filter_mode}")
    logger.info(f"Train/validation split: {args.train_split:.1%}")
    
    train_loader, val_loader = create_dataloaders(
        melody_path=args.melody_path,
        acc_path=args.acc_path,
        batch_size=args.batch_size,
        target_length=args.target_length,
        train_split=args.train_split,
        num_workers=args.num_workers,
        quality_filter_mode=args.quality_filter_mode
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create optimizer and scheduler (same as main model)
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.005)
    
    logger.info(f"Starting training for {args.epochs} epochs ({total_steps} total steps)...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, logger)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device, logger)
        
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch summary
        logger.info(f"Epoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"  Time: {epoch_time:.2f}s, Current LR: {current_lr:.6f}")
        logger.info("-" * 60)
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_loss,
                'train/epoch_accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'epoch_time': epoch_time,
                'learning_rate': current_lr
            })
        
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
            logger.info(f"Saved new best model with val_acc: {val_acc:.2f}%")
        
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
    
    # Training completed
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Log final summary to wandb
    if not args.no_wandb:
        wandb.log({
            'final/best_val_accuracy': best_val_acc,
            'final/total_epochs': args.epochs,
            'final/model_parameters': model_params
        })
        wandb.finish()
        logger.info("Wandb logging finished")


if __name__ == '__main__':
    main()