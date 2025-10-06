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
from datetime import datetime

from discriminative_model import DiscriminativeRewardModel
from data_loader import create_dataloaders

# Import for MIDI export
try:
    import sys
    import os
    # Add both parent directory and preprocess directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_dir = os.path.join(parent_dir, 'preprocess')
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, preprocess_dir)
    
    from preprocess_midi2pt_dataset import tensor_to_midi
    MIDI_EXPORT_AVAILABLE = True
    print("✅ MIDI export available")
except ImportError as e:
    MIDI_EXPORT_AVAILABLE = False
    print(f"❌ MIDI export not available: {e}")
    print("Will export tensor analysis only")


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


def export_samples(train_loader, export_dir, num_samples=8):
    """Export first few samples for verification"""
    os.makedirs(export_dir, exist_ok=True)
    print(f"Exporting {num_samples} samples to {export_dir}...")
    
    # Get first batch
    batch = next(iter(train_loader))
    sequences = batch['sequences']  # [batch_size, 2*seq_len, 12]
    labels = batch['labels']        # [batch_size]
    pitch_shifts = batch['pitch_shifts']  # [batch_size]
    
    # Export up to num_samples
    actual_samples = min(num_samples, sequences.shape[0])
    
    # Create analysis file
    analysis_file = os.path.join(export_dir, "sample_analysis.txt")
    
    with open(analysis_file, "w") as f:
        f.write("TRAINING SAMPLE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        for i in range(actual_samples):
            sequence = sequences[i]  # [2*seq_len, 12]
            label = labels[i].item()
            pitch_shift = pitch_shifts[i].item()
            
            # Split interleaved sequence back to melody and accompaniment
            # Interleaved format: [acc_0, mel_0, acc_1, mel_1, ...]
            melody = sequence[1::2]      # Odd indices: mel_0, mel_1, ...
            accompaniment = sequence[0::2]  # Even indices: acc_0, acc_1, ...
            
            # Analyze content
            mel_notes = count_musical_notes(melody)
            acc_notes = count_musical_notes(accompaniment)
            
            # Get unique pitches to check for key/harmony coherence
            mel_pitches = get_unique_pitches(melody)
            acc_pitches = get_unique_pitches(accompaniment)
            
            label_str = "REAL" if label == 1.0 else "FAKE"
            
            f.write(f"Sample {i}: {label_str} (pitch_shift: {pitch_shift:+d})\n")
            f.write(f"  Melody: {mel_notes} notes, pitches: {sorted(mel_pitches)[:10]}...\n")
            f.write(f"  Accompaniment: {acc_notes} notes, pitches: {sorted(acc_pitches)[:10]}...\n")
            
            # Check for pitch overlap (real pairs should have similar pitches)
            pitch_overlap = len(set(mel_pitches) & set(acc_pitches))
            f.write(f"  Pitch overlap: {pitch_overlap} common pitches\n")
            
            # Save raw tensors for inspection
            tensor_file = os.path.join(export_dir, f"sample_{i:02d}_{label_str}.pt")
            torch.save({
                'melody': melody,
                'accompaniment': accompaniment,
                'label': label,
                'pitch_shift': pitch_shift
            }, tensor_file)
            
            f.write(f"  Saved tensor: {tensor_file}\n")
            
            # Export MIDI files if available
            if MIDI_EXPORT_AVAILABLE:
                try:
                    mel_path = os.path.join(export_dir, f"sample_{i:02d}_{label_str}_melody.mid")
                    acc_path = os.path.join(export_dir, f"sample_{i:02d}_{label_str}_accompaniment.mid")
                    
                    tensor_to_midi(melody, mel_path, tempo=120.0, instrument_program=0)  # Piano
                    tensor_to_midi(accompaniment, acc_path, tempo=120.0, instrument_program=1)  # Electric Piano
                    
                    f.write(f"  Exported MIDI: {mel_path}, {acc_path}\n")
                    
                except Exception as e:
                    f.write(f"  MIDI export failed: {e}\n")
            else:
                f.write(f"  MIDI export not available\n")
            
            f.write("-" * 30 + "\n")
            
            print(f"  Sample {i}: {label_str}, mel={mel_notes} notes, acc={acc_notes} notes, overlap={pitch_overlap}")
    
    print(f"Sample analysis completed. Check: {analysis_file}")
    print(f"Key insights to look for:")
    print(f"  - REAL samples should have similar pitch ranges in melody and accompaniment")
    print(f"  - FAKE samples should have different/clashing pitch ranges")
    print(f"  - If both look similar, the fake generation logic has issues")


def count_musical_notes(tensor):
    """Count non-empty musical notes in a tensor"""
    # tensor shape: [seq_len, 12] -> [seq_len, 4, 3]
    notes = tensor.view(tensor.shape[0], 4, 3)
    count = 0
    for frame_idx in range(notes.shape[0]):
        for note_idx in range(notes.shape[1]):
            pitch = notes[frame_idx, note_idx, 1]
            if pitch not in [254, 255]:  # Not EOS or PAD
                count += 1
    return count


def get_unique_pitches(tensor):
    """Get unique pitches from a tensor"""
    notes = tensor.view(tensor.shape[0], 4, 3)
    pitches = set()
    for frame_idx in range(notes.shape[0]):
        for note_idx in range(notes.shape[1]):
            pitch = notes[frame_idx, note_idx, 1].item()
            if pitch not in [254, 255]:  # Not EOS or PAD
                pitches.add(pitch)
    return list(pitches)


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
        if scheduler is not None:
            scheduler.step()  # Only step if scheduler exists
        
        # Statistics
        total_loss += loss.item()
        predictions = torch.sigmoid(logits) > 0.5
        correct += (predictions == (labels > 0.5)).sum().item()
        total += labels.size(0)
        
        # Log training progress
        if batch_idx % 50 == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            batch_acc = 100. * correct / total if total > 0 else 0
            
            # Log to console and file
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                       f'Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}')
            
            # Wandb logging removed for simplicity
    
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
    parser.add_argument('--target_length', type=int, default=384, help='Sequence length in frames (16 frames = 1 bar, so 384 = 24 bars)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size (default matches main model)')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers (default matches main model)')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads (default matches main model)')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--quality_filter_mode', default='resample', choices=['resample', 'strict'], 
                       help='Quality filtering mode: resample (same samples, retry) or strict (fewer high-quality samples)')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train/validation split ratio')
    parser.add_argument('--export_samples', action='store_true', help='Export first few samples as MIDI for verification')
    parser.add_argument('--export_dir', default='./sample_exports', help='Directory to export sample MIDI files')
    
    # Logging arguments (wandb removed for simplicity)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Wandb logging disabled for simplicity")
    
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
    
    # Export samples for verification if requested
    if args.export_samples:
        logger.info("Exporting sample MIDI files for verification...")
        export_samples(train_loader, args.export_dir, num_samples=8)
        logger.info(f"Samples exported to: {args.export_dir}")
    
    # Create optimizer (simple Adam like in debug - no scheduler for now)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None  # Remove problematic OneCycleLR scheduler
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, logger)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device, logger)
        
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        
        # Log epoch summary
        logger.info(f"Epoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"  Time: {epoch_time:.2f}s, Current LR: {current_lr:.6f}")
        logger.info("-" * 60)
        
        # Wandb logging removed for simplicity
        
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
    
    # Training completed - wandb logging removed for simplicity


if __name__ == '__main__':
    main()