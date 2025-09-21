import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Tuple


class DiscriminativeDataset(Dataset):
    """
    Dataset for discriminative reward model training.
    Creates real and fake melody-accompaniment pairs.
    """
    
    def __init__(self, melody_path: str, acc_path: str, target_length: int = 384):
        """
        Initialize dataset.
        
        Args:
            melody_path: Path to melody .pt file
            acc_path: Path to accompaniment .pt file  
            target_length: Target sequence length for training
        """
        self.target_length = target_length
        
        # Load data
        self.melody_data = torch.load(melody_path, mmap=True)  # [num_sequences, seq_len, 12]
        self.acc_data = torch.load(acc_path, mmap=True)       # [num_sequences, seq_len, 12]
        
        # Load lengths and pitch shift ranges
        melody_length_path = melody_path.replace('.pt', '.length.pt')
        acc_length_path = acc_path.replace('.pt', '.length.pt')
        melody_pitch_range_path = melody_path.replace('.pt', '.pitch_shift_range.pt')
        acc_pitch_range_path = acc_path.replace('.pt', '.pitch_shift_range.pt')
        
        self.melody_lengths = torch.load(melody_length_path, mmap=True)
        self.acc_lengths = torch.load(acc_length_path, mmap=True)
        self.melody_pitch_ranges = torch.load(melody_pitch_range_path, mmap=True)
        self.acc_pitch_ranges = torch.load(acc_pitch_range_path, mmap=True)
        
        # Find valid sequences (both melody and acc have sufficient length)
        min_lengths = torch.minimum(self.melody_lengths, self.acc_lengths)
        valid_mask = min_lengths >= target_length
        self.valid_indices = torch.where(valid_mask)[0]
        
        print(f"Loaded {len(self.melody_data)} total sequences")
        print(f"Found {len(self.valid_indices)} sequences with length >= {target_length}")
        
        # Each valid sequence generates 2 samples: 1 real + 1 fake
        self.num_samples = len(self.valid_indices) * 2
        
    def __len__(self):
        return self.num_samples
    
    def get_sequence_segment(self, data, idx, length):
        """Extract random segment of target_length from sequence"""
        available_length = int(length.item()) if hasattr(length, 'item') else int(length)
        
        # Ensure we have enough length
        if available_length < self.target_length:
            raise ValueError(f"Sequence {idx} has length {available_length} < target_length {self.target_length}")
        
        if available_length == self.target_length:
            start_idx = 0
        else:
            start_idx = random.randint(0, available_length - self.target_length)
        
        segment = data[idx, start_idx:start_idx + self.target_length]
        
        # Validate segment
        if segment.shape[0] != self.target_length:
            raise ValueError(f"Extracted segment has wrong length: {segment.shape[0]} != {self.target_length}")
        
        return segment
    
    def __getitem__(self, idx):
        """
        Get a sample: either real pair (label=1) or fake pair (label=0)
        
        Returns:
            interleaved_sequence: [2*target_length, 12] 
            label: 0 (fake) or 1 (real)
            pitch_shift: random pitch shift value
        """
        # Determine if this is a real or fake sample
        is_real = idx % 2 == 0
        sequence_idx = idx // 2
        
        valid_idx = self.valid_indices[sequence_idx]
        
        if is_real:
            # Real pair: same sequence index for both melody and accompaniment
            mel_idx = valid_idx
            acc_idx = valid_idx
            label = 1.0
        else:
            # Fake pair: different sequence indices
            mel_idx = valid_idx
            
            # Sample a different accompaniment index
            available_indices = list(range(len(self.valid_indices)))
            available_indices.remove(sequence_idx)  # Remove the melody index
            fake_seq_idx = random.choice(available_indices)
            acc_idx = self.valid_indices[fake_seq_idx]
            label = 0.0
        
        # Extract segments
        melody_segment = self.get_sequence_segment(
            self.melody_data, mel_idx, self.melody_lengths[mel_idx]
        )
        acc_segment = self.get_sequence_segment(
            self.acc_data, acc_idx, self.acc_lengths[acc_idx]
        )
        
        # Generate random pitch shift using per-sequence ranges (same as main model)
        mel_range = self.melody_pitch_ranges[mel_idx]  # [min, max]
        acc_range = self.acc_pitch_ranges[acc_idx]     # [min, max]
        
        # Find the valid pitch shift range for both melody and accompaniment
        min_shift = torch.maximum(mel_range[0], acc_range[0])
        max_shift = torch.minimum(mel_range[1], acc_range[1])
        
        # Generate random pitch shift within valid range
        if min_shift <= max_shift:
            pitch_shift = torch.randint(min_shift, max_shift + 1, (1,)).long()
        else:
            # If no valid range, use 0 (no shift)
            pitch_shift = torch.tensor([0]).long()
        
        # Create interleaved sequence [acc_0, mel_0, acc_1, mel_1, ...]
        from discriminative_model import create_interleaved_sequence
        interleaved = create_interleaved_sequence(melody_segment, acc_segment)
        
        return {
            'sequence': interleaved,  # [2*target_length, 12]
            'label': torch.tensor(label, dtype=torch.float32),
            'pitch_shift': pitch_shift[0]  # Scalar pitch shift value
        }


def collate_batch(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched sequences, labels, and pitch shifts
    """
    sequences = torch.stack([item['sequence'] for item in batch])     # [batch_size, 2*seq_len, 12]
    labels = torch.stack([item['label'] for item in batch])          # [batch_size]
    pitch_shifts = torch.stack([item['pitch_shift'] for item in batch])  # [batch_size]
    
    return {
        'sequences': sequences,
        'labels': labels,
        'pitch_shifts': pitch_shifts
    }


def create_dataloaders(melody_path: str, acc_path: str, batch_size: int = 8, 
                      target_length: int = 384, train_split: float = 0.9, 
                      num_workers: int = 4):
    """
    Create train and validation dataloaders.
    
    Args:
        melody_path: Path to melody .pt file
        acc_path: Path to accompaniment .pt file
        batch_size: Batch size for training
        target_length: Sequence length for training
        train_split: Ratio for train/validation split
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = DiscriminativeDataset(melody_path, acc_path, target_length)
    
    # Split into train/val
    total_sequences = len(full_dataset.valid_indices)
    train_size = int(total_sequences * train_split)
    val_size = total_sequences - train_size
    
    # Create train dataset with subset of valid indices
    train_dataset = DiscriminativeDataset(melody_path, acc_path, target_length)
    train_dataset.valid_indices = full_dataset.valid_indices[:train_size]
    train_dataset.num_samples = len(train_dataset.valid_indices) * 2
    
    # Create val dataset with remaining indices
    val_dataset = DiscriminativeDataset(melody_path, acc_path, target_length)
    val_dataset.valid_indices = full_dataset.valid_indices[train_size:]
    val_dataset.num_samples = len(val_dataset.valid_indices) * 2
    
    print(f"Train sequences: {len(train_dataset.valid_indices)} ({train_dataset.num_samples} samples)")
    print(f"Val sequences: {len(val_dataset.valid_indices)} ({val_dataset.num_samples} samples)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    return train_loader, val_loader