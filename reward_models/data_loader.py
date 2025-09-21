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
        
        # Load concatenated data - all sequences are in one big tensor
        self.melody_data = torch.load(melody_path, mmap=True)  # [total_frames, 12]
        self.acc_data = torch.load(acc_path, mmap=True)       # [total_frames, 12]
        
        # Load lengths and pitch shift ranges
        melody_length_path = melody_path.replace('.pt', '.length.pt')
        acc_length_path = acc_path.replace('.pt', '.length.pt')
        melody_pitch_range_path = melody_path.replace('.pt', '.pitch_shift_range.pt')
        acc_pitch_range_path = acc_path.replace('.pt', '.pitch_shift_range.pt')
        
        self.melody_lengths = torch.load(melody_length_path, mmap=True)  # [num_songs]
        self.acc_lengths = torch.load(acc_length_path, mmap=True)        # [num_songs]
        self.melody_pitch_ranges = torch.load(melody_pitch_range_path, mmap=True).reshape(-1, 2)  # [num_songs, 2]
        self.acc_pitch_ranges = torch.load(acc_pitch_range_path, mmap=True).reshape(-1, 2)        # [num_songs, 2]
        
        # Calculate start indices for each song in the concatenated data
        self.melody_starts = torch.zeros(len(self.melody_lengths), dtype=torch.long)
        self.melody_starts[1:] = torch.cumsum(self.melody_lengths[:-1], dim=0)
        
        self.acc_starts = torch.zeros(len(self.acc_lengths), dtype=torch.long)
        self.acc_starts[1:] = torch.cumsum(self.acc_lengths[:-1], dim=0)
        
        # Find valid sequences (both melody and acc have sufficient length)
        min_lengths = torch.minimum(self.melody_lengths, self.acc_lengths)
        valid_mask = min_lengths >= target_length
        self.valid_indices = torch.where(valid_mask)[0]
        
        print(f"Loaded {len(self.melody_lengths)} total songs")
        print(f"Total melody frames: {len(self.melody_data)}")
        print(f"Total acc frames: {len(self.acc_data)}")
        print(f"Found {len(self.valid_indices)} songs with length >= {target_length}")
        
        if len(self.valid_indices) == 0:
            raise ValueError("No valid sequences found! Check your data and target_length.")
        
        # Each valid sequence generates 2 samples: 1 real + 1 fake
        self.num_samples = len(self.valid_indices) * 2
        
    def __len__(self):
        return self.num_samples
    
    def get_sequence_segment(self, data, starts, idx, length):
        """Extract random segment of target_length from concatenated sequence data"""
        available_length = int(length.item()) if hasattr(length, 'item') else int(length)
        song_start = int(starts[idx].item())
        
        # Ensure we have enough length
        if available_length < self.target_length:
            raise ValueError(f"Song {idx} has length {available_length} < target_length {self.target_length}")
        
        # Random start within the song
        if available_length == self.target_length:
            segment_start = 0
        else:
            segment_start = random.randint(0, available_length - self.target_length)
        
        # Calculate absolute indices in the concatenated data
        abs_start = song_start + segment_start
        abs_end = abs_start + self.target_length
        
        # Extract segment from concatenated data
        segment = data[abs_start:abs_end]
        
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
        
        # Extract segments from concatenated data
        melody_segment = self.get_sequence_segment(
            self.melody_data, self.melody_starts, mel_idx, self.melody_lengths[mel_idx]
        )
        acc_segment = self.get_sequence_segment(
            self.acc_data, self.acc_starts, acc_idx, self.acc_lengths[acc_idx]
        )
        
        # Generate random pitch shift using per-sequence ranges (same as main model)
        mel_range = self.melody_pitch_ranges[mel_idx]  # [min, max]
        acc_range = self.acc_pitch_ranges[acc_idx]     # [min, max]
        
        # Find the valid pitch shift range for both melody and accompaniment
        min_shift = torch.maximum(mel_range[0], acc_range[0])
        max_shift = torch.minimum(mel_range[1], acc_range[1])
        
        # Generate random pitch shift within valid range (same format as main model)
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
            'pitch_shift': pitch_shift  # Keep as 1-element tensor like main model
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