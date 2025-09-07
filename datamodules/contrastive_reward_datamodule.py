import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from schema.model_io_schema import ContrastiveBatch, collate_contrastive_batch
import random
import numpy as np
from typing import List, Dict, Tuple
import os


class ContrastiveRewardDataset(Dataset):
    """
    Dataset for contrastive reward model training.
    Creates positive and negative melody-accompaniment pairs.
    """
    
    def __init__(self, data_path: str, target_length: int = 256, negative_sampling_ratio: float = 1.0):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to preprocessed .pt file containing melody and accompaniment data
            target_length: Target sequence length for training
            negative_sampling_ratio: Ratio of negative to positive samples
        """
        self.data_path = data_path
        self.target_length = target_length
        self.negative_sampling_ratio = negative_sampling_ratio
        
        # Load melody and accompaniment data
        self.melody_data = torch.load(data_path.replace("acc", "mel"), mmap=True)
        self.accompaniment_data = torch.load(data_path, mmap=True)
        
        # Load sequence lengths
        length_path = data_path.replace(".pt", ".length.pt")
        if os.path.exists(length_path):
            self.lengths = torch.load(length_path, mmap=True)
            self.start_indices = torch.cumsum(self.lengths, dim=0) - self.lengths
        else:
            # Fallback: assume equal length sequences
            num_sequences = len(self.melody_data) // target_length
            self.lengths = torch.full((num_sequences,), target_length)
            self.start_indices = torch.arange(0, len(self.melody_data), target_length)
        
        # Filter valid sequences (must be at least target_length)
        valid_mask = self.lengths >= target_length
        self.valid_indices = torch.where(valid_mask)[0]
        
        self.num_sequences = len(self.valid_indices)
        
    def __len__(self):
        # Return total samples including both positive and negative pairs
        return int(self.num_sequences * (1 + self.negative_sampling_ratio))
    
    def convert_polyphonic_to_flat_tokens(self, polyphonic_seq):
        """
        Convert polyphonic sequence [seq_len, 12] to flat token sequence [seq_len*12].
        This matches how main models handle polyphonic data.
        """
        # Simply flatten the polyphonic sequence like main models do
        # Ensure values are within valid range [0, 255] and convert to long
        flat_tokens = polyphonic_seq.flatten()
        flat_tokens = torch.clamp(flat_tokens, 0, 255).long()
        return flat_tokens

    def get_sequence_pair(self, melody_idx: int, accompaniment_idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get melody-accompaniment pair by indices.
        
        Args:
            melody_idx: Index of melody sequence
            accompaniment_idx: Index of accompaniment sequence
            
        Returns:
            Tuple of (melody_tokens, accompaniment_tokens, actual_length)
        """
        # Get melody sequence
        mel_start = self.start_indices[melody_idx]
        mel_seq = self.melody_data[mel_start:mel_start + self.target_length]
        
        # Get accompaniment sequence
        acc_start = self.start_indices[accompaniment_idx]
        acc_seq = self.accompaniment_data[acc_start:acc_start + self.target_length]
        
        # Pad polyphonic sequences if necessary
        if len(mel_seq) < self.target_length:
            pad_len = self.target_length - len(mel_seq)
            # Create padding with shape [pad_len, 12] filled with [254, 255, 255, ...] (EOS + padding)
            mel_pad = torch.full((pad_len, 12), 255, dtype=mel_seq.dtype)
            mel_pad[:, 0] = 254  # EOS token in first position
            mel_seq = torch.cat([mel_seq, mel_pad])
        if len(acc_seq) < self.target_length:
            pad_len = self.target_length - len(acc_seq)
            acc_pad = torch.full((pad_len, 12), 255, dtype=acc_seq.dtype)
            acc_pad[:, 0] = 254  # EOS token in first position
            acc_seq = torch.cat([acc_seq, acc_pad])
        
        # Convert polyphonic to flat token sequences (like main models)
        mel_tokens = self.convert_polyphonic_to_flat_tokens(mel_seq)
        acc_tokens = self.convert_polyphonic_to_flat_tokens(acc_seq)
            
        actual_length = min(self.lengths[melody_idx], self.lengths[accompaniment_idx], self.target_length)
        
        # Handle both tensor and int cases
        if hasattr(actual_length, 'item'):
            actual_length = actual_length.item()
        
        return mel_tokens, acc_tokens, actual_length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample.
        
        Returns:
            Dict with keys: 'melody', 'accompaniment', 'label', 'length'
        """
        # Determine if this should be a positive or negative pair
        if idx < self.num_sequences:
            # Positive pair - same sequence index
            seq_idx = self.valid_indices[idx]
            melody_tokens, acc_tokens, length = self.get_sequence_pair(seq_idx, seq_idx)
            label = 1
        else:
            # Negative pair - different sequence indices
            melody_idx = self.valid_indices[random.randint(0, self.num_sequences - 1)]
            acc_idx = self.valid_indices[random.randint(0, self.num_sequences - 1)]
            # Ensure they're different
            while acc_idx == melody_idx:
                acc_idx = self.valid_indices[random.randint(0, self.num_sequences - 1)]
            
            melody_tokens, acc_tokens, length = self.get_sequence_pair(melody_idx, acc_idx)
            label = 0
            
        return {
            'melody': melody_tokens,
            'accompaniment': acc_tokens,
            'label': label,
            'length': length
        }


class ContrastiveRewardDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for contrastive reward model training.
    """
    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str = None,
        batch_size: int = 8,
        num_workers: int = 4,
        target_length: int = 256,
        negative_sampling_ratio: float = 1.0,
        train_split_ratio: int = 10
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path or train_data_path  # Use same data for val if not specified
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_length = target_length
        self.negative_sampling_ratio = negative_sampling_ratio
        self.train_split_ratio = train_split_ratio
        
    def setup(self, stage: str = None):
        """Setup datasets for training and validation."""
        if stage == 'fit' or stage is None:
            # Create full dataset
            full_dataset = ContrastiveRewardDataset(
                self.train_data_path,
                target_length=self.target_length,
                negative_sampling_ratio=self.negative_sampling_ratio
            )
            
            # Split into train/val based on split_ratio (similar to existing pattern)
            total_size = len(full_dataset)
            val_size = total_size // self.train_split_ratio
            train_size = total_size - val_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_contrastive_batch,
            pin_memory=True
        )
        
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_contrastive_batch,
            pin_memory=True
        )