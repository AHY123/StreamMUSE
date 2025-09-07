import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from schema.model_io_schema import DiscriminativeBatch, collate_discriminative_batch
import random
import numpy as np
from typing import List, Dict, Tuple
import os


class DiscriminativeRewardDataset(Dataset):
    """
    Dataset for discriminative reward model training.
    Creates interleaved sequences with real/fake labels.
    """
    
    def __init__(self, data_path: str, target_length: int = 256, negative_ratio: float = 1.0):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to preprocessed .pt file containing melody and accompaniment data
            target_length: Target sequence length for training
            negative_ratio: Ratio of fake to real samples
        """
        self.data_path = data_path
        self.target_length = target_length
        self.negative_ratio = negative_ratio
        
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
        # Return total samples including both real and fake sequences
        return int(self.num_sequences * (1 + self.negative_ratio))
    
    def convert_polyphonic_to_flat_tokens(self, polyphonic_seq):
        """
        Convert polyphonic sequence [seq_len, 12] to flat token sequence [seq_len*12].
        This matches how main models handle polyphonic data.
        """
        # Simply flatten the polyphonic sequence like main models do
        flat_tokens = polyphonic_seq.flatten().long()
        return flat_tokens

    def create_interleaved_sequence(self, melody_tokens: torch.Tensor, accompaniment_tokens: torch.Tensor) -> torch.Tensor:
        """
        Create interleaved sequence from melody and accompaniment tokens.
        
        Args:
            melody_tokens: [seq_len] melody sequence
            accompaniment_tokens: [seq_len] accompaniment sequence
            
        Returns:
            torch.Tensor: [2*seq_len] interleaved sequence
        """
        # Ensure same length (truncate if necessary)
        min_len = min(len(melody_tokens), len(accompaniment_tokens))
        melody_tokens = melody_tokens[:min_len]
        accompaniment_tokens = accompaniment_tokens[:min_len]
        
        # Simple interleaving: alternate tokens
        interleaved = torch.stack([melody_tokens, accompaniment_tokens], dim=1).flatten()
        return interleaved
    
    def get_sequence_data(self, seq_idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get melody and accompaniment sequences by index.
        
        Args:
            seq_idx: Sequence index
            
        Returns:
            Tuple of (melody_tokens, accompaniment_tokens, actual_length)
        """
        start = self.start_indices[seq_idx]
        length = min(self.lengths[seq_idx], self.target_length)
        
        # Get polyphonic sequences
        melody_seq = self.melody_data[start:start + length]
        acc_seq = self.accompaniment_data[start:start + length]
        
        # Pad polyphonic sequences if necessary
        if len(melody_seq) < self.target_length:
            pad_len = self.target_length - len(melody_seq)
            # Create padding with shape [pad_len, 12] filled with [254, 255, 255, ...] (EOS + padding)
            mel_pad = torch.full((pad_len, 12), 255, dtype=melody_seq.dtype)
            mel_pad[:, 0] = 254  # EOS token in first position
            melody_seq = torch.cat([melody_seq, mel_pad])
        if len(acc_seq) < self.target_length:
            pad_len = self.target_length - len(acc_seq)
            acc_pad = torch.full((pad_len, 12), 255, dtype=acc_seq.dtype)
            acc_pad[:, 0] = 254  # EOS token in first position
            acc_seq = torch.cat([acc_seq, acc_pad])
        
        # Convert polyphonic to token sequences
        melody_tokens = self.convert_polyphonic_to_flat_tokens(melody_seq)
        acc_tokens = self.convert_polyphonic_to_flat_tokens(acc_seq)
            
        # Handle both tensor and int cases
        if hasattr(length, 'item'):
            length = length.item()
        
        return melody_tokens, acc_tokens, length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample.
        
        Returns:
            Dict with keys: 'sequence', 'label', 'mask', 'length'
        """
        if idx < self.num_sequences:
            # Real sample - melody and accompaniment from same sequence
            seq_idx = self.valid_indices[idx]
            melody_tokens, acc_tokens, length = self.get_sequence_data(seq_idx)
            label = 1  # Real
        else:
            # Fake sample - melody and accompaniment from different sequences
            melody_idx = self.valid_indices[random.randint(0, self.num_sequences - 1)]
            acc_idx = self.valid_indices[random.randint(0, self.num_sequences - 1)]
            
            # Ensure they're different
            while acc_idx == melody_idx:
                acc_idx = self.valid_indices[random.randint(0, self.num_sequences - 1)]
            
            melody_tokens, _, _ = self.get_sequence_data(melody_idx)
            _, acc_tokens, _ = self.get_sequence_data(acc_idx)
            
            # Use minimum length for safety
            length = min(self.lengths[melody_idx], self.lengths[acc_idx], self.target_length)
            if hasattr(length, 'item'):
                length = length.item()
            label = 0  # Fake
        
        # Create interleaved sequence
        interleaved_seq = self.create_interleaved_sequence(melody_tokens, acc_tokens)
        
        # Create attention mask for non-padding tokens
        attention_mask = (interleaved_seq != 255).float()  # PAD_TOKEN = 255 in polyphonic data
        
        return {
            'sequence': interleaved_seq,
            'label': label,
            'mask': attention_mask,
            'length': len(interleaved_seq)
        }


class DiscriminativeRewardDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for discriminative reward model training.
    """
    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str = None,
        batch_size: int = 8,
        num_workers: int = 4,
        target_length: int = 256,
        negative_ratio: float = 1.0,
        train_split_ratio: int = 10
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path or train_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_length = target_length
        self.negative_ratio = negative_ratio
        self.train_split_ratio = train_split_ratio
        
    def setup(self, stage: str = None):
        """Setup datasets for training and validation."""
        if stage == 'fit' or stage is None:
            # Create full dataset
            full_dataset = DiscriminativeRewardDataset(
                self.train_data_path,
                target_length=self.target_length,
                negative_ratio=self.negative_ratio
            )
            
            # Split into train/val based on split_ratio
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
            collate_fn=collate_discriminative_batch,
            pin_memory=True
        )
        
    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_discriminative_batch,
            pin_memory=True
        )