import torch
from torch.utils.data import Dataset, DataLoader
import random
from typing import Tuple


class DiscriminativeDataset(Dataset):
    """
    Dataset for discriminative reward model training.
    Creates real and fake melody-accompaniment pairs.
    """
    
    def __init__(self, melody_path: str, acc_path: str, target_length: int = 384, 
                 quality_filter_mode: str = "resample"):
        """
        Initialize dataset.
        
        Args:
            melody_path: Path to melody .pt file
            acc_path: Path to accompaniment .pt file  
            target_length: Target sequence length for training
            quality_filter_mode: How to handle quality filtering
                - "resample": Try multiple times to find good segments (same number of samples)
                - "strict": Skip songs that don't have good segments (fewer samples, higher quality)
        """
        self.target_length = target_length
        self.quality_filter_mode = quality_filter_mode
        
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
        length_valid_indices = torch.where(valid_mask)[0]
        
        print(f"Loaded {len(self.melody_lengths)} total songs")
        print(f"Total melody frames: {len(self.melody_data)}")
        print(f"Total acc frames: {len(self.acc_data)}")
        print(f"Found {len(length_valid_indices)} songs with length >= {target_length}")
        
        # Apply quality filtering if in strict mode
        if quality_filter_mode == "strict":
            print(f"Applying strict quality filtering (50% bar occupancy)...")
            quality_valid_indices = []
            
            for idx in length_valid_indices:
                idx = int(idx.item())
                
                # Test if we can find quality segments for both melody and accompaniment
                mel_has_quality = self._test_song_quality(
                    self.melody_data, self.melody_starts, idx, self.melody_lengths[idx]
                )
                acc_has_quality = self._test_song_quality(
                    self.acc_data, self.acc_starts, idx, self.acc_lengths[idx]
                )
                
                if mel_has_quality and acc_has_quality:
                    quality_valid_indices.append(idx)
            
            self.valid_indices = torch.tensor(quality_valid_indices, dtype=torch.long)
            print(f"After quality filtering: {len(self.valid_indices)} songs remain")
        else:
            self.valid_indices = length_valid_indices
            print(f"Using resample mode - will try to find quality segments during training")
        
        if len(self.valid_indices) == 0:
            raise ValueError("No valid sequences found! Check your data and target_length.")
        
        # Each valid sequence generates 2 samples: 1 real + 1 fake
        self.num_samples = len(self.valid_indices) * 2
        
    def __len__(self):
        return self.num_samples
    
    def _test_song_quality(self, data, starts, idx, length, num_tests=3):
        """
        Test if a song can produce quality segments meeting the bar occupancy requirement.
        
        Args:
            data: Concatenated sequence data
            starts: Start indices for each song
            idx: Song index to test
            length: Song length
            num_tests: Number of random segments to test
            
        Returns:
            bool: True if at least one quality segment can be extracted
        """
        available_length = int(length.item()) if hasattr(length, 'item') else int(length)
        song_start = int(starts[idx].item())
        
        if available_length < self.target_length:
            return False
        
        # Try a few random segments to see if any meet quality requirements
        for _ in range(num_tests):
            # Get random bar-aligned start
            if available_length == self.target_length:
                segment_start = 0
            else:
                segment_start = self.get_bar_aligned_segment_start(available_length)
            
            # Extract segment
            abs_start = song_start + segment_start
            abs_end = abs_start + self.target_length
            segment = data[abs_start:abs_end]
            
            # Check if this segment meets quality requirements
            if self.has_sufficient_content(segment):
                return True
        
        return False
    
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
    
    def get_sequence_segment_with_start(self, data, starts, idx, length):
        """Extract random segment and return both segment and start position"""
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
        
        # Extract segment
        absolute_start = song_start + segment_start
        absolute_end = absolute_start + self.target_length
        segment = data[absolute_start:absolute_end]
        
        return segment, segment_start
    
    def get_sequence_segment_fixed_start(self, data, starts, idx, length, fixed_start):
        """Extract segment with fixed start position"""
        available_length = int(length.item()) if hasattr(length, 'item') else int(length)
        song_start = int(starts[idx].item())
        
        # Ensure we have enough length
        if available_length < self.target_length:
            raise ValueError(f"Song {idx} has length {available_length} < target_length {self.target_length}")
        
        # Use fixed start position
        segment_start = fixed_start
        if segment_start + self.target_length > available_length:
            raise ValueError(f"Fixed start {segment_start} + target_length {self.target_length} > song length {available_length}")
        
        # Extract segment
        absolute_start = song_start + segment_start
        absolute_end = absolute_start + self.target_length
        segment = data[absolute_start:absolute_end]
        
        return segment
    
    def has_sufficient_content(self, segment, min_bar_occupancy=0.5):
        """
        Check if a segment has sufficient musical content.
        Requires at least 50% of bars to have at least 1 note.
        
        Args:
            segment: [seq_len, 12] tensor representing musical sequence
            min_bar_occupancy: Minimum fraction of bars that must have notes (default: 0.5 = 50%)
            
        Returns:
            bool: True if segment has sufficient content
        """
        # Reshape to [seq_len, 4, 3] (4 notes per frame, 3 values each: program, pitch, duration)
        notes = segment.view(segment.shape[0], 4, 3)
        
        # Group frames into bars (16 frames = 1 bar)
        bar_size = 16
        num_bars = segment.shape[0] // bar_size
        
        if num_bars == 0:
            return False  # Segment too short to contain even one bar
        
        bars_with_notes = 0
        
        for bar_idx in range(num_bars):
            bar_start = bar_idx * bar_size
            bar_end = min(bar_start + bar_size, segment.shape[0])
            
            # Check if this bar has any notes
            bar_has_notes = False
            for frame_idx in range(bar_start, bar_end):
                for note_idx in range(notes.shape[1]):  # 4 notes per frame
                    pitch = notes[frame_idx, note_idx, 1]
                    if pitch not in [254, 255]:  # Not EOS or PAD
                        bar_has_notes = True
                        break
                if bar_has_notes:
                    break
            
            if bar_has_notes:
                bars_with_notes += 1
        
        # Calculate occupancy rate
        occupancy_rate = bars_with_notes / num_bars
        
        return occupancy_rate >= min_bar_occupancy
    
    def get_bar_aligned_segment_start(self, available_length):
        """
        Get a random segment start that aligns with bar boundaries.
        
        Args:
            available_length: Total length of the song in frames
            
        Returns:
            int: Bar-aligned segment start position
        """
        # Calculate maximum possible start that leaves room for target_length
        max_start = available_length - self.target_length
        
        # Align to bar boundaries (16 frames = 1 bar)
        # Find all valid bar-aligned starts
        bar_size = 16
        max_start_bars = max_start // bar_size
        
        if max_start_bars <= 0:
            # If song is too short for bar alignment, use frame 0
            return 0
        
        # Choose random bar and convert to frames
        random_bar = random.randint(0, max_start_bars)
        return random_bar * bar_size
    
    def get_sequence_segment_with_quality_check(self, data, starts, idx, length, max_attempts=10):
        """
        Extract segment with bar alignment and quality checks.
        
        Args:
            data: Concatenated sequence data
            starts: Start indices for each song
            idx: Song index
            length: Song length
            max_attempts: Maximum attempts to find a good segment
            
        Returns:
            segment: [target_length, 12] tensor
        """
        available_length = int(length.item()) if hasattr(length, 'item') else int(length)
        song_start = int(starts[idx].item())
        
        # Ensure we have enough length
        if available_length < self.target_length:
            raise ValueError(f"Song {idx} has length {available_length} < target_length {self.target_length}")
        
        # Try multiple times to get a good segment
        for attempt in range(max_attempts):
            # Get bar-aligned segment start
            if available_length == self.target_length:
                segment_start = 0
            else:
                segment_start = self.get_bar_aligned_segment_start(available_length)
            
            # Extract segment
            abs_start = song_start + segment_start
            abs_end = abs_start + self.target_length
            segment = data[abs_start:abs_end]
            
            # Validate segment length
            if segment.shape[0] != self.target_length:
                raise ValueError(f"Extracted segment has wrong length: {segment.shape[0]} != {self.target_length}")
            
            # Check if segment has sufficient musical content
            if self.has_sufficient_content(segment):
                return segment
            
            # If this attempt failed and we have more attempts, try again
            if attempt < max_attempts - 1:
                continue
        
        # If all attempts failed, behavior depends on filter mode
        if self.quality_filter_mode == "strict":
            raise ValueError(f"Could not find quality segment for song {idx} after {max_attempts} attempts in strict mode")
        else:
            # In resample mode, return the last segment anyway (fallback)
            print(f"Warning: Could not find segment with sufficient content for song {idx} after {max_attempts} attempts")
            return segment
    
    def get_sequence_segment_with_start_and_quality(self, data, starts, idx, length, max_attempts=10):
        """
        Extract segment with bar alignment, quality checks, and return start position.
        """
        available_length = int(length.item()) if hasattr(length, 'item') else int(length)
        song_start = int(starts[idx].item())
        
        # Ensure we have enough length
        if available_length < self.target_length:
            raise ValueError(f"Song {idx} has length {available_length} < target_length {self.target_length}")
        
        # Try multiple times to get a good segment
        for attempt in range(max_attempts):
            # Get bar-aligned segment start
            if available_length == self.target_length:
                segment_start = 0
            else:
                segment_start = self.get_bar_aligned_segment_start(available_length)
            
            # Extract segment
            abs_start = song_start + segment_start
            abs_end = abs_start + self.target_length
            segment = data[abs_start:abs_end]
            
            # Check if segment has sufficient musical content
            if self.has_sufficient_content(segment):
                return segment, segment_start
            
            # If this attempt failed and we have more attempts, try again
            if attempt < max_attempts - 1:
                continue
        
        # If all attempts failed, behavior depends on filter mode
        if self.quality_filter_mode == "strict":
            raise ValueError(f"Could not find quality segment for song {idx} after {max_attempts} attempts in strict mode")
        else:
            # In resample mode, return the last segment anyway (fallback)
            print(f"Warning: Could not find segment with sufficient content for song {idx} after {max_attempts} attempts")
            return segment, segment_start
    
    def get_sequence_segment_fixed_start_bar_aligned(self, data, starts, idx, length, fixed_start):
        """
        Extract segment with fixed start position, ensuring bar alignment.
        """
        available_length = int(length.item()) if hasattr(length, 'item') else int(length)
        song_start = int(starts[idx].item())
        
        # Ensure we have enough length
        if available_length < self.target_length:
            raise ValueError(f"Song {idx} has length {available_length} < target_length {self.target_length}")
        
        # Ensure fixed start is bar-aligned
        bar_size = 16
        aligned_start = (fixed_start // bar_size) * bar_size
        
        # Make sure the aligned start still fits
        if aligned_start + self.target_length > available_length:
            # If alignment pushes us out of bounds, use the previous bar
            aligned_start = max(0, aligned_start - bar_size)
        
        # Extract segment
        absolute_start = song_start + aligned_start
        absolute_end = absolute_start + self.target_length
        segment = data[absolute_start:absolute_end]
        
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
            
            if len(available_indices) > 0:
                # Normal case: choose different song
                fake_seq_idx = random.choice(available_indices)
                acc_idx = self.valid_indices[fake_seq_idx]
            else:
                # Edge case: only 1 song available, use same song but different segments
                # This creates fake pairs by using different timeframes from the same song
                fake_seq_idx = sequence_idx
                acc_idx = valid_idx
                
            label = 0.0
        
        # Extract segments from concatenated data with quality checks and bar alignment
        if is_real:
            # For REAL samples: use SAME segment start for both melody and accompaniment
            # First get melody segment with quality check and bar alignment
            melody_segment, segment_start = self.get_sequence_segment_with_start_and_quality(
                self.melody_data, self.melody_starts, mel_idx, self.melody_lengths[mel_idx]
            )
            # Use the same segment start for accompaniment, ensuring bar alignment
            acc_segment = self.get_sequence_segment_fixed_start_bar_aligned(
                self.acc_data, self.acc_starts, acc_idx, self.acc_lengths[acc_idx], segment_start
            )
        else:
            # For FAKE samples: use different random segments with quality checks
            if mel_idx == acc_idx:
                # Edge case: same song, ensure different segments for fake pair
                melody_segment, mel_start = self.get_sequence_segment_with_start_and_quality(
                    self.melody_data, self.melody_starts, mel_idx, self.melody_lengths[mel_idx]
                )
                
                # For accompaniment, try to get a segment that doesn't overlap with melody
                # This ensures the fake pair is actually from different timeframes
                max_attempts = 10
                for attempt in range(max_attempts):
                    acc_segment, acc_start = self.get_sequence_segment_with_start_and_quality(
                        self.acc_data, self.acc_starts, acc_idx, self.acc_lengths[acc_idx]
                    )
                    
                    # Check if segments overlap (if they don't overlap much, it's a good fake pair)
                    if abs(mel_start - acc_start) >= self.target_length // 2:  # At least 50% non-overlap
                        break
                    
                    # If last attempt, use whatever we got
                    if attempt == max_attempts - 1:
                        break
            else:
                # Normal case: different songs
                melody_segment = self.get_sequence_segment_with_quality_check(
                    self.melody_data, self.melody_starts, mel_idx, self.melody_lengths[mel_idx]
                )
                acc_segment = self.get_sequence_segment_with_quality_check(
                    self.acc_data, self.acc_starts, acc_idx, self.acc_lengths[acc_idx]
                )
        
        # Generate random pitch shift using per-sequence ranges (same as main model)
        mel_range = self.melody_pitch_ranges[mel_idx]  # [min, max]
        acc_range = self.acc_pitch_ranges[acc_idx]     # [min, max]
        
        # Find the valid pitch shift range for both melody and accompaniment
        min_shift = torch.maximum(mel_range[0], acc_range[0])
        max_shift = torch.minimum(mel_range[1], acc_range[1])
        
        # Limit pitch shifts to reasonable range (±6 semitones = half octave)
        min_shift = torch.clamp(min_shift, -6, 6)
        max_shift = torch.clamp(max_shift, -6, 6)
        
        # Generate random pitch shift within valid range (same format as main model)
        if min_shift <= max_shift:
            pitch_shift = torch.randint(min_shift, max_shift + 1, (1,)).long()
        else:
            # If no valid range, use 0 (no shift)
            pitch_shift = torch.tensor([0]).long()
        
        # Create interleaved sequence [acc_0, mel_0, acc_1, mel_1, ...]
        try:
            from reward_models.discriminative_model import create_interleaved_sequence
        except ImportError:
            # Fallback for debugging - avoid heavy transformer imports
            import sys
            import os
            debug_scripts_path = os.path.join(os.path.dirname(__file__), "debug_scripts")
            if debug_scripts_path not in sys.path:
                sys.path.append(debug_scripts_path)
            from simple_interleave import create_interleaved_sequence
        interleaved = create_interleaved_sequence(melody_segment, acc_segment)
        
        return {
            'sequence': interleaved,  # [2*target_length, 12]
            'label': torch.tensor(label, dtype=torch.float32),
            'pitch_shift': pitch_shift.squeeze()  # Remove extra dimension: [1] -> scalar
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
                      num_workers: int = 4, quality_filter_mode: str = "resample"):
    """
    Create train and validation dataloaders.
    
    Args:
        melody_path: Path to melody .pt file
        acc_path: Path to accompaniment .pt file
        batch_size: Batch size for training
        target_length: Sequence length for training
        train_split: Ratio for train/validation split
        num_workers: Number of workers for data loading
        quality_filter_mode: "resample" (try multiple times) or "strict" (fewer high-quality samples)
        
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = DiscriminativeDataset(melody_path, acc_path, target_length, quality_filter_mode)
    
    # Split into train/val
    total_sequences = len(full_dataset.valid_indices)
    train_size = int(total_sequences * train_split)
    val_size = total_sequences - train_size
    
    # Create train dataset with subset of valid indices
    train_dataset = DiscriminativeDataset(melody_path, acc_path, target_length, quality_filter_mode)
    train_dataset.valid_indices = full_dataset.valid_indices[:train_size]
    train_dataset.num_samples = len(train_dataset.valid_indices) * 2
    
    # Create val dataset with remaining indices
    val_dataset = DiscriminativeDataset(melody_path, acc_path, target_length, quality_filter_mode)
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