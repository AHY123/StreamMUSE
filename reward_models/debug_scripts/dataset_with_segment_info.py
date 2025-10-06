#!/usr/bin/env python3
"""
Modified DiscriminativeDataset that returns exact segment information.
"""

import torch
import random
from reward_models.data_loader import DiscriminativeDataset


class DiscriminativeDatasetWithSegmentInfo(DiscriminativeDataset):
    """
    Extended dataset that returns exact segment start positions for debugging.
    """
    
    def get_sequence_segment_with_info(self, data, starts, idx, length):
        """Extract random segment and return start position info"""
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
        
        # Return segment and position info
        return segment, {
            'song_idx': idx,
            'segment_start_in_song': segment_start,
            'segment_length': self.target_length,
            'song_length': available_length,
            'absolute_start': absolute_start,
            'absolute_end': absolute_end
        }
    
    def get_sequence_segment_with_fixed_start(self, data, starts, idx, length, fixed_start):
        """Extract segment with a fixed start position (for matching real pairs)"""
        available_length = int(length.item()) if hasattr(length, 'item') else int(length)
        song_start = int(starts[idx].item())
        
        # Ensure we have enough length
        if available_length < self.target_length:
            raise ValueError(f"Song {idx} has length {available_length} < target_length {self.target_length}")
        
        # Use the provided fixed start position
        segment_start = fixed_start
        
        # Ensure the fixed start is valid
        if segment_start + self.target_length > available_length:
            raise ValueError(f"Fixed start {segment_start} + target_length {self.target_length} > song length {available_length}")
        
        # Extract segment
        absolute_start = song_start + segment_start
        absolute_end = absolute_start + self.target_length
        segment = data[absolute_start:absolute_end]
        
        # Return segment and position info
        return segment, {
            'song_idx': idx,
            'segment_start_in_song': segment_start,
            'segment_length': self.target_length,
            'song_length': available_length,
            'absolute_start': absolute_start,
            'absolute_end': absolute_end
        }
    
    def __getitem__(self, idx):
        """
        Get a sample with detailed segment information.
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
        
        # Extract segments with position info
        if is_real:
            # For REAL samples: use SAME segment start for both melody and accompaniment
            # First get melody segment
            melody_segment, mel_info = self.get_sequence_segment_with_info(
                self.melody_data, self.melody_starts, mel_idx, self.melody_lengths[mel_idx]
            )
            # Then use the SAME segment start for accompaniment (force same segment)
            acc_segment, acc_info = self.get_sequence_segment_with_fixed_start(
                self.acc_data, self.acc_starts, acc_idx, self.acc_lengths[acc_idx],
                mel_info['segment_start_in_song']  # Use same start as melody
            )
        else:
            # For FAKE samples: use different random segments (current behavior)
            melody_segment, mel_info = self.get_sequence_segment_with_info(
                self.melody_data, self.melody_starts, mel_idx, self.melody_lengths[mel_idx]
            )
            acc_segment, acc_info = self.get_sequence_segment_with_info(
                self.acc_data, self.acc_starts, acc_idx, self.acc_lengths[acc_idx]
            )
        
        # DISABLE PITCH SHIFTS FOR DEBUGGING
        # Set pitch shift to 0 to test raw segments without transposition
        pitch_shift = torch.tensor([0]).long()
        
        # Original pitch shift code (commented out for debugging):
        # mel_range = self.melody_pitch_ranges[mel_idx]  # [min, max]
        # acc_range = self.acc_pitch_ranges[acc_idx]     # [min, max]
        # min_shift = torch.maximum(mel_range[0], acc_range[0])
        # max_shift = torch.minimum(mel_range[1], acc_range[1])
        # min_shift = torch.clamp(min_shift, -6, 6)
        # max_shift = torch.clamp(max_shift, -6, 6)
        # if min_shift <= max_shift:
        #     pitch_shift = torch.randint(min_shift, max_shift + 1, (1,)).long()
        # else:
        #     pitch_shift = torch.tensor([0]).long()
        
        # Create interleaved sequence
        try:
            from reward_models.discriminative_model import create_interleaved_sequence
        except ImportError:
            # Fallback for debugging
            import sys
            import os
            debug_scripts_path = os.path.join(os.path.dirname(__file__))
            if debug_scripts_path not in sys.path:
                sys.path.append(debug_scripts_path)
            from simple_interleave import create_interleaved_sequence
        
        interleaved = create_interleaved_sequence(melody_segment, acc_segment)
        
        return {
            'sequence': interleaved,  # [2*target_length, 12]
            'label': torch.tensor(label, dtype=torch.float32),
            'pitch_shift': pitch_shift,
            'melody_info': mel_info,
            'acc_info': acc_info,
            'is_real': is_real
        }