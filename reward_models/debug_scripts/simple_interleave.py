#!/usr/bin/env python3
"""
Simple standalone interleaving function for debugging.
Avoids heavy transformer imports.
"""

import torch


def create_interleaved_sequence(melody, accompaniment):
    """
    Create interleaved sequence: [acc_0, mel_0, acc_1, mel_1, ...]
    
    Args:
        melody: [seq_len, 12]
        accompaniment: [seq_len, 12] 
        
    Returns:
        interleaved: [2*seq_len, 12]
    """
    seq_len = melody.shape[0]
    assert melody.shape == accompaniment.shape, f"Shape mismatch: melody {melody.shape} vs acc {accompaniment.shape}"
    
    # Create interleaved tensor
    interleaved = torch.zeros(2 * seq_len, 12, dtype=melody.dtype)
    
    # Fill interleaved: [acc_0, mel_0, acc_1, mel_1, ...]
    interleaved[0::2] = accompaniment  # Even indices: accompaniment
    interleaved[1::2] = melody         # Odd indices: melody
    
    return interleaved