#!/usr/bin/env python3
"""
Detailed debugging of the data pipeline to understand why MIDI exports have very few notes.
Traces data flow from raw .pt files through dataloader to final output.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.data_loader import create_dataloaders, DiscriminativeDataset


def search_for_music():
    """Search through the data to find frames with actual musical content"""
    print("🎵 SEARCHING FOR ACTUAL MUSICAL CONTENT")
    print("=" * 60)
    
    melody_data = torch.load("data/reward_training_mel_cp4.pt", mmap=True)
    acc_data = torch.load("data/reward_training_acc_cp4.pt", mmap=True)
    
    print("Searching for non-empty frames...")
    
    # Search through first 10000 frames to find music
    frames_checked = 0
    music_frames_found = 0
    
    for i in range(min(10000, melody_data.shape[0])):
        mel_frame = melody_data[i].view(4, 3)
        acc_frame = acc_data[i].view(4, 3)
        
        # Count real notes (not PAD/EOS)
        mel_notes = sum(1 for note in mel_frame if note[1] != 255 and note[1] != 254)
        acc_notes = sum(1 for note in acc_frame if note[1] != 255 and note[1] != 254)
        
        if mel_notes > 0 or acc_notes > 0:
            music_frames_found += 1
            if music_frames_found <= 5:  # Show first 5 musical frames
                print(f"\nFrame {i} - Found music! Mel notes: {mel_notes}, Acc notes: {acc_notes}")
                print(f"  Melody: {mel_frame.tolist()}")
                print(f"  Acc: {acc_frame.tolist()}")
        
        frames_checked += 1
        
        if frames_checked % 1000 == 0:
            print(f"  Checked {frames_checked} frames, found {music_frames_found} with music")
    
    print(f"\nSummary: Found {music_frames_found} musical frames out of {frames_checked} checked")
    if music_frames_found == 0:
        print("❌ NO MUSICAL CONTENT FOUND - This explains the empty MIDI exports!")
    else:
        print(f"✅ Musical content exists at approximately {100*music_frames_found/frames_checked:.2f}% density")


def debug_raw_data():
    """Debug the raw .pt files to see what they contain"""
    print("🔍 DEBUGGING RAW DATA FILES")
    print("=" * 60)
    
    # Load raw data
    melody_path = "data/reward_training_mel_cp4.pt"
    acc_path = "data/reward_training_acc_cp4.pt"
    
    print(f"Loading: {melody_path}")
    melody_data = torch.load(melody_path, mmap=True)
    print(f"Melody shape: {melody_data.shape}")
    print(f"Melody range: [{melody_data.min()}, {melody_data.max()}]")
    
    print(f"\nLoading: {acc_path}")
    acc_data = torch.load(acc_path, mmap=True)
    print(f"Acc shape: {acc_data.shape}")
    print(f"Acc range: [{acc_data.min()}, {acc_data.max()}]")
    
    # Look at first few frames of melody
    print(f"\nFirst 5 frames of melody:")
    for i in range(min(5, melody_data.shape[0])):
        frame = melody_data[i]  # [12]
        frame_reshaped = frame.view(4, 3)  # [4, 3] = 4 notes per frame
        print(f"  Frame {i}: {frame.tolist()}")
        print(f"    Reshaped: {frame_reshaped.tolist()}")
        
        # Count non-pad notes
        non_pad_notes = 0
        for note_idx in range(4):
            prog, pitch, dur = frame_reshaped[note_idx]
            if pitch != 255:  # Not PAD
                non_pad_notes += 1
                print(f"      Note {note_idx}: prog={prog}, pitch={pitch}, dur={dur}")
        print(f"    Non-pad notes: {non_pad_notes}")
    
    # Same for accompaniment
    print(f"\nFirst 5 frames of accompaniment:")
    for i in range(min(5, acc_data.shape[0])):
        frame = acc_data[i]  # [12]
        frame_reshaped = frame.view(4, 3)  # [4, 3] = 4 notes per frame
        print(f"  Frame {i}: {frame.tolist()}")
        
        # Count non-pad notes
        non_pad_notes = 0
        for note_idx in range(4):
            prog, pitch, dur = frame_reshaped[note_idx]
            if pitch != 255:  # Not PAD
                non_pad_notes += 1
                print(f"      Note {note_idx}: prog={prog}, pitch={pitch}, dur={dur}")
        print(f"    Non-pad notes: {non_pad_notes}")


def debug_dataset_creation():
    """Debug the dataset creation and song extraction"""
    print("\n🔍 DEBUGGING DATASET CREATION")
    print("=" * 60)
    
    # Create dataset object
    dataset = DiscriminativeDataset(
        melody_path="data/reward_training_mel_cp4.pt",
        acc_path="data/reward_training_acc_cp4.pt",
        target_length=64
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Total melody frames: {dataset.melody_data.shape[0]}")
    print(f"Total acc frames: {dataset.acc_data.shape[0]}")
    print(f"Number of valid songs: {len(dataset.valid_indices)}")
    print(f"Valid song indices (first 10): {dataset.valid_indices[:10]}")
    print(f"Melody starts (first 10): {dataset.melody_starts[:10]}")
    print(f"Melody lengths (first 10): {dataset.melody_lengths[:10]}")
    
    # Look at a few songs
    print(f"\nAnalyzing first 3 valid songs:")
    for i in range(min(3, len(dataset.valid_indices))):
        song_idx = dataset.valid_indices[i]
        start_idx = int(dataset.melody_starts[song_idx].item())
        song_length = int(dataset.melody_lengths[song_idx].item())
        
        end_idx = start_idx + song_length
        
        print(f"\nSong {song_idx}:")
        print(f"  Frames: {start_idx} to {end_idx} (length={song_length})")
        
        # Check if song is long enough
        min_length = dataset.target_length
        if song_length >= min_length:
            print(f"  ✅ Long enough (>= {min_length})")
            
            # Look at melody and acc for this song
            mel_segment = dataset.melody_data[start_idx:start_idx + min_length]
            acc_segment = dataset.acc_data[start_idx:start_idx + min_length]
            
            # Count non-pad notes in first 10 frames
            mel_notes = 0
            acc_notes = 0
            for frame_idx in range(min(10, min_length)):
                mel_frame = mel_segment[frame_idx].view(4, 3)
                acc_frame = acc_segment[frame_idx].view(4, 3)
                
                for note_idx in range(4):
                    if mel_frame[note_idx][1] != 255:  # pitch != PAD
                        mel_notes += 1
                    if acc_frame[note_idx][1] != 255:  # pitch != PAD
                        acc_notes += 1
            
            print(f"  Melody notes in first 10 frames: {mel_notes}")
            print(f"  Acc notes in first 10 frames: {acc_notes}")
        else:
            print(f"  ❌ Too short (< {min_length})")


def debug_dataloader_output():
    """Debug what comes out of the dataloader"""
    print("\n🔍 DEBUGGING DATALOADER OUTPUT")
    print("=" * 60)
    
    # Create dataloader
    train_loader, _ = create_dataloaders(
        melody_path="data/reward_training_mel_cp4.pt",
        acc_path="data/reward_training_acc_cp4.pt",
        batch_size=2,
        target_length=64,
        num_workers=0
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    sequences = batch['sequences']  # [batch_size, 2*seq_len, 12]
    labels = batch['labels']
    pitch_shifts = batch['pitch_shifts'].squeeze(-1)
    
    print(f"Batch sequences shape: {sequences.shape}")
    print(f"Labels: {labels.tolist()}")
    print(f"Pitch shifts: {pitch_shifts.tolist()}")
    
    # Analyze each sample in the batch
    for sample_idx in range(sequences.shape[0]):
        sequence = sequences[sample_idx]  # [2*seq_len, 12]
        label = labels[sample_idx].item()
        pitch_shift = pitch_shifts[sample_idx].item()
        
        print(f"\nSample {sample_idx} (label={label}, pitch_shift={pitch_shift}):")
        print(f"  Sequence shape: {sequence.shape}")
        
        # Split into melody and accompaniment
        seq_len = sequence.shape[0] // 2
        melody = sequence[1::2]      # Odd indices: melody  
        accompaniment = sequence[0::2]  # Even indices: accompaniment
        
        print(f"  After splitting:")
        print(f"    Melody shape: {melody.shape}")
        print(f"    Acc shape: {accompaniment.shape}")
        
        # Count non-pad notes in first 10 frames
        mel_notes = 0
        acc_notes = 0
        
        for frame_idx in range(min(10, seq_len)):
            mel_frame = melody[frame_idx].view(4, 3)
            acc_frame = accompaniment[frame_idx].view(4, 3)
            
            for note_idx in range(4):
                if mel_frame[note_idx][1] != 255:  # pitch != PAD
                    mel_notes += 1
                if acc_frame[note_idx][1] != 255:  # pitch != PAD
                    acc_notes += 1
        
        print(f"    Melody notes in first 10 frames: {mel_notes}")
        print(f"    Acc notes in first 10 frames: {acc_notes}")
        
        # Show first few frames in detail
        print(f"    First 3 melody frames:")
        for frame_idx in range(min(3, seq_len)):
            frame = melody[frame_idx].view(4, 3)
            print(f"      Frame {frame_idx}: {frame.tolist()}")
        
        print(f"    First 3 acc frames:")
        for frame_idx in range(min(3, seq_len)):
            frame = accompaniment[frame_idx].view(4, 3)
            print(f"      Frame {frame_idx}: {frame.tolist()}")


def debug_interleaving_logic():
    """Debug the interleaving logic specifically"""
    print("\n🔍 DEBUGGING INTERLEAVING LOGIC")
    print("=" * 60)
    
    # Create simple test data
    mel_test = torch.tensor([
        [1, 60, 4, 255, 255, 255, 255, 255, 255, 255, 255, 255],  # 1 note
        [1, 62, 4, 255, 255, 255, 255, 255, 255, 255, 255, 255],  # 1 note
    ])  # [2, 12]
    
    acc_test = torch.tensor([
        [1, 48, 8, 1, 52, 8, 255, 255, 255, 255, 255, 255],  # 2 notes  
        [1, 50, 8, 1, 54, 8, 255, 255, 255, 255, 255, 255],  # 2 notes
    ])  # [2, 12]
    
    print("Test melody:")
    for i, frame in enumerate(mel_test):
        print(f"  Frame {i}: {frame.view(4, 3).tolist()}")
    
    print("Test accompaniment:")
    for i, frame in enumerate(acc_test):
        print(f"  Frame {i}: {frame.view(4, 3).tolist()}")
    
    # Import the interleaving function
    from reward_models.discriminative_model import create_interleaved_sequence
    
    # Test interleaving
    interleaved = create_interleaved_sequence(mel_test, acc_test)
    print(f"\nInterleaved result shape: {interleaved.shape}")
    
    # Show the interleaved result
    print("Interleaved sequence:")
    for i, frame in enumerate(interleaved):
        frame_type = "ACC" if i % 2 == 0 else "MEL"
        print(f"  Frame {i} ({frame_type}): {frame.view(4, 3).tolist()}")
    
    # Test de-interleaving
    seq_len = interleaved.shape[0] // 2
    extracted_mel = interleaved[1::2]  # Odd indices
    extracted_acc = interleaved[0::2]  # Even indices
    
    print(f"\nDe-interleaved melody:")
    for i, frame in enumerate(extracted_mel):
        print(f"  Frame {i}: {frame.view(4, 3).tolist()}")
    
    print(f"De-interleaved accompaniment:")
    for i, frame in enumerate(extracted_acc):
        print(f"  Frame {i}: {frame.view(4, 3).tolist()}")
    
    # Check if they match the original
    mel_match = torch.equal(extracted_mel, mel_test)
    acc_match = torch.equal(extracted_acc, acc_test)
    
    print(f"\nMatching check:")
    print(f"  Melody matches: {mel_match}")
    print(f"  Acc matches: {acc_match}")


def main():
    print("🐛 COMPREHENSIVE DATA PIPELINE DEBUG")
    print("=" * 60)
    
    search_for_music()
    debug_raw_data()
    debug_dataset_creation()
    debug_dataloader_output()
    debug_interleaving_logic()
    
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    print("Review the output above to identify:")
    print("1. Are the raw .pt files empty or mostly PAD tokens?")
    print("2. Is the song boundary detection working?")
    print("3. Is the interleaving/de-interleaving logic correct?")
    print("4. Are the dataloader samples actually getting real notes?")


if __name__ == "__main__":
    main()