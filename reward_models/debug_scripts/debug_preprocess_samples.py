#!/usr/bin/env python3
"""
Phase 1: Debug preprocessing pipeline with 5 known good samples.
Converts the mini samples through exact same preprocessing as main model.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "preprocess"))

from preprocess_midi2pt_dataset import preprocess_midi, DURATION_TEMPLATES, tensor_to_midi
import pretty_midi


def debug_preprocess_mini_samples():
    """
    Process 5 mini samples through the exact same preprocessing pipeline as main model.
    Creates debug .pt files and verification outputs.
    """
    print("🔧 PHASE 1: DEBUGGING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Input and output paths
    input_dir = Path("debug_mini_samples")
    output_dir = Path("reward_models/debug_data")
    verification_dir = Path("reward_models/debug_outputs")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Verification directory: {verification_dir.absolute()}")
    
    # Check input files exist
    mel_dir = input_dir / "mel"
    acc_dir = input_dir / "acc"
    
    if not mel_dir.exists() or not acc_dir.exists():
        print(f"❌ Error: Could not find mel/ and acc/ directories in {input_dir}")
        return
    
    mel_files = sorted(list(mel_dir.glob("*.mid")))
    acc_files = sorted(list(acc_dir.glob("*.mid")))
    
    print(f"Found {len(mel_files)} melody files: {[f.name for f in mel_files]}")
    print(f"Found {len(acc_files)} accompaniment files: {[f.name for f in acc_files]}")
    
    if len(mel_files) != len(acc_files):
        print(f"❌ Error: Mismatch in number of melody ({len(mel_files)}) vs acc ({len(acc_files)}) files")
        return
    
    # Process each file pair
    all_mel_data = []
    all_acc_data = []
    mel_lengths = []
    acc_lengths = []
    mel_pitch_ranges = []
    acc_pitch_ranges = []
    song_names = []
    
    print(f"\n🎵 Processing {len(mel_files)} sample pairs...")
    
    for i, (mel_file, acc_file) in enumerate(zip(mel_files, acc_files)):
        print(f"\n--- Processing sample {i+1}: {mel_file.stem} ---")
        
        # Verify file names match
        if mel_file.stem != acc_file.stem:
            print(f"⚠️  Warning: Melody file {mel_file.stem} doesn't match acc file {acc_file.stem}")
        
        song_names.append(mel_file.stem)
        
        # Process melody (using exact same parameters as reward training data)
        print(f"Processing melody: {mel_file}")
        try:
            mel_tensor, mel_pitch_range = preprocess_midi(str(mel_file), max_polyphony=4, beat_div=4, ins_ids='all')
            if mel_tensor is not None:
                print(f"  ✅ Melody processed: shape={mel_tensor.shape}, pitch_range={mel_pitch_range}")
                all_mel_data.append(mel_tensor)
                mel_lengths.append(mel_tensor.shape[0])
                mel_pitch_ranges.append(mel_pitch_range)
                
                # Show first few frames
                print(f"  First 3 frames:")
                for frame_idx in range(min(3, mel_tensor.shape[0])):
                    frame = mel_tensor[frame_idx].view(4, 3)
                    non_pad = sum(1 for note in frame if note[1] != 255)
                    print(f"    Frame {frame_idx}: {frame.tolist()} ({non_pad} notes)")
            else:
                print(f"  ❌ Failed to process melody")
                return
        except Exception as e:
            print(f"  ❌ Error processing melody: {e}")
            return
        
        # Process accompaniment (using exact same parameters as reward training data)
        print(f"Processing accompaniment: {acc_file}")
        try:
            acc_tensor, acc_pitch_range = preprocess_midi(str(acc_file), max_polyphony=4, beat_div=4, ins_ids='all')
            if acc_tensor is not None:
                print(f"  ✅ Accompaniment processed: shape={acc_tensor.shape}, pitch_range={acc_pitch_range}")
                all_acc_data.append(acc_tensor)
                acc_lengths.append(acc_tensor.shape[0])
                acc_pitch_ranges.append(acc_pitch_range)
                
                # Show first few frames
                print(f"  First 3 frames:")
                for frame_idx in range(min(3, acc_tensor.shape[0])):
                    frame = acc_tensor[frame_idx].view(4, 3)
                    non_pad = sum(1 for note in frame if note[1] != 255)
                    print(f"    Frame {frame_idx}: {frame.tolist()} ({non_pad} notes)")
            else:
                print(f"  ❌ Failed to process accompaniment")
                return
        except Exception as e:
            print(f"  ❌ Error processing accompaniment: {e}")
            return
        
        print(f"  Melody length: {mel_lengths[-1]}, Acc length: {acc_lengths[-1]}")
    
    # Concatenate all data (same format as main dataset)
    print(f"\n🔗 Concatenating data...")
    mel_data_combined = torch.cat(all_mel_data, dim=0)
    acc_data_combined = torch.cat(all_acc_data, dim=0)
    
    print(f"Combined melody shape: {mel_data_combined.shape}")
    print(f"Combined acc shape: {acc_data_combined.shape}")
    print(f"Melody range: [{mel_data_combined.min()}, {mel_data_combined.max()}]")
    print(f"Acc range: [{acc_data_combined.min()}, {acc_data_combined.max()}]")
    
    # Create length tensors (cumulative starts for each song)
    mel_starts = torch.cumsum(torch.tensor([0] + mel_lengths[:-1]), dim=0)
    acc_starts = torch.cumsum(torch.tensor([0] + acc_lengths[:-1]), dim=0)
    
    print(f"Song starts - Melody: {mel_starts.tolist()}")
    print(f"Song starts - Acc: {acc_starts.tolist()}")
    print(f"Song lengths - Melody: {mel_lengths}")
    print(f"Song lengths - Acc: {acc_lengths}")
    
    # Convert pitch ranges to tensors
    mel_pitch_ranges_tensor = torch.stack(mel_pitch_ranges)  # [num_songs, 2]
    acc_pitch_ranges_tensor = torch.stack(acc_pitch_ranges)  # [num_songs, 2]
    
    print(f"Pitch ranges:")
    print(f"  Melody: {mel_pitch_ranges_tensor}")
    print(f"  Acc: {acc_pitch_ranges_tensor}")
    
    # Save processed data
    print(f"\n💾 Saving processed data...")
    torch.save(mel_data_combined, output_dir / "debug_mel.pt")
    torch.save(acc_data_combined, output_dir / "debug_acc.pt")
    torch.save(torch.tensor(mel_lengths), output_dir / "debug_mel.length.pt")
    torch.save(torch.tensor(acc_lengths), output_dir / "debug_acc.length.pt")
    torch.save(mel_pitch_ranges_tensor, output_dir / "debug_mel.pitch_shift_range.pt")
    torch.save(acc_pitch_ranges_tensor, output_dir / "debug_acc.pitch_shift_range.pt")
    
    print(f"✅ Saved to {output_dir}/")
    print(f"  - debug_mel.pt: {mel_data_combined.shape}")
    print(f"  - debug_acc.pt: {acc_data_combined.shape}")
    print(f"  - debug_mel.length.pt: {len(mel_lengths)} songs")
    print(f"  - debug_acc.length.pt: {len(acc_lengths)} songs")
    print(f"  - debug_mel.pitch_shift_range.pt: {mel_pitch_ranges_tensor.shape}")
    print(f"  - debug_acc.pitch_shift_range.pt: {acc_pitch_ranges_tensor.shape}")
    
    # Verification: Export processed data back to MIDI
    print(f"\n🔍 VERIFICATION: Exporting processed data back to MIDI...")
    
    mel_start_idx = 0
    acc_start_idx = 0
    
    for i, song_name in enumerate(song_names):
        mel_len = mel_lengths[i]
        acc_len = acc_lengths[i]
        
        # Extract song data
        mel_song = mel_data_combined[mel_start_idx:mel_start_idx + mel_len]
        acc_song = acc_data_combined[acc_start_idx:acc_start_idx + acc_len]
        
        # Export to MIDI for verification
        mel_verify_path = verification_dir / f"{song_name}_melody_reconstructed.mid"
        acc_verify_path = verification_dir / f"{song_name}_acc_reconstructed.mid"
        
        try:
            tensor_to_midi(mel_song, str(mel_verify_path), tempo=120.0, instrument_program=0)
            tensor_to_midi(acc_song, str(acc_verify_path), tempo=120.0, instrument_program=1)
            print(f"  ✅ Exported {song_name}: melody & accompaniment verification files")
        except Exception as e:
            print(f"  ❌ Failed to export {song_name}: {e}")
        
        mel_start_idx += mel_len
        acc_start_idx += acc_len
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("=" * 40)
    print(f"Total samples processed: {len(song_names)}")
    print(f"Total melody frames: {mel_data_combined.shape[0]}")
    print(f"Total acc frames: {acc_data_combined.shape[0]}")
    print(f"Average melody length: {np.mean(mel_lengths):.1f} frames")
    print(f"Average acc length: {np.mean(acc_lengths):.1f} frames")
    
    # Check for musical content
    mel_musical_frames = 0
    acc_musical_frames = 0
    
    for frame_idx in range(mel_data_combined.shape[0]):
        mel_frame = mel_data_combined[frame_idx].view(4, 3)
        if any(note[1] not in [254, 255] for note in mel_frame):  # Not EOS/PAD
            mel_musical_frames += 1
    
    for frame_idx in range(acc_data_combined.shape[0]):
        acc_frame = acc_data_combined[frame_idx].view(4, 3)
        if any(note[1] not in [254, 255] for note in acc_frame):  # Not EOS/PAD
            acc_musical_frames += 1
    
    mel_density = 100 * mel_musical_frames / mel_data_combined.shape[0]
    acc_density = 100 * acc_musical_frames / acc_data_combined.shape[0]
    
    print(f"Melody musical content: {mel_musical_frames}/{mel_data_combined.shape[0]} ({mel_density:.1f}%)")
    print(f"Acc musical content: {acc_musical_frames}/{acc_data_combined.shape[0]} ({acc_density:.1f}%)")
    
    if mel_musical_frames == 0:
        print("❌ WARNING: No musical content found in melody data!")
    if acc_musical_frames == 0:
        print("❌ WARNING: No musical content found in accompaniment data!")
    
    if mel_musical_frames > 0 and acc_musical_frames > 0:
        print("✅ Both melody and accompaniment have musical content")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Listen to verification files in {verification_dir}/")
    print(f"2. Compare with original files in {input_dir}/")
    print(f"3. If they sound similar, preprocessing is working correctly")
    print(f"4. If not, there's an issue in the preprocessing pipeline")
    print(f"5. Run Phase 2: debug_dataset_mini.py")


if __name__ == "__main__":
    debug_preprocess_mini_samples()