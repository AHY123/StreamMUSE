#!/usr/bin/env python3
"""
Detailed debugging of real/fake sampling logic.
Shows exactly which segments are extracted and from where.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "preprocess"))

from reward_models.data_loader import DiscriminativeDataset
from preprocess_midi2pt_dataset import tensor_to_midi


def debug_detailed_sampling():
    """
    Debug the exact sampling logic with detailed logging.
    """
    print("🔍 DETAILED REAL/FAKE SAMPLING DEBUG")
    print("=" * 60)
    
    # Paths
    data_dir = Path("reward_models/debug_data")
    output_dir = Path("reward_models/debug_outputs/detailed_sampling")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Load raw data to understand song structure
    melody_data = torch.load(data_dir / "debug_mel.pt")
    acc_data = torch.load(data_dir / "debug_acc.pt")
    mel_lengths = torch.load(data_dir / "debug_mel.length.pt")
    acc_lengths = torch.load(data_dir / "debug_acc.length.pt")
    
    print(f"\nDataset structure:")
    print(f"  Total songs: {len(mel_lengths)}")
    print(f"  Song lengths: {mel_lengths.tolist()}")
    
    # Calculate song boundaries
    mel_starts = torch.cumsum(torch.cat([torch.tensor([0]), mel_lengths[:-1]]), dim=0)
    
    print(f"  Song boundaries:")
    for i in range(len(mel_lengths)):
        start = mel_starts[i].item()
        length = mel_lengths[i].item()
        end = start + length
        bars = length / 16  # Convert ticks to bars
        print(f"    Song {i}: frames {start}-{end} (length={length}, {bars:.1f} bars)")
    
    # Create dataset
    target_length = 128  # 8 bars
    bars = target_length / 16
    print(f"\nCreating dataset with target_length={target_length} ({bars:.0f} bars)")
    
    # First, let's modify the pitch shift ranges to be less aggressive
    mel_pitch_ranges = torch.load(data_dir / "debug_mel.pitch_shift_range.pt")
    acc_pitch_ranges = torch.load(data_dir / "debug_acc.pitch_shift_range.pt")
    
    print(f"\nOriginal pitch ranges:")
    print(f"  Melody: {mel_pitch_ranges}")
    print(f"  Acc: {acc_pitch_ranges}")
    
    # Make pitch shifts less aggressive (limit to ±6 semitones = half octave)
    mel_pitch_ranges_limited = torch.clamp(mel_pitch_ranges, -6, 6)
    acc_pitch_ranges_limited = torch.clamp(acc_pitch_ranges, -6, 6)
    
    print(f"\nLimited pitch ranges (±6 semitones):")
    print(f"  Melody: {mel_pitch_ranges_limited}")
    print(f"  Acc: {acc_pitch_ranges_limited}")
    
    # Save limited ranges
    torch.save(mel_pitch_ranges_limited, data_dir / "debug_mel.pitch_shift_range.pt")
    torch.save(acc_pitch_ranges_limited, data_dir / "debug_acc.pitch_shift_range.pt")
    
    dataset = DiscriminativeDataset(
        melody_path=str(data_dir / "debug_mel.pt"),
        acc_path=str(data_dir / "debug_acc.pt"),
        target_length=target_length
    )
    
    print(f"✅ Dataset created successfully")
    print(f"  Valid songs: {len(dataset.valid_indices)}")
    print(f"  Total samples: {len(dataset)} (should be 2x valid songs)")
    
    # Manually extract and analyze several samples
    num_samples_to_test = 8  # Test 4 real + 4 fake
    
    print(f"\n📊 DETAILED SAMPLE ANALYSIS:")
    print("=" * 60)
    
    for sample_idx in range(num_samples_to_test):
        print(f"\n--- SAMPLE {sample_idx} ---")
        
        # Predict what the sample should be
        is_real = sample_idx % 2 == 0
        sequence_idx = sample_idx // 2
        valid_idx = dataset.valid_indices[sequence_idx]
        
        print(f"Index logic:")
        print(f"  sample_idx={sample_idx} → is_real={is_real}, sequence_idx={sequence_idx}")
        print(f"  valid_idx={valid_idx} (song {valid_idx})")
        
        # Get the actual sample
        sample = dataset[sample_idx]
        sequences = sample['sequence']
        label = sample['label'].item()
        pitch_shift = sample['pitch_shift'].item()
        
        label_str = "REAL" if label == 1.0 else "FAKE"
        print(f"Actual result: label={label} ({label_str}), pitch_shift={pitch_shift}")
        
        # Split back into melody and accompaniment
        melody = sequences[1::2]  # Odd indices
        accompaniment = sequences[0::2]  # Even indices
        
        # Analyze which parts of which songs were used
        if is_real:
            print(f"Expected: Melody and Acc both from song {valid_idx}")
            # For real samples, both come from same song and same segment
            # We need to figure out which segment was randomly selected
            song_length = mel_lengths[valid_idx].item()
            max_start = song_length - target_length
            print(f"  Song {valid_idx} length: {song_length} frames ({song_length/16:.1f} bars)")
            print(f"  Possible segment starts: 0 to {max_start} frames")
            print(f"  Segment length: {target_length} frames ({target_length/16:.0f} bars)")
            
            # To find the exact segment, we'd need to modify the dataset to return this info
            # For now, show the range of possibilities
            start_bar_min = 0
            start_bar_max = max_start / 16
            end_bar_min = target_length / 16
            end_bar_max = song_length / 16
            print(f"  Extracted segment: somewhere between bars {start_bar_min:.1f}-{end_bar_min:.1f} and bars {start_bar_max:.1f}-{end_bar_max:.1f}")
            
        else:
            print(f"Expected: Melody from song {valid_idx}, Acc from different song")
            # For fake samples, melody and acc come from different songs and different segments
            mel_song_length = mel_lengths[valid_idx].item()
            mel_max_start = mel_song_length - target_length
            print(f"  Melody from song {valid_idx}: length {mel_song_length} frames ({mel_song_length/16:.1f} bars)")
            print(f"  Melody segment range: bars 0-{target_length/16:.0f} to {mel_max_start/16:.1f}-{mel_song_length/16:.1f}")
            print(f"  Acc from different song (unknown which one without modifying dataset)")
        
        # Try to identify which segments were extracted by checking content
        print(f"Musical content analysis:")
        mel_notes = sum(1 for frame_idx in range(melody.shape[0])
                       for note in melody[frame_idx].view(4, 3)
                       if note[1] not in [254, 255])
        acc_notes = sum(1 for frame_idx in range(accompaniment.shape[0])
                       for note in accompaniment[frame_idx].view(4, 3)
                       if note[1] not in [254, 255])
        
        print(f"  Melody notes: {mel_notes}")
        print(f"  Acc notes: {acc_notes}")
        
        if mel_notes == 0:
            print(f"  ⚠️  WARNING: No melody content!")
        if acc_notes == 0:
            print(f"  ⚠️  WARNING: No accompaniment content!")
        
        # Export for manual verification
        sample_name = f"sample{sample_idx:02d}_{label_str}_shift{pitch_shift:+d}"
        mel_path = output_dir / f"{sample_name}_melody.mid"
        acc_path = output_dir / f"{sample_name}_acc.mid"
        
        try:
            tensor_to_midi(melody, str(mel_path), tempo=120.0, instrument_program=0)
            tensor_to_midi(accompaniment, str(acc_path), tempo=120.0, instrument_program=1)
            print(f"  ✅ Exported: {sample_name}_melody.mid & {sample_name}_acc.mid")
        except Exception as e:
            print(f"  ❌ Export failed: {e}")
    
    print(f"\n🎯 MANUAL VERIFICATION INSTRUCTIONS:")
    print("=" * 60)
    print(f"Listen to files in: {output_dir.absolute()}")
    print()
    print("Real samples (should sound coherent):")
    for i in range(0, num_samples_to_test, 2):
        print(f"  - sample{i:02d}_REAL_*")
    print()
    print("Fake samples (should sound incoherent):")
    for i in range(1, num_samples_to_test, 2):
        print(f"  - sample{i:02d}_FAKE_*")
    print()
    print("Check that:")
    print("  1. Real samples: melody and accompaniment harmonically fit together")
    print("  2. Fake samples: melody and accompaniment clash or don't fit")
    print("  3. Pitch shifts are reasonable (±6 semitones max)")
    print("  4. Both melody and accompaniment have musical content")


def analyze_segment_locations():
    """
    Try to identify exactly which bars were extracted from each song.
    """
    print(f"\n🎵 SEGMENT LOCATION ANALYSIS")
    print("=" * 60)
    
    data_dir = Path("reward_models/debug_data")
    
    # Load data
    melody_data = torch.load(data_dir / "debug_mel.pt")
    mel_lengths = torch.load(data_dir / "debug_mel.length.pt")
    mel_starts = torch.cumsum(torch.cat([torch.tensor([0]), mel_lengths[:-1]]), dim=0)
    
    target_length = 128
    
    print(f"For target_length={target_length} ({target_length/16:.0f} bars):")
    
    # Simulate the segment extraction logic
    for song_idx in range(len(mel_lengths)):
        start_idx = mel_starts[song_idx].item()
        song_length = mel_lengths[song_idx].item()
        
        if song_length >= target_length:
            # This is how DiscriminativeDataset extracts segments
            max_start = song_length - target_length
            
            print(f"\nSong {song_idx} (length={song_length}, {song_length/16:.1f} bars):")
            print(f"  Absolute frames: {start_idx} to {start_idx + song_length}")
            print(f"  Possible segment starts: 0 to {max_start} (within song)")
            print(f"  Segment length: {target_length} frames ({target_length/16:.0f} bars)")
            
            # Show a few example extractions
            for example_start in [0, max_start//2, max_start]:
                if example_start <= max_start:
                    abs_start = start_idx + example_start
                    abs_end = abs_start + target_length
                    rel_start_bars = example_start / 16
                    rel_end_bars = (example_start + target_length) / 16
                    print(f"    Example: relative bars {rel_start_bars:.1f}-{rel_end_bars:.1f} → absolute frames {abs_start}-{abs_end}")
        else:
            print(f"\nSong {song_idx}: Too short (length={song_length}, {song_length/16:.1f} bars) - skipped")


if __name__ == "__main__":
    debug_detailed_sampling()
    analyze_segment_locations()