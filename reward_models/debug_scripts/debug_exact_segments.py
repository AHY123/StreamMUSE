#!/usr/bin/env python3
"""
Debug script that shows EXACT bar ranges for each extracted segment.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "preprocess"))

from dataset_with_segment_info import DiscriminativeDatasetWithSegmentInfo
from preprocess_midi2pt_dataset import tensor_to_midi


def debug_exact_segments():
    """
    Show exactly which bars are extracted from each song.
    """
    print("🎯 EXACT SEGMENT BAR RANGES DEBUG")
    print("=" * 60)
    
    # Paths
    data_dir = Path("reward_models/debug_data")
    output_dir = Path("reward_models/debug_outputs/exact_segments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create enhanced dataset that returns segment info
    target_length = 128  # 8 bars
    
    dataset = DiscriminativeDatasetWithSegmentInfo(
        melody_path=str(data_dir / "debug_mel.pt"),
        acc_path=str(data_dir / "debug_acc.pt"),
        target_length=target_length
    )
    
    print(f"Dataset created with target_length={target_length} ({target_length/16:.0f} bars)")
    print(f"Valid songs: {len(dataset.valid_indices)}")
    
    # Test several samples
    num_samples = 8
    
    for sample_idx in range(num_samples):
        print(f"\n{'='*50}")
        print(f"SAMPLE {sample_idx}")
        print(f"{'='*50}")
        
        # Get sample with detailed info
        sample = dataset[sample_idx]
        
        sequences = sample['sequence']
        label = sample['label'].item()
        pitch_shift = sample['pitch_shift'].item()
        mel_info = sample['melody_info']
        acc_info = sample['acc_info']
        is_real = sample['is_real']
        
        label_str = "REAL" if label == 1.0 else "FAKE"
        
        print(f"Label: {label} ({label_str})")
        print(f"Pitch shift: {pitch_shift} semitones")
        print(f"Is real pair: {is_real}")
        
        # Show EXACT melody segment info
        print(f"\n🎵 MELODY SEGMENT:")
        mel_start_bar = mel_info['segment_start_in_song'] / 16
        mel_end_bar = (mel_info['segment_start_in_song'] + mel_info['segment_length']) / 16
        mel_song_total_bars = mel_info['song_length'] / 16
        
        print(f"  Song: {mel_info['song_idx']}")
        print(f"  Song total length: {mel_info['song_length']} frames ({mel_song_total_bars:.1f} bars)")
        print(f"  Extracted segment: frames {mel_info['segment_start_in_song']}-{mel_info['segment_start_in_song'] + mel_info['segment_length']}")
        print(f"  EXACT BAR RANGE: {mel_start_bar:.1f} to {mel_end_bar:.1f} bars")
        
        # Show EXACT accompaniment segment info
        print(f"\n🎹 ACCOMPANIMENT SEGMENT:")
        acc_start_bar = acc_info['segment_start_in_song'] / 16
        acc_end_bar = (acc_info['segment_start_in_song'] + acc_info['segment_length']) / 16
        acc_song_total_bars = acc_info['song_length'] / 16
        
        print(f"  Song: {acc_info['song_idx']}")
        print(f"  Song total length: {acc_info['song_length']} frames ({acc_song_total_bars:.1f} bars)")
        print(f"  Extracted segment: frames {acc_info['segment_start_in_song']}-{acc_info['segment_start_in_song'] + acc_info['segment_length']}")
        print(f"  EXACT BAR RANGE: {acc_start_bar:.1f} to {acc_end_bar:.1f} bars")
        
        # Summary
        if is_real:
            if mel_info['song_idx'] == acc_info['song_idx'] and mel_info['segment_start_in_song'] == acc_info['segment_start_in_song']:
                print(f"\n✅ REAL PAIR CONFIRMED:")
                print(f"   Both melody and accompaniment from song {mel_info['song_idx']}")
                print(f"   Both from bars {mel_start_bar:.1f} to {mel_end_bar:.1f}")
            else:
                print(f"\n❌ ERROR: Real pair but segments don't match!")
        else:
            print(f"\n✅ FAKE PAIR CONFIRMED:")
            print(f"   Melody: song {mel_info['song_idx']}, bars {mel_start_bar:.1f}-{mel_end_bar:.1f}")
            print(f"   Accompaniment: song {acc_info['song_idx']}, bars {acc_start_bar:.1f}-{acc_end_bar:.1f}")
        
        # Musical content check
        melody = sequences[1::2]  # Odd indices
        accompaniment = sequences[0::2]  # Even indices
        
        mel_notes = sum(1 for frame_idx in range(melody.shape[0])
                       for note in melody[frame_idx].view(4, 3)
                       if note[1] not in [254, 255])
        acc_notes = sum(1 for frame_idx in range(accompaniment.shape[0])
                       for note in accompaniment[frame_idx].view(4, 3)
                       if note[1] not in [254, 255])
        
        print(f"\nMusical content:")
        print(f"  Melody notes: {mel_notes}")
        print(f"  Accompaniment notes: {acc_notes}")
        
        # Export MIDI files
        sample_name = f"sample{sample_idx:02d}_{label_str}_shift{pitch_shift:+d}"
        sample_name += f"_mel_song{mel_info['song_idx']}_bars{mel_start_bar:.1f}-{mel_end_bar:.1f}"
        sample_name += f"_acc_song{acc_info['song_idx']}_bars{acc_start_bar:.1f}-{acc_end_bar:.1f}"
        
        mel_path = output_dir / f"{sample_name}_melody.mid"
        acc_path = output_dir / f"{sample_name}_acc.mid"
        
        try:
            tensor_to_midi(melody, str(mel_path), tempo=120.0, instrument_program=0)
            tensor_to_midi(accompaniment, str(acc_path), tempo=120.0, instrument_program=1)
            print(f"✅ Exported detailed MIDI files")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    print(f"\n🎯 VERIFICATION SUMMARY:")
    print("="*60)
    print(f"MIDI files exported to: {output_dir.absolute()}")
    print()
    print("File names now show EXACT segment information:")
    print("  sample00_REAL_shift+2_mel_song0_bars3.2-11.2_acc_song0_bars3.2-11.2_melody.mid")
    print("  sample01_FAKE_shift-1_mel_song1_bars0.0-8.0_acc_song3_bars5.5-13.5_melody.mid")
    print()
    print("This tells you:")
    print("  - Which sample number")
    print("  - If it's REAL (same song/bars) or FAKE (different songs/bars)")
    print("  - Pitch shift amount")
    print("  - EXACT source: which song and which bar range")


if __name__ == "__main__":
    debug_exact_segments()