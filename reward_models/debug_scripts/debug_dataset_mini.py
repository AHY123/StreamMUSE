#!/usr/bin/env python3
"""
Phase 2: Debug dataset creation and segment extraction with 5 known good samples.
Tests DiscriminativeDataset creation, segment extraction, and real/fake pair generation.
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


def debug_dataset_mini():
    """
    Test dataset creation with our 5 processed samples.
    """
    print("🔧 PHASE 2: DEBUGGING DATASET CREATION & SEGMENT EXTRACTION")
    print("=" * 60)
    
    # Paths
    data_dir = Path("reward_models/debug_data")
    output_dir = Path("reward_models/debug_outputs/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Check that Phase 1 outputs exist
    required_files = [
        "debug_mel.pt", "debug_acc.pt", 
        "debug_mel.length.pt", "debug_acc.length.pt",
        "debug_mel.pitch_shift_range.pt", "debug_acc.pitch_shift_range.pt"
    ]
    for file in required_files:
        if not (data_dir / file).exists():
            print(f"❌ Error: Missing {file}. Please run Phase 1 first.")
            return
    
    print("✅ Phase 1 outputs found")
    
    # Test different target lengths representing musical phrases
    # 4 ticks = 1 beat, 16 ticks = 1 bar, 192 ticks = 12 bars
    test_lengths = [64, 128, 192]  # 4 bars, 8 bars, 12 bars
    
    for target_length in test_lengths:
        print(f"\n" + "="*40)
        bars = target_length / 16  # Convert ticks to bars
        print(f"TESTING TARGET_LENGTH = {target_length} ({bars:.0f} bars)")
        print("="*40)
        
        try:
            # Create dataset
            dataset = DiscriminativeDataset(
                melody_path=str(data_dir / "debug_mel.pt"),
                acc_path=str(data_dir / "debug_acc.pt"),
                target_length=target_length
            )
            
            print(f"✅ Dataset created successfully")
            print(f"  Dataset length: {len(dataset)} samples")
            print(f"  Valid songs: {len(dataset.valid_indices)}")
            print(f"  Song indices: {dataset.valid_indices[:5]}...")  # Show first 5
            
            if len(dataset.valid_indices) == 0:
                print(f"❌ No valid songs found for target_length={target_length}")
                continue
            
            # Test extracting several samples
            num_test_samples = min(6, len(dataset))  # Test up to 6 samples
            print(f"\n📊 Testing {num_test_samples} sample extractions:")
            
            for sample_idx in range(num_test_samples):
                print(f"\n--- Sample {sample_idx} ---")
                
                try:
                    sample = dataset[sample_idx]
                    sequences = sample['sequence']  # [2*target_length, 12] (note: singular!)
                    labels = sample['label']        # scalar
                    pitch_shifts = sample['pitch_shift']  # [1]
                    
                    label_str = "REAL" if labels.item() == 1.0 else "FAKE"
                    pitch_shift = pitch_shifts.item()
                    
                    print(f"  Label: {labels.item()} ({label_str})")
                    print(f"  Pitch shift: {pitch_shift}")
                    print(f"  Sequences shape: {sequences.shape}")
                    print(f"  Expected shape: [{ 2 * target_length}, 12]")
                    
                    # Split interleaved sequence back into melody and accompaniment
                    melody = sequences[1::2]      # Odd indices: melody  
                    accompaniment = sequences[0::2]  # Even indices: accompaniment
                    
                    print(f"  After splitting:")
                    print(f"    Melody shape: {melody.shape}")
                    print(f"    Acc shape: {accompaniment.shape}")
                    
                    # Count musical content
                    mel_notes = sum(1 for frame_idx in range(melody.shape[0]) 
                                  for note in melody[frame_idx].view(4, 3) 
                                  if note[1] not in [254, 255])
                    acc_notes = sum(1 for frame_idx in range(accompaniment.shape[0]) 
                                  for note in accompaniment[frame_idx].view(4, 3) 
                                  if note[1] not in [254, 255])
                    
                    print(f"    Melody musical notes: {mel_notes}")
                    print(f"    Acc musical notes: {acc_notes}")
                    
                    if mel_notes == 0 and acc_notes == 0:
                        print(f"    ⚠️  WARNING: No musical content found!")
                    elif mel_notes == 0:
                        print(f"    ⚠️  WARNING: No melody content found!")
                    elif acc_notes == 0:
                        print(f"    ⚠️  WARNING: No accompaniment content found!")
                    else:
                        print(f"    ✅ Both melody and acc have musical content")
                    
                    # Export to MIDI for verification
                    sample_name = f"len{target_length}_sample{sample_idx}_{label_str}_shift{pitch_shift:+d}"
                    mel_path = output_dir / f"{sample_name}_melody.mid"
                    acc_path = output_dir / f"{sample_name}_acc.mid"
                    
                    try:
                        tensor_to_midi(melody, str(mel_path), tempo=120.0, instrument_program=0)
                        tensor_to_midi(accompaniment, str(acc_path), tempo=120.0, instrument_program=1)
                        print(f"    ✅ Exported: {sample_name}_melody.mid & {sample_name}_acc.mid")
                    except Exception as e:
                        print(f"    ❌ Export failed: {e}")
                
                except Exception as e:
                    print(f"  ❌ Failed to extract sample {sample_idx}: {e}")
            
            # Test dataset statistics
            print(f"\n📈 DATASET STATISTICS (target_length={target_length}):")
            
            # Count real vs fake samples by checking multiple samples
            test_sample_count = min(20, len(dataset))
            real_count = 0
            fake_count = 0
            
            for i in range(test_sample_count):
                sample = dataset[i]
                if sample['label'].item() == 1.0:
                    real_count += 1
                else:
                    fake_count += 1
            
            print(f"  Sample distribution (first {test_sample_count}):")
            print(f"    Real samples: {real_count}")
            print(f"    Fake samples: {fake_count}")
            print(f"    Balance: {100*real_count/test_sample_count:.1f}% real, {100*fake_count/test_sample_count:.1f}% fake")
            
            if abs(real_count - fake_count) > test_sample_count * 0.2:
                print(f"    ⚠️  WARNING: Significant imbalance detected!")
            else:
                print(f"    ✅ Good balance between real and fake samples")
                
        except Exception as e:
            print(f"❌ Failed to create dataset with target_length={target_length}: {e}")
    
    print(f"\n🎯 PHASE 2 VERIFICATION INSTRUCTIONS:")
    print("="*60)
    print(f"Check the MIDI files in: {output_dir.absolute()}")
    print()
    print("For REAL samples:")
    print("  - Melody and accompaniment should sound musically coherent")
    print("  - They should harmonically fit together")
    print("  - Should sound like they belong to the same piece")
    print()
    print("For FAKE samples:")
    print("  - Melody and accompaniment should sound musically incoherent")
    print("  - They should clash harmonically or rhythmically")
    print("  - Should sound like they're from different pieces")
    print()
    print("Pitch shifts:")
    print("  - Both melody and accompaniment should be shifted by the same amount")
    print("  - Negative shifts = lower pitch, positive shifts = higher pitch")
    print()
    print("If REAL samples sound coherent and FAKE samples sound incoherent,")
    print("then dataset creation is working correctly!")


def debug_segment_extraction_details():
    """
    Debug the specific segment extraction logic in detail.
    """
    print(f"\n🔍 DETAILED SEGMENT EXTRACTION DEBUG")
    print("="*60)
    
    data_dir = Path("reward_models/debug_data")
    
    # Load raw data to understand song boundaries
    melody_data = torch.load(data_dir / "debug_mel.pt")
    acc_data = torch.load(data_dir / "debug_acc.pt")
    mel_lengths = torch.load(data_dir / "debug_mel.length.pt")
    acc_lengths = torch.load(data_dir / "debug_acc.length.pt")
    
    print(f"Raw data shapes:")
    print(f"  Melody: {melody_data.shape}")
    print(f"  Acc: {acc_data.shape}")
    print(f"  Song lengths: {len(mel_lengths)} songs")
    
    # Show song boundaries
    mel_starts = torch.cumsum(torch.cat([torch.tensor([0]), mel_lengths[:-1]]), dim=0)
    print(f"\nSong boundaries (melody):")
    for i in range(len(mel_lengths)):
        start_idx = mel_starts[i].item()
        length = mel_lengths[i].item()
        end_idx = start_idx + length
        print(f"  Song {i}: frames {start_idx}-{end_idx} (length={length})")
    
    # Test segment extraction manually
    target_length = 128  # 8 bars - good length for testing
    bars = target_length / 16
    print(f"\nManual segment extraction test (target_length={target_length}, {bars:.0f} bars):")
    
    for song_idx in range(min(3, len(mel_lengths))):  # Test first 3 songs
        start_idx = mel_starts[song_idx].item()
        song_length = mel_lengths[song_idx].item()
        
        print(f"\nSong {song_idx}:")
        print(f"  Start: {start_idx}, Length: {song_length}")
        
        if song_length >= target_length:
            # Extract a segment (like dataset does)
            max_start = song_length - target_length
            segment_start = max_start // 2  # Take middle segment for reproducibility
            absolute_start = start_idx + segment_start
            absolute_end = absolute_start + target_length
            
            print(f"  Extracting segment: frames {absolute_start}-{absolute_end}")
            
            mel_segment = melody_data[absolute_start:absolute_end]
            acc_segment = acc_data[absolute_start:absolute_end]
            
            # Count musical content in segment
            mel_notes = sum(1 for frame_idx in range(target_length)
                          for note in mel_segment[frame_idx].view(4, 3)
                          if note[1] not in [254, 255])
            acc_notes = sum(1 for frame_idx in range(target_length)
                          for note in acc_segment[frame_idx].view(4, 3)
                          if note[1] not in [254, 255])
            
            print(f"  Musical content in segment:")
            print(f"    Melody notes: {mel_notes}")
            print(f"    Acc notes: {acc_notes}")
            
            if mel_notes > 0 and acc_notes > 0:
                print(f"    ✅ Good segment with both melody and acc")
            elif mel_notes > 0:
                print(f"    ⚠️  Only melody, no acc")
            elif acc_notes > 0:
                print(f"    ⚠️  Only acc, no melody")
            else:
                print(f"    ❌ Empty segment!")
        else:
            print(f"  ❌ Song too short for target_length={target_length}")


if __name__ == "__main__":
    debug_dataset_mini()
    debug_segment_extraction_details()