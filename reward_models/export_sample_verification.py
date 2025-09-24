#!/usr/bin/env python3
"""
Export training samples as MIDI files to manually verify real/fake labels.
Creates MIDI files with melody+accompaniment tracks labeled by what the model thinks is real/fake.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.data_loader import create_dataloaders

# Import the existing tested tensor_to_midi function
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocess"))
from preprocess_midi2pt_dataset import tensor_to_midi


def export_samples():
    print("🎵 EXPORTING SAMPLE VERIFICATION MIDI FILES")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("sample_verification")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Create dataloader
    train_loader, _ = create_dataloaders(
        melody_path="data/reward_training_mel_cp4.pt",
        acc_path="data/reward_training_acc_cp4.pt",
        batch_size=20,
        target_length=64,  # Shorter for easier manual verification
        num_workers=0
    )
    
    # Get first batch
    batch = next(iter(train_loader))
    sequences = batch['sequences']  # [batch_size, 2*seq_len, 12]
    labels = batch['labels']        # [batch_size] 
    pitch_shifts = batch['pitch_shifts'].squeeze(-1)  # [batch_size]
    
    print(f"Batch shape: {sequences.shape}")
    print(f"Labels: {labels.tolist()}")
    print(f"Pitch shifts: {pitch_shifts.tolist()}")
    
    # Export first 10 samples
    num_samples = min(10, sequences.shape[0])
    
    for i in range(num_samples):
        sequence = sequences[i]  # [2*seq_len, 12]
        label = labels[i].item()
        pitch_shift = pitch_shifts[i].item()
        
        # Split interleaved sequence back into melody and accompaniment
        # Format: [acc_0, mel_0, acc_1, mel_1, ...] 
        seq_len = sequence.shape[0] // 2
        
        # Extract melody and accompaniment
        melody = sequence[1::2]      # Odd indices: melody
        accompaniment = sequence[0::2]  # Even indices: accompaniment
        
        print(f"\nSample {i}:")
        print(f"  Label: {label} ({'REAL' if label == 1.0 else 'FAKE'})")
        print(f"  Pitch shift: {pitch_shift}")
        print(f"  Melody shape: {melody.shape}")
        print(f"  Acc shape: {accompaniment.shape}")
        
        # Create filenames
        label_str = "REAL" if label == 1.0 else "FAKE"
        mel_filename = f"sample_{i:02d}_{label_str}_melody_shift{pitch_shift:+d}.mid"
        acc_filename = f"sample_{i:02d}_{label_str}_accompaniment_shift{pitch_shift:+d}.mid"
        
        try:
            # Export melody (use program 0 = piano for melody)
            tensor_to_midi(melody, str(output_dir / mel_filename), tempo=120.0, instrument_program=0)
            
            # Export accompaniment (use program 1 = bright piano for accompaniment)
            tensor_to_midi(accompaniment, str(output_dir / acc_filename), tempo=120.0, instrument_program=1)
            
            print(f"  ✅ Exported: {mel_filename} & {acc_filename}")
            
        except Exception as e:
            print(f"  ❌ Failed to export sample {i}: {e}")
    
    print(f"\n" + "=" * 60)
    print("VERIFICATION INSTRUCTIONS")
    print("=" * 60)
    print(f"Check the MIDI files in: {output_dir.absolute()}")
    print()
    print("For REAL samples:")
    print("  - Melody and accompaniment should be from the SAME song")
    print("  - They should sound musically coherent together")
    print("  - Harmonies should make sense")
    print()
    print("For FAKE samples:")
    print("  - Melody and accompaniment should be from DIFFERENT songs") 
    print("  - They should sound musically incoherent/clashing")
    print("  - Harmonies should NOT make sense together")
    print()
    print("If this labeling is wrong, we found the bug!")
    
    # Also create a summary file
    summary_file = output_dir / "sample_summary.txt"
    with open(summary_file, "w") as f:
        f.write("SAMPLE VERIFICATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        for i in range(num_samples):
            label = labels[i].item()
            pitch_shift = pitch_shifts[i].item()
            label_str = "REAL" if label == 1.0 else "FAKE"
            
            f.write(f"Sample {i:02d}: {label_str} (label={label}, pitch_shift={pitch_shift:+d})\n")
            f.write(f"  Melody: sample_{i:02d}_{label_str}_melody_shift{pitch_shift:+d}.mid\n")
            f.write(f"  Accompaniment: sample_{i:02d}_{label_str}_accompaniment_shift{pitch_shift:+d}.mid\n\n")
        
        f.write("\nVERIFICATION CHECKLIST:\n")
        f.write("□ REAL samples: melody + acc from same song, sound coherent\n")
        f.write("□ FAKE samples: melody + acc from different songs, sound incoherent\n")
        f.write("□ Pitch shifts applied correctly to both melody and accompaniment\n")
    
    print(f"📄 Created summary file: {summary_file}")


if __name__ == "__main__":
    export_samples()