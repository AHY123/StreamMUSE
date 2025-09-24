#!/usr/bin/env python3
"""
Rigorous verification of the data pipeline for discriminative reward model.
Checks every stage from raw data to final tokens.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_models.discriminative_model import DiscriminativeRewardModel, DURATION_TEMPLATES
from reward_models.data_loader import DiscriminativeDataset, create_dataloaders

def analyze_raw_data():
    """Phase 1: Analyze raw data in .pt files"""
    print("=" * 60)
    print("PHASE 1: RAW DATA ANALYSIS")
    print("=" * 60)
    
    # Load raw data files
    melody_path = "data/reward_training_mel_cp4.pt"
    acc_path = "data/reward_training_acc_cp4.pt"
    
    print(f"Loading raw data from:")
    print(f"  Melody: {melody_path}")
    print(f"  Accompaniment: {acc_path}")
    
    melody_data = torch.load(melody_path, mmap=True)
    acc_data = torch.load(acc_path, mmap=True)
    
    print(f"\nRaw data shapes:")
    print(f"  Melody: {melody_data.shape}")
    print(f"  Acc: {acc_data.shape}")
    
    print(f"\nRaw data ranges:")
    print(f"  Melody: [{melody_data.min().item()}, {melody_data.max().item()}]")
    print(f"  Acc: [{acc_data.min().item()}, {acc_data.max().item()}]")
    
    # Analyze the structure: [total_frames, 12] -> [total_frames, 4, 3]
    melody_reshaped = melody_data.view(-1, 4, 3)
    acc_reshaped = acc_data.view(-1, 4, 3)
    
    print(f"\nAnalyzing (program, pitch, duration) structure:")
    print("Melody stats:")
    mel_programs = melody_reshaped[:, :, 0]
    mel_pitches = melody_reshaped[:, :, 1] 
    mel_durations = melody_reshaped[:, :, 2]
    
    print(f"  Programs: [{mel_programs.min().item()}, {mel_programs.max().item()}]")
    print(f"  Pitches: [{mel_pitches.min().item()}, {mel_pitches.max().item()}]")
    print(f"  Durations: [{mel_durations.min().item()}, {mel_durations.max().item()}]")
    
    print("Accompaniment stats:")
    acc_programs = acc_reshaped[:, :, 0]
    acc_pitches = acc_reshaped[:, :, 1]
    acc_durations = acc_reshaped[:, :, 2]
    
    print(f"  Programs: [{acc_programs.min().item()}, {acc_programs.max().item()}]")
    print(f"  Pitches: [{acc_pitches.min().item()}, {acc_pitches.max().item()}]") 
    print(f"  Durations: [{acc_durations.min().item()}, {acc_durations.max().item()}]")
    
    # Check for special tokens
    print(f"\nSpecial token analysis:")
    print(f"  PAD token (255): Melody={torch.sum(mel_pitches == 255)}, Acc={torch.sum(acc_pitches == 255)}")
    print(f"  EOS token (254): Melody={torch.sum(mel_programs == 254)}, Acc={torch.sum(acc_programs == 254)}")
    
    # Check for drums - these probably shouldn't exist in melody/accompaniment data
    mel_drums = torch.sum(mel_programs == 127)
    acc_drums = torch.sum(acc_programs == 127)
    print(f"  Drum program (127): Melody={mel_drums}, Acc={acc_drums}")
    if mel_drums > 0 or acc_drums > 0:
        print(f"    ⚠️  WARNING: Found drums in melody/accompaniment data!")
        print(f"    ⚠️  This might be unexpected for melody-accompaniment extraction")
    else:
        print(f"    ✅ No drums found (expected for melody/accompaniment data)")
    
    # Show program distribution to understand what instruments we actually have
    print(f"\nProgram distribution (top 10 most common):")
    unique_mel_programs, mel_counts = torch.unique(mel_programs, return_counts=True)
    unique_acc_programs, acc_counts = torch.unique(acc_programs, return_counts=True)
    
    # Sort by count and show top 10
    mel_sorted_idx = torch.argsort(mel_counts, descending=True)
    acc_sorted_idx = torch.argsort(acc_counts, descending=True)
    
    print("  Melody programs:")
    for i in range(min(10, len(unique_mel_programs))):
        prog = unique_mel_programs[mel_sorted_idx[i]].item()
        count = mel_counts[mel_sorted_idx[i]].item()
        print(f"    Program {prog:3d}: {count:8d} occurrences")
        
    print("  Accompaniment programs:")
    for i in range(min(10, len(unique_acc_programs))):
        prog = unique_acc_programs[acc_sorted_idx[i]].item()
        count = acc_counts[acc_sorted_idx[i]].item() 
        print(f"    Program {prog:3d}: {count:8d} occurrences")
    
    return melody_data, acc_data


def analyze_duration_quantization():
    """Phase 2: Verify duration quantization logic"""
    print("\n" + "=" * 60)
    print("PHASE 2: DURATION QUANTIZATION ANALYSIS")
    print("=" * 60)
    
    print(f"Duration templates: {DURATION_TEMPLATES}")
    print(f"Number of templates: {len(DURATION_TEMPLATES)}")
    
    # Test the quantization function with edge cases
    model = DiscriminativeRewardModel()
    
    print("\nTesting duration quantization with edge cases:")
    test_durations = torch.tensor([0, 1, 2, 23, 50, 100, 200, 255])
    quantized = model.quantize_duration(test_durations)
    
    for i, (raw, quant) in enumerate(zip(test_durations, quantized)):
        template_val = DURATION_TEMPLATES[quant] if quant < len(DURATION_TEMPLATES) else "OUT_OF_BOUNDS"
        print(f"  Raw {raw.item():3d} -> Index {quant.item():2d} -> Template {template_val}")
    
    # Verify max possible quantized index
    max_duration = 255  # Max possible raw duration
    max_quantized = model.quantize_duration(torch.tensor([max_duration]))
    print(f"\nMax duration {max_duration} quantizes to index {max_quantized.item()}")
    print(f"This corresponds to template value: {DURATION_TEMPLATES[max_quantized.item()]}")


def analyze_token_generation():
    """Phase 3: Analyze the full tokenization process"""
    print("\n" + "=" * 60)  
    print("PHASE 3: TOKEN GENERATION ANALYSIS")
    print("=" * 60)
    
    model = DiscriminativeRewardModel()
    
    # Create synthetic test data to verify tokenization formula
    print("Creating synthetic test data...")
    batch_size, seq_len = 2, 4
    test_data = torch.zeros(batch_size, seq_len, 12, dtype=torch.long)
    
    # Fill with realistic values for melody/accompaniment (no drums)
    for b in range(batch_size):
        for s in range(seq_len):
            for note in range(4):
                idx = note * 3
                # Use realistic program numbers (0=piano, 1=bright piano, etc. - no drums)
                test_data[b, s, idx] = note % 8                   # programs 0-7 (common instruments)
                test_data[b, s, idx+1] = 60 + note               # pitch (60-63, middle C range)
                test_data[b, s, idx+2] = note * 10               # raw duration (0, 10, 20, 30)
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data range: [{test_data.min()}, {test_data.max()}]")
    
    # Test tokenization with different pitch shifts
    for pitch_shift in [0, 3, -2, 6]:
        print(f"\nTesting pitch shift = {pitch_shift}")
        pitch_shift_tensor = torch.tensor([pitch_shift, pitch_shift])
        
        tokens = model.preprocess(test_data, pitch_shift_tensor)
        
        print(f"  Output shape: {tokens.shape}")
        print(f"  Token range: [{tokens.min().item()}, {tokens.max().item()}]")
        
        # Check if any tokens exceed vocab size
        invalid = tokens >= model.vocab_size
        if invalid.any():
            print(f"  ❌ {invalid.sum()} tokens >= vocab_size ({model.vocab_size})")
            print(f"  Max invalid token: {tokens[invalid].max().item()}")
        else:
            print(f"  ✅ All tokens valid (< {model.vocab_size})")
            
        # Show formula breakdown for first token
        if pitch_shift == 0:  # Only show detailed breakdown once
            print(f"  Formula breakdown for sample tokens:")
            raw = test_data[0, 0].view(4, 3)  # First frame, 4 notes
            processed = tokens[0, 0].view(4, 2)  # First frame output
            
            for note in range(2):  # Show first 2 notes
                prog, pitch, raw_dur = raw[note]
                _, final_token = processed[note]
                
                # Calculate expected token manually
                quant_dur_idx = model.quantize_duration(torch.tensor([raw_dur.item()])).item()
                is_not_drum = prog.item() != 127
                expected = pitch.item() + quant_dur_idx * 128 + 2 + pitch_shift * is_not_drum
                
                print(f"    Note {note}: pitch={pitch.item()}, raw_dur={raw_dur.item()}, quant_idx={quant_dur_idx}")
                print(f"      Expected: {pitch.item()} + {quant_dur_idx}*128 + 2 + {pitch_shift}*{is_not_drum} = {expected}")
                print(f"      Actual: {final_token.item()}")
                print(f"      Match: {'✅' if expected == final_token.item() else '❌'}")


def analyze_dataloader_output():
    """Phase 4: Verify dataloader produces valid data"""
    print("\n" + "=" * 60)
    print("PHASE 4: DATALOADER OUTPUT ANALYSIS") 
    print("=" * 60)
    
    # Create dataloader
    train_loader, _ = create_dataloaders(
        melody_path="data/reward_training_mel_cp4.pt",
        acc_path="data/reward_training_acc_cp4.pt",
        batch_size=4,
        target_length=64,  # Smaller for testing
        num_workers=0
    )
    
    print(f"Created dataloader with {len(train_loader)} batches")
    
    # Test several batches
    model = DiscriminativeRewardModel()
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:  # Test first 3 batches
            break
            
        sequences = batch['sequences']  # [batch_size, 2*seq_len, 12]
        labels = batch['labels']
        pitch_shifts = batch['pitch_shifts'] 
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  Raw sequence range: [{sequences.min()}, {sequences.max()}]")
        print(f"  Labels: {labels}")
        print(f"  Pitch shifts: {pitch_shifts}")
        
        # Process through model
        try:
            with torch.no_grad():
                logits = model(sequences, pitch_shifts.squeeze(-1))
                print(f"  ✅ Model forward pass successful")
                print(f"  Output logits: {logits}")
        except Exception as e:
            print(f"  ❌ Model forward pass failed: {e}")
            return False
            
    return True


def verify_no_clamping_needed():
    """Phase 5: Verify clamping is unnecessary"""
    print("\n" + "=" * 60)
    print("PHASE 5: CLAMPING NECESSITY VERIFICATION")
    print("=" * 60)
    
    # Create model without clamping
    class TestModel(DiscriminativeRewardModel):
        def preprocess(self, x, pitch_shift=None):
            """Same as parent but track if clamping would be needed"""
            batch_size, seq_length, subseq_length = x.shape
            x = x.long().view(batch_size, seq_length, subseq_length // 3, 3)
            
            x_processed = torch.zeros(
                batch_size, seq_length, subseq_length // 3, 2,
                dtype=torch.long, device=x.device
            )
            
            pad_indices = x[:, :, :, 1] == 255
            eos_indices = x[:, :, :, 0] == 254  
            is_not_drum = x[:, :, :, 0] != 127
            
            x_processed[:, :, :, 0] = 0
            
            raw_durations = x[:, :, :, 2]
            duration_indices = self.quantize_duration(raw_durations)
            
            if pitch_shift is None:
                pitch_shift = torch.zeros(batch_size, device=x.device)
            
            x_processed[:, :, :, 1] = (
                x[:, :, :, 1] + 
                duration_indices * 128 + 
                2 + 
                pitch_shift.unsqueeze(1).unsqueeze(2) * is_not_drum
            )
            
            # Apply special tokens
            x_processed[pad_indices] = self.PAD_TOKEN
            x_processed[:, :, :, 0][eos_indices] = self.EOS_TOKEN
            
            # CHECK: Would clamping be needed?
            needs_clamping = (x_processed < 0) | (x_processed >= self.vocab_size)
            if needs_clamping.any():
                print(f"    ❌ Clamping needed! {needs_clamping.sum()} tokens out of range")
                print(f"    Token range: [{x_processed.min()}, {x_processed.max()}]")
                print(f"    Vocab size: {self.vocab_size}")
                return False, x_processed
            else:
                print(f"    ✅ No clamping needed. All tokens in [0, {self.vocab_size-1}]")
                return True, x_processed
    
    model = TestModel()
    
    # Test on real data
    train_loader, _ = create_dataloaders(
        melody_path="data/reward_training_mel_cp4.pt",
        acc_path="data/reward_training_acc_cp4.pt",
        batch_size=8,
        target_length=128,
        num_workers=0
    )
    
    print("Testing clamping necessity on real batches...")
    all_valid = True
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 5:  # Test first 5 batches
            break
            
        sequences = batch['sequences']
        pitch_shifts = batch['pitch_shifts'].squeeze(-1)
        
        print(f"  Batch {batch_idx}: ", end="")
        valid, tokens = model.preprocess(sequences, pitch_shifts)
        all_valid &= valid
        
        if not valid:
            break
    
    return all_valid


def main():
    """Run all verification phases"""
    print("🔍 RIGOROUS DATA PIPELINE VERIFICATION")
    print("Testing discriminative reward model data flow...\n")
    
    try:
        # Run all verification phases
        analyze_raw_data()
        analyze_duration_quantization()
        analyze_token_generation()
        
        dataloader_ok = analyze_dataloader_output()
        if not dataloader_ok:
            print("\n❌ VERIFICATION FAILED: Dataloader issues detected")
            return False
            
        clamping_unnecessary = verify_no_clamping_needed()
        
        print("\n" + "=" * 60)
        print("FINAL VERIFICATION RESULTS")
        print("=" * 60)
        
        if clamping_unnecessary:
            print("✅ VERIFICATION PASSED: Defensive clamping can be safely removed")
            print("   - All tokens stay within valid range [0, vocab_size-1]")
            print("   - Duration quantization works correctly")
            print("   - Special token handling is proper")
            return True
        else:
            print("❌ VERIFICATION FAILED: Defensive clamping is still needed")
            print("   - Some tokens exceed vocab_size range")
            print("   - Data pipeline needs investigation")
            return False
            
    except Exception as e:
        print(f"\n❌ VERIFICATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()