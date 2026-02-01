import sys
import os
import torch
import numpy as np
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.inference_engines.transformer_engine_lekai import InferenceEngineLekai
from lekai_model.PianoDataset import PianoDataset, encode_bpm, process_measure_with_beat_interleaving
from lekai_model.config import ModelConfig
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM


def compare_tokens(npz_dir, model_path, engine_config_path=None):
    # Load Model Config first
    config = ModelConfig()

    # 1. Setup V2-style components
    # Pass config and data_dir
    dataset = PianoDataset(data_dir=npz_dir, config=config, cache_lengths=False)

    # Load Model (Shared)
    llama_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        rope_theta=config.rope_theta,
        attention_dropout=config.dropout,
        use_cache=True,
        initializer_range=0.02,
    )
    model = LlamaForCausalLM(llama_config)
    state_dict = load_file(model_path)
    # Fix key mismatch (strip 'model.' prefix if checkpoint is wrapped)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.model."):
            new_state_dict[k[6:]] = v  # Remove first 'model.'
        elif k.startswith("model.lm_head"):
            new_state_dict[k[6:]] = v  # Remove first 'model.'
        else:
            new_state_dict[k] = v

    # Fallback: if keys don't match pattern, try straightforward load or smarter replacement
    # Simpler approach: check if ALL keys start with 'model.' and target doesn't expect it?
    # Actually, LlamaForCausalLM IS 'model'.
    # If checkpoint comes from 'LitLlama' where self.model = LlamaForCausalLM.
    # Then keys are 'model.....'.
    # We just strip 'model.'.

    final_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            # Check if this "model." is the wrapper or the LlamaModel attribute?
            # LlamaForCausalLM has attribute 'model'.
            # Keys should be 'model.embed_tokens...'
            # If checkpoint has 'model.model.embed_tokens', it's wrapped.
            if k.startswith("model.model."):
                final_dict[k[6:]] = v
            elif k.startswith("model.lm_head"):
                final_dict[k[6:]] = v
            else:
                # Checkpoint has 'model.something' but not 'model.model'.
                # e.g. 'model.layers'.
                # This matches LlamaForCausalLM structure.
                final_dict[k] = v
        else:
            final_dict[k] = v

    model.load_state_dict(final_dict)
    model.eval()

    # 2. Setup Engine
    # Assuming default config if not provided
    engine = InferenceEngineLekai(
        checkpoint_path=model_path,
        model_size="tiny",  # Dummy argument
        inference_mode="stateful",
    )
    # Hack: Inject the SAME model instance to avoid reloading if possible?
    # Engine loads its own. That's fine.

    # 3. Select a sample
    idx = 0
    print(f"Comparing on Sample Index {idx}: {dataset.data_files[idx]}")

    # --- GET V2 TOKENS ---
    # We need to capture the tokens constructed inside generate_accompaniment_v2
    # Since we can't easily hook into it without modifying it,
    # we will REPLICATE the construction logic here identically.

    # Load NPZ
    file_path = os.path.join(dataset.root_dir, dataset.data_files[idx])
    save_dict = np.load(file_path, allow_pickle=True)
    metadata = save_dict["metadata"].item()
    bpm_value = metadata["bpm"]
    num_measures = metadata["num_measures"]
    time_sig_idx = metadata["time_signature_idx"]
    if time_sig_idx == 9:
        time_sig_idx = 4

    # Construct Part0/Part1 Lists (V2 Logic)
    part0_beats_list = []
    part1_beats_gt_list = []
    bar_token = 255

    for i in range(num_measures):
        measure = save_dict[f"measure_{i}"]
        p0, p1 = process_measure_with_beat_interleaving(measure, tokenizer=dataset.tokenizer, timesteps_per_beat=4)
        part0_beats_list.append(torch.tensor([bar_token], dtype=torch.long))
        part0_beats_list.extend(p0)
        part1_beats_gt_list.append(torch.tensor([bar_token], dtype=torch.long))
        part1_beats_gt_list.extend(p1)

    # Initial Tokens
    bpm_token = encode_bpm(bpm_value) + dataset.bpm_offset_id
    v2_seq = [
        torch.tensor([dataset.bos_token]),
        torch.tensor([time_sig_idx + dataset.time_sig_offset_id]),
        torch.tensor([bpm_token]),
        torch.tensor([173]),  # Start PAD
    ]

    # Let's verify the first measure (4 beats) + 1 beat injection
    # delay_beats = -1 (Stagger)
    # Beat 0:
    #   Part0: [PAD] (since b=0 and stagger)
    #   Part1: Acc[0] (Target)

    print("\n--- Constructing V2 Sequence (First 4 Beats) ---")
    # This logic mimics batch_generate loop.
    # Beat 0
    # append Part0[0] -> Stagger -> PAD
    v2_seq.append(torch.tensor([173]))
    # append Part1[0] (Bar[0] + Notes?)
    # Wait, process_measure output includes Bar token!
    # No, process_measure output returns BEATS.
    # V2 loop:
    #   part0_beats_list includes Bars interleaved.
    #   Index 0 of part0_beats_list IS the Bar token.
    #   Index 1 is Beat 0.

    # So V2 Input for Beat 0 generation:
    # [BOS, TS, BPM, PAD, PAD] -> Generate Acc[0]?
    # Wait, Part1 Loop:
    #   delay_beats=-1.
    #   part0_idx = 0.
    #   In loop:
    #     position=0 (Part0). delay_beats<0.
    #     part0_idx (0) <= -delay_beats (1).
    #     Insert PAD. part0_idx+=1. pos=1.
    #     position=1 (Part1). part1_idx(0) < gt_prefix.
    #     Insert GT Beat 0.

    # Wait, I shouldn't just guess. I should run the ACTUAL function and print `input_ids`.
    # But I can't modify the source easily without dirtying it.
    # I will replicate the loop logic carefully.

    # ... (Replication logic omitted for brevity in thought, implementing in code)
    # Actually, to be 100% sure, I will define a helper that mimics `batch_generate` logic EXACTLY.

    # --- GET ENGINE TOKENS ---
    # Engine.inject_context_beat(mel_notes, acc_notes...)
    # I need to convert NPZ back to notes to feed the Engine.
    # Or bypass conversion and feed tokens? Engine expects NOTES.

    print("\n--- Simulating V2 Token Generation ---")

    # Simulation variables
    v2_generated = torch.cat(v2_seq).unsqueeze(0)
    part0_idx = 0
    part1_idx = 0
    delay_beats = -1
    gt_prefix_beats = 16
    pad_marker = 173
    accumulated_v2_tokens = []  # Store token lists for comparison

    # We will simulate the loop until we have constructed context for Beat 16
    # This matches the "Prompt Injection" phase

    # V2 Loop Simulation
    position = 0  # 0=Part0, 1=Part1

    # We want to capture the state of 'generated' just before the model would be called for Beat X
    # Actually, we can just capture the sequence of tokens appended.

    # Initial manual tokens
    accumulated_v2_tokens.extend([t.tolist() for t in v2_seq])

    max_steps = 200  # Safety
    steps = 0

    curr_beat_processed = 0

    while steps < max_steps:
        # Part 0
        if position == 0:
            if delay_beats < 0 and part0_idx <= -delay_beats:
                # PAD
                t = torch.tensor([pad_marker])
                accumulated_v2_tokens.append(t.tolist())
                part0_idx += 1
                position = 1
            elif part0_idx < len(part0_beats_list):
                t = part0_beats_list[part0_idx]
                accumulated_v2_tokens.append(t.tolist())
                part0_idx += 1
                position = 1
        # Part 1
        else:
            if delay_beats >= 0 and part0_idx <= delay_beats:
                # PAD
                t = torch.tensor([pad_marker])
                accumulated_v2_tokens.append(t.tolist())
                part1_idx += 1
                position = 0
            elif part1_idx < gt_prefix_beats:
                # GT Injection
                t = part1_beats_gt_list[part1_idx]
                accumulated_v2_tokens.append(t.tolist())
                part1_idx += 1
                position = 0
            else:
                # Generation Start
                print("V2 Simulation reached generation start.")
                break
        steps += 1

    print(f"V2 Simulation generated {len(accumulated_v2_tokens)} token chunks.")

    print("\n--- Generating Engine Tokens ---")
    # Setup Engine Data
    # Convert NPZ content to Notes for Engine
    # Note: Engine expects notes relative to start_tick, but we shifted everything to 0.

    # Extract ALL notes from NPZ structure for simplicity (mimic helper)
    # We actually need to re-parse using tokenizer or just use the beats list if we trust engine internals?
    # Engine.inject_context_beat takes NOTES.
    # We should use `midi_converter` to get notes from the SAME beats we used for V2?
    # No, that's circular. We should get notes from NPZ using Standard Midi Converter.

    # Load NPZ -> Pianoroll -> Notes
    # Or rely on `dataset` tokenizer to decompress?
    # `process_measure_with_beat_interleaving` already gets tokens.
    # We need EVENTS/NOTES.

    # Let's use `MidiConverter` if available to parse `save_dict`?
    # No, `save_dict` has tokens.
    # We need raw notes.
    # NPZ *also* has 'measures' which are compressed tokens.
    # The original MIDI is needed?.
    # Wait, `PianoDataset` loads compressed tokens.
    # If we want to test Engine, we usually start from MIDI/Notes.
    # But here we have NPZ.
    # We can reconstruct notes from the V2 tokens we just loaded!
    # Yes! `process_part_beats_to_pianoroll` -> `pianoroll_to_notes`.

    from lekai_model.Token2Midi import process_part_beats_to_pianoroll, pianoroll_to_midi_notes

    # Helper to get notes from beats list (V2 style)
    def get_notes_from_beats(beats_list, is_part1=False):
        # Remove bar tokens for decompression
        clean_beats = [b for b in beats_list if not (b.numel() == 1 and b.item() == 255)]
        pr = process_part_beats_to_pianoroll(
            clean_beats, tokenizer=dataset.tokenizer, split_marker_id_1=170, split_marker_id_2=171 if is_part1 else 170
        )
        # Convert to Notes
        # We need a tempo but standard is 120 for relative ticks?
        # Engine expects ticks. `pianoroll_to_notes` in Engine uses MidiConverter.
        # `Token2Midi.pianoroll_to_midi_notes` returns PrettyMIDI objects (seconds).
        # We need TICKS.
        # We can write a quick `pianoroll_to_tick_notes` helper.

        notes = []
        # pr: (2, 88, T)
        sustain = pr[0]
        onset = pr[1]
        T = pr.shape[2]

        for p in range(88):
            pitch = p + 21
            # Find onsets
            on_locs = np.where(onset[p] > 0)[0]
            for t_on in on_locs:
                dur = 1
                while (t_on + dur < T) and (sustain[p, t_on + dur] > 0):
                    dur += 1
                notes.append({"pitch": pitch, "tick": int(t_on), "duration": int(dur), "velocity": 80})

        return sorted(notes, key=lambda x: x["tick"])

    print("Extracting Notes from V2 Data...")
    mel_notes = get_notes_from_beats(part0_beats_list, is_part1=False)
    acc_notes = get_notes_from_beats(part1_beats_gt_list, is_part1=True)

    print(f"Extracted {len(mel_notes)} Mel notes, {len(acc_notes)} Acc notes.")

    # Configure Engine
    engine.sequence_start_beat = 0
    engine.delay_beats = -1
    engine.clear_history()

    # Run Inject up to beat 16
    ticks_per_beat = 480 // 4  # Wait, V2 uses T=4 per beat?
    # Check `process_measure_with_beat_interleaving`: `timesteps_per_beat=4`.
    # So resolution is 4 steps per beat.
    # Engine config: `ticks_per_beat`?
    # Engine usually uses 480 ticks/beat and quantizes?
    # Or does it work in "Token Steps"?
    # `engine.ticks_per_beat` comes from MidiConverter.
    # `MidiToNpzConverter` (used by Engine) usually has `resolution` (steps per quarter).
    # If standard is 120 ticks/beat.
    # The V2 tokens are 4 steps/beat. = 12 steps/beat?
    # `PianoRollTokenizer` patch_w=4.
    # We need to ensure Engine's `ticks_per_beat` matches V2's 4-step resolution.
    # Engine `inject_context_beat` uses `_get_mel_pianoroll_for_beat`.
    # It converts notes (ticks) -> pianoroll (pixels/tokens).
    # We need to map our High-Res Notes (extracted from V2 Tokens which are Low-Res?)
    # Wait, `get_notes_from_beats` returned notes in STEP units (0..T).
    # Engine expects TICKS.
    # We need to scale Steps -> Ticks.
    # `patch_w=4` means 1 token = 4 pixels? No.
    # `timesteps_per_beat=4`.
    # So 1 beat = 4 steps.
    # Engine config usually 480 ticks/beat.
    # So Scale = 480 / 4 = 120 ticks per step.

    # Engine expects 'step-based' ticks because tokenizer expects step-based resolution
    # Standard resolution=4.
    # We extracted notes from V2 Tokens (Resolution=4).
    # So 'tick' 0..4 is correct for Engine if we leave ticks_per_beat default.

    # Check Config Consistency
    print(f"Config Debug:")
    print(f"  V2 Dataset BOS: {dataset.bos_token}")
    print(f"  Engine Config BOS: {engine.config.bos_token_id}")
    print(f"  Dataset Bar: {dataset.bar_token}")
    print(f"  Engine Bar: {engine.config.bar_token_id}")

    # Remove Scaling logic - Pass raw steps
    # engine.ticks_per_beat default is 4 in __init__

    # Run Inject
    # We verify beat by beat

    print("\n--- Walking through Beats ---")

    engine_tokens_flat = []

    # Engine uses ticks_per_beat = 4 internally
    ticks_per_beat = 4

    # Initial Engine Tokens
    # BOS, TS, BPM, PAD (Start)
    # Engine builds `input_ids` in `inject_context_beat` fresh every time?
    # No, `stateful` mode keeps history?
    # Actually `inject_context_beat` builds the FULL SEQUENCE up to current beat if reset.
    # If we loop 0..15 calling inject.
    # The Last Call (Inject Beat 15) will generate the sequence 0..15.
    # We should capture THAT sequence.

    # But wait, `inject_context_beat` returns logic inside `input_ids`.
    # It doesn't return the tokens. It feeds model.
    # We need to intercept `input_ids`.
    # We can inspect `engine.past_key_values`? No.
    # We can MONKEY PATCH `engine.model` to capture input?
    # Yes.

    captured_inputs = []
    original_forward = engine.model.forward

    def mock_forward(input_ids, **kwargs):
        captured_inputs.append(input_ids.cpu().tolist())
        return original_forward(input_ids, **kwargs)

    engine.model.forward = mock_forward

    # Run Injection for 16 beats
    # We assume prompts are beat 0..15.
    # We call inject for each.

    for b in range(16):
        b_start = b * 480
        b_end = (b + 1) * 480
        m_notes = [n for n in mel_notes if b_start <= n["tick"] < b_end]
        a_notes = [n for n in acc_notes if b_start <= n["tick"] < b_end]

        # Adjust 'tick' to be relative? No, Engine handles normalization.
        # But we must pass notes that are ALREADY shifted to 0 (which they are).

        if b == 0:
            print(f"Beat 0 Debug:")
            print(f"  Start: {b_start}, End: {b_end}")
            print(f"  Mel Notes in range: {len(m_notes)}")
            if m_notes:
                print(f"  First Note: {m_notes[0]}")
            print(f"  Acc Notes in range: {len(a_notes)}")

        engine.inject_context_beat(m_notes, a_notes, b_start, bpm=bpm_value, time_sig=(4, 4))

    print(f"Captured {len(captured_inputs)} forward passes.")

    # The LAST forward pass should contain the full prompt sequence.
    last_input = captured_inputs[-1][0]  # Batch 0

    # Flatten V2 tokens
    v2_flat = []
    for chunk in accumulated_v2_tokens:
        v2_flat.extend(chunk)

    print(f"\nV2 Total Tokens: {len(v2_flat)}")
    print(f"Engine Total Tokens: {len(last_input)}")

    # Compare!
    limit = min(len(v2_flat), len(last_input))
    mismatch_count = 0
    first_mismatch = -1

    print("\n--- DETAILED COMPARISON ---")
    for i in range(limit):
        t1 = v2_flat[i]
        t2 = last_input[i]
        if t1 != t2:
            mismatch_count += 1
            if first_mismatch == -1:
                first_mismatch = i
            # Print context around mismatch
            if mismatch_count <= 5:
                print(f"Mismatch at {i}: V2={t1} vs Engine={t2}")
                # Print context
                start = max(0, i - 5)
                end = min(limit, i + 5)
                print(f"  Ctx V2: {v2_flat[start:end]}")
                print(f"  Ctx Eng: {last_input[start:end]}")

    if len(v2_flat) != len(last_input):
        print(f"Length Mismatch! V2={len(v2_flat)}, Eng={len(last_input)}")

    if mismatch_count == 0 and len(v2_flat) == len(last_input):
        print("\nSUCCESS! Sequences are IDENTICAL.")
    else:
        print(f"\nFAILED! Found {mismatch_count} mismatches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    compare_tokens(args.npz_dir, args.model)
