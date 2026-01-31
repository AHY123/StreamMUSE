import os
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from app.inference_engines.transformer_engine_lekai import InferenceEngineLekai, events_to_notes, notes_to_events
from mid2pianoroll_v2 import MidiToNpzConverter
import pretty_midi


def save_midi(acc_notes, melody_notes, output_path, bpm=120, mel_program=0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Melody Track
    inst_mel = pretty_midi.Instrument(program=mel_program, name="Melody")
    seconds_per_beat = 60.0 / bpm
    seconds_per_tick = seconds_per_beat / 4

    for note in melody_notes:
        start_time = note["tick"] * seconds_per_tick
        duration_time = note["duration"] * seconds_per_tick
        end_time = start_time + duration_time
        pm_note = pretty_midi.Note(velocity=80, pitch=int(note["pitch"]), start=start_time, end=end_time)
        inst_mel.notes.append(pm_note)
    pm.instruments.append(inst_mel)

    # Accompaniment Track
    inst_acc = pretty_midi.Instrument(program=0, name="Accompaniment")
    for note in acc_notes:
        start_time = note["tick"] * seconds_per_tick
        duration_time = note["duration"] * seconds_per_tick
        end_time = start_time + duration_time
        pm_note = pretty_midi.Note(velocity=70, pitch=int(note["pitch"]), start=start_time, end=end_time)
        inst_acc.notes.append(pm_note)
    pm.instruments.append(inst_acc)

    pm.write(output_path)
    print(f"Saved MIDI to {output_path}")


def run_test(run_name, model_path, prompt_beats=0, delay_beats=-1):
    output_dir = os.path.join(PROJECT_ROOT, "lekai_alignment/outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    data_root = os.path.join(PROJECT_ROOT, "lekai_alignment/data")
    paired_dir = os.path.join(data_root, "paired")
    mini_dir = os.path.join(data_root, "mini_midis")

    # Initialize Engine
    print(f"Initializing engine with model: {model_path}")
    try:
        engine = InferenceEngineLekai(
            checkpoint_path=model_path,
            model_size="llama",
            inference_mode="stateful",
            delay_beats=delay_beats,
        )
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # Process Paired Data
    mel_dir = os.path.join(paired_dir, "mel")
    acc_dir = os.path.join(paired_dir, "acc")

    if os.path.exists(mel_dir):
        print(f"\n=== Processing Paired Data from {mel_dir} ===")
        for filename in sorted(os.listdir(mel_dir)):
            if not (filename.endswith(".mid") or filename.endswith(".midi")):
                continue

            input_midi = os.path.join(mel_dir, filename)
            input_acc = os.path.join(acc_dir, filename) if os.path.exists(os.path.join(acc_dir, filename)) else None

            output_path = os.path.join(output_dir, f"paired_{filename}")
            # Skip if input_acc exists? No, we use it for prompt if requested, but mainly for GT comparison if we want.
            # Here we just generate.
            process_single_file(engine, input_midi, output_path, input_acc, prompt_beats)

    # Process Mini Data
    if os.path.exists(mini_dir):
        print(f"\n=== Processing Mini Data from {mini_dir} ===")
        # Limit to first 5 for speed unless specified
        files = sorted([f for f in os.listdir(mini_dir) if f.endswith(".mid") or f.endswith(".midi")])[:5]
        for filename in files:
            input_midi = os.path.join(mini_dir, filename)
            output_path = os.path.join(output_dir, f"mini_{filename}")
            process_single_file(engine, input_midi, output_path, None, 0)  # No prompt for mini


def process_single_file(engine, input_midi_path, output_path, input_acc_midi_path, prompt_beats):
    print(f"Processing {os.path.basename(input_midi_path)}...")

    # Load Notes
    converter = MidiToNpzConverter()
    melody_notes, acc_notes, metadata = converter.get_aligned_notes(input_midi_path, input_acc_midi_path)

    if not melody_notes:
        print("  Failed to load notes.")
        return

    bpm = metadata.get("bpm", 120)
    time_sig = metadata.get("time_sig", (4, 4))

    max_tick_mel = max([n["tick"] + n["duration"] for n in melody_notes]) if melody_notes else 0
    max_tick_acc = max([n["tick"] + n["duration"] for n in acc_notes]) if acc_notes else 0
    max_tick = max(max_tick_mel, max_tick_acc)

    # Clear Engine State
    engine.clear_history()
    # Ensure history lists are clear
    engine.melody_event_history = []
    engine.accompaniment_history = []
    engine._active_melody_pitches = set()
    engine._active_acc_pitches = set()

    ticks_per_beat = 4

    # Skip Silence & Shift Logic
    # To match inference_v2 output structure, we should SHIFT the input notes
    # so the first measure starts at tick 0.

    start_shift_tick = 0
    if melody_notes:
        first_note_tick = min(n["tick"] for n in melody_notes)
        beats_per_measure = 4
        start_measure = (first_note_tick // ticks_per_beat) // beats_per_measure
        start_shift_tick = max(0, start_measure * beats_per_measure * ticks_per_beat)

        print(f"  Shifting Input: -{start_shift_tick} ticks (Start Measure: {start_measure})")

        # Apply Shift
        for n in melody_notes:
            n["tick"] -= start_shift_tick

        if acc_notes:
            for n in acc_notes:
                n["tick"] -= start_shift_tick

    # Recalculate max_tick after shift
    max_tick_mel = max([n["tick"] + n["duration"] for n in melody_notes]) if melody_notes else 0
    max_tick_acc = max([n["tick"] + n["duration"] for n in acc_notes]) if acc_notes else 0
    max_tick = max(max_tick_mel, max_tick_acc)

    # Reset engine start beat to 0 (since we shifted data)
    engine.set_sequence_start_beat(0)

    total_beats = (max_tick + ticks_per_beat - 1) // ticks_per_beat + 4

    all_generated_notes = []

    # Loop
    for beat in range(total_beats):
        start_tick = beat * ticks_per_beat
        end_tick = (beat + 1) * ticks_per_beat

        new_mel_notes = [n for n in melody_notes if start_tick <= n["tick"] < end_tick]

        if beat < prompt_beats and input_acc_midi_path:
            new_acc_notes = [n for n in acc_notes if start_tick <= n["tick"] < end_tick]

            # Prompting Logic: Inject GT Acc
            # print(f"  [Prompt] Beat {beat}: Injecting GT Acc")
            engine.inject_context_beat(
                melody_notes=new_mel_notes,
                acc_notes=new_acc_notes,
                generation_start_tick=start_tick,
                bpm=bpm,
                time_sig=time_sig,
            )
            # Add to generated list for output (so the file is complete)
            if new_acc_notes:
                all_generated_notes.extend(new_acc_notes)

        else:
            # Generation
            mel_events = notes_to_events(new_mel_notes)

            try:
                # engine.generate_accompaniment returns:
                # (relative_generated_events, preprocess_start, inference_start, inference_end, postprocess_start, pr)
                generated_events, _, _, _, _, _ = engine.generate_accompaniment(
                    melody_notes=mel_events, generation_start_tick=start_tick, bpm=bpm, time_sig=time_sig
                )

                if generated_events:
                    # Convert events to notes immediately for this beat
                    beat_notes = events_to_notes(generated_events)
                    all_generated_notes.extend(beat_notes)

            except Exception as e:
                print(f"Error at beat {beat}: {e}")
                import traceback

                traceback.print_exc()
                break

    # Save Result
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_midi(all_generated_notes, melody_notes, output_path, bpm=bpm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="run_001_baseline", help="Run name")
    parser.add_argument(
        "--model",
        type=str,
        default="rt/RT_Accompaniment/checkpoints/epoch_4_1104_1204/model.safetensors",
        help="Model path",
    )
    parser.add_argument("--delay_beats", type=int, default=-1, help="Delay beats (default -1)")
    parser.add_argument("--prompt_beats", type=int, default=0, help="Prompt beats (default 0)")
    args = parser.parse_args()

    run_test(args.name, args.model, prompt_beats=args.prompt_beats, delay_beats=args.delay_beats)
