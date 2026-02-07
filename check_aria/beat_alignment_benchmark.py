import pretty_midi
import soundfile as sf
import librosa
import madmom
import numpy as np
import os
from pathlib import Path

# ==============================================================================
#  HELPER FUNCTIONS (with final tempo-writing logic)
# ==============================================================================

def synthesize_midi_to_wav(midi_path: str, wav_path: str, soundfont: str) -> bool:
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio_data = midi_data.fluidsynth(fs=44100, sf2_path=soundfont)
        sf.write(wav_path, audio_data, 44100)
        return True
    except Exception as e:
        print(f"    - ERROR: Could not synthesize MIDI '{Path(midi_path).name}'. Skipping. Reason: {e}")
        return False

# --- Tempo Writing Function (The New Fix) ---
def update_midi_tempo(midi_data: pretty_midi.PrettyMIDI, new_bpm: float):
    """Clears all old tempo changes and sets a new, constant BPM."""
    # Accessing internal lists, as pretty_midi lacks a direct public setter
    midi_data._tempo_changes = []
    midi_data._tick_to_time = []
    # Add the new tempo change at the beginning of the file
    new_tempo = pretty_midi.bpm_to_tempo(new_bpm)
    midi_data._add_tempo_change(int(new_tempo), 0)


# --- Librosa Path Functions (Updated to return BPM) ---
def get_beats_with_librosa(wav_path: str) -> np.ndarray:
    y, sr = librosa.load(wav_path)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"    - Librosa found {len(beat_times)} human-timed beats.")
    return beat_times

def regularize_beat_grid(beat_times: np.ndarray) -> tuple[np.ndarray, float]:
    """Creates a constant tempo grid and now also returns the calculated BPM."""
    if len(beat_times) < 2: return beat_times, 120.0 # Return beats and a default BPM
    beat_intervals = np.diff(beat_times)
    avg_interval = np.mean(beat_intervals)
    first_beat = beat_times[0]
    num_beats = len(beat_times)
    regular_beats = np.arange(num_beats) * avg_interval + first_beat
    avg_bpm = 60 / avg_interval
    print(f"    - Librosa Regularized grid to a constant {avg_bpm:.2f} BPM.")
    return regular_beats, avg_bpm

# --- Madmom Path Functions (Updated to return BPM) ---
def get_beats_and_downbeats_with_madmom(wav_path: str) -> np.ndarray:
    rnn_proc = madmom.features.downbeats.RNNDownBeatProcessor()
    activations = rnn_proc(wav_path)
    dbn_proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    beats = dbn_proc(activations)
    if beats.size > 0:
        beats = beats[beats[:, 1] > 0] 
    print(f"    - Madmom found {len(beats)} beats with downbeat info.")
    return beats

def create_daw_aligned_grid(beat_activations: np.ndarray) -> tuple[np.ndarray, float]:
    """Creates a DAW-aligned grid and now also returns the calculated BPM."""
    if len(beat_activations) < 2: return beat_activations[:, 0], 120.0
    beat_times = beat_activations[:, 0]
    beat_intervals = np.diff(beat_times)
    avg_interval = np.mean(beat_intervals)
    downbeat_indices = np.where(beat_activations[:, 1] == 1)[0]
    first_anchor_time = beat_times[downbeat_indices[0]] if len(downbeat_indices) > 0 else beat_times[0]
    first_beat_index = np.argmin(np.abs(beat_times - first_anchor_time))
    regular_beats = (np.arange(len(beat_times)) - first_beat_index) * avg_interval + first_anchor_time
    avg_bpm = 60 / avg_interval
    print(f"    - Madmom Created DAW-aligned grid at a constant {avg_bpm:.2f} BPM.")
    return regular_beats, avg_bpm

# --- Shared Quantization Function (No changes needed) ---
def quantize_midi_to_grid(midi_data: pretty_midi.PrettyMIDI, beat_times: np.ndarray, strength: float, subdivisions: int) -> pretty_midi.PrettyMIDI:
    grid_times = []
    for i in range(len(beat_times) - 1):
        beat_duration = beat_times[i+1] - beat_times[i]
        subdivision_duration = beat_duration / subdivisions
        for j in range(subdivisions):
            grid_times.append(beat_times[i] + j * subdivision_duration)
    grid_times.append(beat_times[-1])
    grid_times = np.array(grid_times)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            original_start = note.start
            duration = note.end - note.start
            closest_grid_time = grid_times[np.argmin(np.abs(grid_times - original_start))]
            new_start = original_start + (closest_grid_time - original_start) * strength
            note.start = new_start
            note.end = new_start + duration
    return midi_data

# ==============================================================================
#  MAIN WORKFLOW ORCHESTRATOR
# ==============================================================================
def process_directory_for_comparison(
    input_dir: str,
    output_dir_madmom: str,
    output_dir_librosa: str,
    quantize_strength: float = 1.0,
    quantize_subdivisions: int = 4
):
    input_path = Path(input_dir)
    madmom_path = Path(output_dir_madmom)
    librosa_path = Path(output_dir_librosa)
    madmom_path.mkdir(exist_ok=True)
    librosa_path.mkdir(exist_ok=True)
    soundfont = "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE_Fork2/FluidR3_GM.sf2"

    if not os.path.exists(soundfont):
        print(f"FATAL ERROR: The SoundFont file was not found at the specified path: '{soundfont}'")
        return

    midi_files = list(input_path.glob('*.mid')) + list(input_path.glob('*.midi'))
    if not midi_files:
        print(f"No MIDI files found in '{input_dir}'.")
        return

    print(f"Found {len(midi_files)} MIDI files to process.\n")
    
    for i, midi_file in enumerate(midi_files):
        print(f"--- Processing file {i+1}/{len(midi_files)}: {midi_file.name} ---")
        temp_wav_path = "temp_audio_for_analysis.wav"

        if not synthesize_midi_to_wav(str(midi_file), temp_wav_path, soundfont):
            continue

        # --- MADMOM PATH ---
        try:
            beat_activations = get_beats_and_downbeats_with_madmom(temp_wav_path)
            if len(beat_activations) > 0:
                metronome_grid, new_bpm = create_daw_aligned_grid(beat_activations)
                original_midi = pretty_midi.PrettyMIDI(str(midi_file))
                quantized_midi = quantize_midi_to_grid(original_midi, metronome_grid, quantize_strength, quantize_subdivisions)
                # ** NEW STEP: Write the new BPM to the MIDI file **
                update_midi_tempo(quantized_midi, new_bpm)
                output_path = madmom_path / f'madmom_{midi_file.name}'
                quantized_midi.write(str(output_path))
                print(f"    -> Saved Madmom (Downbeat Aligned) version to '{output_path}'")
            else:
                print("    - WARNING: Madmom found no beats. Skipping Madmom processing.")
        except Exception as e:
            print(f"    - ERROR: Failed during Madmom processing. Reason: {e}")

        # --- LIBROSA PATH ---
        try:
            human_beats = get_beats_with_librosa(temp_wav_path)
            if len(human_beats) > 0:
                metronome_grid, new_bpm = regularize_beat_grid(human_beats)
                original_midi = pretty_midi.PrettyMIDI(str(midi_file))
                quantized_midi = quantize_midi_to_grid(original_midi, metronome_grid, quantize_strength, quantize_subdivisions)
                # ** NEW STEP: Write the new BPM to the MIDI file **
                update_midi_tempo(quantized_midi, new_bpm)
                output_path = librosa_path / f'librosa_{midi_file.name}'
                quantized_midi.write(str(output_path))
                print(f"    -> Saved Librosa (Beat Aligned) version to '{output_path}'")
            else:
                print("    - WARNING: Librosa found no beats. Skipping Librosa processing.")
        except Exception as e:
            print(f"    - ERROR: Failed during Librosa processing. Reason: {e}")

        os.remove(temp_wav_path)
        print("-" * (len(midi_file.name) + 30) + "\n")
        
    print("✅ Workflow complete.")

if __name__ == "__main__":
    INPUT_DIRECTORY = "try_align/messy_midi"
    OUTPUT_MADMOM = "try_align/quantized_madmom"
    OUTPUT_LIBROSA = "try_align/quantized_librosa"
    STRENGTH = 1.0

    Path(INPUT_DIRECTORY).mkdir(exist_ok=True)

    process_directory_for_comparison(
        input_dir=INPUT_DIRECTORY,
        output_dir_madmom=OUTPUT_MADMOM,
        output_dir_librosa=OUTPUT_LIBROSA,
        quantize_strength=STRENGTH
    )