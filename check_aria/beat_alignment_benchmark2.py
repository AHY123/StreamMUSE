import pretty_midi
import soundfile as sf
import librosa
import madmom
import numpy as np
import os
from pathlib import Path

# ==============================================================================
#  1. ANALYSIS & REBUILD FUNCTIONS
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
    if len(beat_activations) < 2: return beat_activations[:, 0], 120.0
    beat_times = beat_activations[:, 0]
    beat_intervals = np.diff(beat_times)
    avg_interval = np.mean(beat_intervals)
    avg_bpm = 60 / avg_interval
    downbeat_indices = np.where(beat_activations[:, 1] == 1)[0]
    first_anchor_time = beat_times[downbeat_indices[0]] if len(downbeat_indices) > 0 else beat_times[0]
    first_beat_index = np.argmin(np.abs(beat_times - first_anchor_time))
    regular_beats = (np.arange(len(beat_times)) - first_beat_index) * avg_interval + first_anchor_time
    print(f"    - Created DAW-aligned grid at a constant {avg_bpm:.2f} BPM.")
    return regular_beats, avg_bpm

def verify_midi_stats(file_path: str):
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        tempo = midi_data.estimate_tempo()
        duration = midi_data.get_end_time()
        print("    - ✅ Verification Stats:")
        print(f"    -   Embedded Tempo: {tempo:.2f} BPM")
        print(f"    -   Duration: {duration:.2f} seconds")
        print(f"    -   Tracks Found ({len(midi_data.instruments)}):")
        for i, inst in enumerate(midi_data.instruments):
            print(f"    -     {i+1}. Name: '{inst.name}', Notes: {len(inst.notes)}, Is Drum: {inst.is_drum}")
    except Exception as e:
        print(f"    - ❌ ERROR: Could not verify MIDI file. Reason: {e}")

# --- UNIFIED REBUILD FUNCTION (with updated drum logic) ---
def rebuild_quantized_midi(
    original_midi: pretty_midi.PrettyMIDI,
    beat_activations: np.ndarray,
    quantization_grid: np.ndarray,
    new_bpm: float = None
) -> pretty_midi.PrettyMIDI:
    if new_bpm:
        new_midi = pretty_midi.PrettyMIDI(initial_tempo=new_bpm)
    else:
        new_midi = pretty_midi.PrettyMIDI()
    
    grid_times = []
    for i in range(len(quantization_grid) - 1):
        beat_duration = quantization_grid[i+1] - quantization_grid[i]
        for j in range(4): # 16th note subdivisions
            grid_times.append(quantization_grid[i] + j * (beat_duration / 4.0))
    grid_times.append(quantization_grid[-1])
    grid_times = np.array(grid_times)

    for old_instrument in original_midi.instruments:
        new_instrument = pretty_midi.Instrument(program=old_instrument.program, is_drum=old_instrument.is_drum, name=old_instrument.name)
        for old_note in old_instrument.notes:
            closest_grid_time = grid_times[np.argmin(np.abs(grid_times - old_note.start))]
            duration = old_note.end - old_note.start
            new_note = pretty_midi.Note(velocity=old_note.velocity, pitch=old_note.pitch, start=closest_grid_time, end=closest_grid_time + duration)
            new_instrument.notes.append(new_note)
        new_midi.instruments.append(new_instrument)
        
    drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="Metronome")
    KICK_PITCH, SNARE_PITCH = 36, 38
    for time, beat_num in beat_activations:
        start_time = grid_times[np.argmin(np.abs(grid_times - time))]
        
        # --- THIS IS THE MODIFIED LINE ---
        # Kick on beats 1 and 3, Snare on beats 2 and 4
        pitch = KICK_PITCH if beat_num in [1.0, 3.0] else SNARE_PITCH
        
        drum_note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=start_time + 0.1)
        drum_track.notes.append(drum_note)
        
    new_midi.instruments.append(drum_track)
    print("    - Added metronome drum track.")
    return new_midi

# ==============================================================================
#  2. MAIN WORKFLOW
# ==============================================================================

def process_directory_final(input_dir: str, output_dir: str, soundfont: str, quantization_mode: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if quantization_mode not in ['metronome', 'dynamic']:
        print(f"FATAL ERROR: Invalid quantization_mode '{quantization_mode}'. Choose 'metronome' or 'dynamic'.")
        return
        
    print(f"Starting workflow in '{quantization_mode}' mode.")
    
    midi_files = list(input_path.glob('*.mid')) + list(input_path.glob('*.midi'))
    print(f"Found {len(midi_files)} MIDI files to process.\n")
    
    for i, midi_file in enumerate(midi_files):
        print(f"--- Processing file {i+1}/{len(midi_files)}: {midi_file.name} ---")
        temp_wav_path = "temp_audio_for_analysis.wav"

        if not synthesize_midi_to_wav(str(midi_file), temp_wav_path, soundfont):
            continue

        try:
            beat_activations = get_beats_and_downbeats_with_madmom(temp_wav_path)
            if len(beat_activations) == 0:
                print("    - WARNING: Madmom found no beats. Skipping.")
                continue
            
            if quantization_mode == 'metronome':
                quantization_grid, new_bpm = create_daw_aligned_grid(beat_activations)
            else: # 'dynamic' mode
                quantization_grid = beat_activations[:, 0]
                new_bpm = None
                print("    - Using DYNAMIC grid from performer's timing.")
            
            original_midi = pretty_midi.PrettyMIDI(str(midi_file))
            rebuilt_midi = rebuild_quantized_midi(original_midi, beat_activations, quantization_grid, new_bpm)
            
            final_output_path = output_path / midi_file.name
            rebuilt_midi.write(str(final_output_path))
            print(f"    -> Saved '{quantization_mode}' aligned MIDI to '{final_output_path}'")
            
            verify_midi_stats(str(final_output_path))
            
        except Exception as e:
            print(f"    - ❌ ERROR: Failed during processing. Reason: {e}")
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        print("-" * (len(midi_file.name) + 24) + "\n")
        
    print("🎉 Workflow complete.")

if __name__ == "__main__":
    # --- CHOOSE YOUR SETTINGS ---
    MODE = 'dynamic'

    INPUT_DIRECTORY = "try_align/messy_midi"
    OUTPUT_DIRECTORY = f"try_align/quantized_{MODE}"
    SOUNDFONT_PATH = "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE_Fork2/FluidR3_GM.sf2"

    Path(INPUT_DIRECTORY).mkdir(exist_ok=True)
    
    if not os.path.exists(SOUNDFONT_PATH):
        print(f"FATAL ERROR: SoundFont not found at '{SOUNDFONT_PATH}'. Please update the path.")
    else:
        process_directory_final(
            input_dir=INPUT_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            soundfont=SOUNDFONT_PATH,
            quantization_mode=MODE
        )