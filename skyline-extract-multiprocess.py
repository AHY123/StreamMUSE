from symusic import Score, Note, Track
from typing import List, Tuple
import argparse
import os
import multiprocessing
from tqdm import tqdm

def extract_midi_skyline(
    midi_file_path: str,
    max_skyline_drop: int = 12,
    max_rest_duration_sec: float = 2.0
) -> Tuple[Score, List[Note], List[Note]]:
    """
    Separates a MIDI file into a 'skyline' melody and the 'accompaniment' notes.
    """
    try:
        score = Score(midi_file_path)
    except Exception as e:
        # CHANGE: Use a flushed print for reliable output from child processes.
        print(f"\n[ERROR] Error loading MIDI file '{midi_file_path}': {e}", flush=True)
        return Score(ticks_per_quarter=480), [], []

    all_notes = [note for track in score.tracks for note in track.notes]
    if not all_notes:
        return score, [], []

    qpm = score.tempos[0].qpm if score.tempos else 120.0
    ticks_per_second = score.ticks_per_quarter * (qpm / 60)
    max_rest_duration_ticks = max_rest_duration_sec * ticks_per_second

    event_times = sorted(list(set(t for note in all_notes for t in (note.start, note.end))))

    skyline_notes: List[Note] = []
    accompaniment_notes: List[Note] = []
    last_accomp_note_for_pitch = {}

    for i in range(len(event_times) - 1):
        start_time = event_times[i]
        end_time = event_times[i+1]
        if start_time >= end_time: continue
        active_notes_in_interval = [note for note in all_notes if note.start <= start_time and note.end > start_time]
        if not active_notes_in_interval: continue
        highest_note = max(active_notes_in_interval, key=lambda note: note.pitch)
        is_valid_skyline_note = True
        if skyline_notes:
            time_since_last_skyline = start_time - skyline_notes[-1].end
            if time_since_last_skyline < max_rest_duration_ticks:
                last_skyline_pitch = skyline_notes[-1].pitch
                if last_skyline_pitch - highest_note.pitch > max_skyline_drop:
                    is_valid_skyline_note = False
        for note in active_notes_in_interval:
            is_the_skyline_note = (note is highest_note and is_valid_skyline_note)
            if is_the_skyline_note:
                if (skyline_notes and skyline_notes[-1].pitch == note.pitch and skyline_notes[-1].end == start_time):
                    skyline_notes[-1].duration += (end_time - start_time)
                else:
                    skyline_notes.append(Note(time=start_time, duration=end_time - start_time, pitch=note.pitch, velocity=note.velocity))
            else:
                last_accomp_note = last_accomp_note_for_pitch.get(note.pitch)
                if (last_accomp_note and last_accomp_note.end == start_time):
                    last_accomp_note.duration += (end_time - start_time)
                else:
                    new_accomp_note = Note(time=start_time, duration=end_time - start_time, pitch=note.pitch, velocity=note.velocity)
                    accompaniment_notes.append(new_accomp_note)
                    last_accomp_note_for_pitch[note.pitch] = new_accomp_note
    return score, skyline_notes, accompaniment_notes


def save_notes_to_midi(notes: list[Note], original_score: Score, output_path: str, track_name: str):
    # --- FIX ---
    # The Score constructor no longer accepts ticks_per_quarter directly.
    # First, create the Score object.
    new_score = Score()
    # Then, set the attribute.
    new_score.ticks_per_quarter = original_score.ticks_per_quarter

    track = Track(name=track_name, program=0, is_drum=False)
    track.notes.extend(notes)
    track.notes.sort(key=lambda note: note.start)
    new_score.tracks.append(track)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_score.dump_midi(output_path)
    except Exception as e:
        print(f"\n[ERROR] Error saving MIDI file to '{output_path}': {e}", flush=True)



def process_file_wrapper(input_path: str, input_dir: str, right_hand_dir: str, left_hand_dir: str) -> str:
    """
    Orchestrates the processing for a single file and returns a status string.
    This is called by the main worker function.
    """
    # CHANGE: Use a flushed print for reliable diagnostic output from child processes.
    print(f"[PID: {os.getpid()}] Starting: {os.path.relpath(input_path, input_dir)}", flush=True)

    try:
        original_score, right_hand_notes, left_hand_notes = extract_midi_skyline(input_path)
        if original_score is None or (not right_hand_notes and not left_hand_notes):
            return f"Skipped (no notes or load error): {os.path.relpath(input_path, input_dir)}"

        relative_path_no_ext = os.path.splitext(os.path.relpath(input_path, input_dir))[0]
        output_filename = relative_path_no_ext.replace(os.sep, '_') + '.mid'
        right_hand_output_path = os.path.join(right_hand_dir, output_filename)
        left_hand_output_path = os.path.join(left_hand_dir, output_filename)

        if right_hand_notes:
            save_notes_to_midi(right_hand_notes, original_score, right_hand_output_path, "Melody")
        if left_hand_notes:
            save_notes_to_midi(left_hand_notes, original_score, left_hand_output_path, "Accompaniment")

        return f"Success: {os.path.relpath(input_path, input_dir)}"
    except Exception as e:
        return f"Failed: {os.path.relpath(input_path, input_dir)} with error: {e}"

def worker(task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue, input_dir: str, right_hand_dir: str, left_hand_dir: str):
    """
    The main function for each worker process.
    """
    # The worker loop continues until it receives a `None` sentinel value
    for file_path in iter(task_queue.get, None):
        try:
            result = process_file_wrapper(file_path, input_dir, right_hand_dir, left_hand_dir)
            result_queue.put(result)
        except Exception as e:
            result_queue.put(f"WORKER CRASH on {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrently extract skyline and accompaniment from MIDI files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the root directory containing input MIDI files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where separated MIDI files will be saved.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes. Defaults to CPU count.")
    args = parser.parse_args()

    left_hand_dir = os.path.join(args.output_dir, "left_hand")
    right_hand_dir = os.path.join(args.output_dir, "right_hand")
    os.makedirs(left_hand_dir, exist_ok=True)
    os.makedirs(right_hand_dir, exist_ok=True)

    all_midi_files = []
    print(f"Searching for MIDI files in '{args.input_dir}'...")
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                all_midi_files.append(os.path.join(root, file))

    if not all_midi_files:
        print(f"No MIDI files found in '{args.input_dir}'.")
    else:
        num_files = len(all_midi_files)
        num_workers = min(args.workers, num_files) if num_files > 0 else 0
        print(f"Found {num_files} MIDI files to process using {num_workers} workers.")

        if num_workers > 0:
            task_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()

            processes = []
            for _ in range(num_workers):
                p = multiprocessing.Process(
                    target=worker,
                    args=(task_queue, result_queue, args.input_dir, right_hand_dir, left_hand_dir)
                )
                p.start()
                processes.append(p)

            for path in all_midi_files:
                task_queue.put(path)

            for _ in range(num_workers):
                task_queue.put(None)

            with tqdm(total=num_files, desc="Processing MIDI files") as pbar:
                for _ in range(num_files):
                    result = result_queue.get()
                    # CHANGE: Use tqdm.write to print results without breaking the progress bar
                    tqdm.write(result)
                    pbar.update(1)

            for p in processes:
                p.join()

        print("\n--- Process Complete ---")
        print(f"Left hand files saved in: '{left_hand_dir}'")
        print(f"Right hand files saved in: '{right_hand_dir}'")