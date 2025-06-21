from symusic import Score, Note, Track
from typing import List, Tuple
import argparse
import os
import glob
import concurrent.futures
from tqdm import tqdm

def extract_midi_skyline(
    midi_file_path: str, 
    max_skyline_drop: int = 12,
    max_rest_duration_sec: float = 2.0
) -> Tuple[Score, List[Note], List[Note]]:
    """
    Separates a MIDI file into a 'skyline' melody and the 'accompaniment' notes.

    - The skyline is the highest active pitch, but avoids sudden large drops.
    - The accompaniment consists of ALL other notes that are not part of the skyline.

    :param midi_file_path: Path to the MIDI file.
    :param max_skyline_drop: Max semitones the skyline can drop between consecutive notes.
    :param max_rest_duration_sec: Max duration of a rest in seconds before the skyline
                                  pitch restriction is reset.
    :return: A tuple of (original Score, skyline notes, accompaniment notes).
    """
    try:
        score = Score(midi_file_path)
    except Exception as e:
        print(f"Error loading MIDI file '{midi_file_path}': {e}")
        return Score(ticks_per_quarter=480), [], []

    all_notes = [note for track in score.tracks for note in track.notes]
    if not all_notes:
        return score, [], []

    # Convert max rest duration from seconds to MIDI ticks
    # This uses the first tempo marking as a reference.
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
        
        if start_time >= end_time:
            continue

        active_notes_in_interval = [
            note for note in all_notes 
            if note.start <= start_time and note.end > start_time
        ]
        
        if not active_notes_in_interval:
            continue

        highest_note = max(active_notes_in_interval, key=lambda note: note.pitch)
        
        # --- Check for significant skyline drops, but only after short rests ---
        is_valid_skyline_note = True
        if skyline_notes:
            time_since_last_skyline = start_time - skyline_notes[-1].end
            # Only apply the drop rule if the rest was shorter than the threshold
            if time_since_last_skyline < max_rest_duration_ticks:
                last_skyline_pitch = skyline_notes[-1].pitch
                if last_skyline_pitch - highest_note.pitch > max_skyline_drop:
                    is_valid_skyline_note = False
        
        # --- Process each note in the interval ---
        for note in active_notes_in_interval:
            is_the_skyline_note = (note is highest_note and is_valid_skyline_note)

            if is_the_skyline_note:
                # --- 1. Add to Skyline (with merging) ---
                if (skyline_notes and 
                    skyline_notes[-1].pitch == note.pitch and
                    skyline_notes[-1].end == start_time):
                    skyline_notes[-1].duration += (end_time - start_time)
                else:
                    skyline_notes.append(Note(
                        time=start_time,
                        duration=end_time - start_time,
                        pitch=note.pitch,
                        velocity=note.velocity
                    ))
            else:
                # --- 2. Add to Accompaniment (with merging) ---
                last_accomp_note = last_accomp_note_for_pitch.get(note.pitch)
                if (last_accomp_note and last_accomp_note.end == start_time):
                    last_accomp_note.duration += (end_time - start_time)
                else:
                    new_accomp_note = Note(
                        time=start_time,
                        duration=end_time - start_time,
                        pitch=note.pitch,
                        velocity=note.velocity
                    )
                    accompaniment_notes.append(new_accomp_note)
                    last_accomp_note_for_pitch[note.pitch] = new_accomp_note

    return score, skyline_notes, accompaniment_notes


def save_notes_to_midi(
    notes: list[Note],
    original_score: Score,
    output_path: str,
    track_name: str
):
    """
    Saves a list of symusic.Note objects to a new MIDI file using the symusic library.
    This is the definitive version that correctly creates a Score and saves it.
    """
    # 1. Create a score object using the Score() factory function.
    new_score = Score()
    
    # 2. Set the 'ticks_per_quarter' attribute on the newly created object.
    new_score.ticks_per_quarter = original_score.ticks_per_quarter

    # 3. Create a new track for the notes.
    track = Track(name=track_name, program=0, is_drum=False)

    # 4. Add the notes to the track's note list.
    track.notes.extend(notes)
    
    # 5. IMPORTANT: Sort notes by start time for a valid MIDI file.
    track.notes.sort(key=lambda note: note.start)
    
    # 6. Add the completed track to the score.
    new_score.tracks.append(track)
    
    try:
        # Create the directory if it doesn't exist.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 7. Use the correct .dump_midi() method to save the file.
        new_score.dump_midi(output_path)
        
    except Exception as e:
        print(f"Error saving MIDI file with symusic to '{output_path}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract right-hand (>=C4) and left-hand (<C4) notes from all MIDI files in a directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the root directory containing input MIDI files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where separated MIDI files will be saved.")
    args = parser.parse_args()

    # --- NEW LOGIC: Define the specific output folders ---
    left_hand_dir = os.path.join(args.output_dir, "left_hand")
    right_hand_dir = os.path.join(args.output_dir, "right_hand")

    # --- NEW LOGIC: Create these folders at the start ---
    os.makedirs(left_hand_dir, exist_ok=True)
    os.makedirs(right_hand_dir, exist_ok=True)

    # Find all MIDI files in the input directory and its subdirectories
    all_midi_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                all_midi_files.append(os.path.join(root, file))

    if not all_midi_files:
        print(f"No MIDI files found in '{args.input_dir}'.")
    else:

        print(f"Found {len(all_midi_files)} MIDI files to process.")

        # Process each MIDI file
        for input_path in tqdm(all_midi_files, desc="Processing MIDI files"):
            original_score, right_hand_notes, left_hand_notes = extract_midi_skyline(input_path)

            if original_score is None:
                continue

            # --- MODIFIED LOGIC: Create a unique flat filename ---
            # This prevents collisions if you have files with the same name in different subfolders.
            # For example, 'subdir/song.mid' becomes 'subdir_song.mid'
            relative_path_no_ext = os.path.splitext(os.path.relpath(input_path, args.input_dir))[0]
            output_filename = relative_path_no_ext.replace(os.sep, '_') + '.mid'

            # --- MODIFIED LOGIC: Define output paths for the new structure ---
            right_hand_output_path = os.path.join(right_hand_dir, output_filename)
            left_hand_output_path = os.path.join(left_hand_dir, output_filename)
            
            # Save the new MIDI files to their respective folders
            if right_hand_notes:
                save_notes_to_midi(right_hand_notes, original_score, right_hand_output_path, "Melody")
            if left_hand_notes:
                save_notes_to_midi(left_hand_notes, original_score, left_hand_output_path, "Accompaniment")

        print("\nExtraction process complete!")
        print(f"Left hand files saved in: '{left_hand_dir}'")
        print(f"Right hand files saved in: '{right_hand_dir}'")