from symusic import Score, Note, Track
from typing import List, Tuple
import argparse
import os
import glob
import concurrent.futures
from tqdm import tqdm

def extract_midi_skyline(midi_file_path: str, pitch_barrier: int = 12) -> Tuple[Score, List[Note], List[Note]]:
    """
    Separates a MIDI file into a 'skyline' melody and a filtered 'accompaniment'.

    - The skyline is the highest active pitch at any moment in time.
    - The accompaniment consists of all other notes playing concurrently that are
      within a specified pitch range below the skyline note.

    :param midi_file_path: Path to the MIDI file.
    :param pitch_barrier: The maximum number of semitones below the skyline note
                          that an accompaniment note can be. Defaults to 24 (2 octaves).
    :return: A tuple containing:
             - The original symusic.Score object.
             - A list of symusic.Note objects for the skyline melody.
             - A list of symusic.Note objects for the filtered accompaniment.
    """
    try:
        score = Score(midi_file_path)
    except Exception as e:
        print(f"Error loading MIDI file '{midi_file_path}': {e}")
        return Score(ticks_per_quarter=480), [], []

    all_notes = [note for track in score.tracks for note in track.notes]
    if not all_notes:
        return score, [], []

    # Create a sorted list of all unique note start and end times
    event_times = sorted(list(set(t for note in all_notes for t in (note.start, note.end))))

    skyline_notes: List[Note] = []
    accompaniment_notes: List[Note] = []
    
    # This dictionary tracks the last added accompaniment note for each pitch to merge them correctly
    last_accomp_note_for_pitch = {}

    for i in range(len(event_times) - 1):
        start_time = event_times[i]
        end_time = event_times[i+1]
        
        if start_time >= end_time:
            continue

        # Find all notes that are active during this time slice
        active_notes_in_interval = [
            note for note in all_notes 
            if note.start <= start_time and note.end > start_time
        ]
        
        if not active_notes_in_interval:
            continue

        # Find the single highest note in the interval
        highest_note = max(active_notes_in_interval, key=lambda note: note.pitch)
        
        # --- 1. Process the Skyline Note ---
        # If the new skyline segment has the same pitch as the last one and is contiguous, merge them
        if (skyline_notes and 
            skyline_notes[-1].pitch == highest_note.pitch and
            skyline_notes[-1].end == start_time):
            skyline_notes[-1].duration += (end_time - start_time)
        else:
            # Create a new skyline note segment
            skyline_notes.append(Note(
                time=start_time,
                duration=end_time - start_time,
                pitch=highest_note.pitch,
                velocity=highest_note.velocity
            ))

        # --- 2. Process Accompaniment Notes ---
        for note in active_notes_in_interval:
            # An accompaniment note must not be the highest note AND must be within the pitch barrier
            if note is not highest_note and (highest_note.pitch - note.pitch <= pitch_barrier):
                
                # Check if this note can be merged with a previous segment of the same pitch
                last_note = last_accomp_note_for_pitch.get(note.pitch)
                if (last_note and last_note.end == start_time):
                    last_note.duration += (end_time - start_time)
                else:
                    # Create a new accompaniment note segment
                    new_accomp_note = Note(
                        time=start_time,
                        duration=end_time - start_time,
                        pitch=note.pitch,
                        velocity=note.velocity
                    )
                    accompaniment_notes.append(new_accomp_note)
                    # Track this new note for potential future merges
                    last_accomp_note_for_pitch[note.pitch] = new_accomp_note

    return score, skyline_notes, accompaniment_notes


def save_notes_to_midi(
    notes: list[Note],
    original_score: Score,
    output_path: str,
    track_name: str
):
    new_score = Score(ticks_per_quarter=original_score.ticks_per_quarter)
    track = Track(name=track_name, program=0, is_drum=False)
    track.notes.extend(notes)
    track.notes.sort(key=lambda note: note.start)
    new_score.tracks.append(track)
    try:
        # 目录创建可以放在主逻辑中一次性完成，这里可以省略
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_score.dump_midi(output_path)
    except Exception as e:
        print(f"Error saving MIDI file to '{output_path}': {e}")

# --- 封装的“工作函数” ---
def process_single_file(input_path: str, args):
    """
    处理单个文件的完整逻辑，方便并行调用。
    """
    try:
        original_score, right_hand_notes, left_hand_notes = extract_midi(input_path)

        if original_score is None or (not right_hand_notes and not left_hand_notes):
            return f"Skipped (no data): {input_path}"

        # --- 计算输出路径 ---
        left_hand_dir = os.path.join(args.output_dir, "acc")
        right_hand_dir = os.path.join(args.output_dir, "mel")
        
        relative_path_no_ext = os.path.splitext(os.path.relpath(input_path, args.input_dir))[0]
        output_filename = relative_path_no_ext.replace(os.sep, '_') + '.mid'

        right_hand_output_path = os.path.join(right_hand_dir, output_filename)
        left_hand_output_path = os.path.join(left_hand_dir, output_filename)
        
        # --- 保存文件 ---
        if right_hand_notes:
            save_notes_to_midi(right_hand_notes, original_score, right_hand_output_path, "Right Hand")
        if left_hand_notes:
            save_notes_to_midi(left_hand_notes, original_score, left_hand_output_path, "Left Hand")
            
        return f"Success: {input_path}"
    except Exception as e:
        return f"Failed: {input_path} with error: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract right-hand (>=C4) and left-hand (<C4) notes from MIDI files concurrently.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the root directory containing input MIDI files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where separated MIDI files will be saved.")
    # 添加一个控制进程数的参数
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes to use. Defaults to the number of CPU cores.")
    args = parser.parse_args()

    left_hand_dir = os.path.join(args.output_dir, "acc")
    right_hand_dir = os.path.join(args.output_dir, "mel")

    os.makedirs(left_hand_dir, exist_ok=True)
    os.makedirs(right_hand_dir, exist_ok=True)

    print(f"Searching for MIDI files in '{args.input_dir}'...")
    search_pattern = os.path.join(args.input_dir, '**', '*.mid')
    all_midi_files = glob.glob(search_pattern, recursive=True)
    search_pattern_midi = os.path.join(args.input_dir, '**', '*.midi')
    all_midi_files.extend(glob.glob(search_pattern_midi, recursive=True))

    if not all_midi_files:
        print(f"No MIDI files found in '{args.input_dir}'.")
    else:
        print(f"Found {len(all_midi_files)} total MIDI files.")
        
        # --- 这里是并发处理的核心 ---
        print(f"\nProcessing {len(all_midi_files)} files using multiple processes...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            # 创建一个 future 列表，将每个文件的处理任务提交给进程池
            futures = [executor.submit(process_single_file, midi_file, args) for midi_file in all_midi_files]
            
            # 使用 tqdm 来显示处理进度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_midi_files)):
                # (可选) 您可以在这里处理每个任务返回的结果，例如记录失败的文件
                # print(future.result())
                pass

        print("\n--- Concurrency Process Complete! ---")
        print(f"Left hand files saved in: '{left_hand_dir}'")
        print(f"Right hand files saved in: '{right_hand_dir}'")