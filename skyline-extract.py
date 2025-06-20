from symusic import Score, Note, Track
from typing import List, Tuple
import argparse
import os
import glob
import concurrent.futures
from tqdm import tqdm

def extract_midi_skyline(midi_file_path: str, max_skyline_drop: int = 12, max_rest_duration_sec: float = 2.0) -> Tuple[Score, List[Note], List[Note]]:
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