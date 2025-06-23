from collections import deque
import shutil
import time

class CLIOutputHandler:
    """
    Handles displaying a rolling, multi-line view of the musical interaction,
    showing one full beat's worth of ticks.
    """
    # REMOVED the limited NOTE_NAMES dictionary.

    def __init__(self, ticks_per_beat: int):
        """
        Initializes the handler with a display buffer for the rolling view.
        """
        # DYNAMIC: The number of rows is set here from the argument.
        self.ticks_per_beat = ticks_per_beat
        
        # DYNAMIC: The display buffer's max length (and thus row count)
        # is set using the ticks_per_beat variable.
        self.display_buffer = deque(['...'] * self.ticks_per_beat, maxlen=self.ticks_per_beat)
        
        self.is_first_display = True
        self.active_model_notes = set()
        self.all_inference_times = []

    def _midi_to_note_name(self, pitch: int) -> str:
        """
        Converts a MIDI pitch number (0-127) to its scientific pitch notation.
        e.g., 60 -> "C4", 69 -> "A4", 38 -> "D2"
        """
        if not 0 <= pitch <= 127:
            return "?"
        
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (pitch // 12) - 1
        note_index = pitch % 12
        return f"{note_names[note_index]}{octave}"

    def update_and_display(self, tick_count, debug_info, model_events_this_tick):
        """
        Formats the data for the current tick, including a new debug line,
        into fixed-width columns for a clean, table-like display.
        """
        # First, update the state of active model notes
        for event in model_events_this_tick:
            if event['type'] == 'note_on':
                self.active_model_notes.add(event['pitch'])
            elif event['type'] == 'note_off':
                self.active_model_notes.discard(event['pitch'])

        # --- Format Debug Line ---
        events_str = ", ".join([f"{e['key']}({e['pitch']})" for e in debug_info["events_this_tick"]]) or "None"
        trigger_str = "YES" if debug_info["inference_triggered"] else "NO"
        debug_line = f"[DEBUG] Tick Events: [{events_str}] | Inference Triggered: {trigger_str}"
        
        # Append benchmarking info to the debug line
        if "warmup_time" in debug_info:
            debug_line += f" | Warmup: {debug_info['warmup_time']:.4f}s"
        elif "last_inference_time" in debug_info:
            last_time = debug_info['last_inference_time']
            avg_time = debug_info['avg_inference_time']
            count = debug_info['inference_count']
            debug_line += f" | Last Inf: {last_time:.4f}s | Avg Inf ({count}): {avg_time:.4f}s"
            self.all_inference_times.append(last_time)
        
        # --- Format Main Display Line ---
        timeline_col_width = 12
        user_col_width = 25
        model_col_width = 40

        beat_num = (tick_count // self.ticks_per_beat) + 1
        sub_tick = (tick_count % self.ticks_per_beat) + 1
        beat_str = f"Beat {beat_num}.{sub_tick}"
        timeline_col = f"{beat_str:<{timeline_col_width}}"

        user_str = ""
        if debug_info["events_this_tick"]:
            first_event = debug_info["events_this_tick"][0]
            user_note_name = self._midi_to_note_name(first_event["pitch"])
            user_str = f"You ({first_event['key']}): {user_note_name}"
        user_col = f"{user_str:<{user_col_width}}"

        model_str = ""
        if self.active_model_notes:
            sorted_notes = sorted(list(self.active_model_notes))
            accompaniment_str = ", ".join([self._midi_to_note_name(p) for p in sorted_notes])
            model_str = f"Model Playing: {accompaniment_str}"
        model_col = f"{model_str:<{model_col_width}}"

        new_line = f"{timeline_col} | {user_col} | {model_col}"
        
        self.display_buffer.append(new_line)

        # --- Redraw Terminal ---
        if not self.is_first_display:
            # Move cursor up to the top of the entire display area (log + debug line)
            print(f"\x1b[{self.ticks_per_beat + 1}A", end="")
        
        self.is_first_display = False

        # Print the debug line first
        terminal_width = shutil.get_terminal_size().columns
        print(f"\x1b[2K{debug_line.ljust(terminal_width)}", flush=True)

        # Then print the rolling log
        for line in self.display_buffer:
            print(f"\x1b[2K{line}", flush=True)

    def save_log_on_exit(self, log_file_prefix="inference_log"):
            """Saves the collected inference times to a file on exit."""
            if not self.all_inference_times:
                print("\nNo inference times to log.")
                return

            avg_time = sum(self.all_inference_times) / len(self.all_inference_times)
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = f"{log_file_prefix}_{timestamp}.txt"

            try:
                with open(log_file, "w") as f:
                    f.write("--- StreamMUSE Inference Benchmark Log ---\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Total Inferences Logged: {len(self.all_inference_times)}\n")
                    f.write(f"Average Inference Time: {avg_time:.4f}s\n")
                    f.write(f"Min Inference Time: {min(self.all_inference_times):.4f}s\n")
                    f.write(f"Max Inference Time: {max(self.all_inference_times):.4f}s\n")
                    f.write("\n--- Raw Data (seconds) ---\n")
                    for i, t in enumerate(self.all_inference_times):
                        f.write(f"Inference {i+1}: {t:.6f}\n")
                
                print(f"\nInference log saved to {log_file}")
            except IOError as e:
                print(f"\nError saving log file: {e}")