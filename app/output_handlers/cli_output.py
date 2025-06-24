from collections import deque
import shutil
import time

class CLIOutputHandler:
    """
    Handles displaying a persistent log with updating status lines at the bottom.
    """
    def __init__(self, ticks_per_beat: int, log_display_count: int = 10):
        self.ticks_per_beat = ticks_per_beat
        self.log_display_count = log_display_count
        self.status_line_count = 4  # Separator, Inputs, Output, Status
        self.total_managed_lines = self.log_display_count + self.status_line_count

        # Use a deque for efficient appending and keeping a fixed-size history
        self.log_history = deque(maxlen=200)  # Store more than we display
        self.last_model_output_str = "None"
        self.is_first_display = True
        self.all_inference_times = []  # For saving log on exit
        self.all_times = []

    def _midi_to_note_name(self, pitch: int) -> str:
        """
        Converts a MIDI pitch number (0-127) to its scientific pitch notation.
        e.g., 60 -> "C4", 69 -> "A4", 38 -> "D2"
        """
        if not 0 <= pitch <= 127:
            return "N/A"
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (pitch // 12) - 1
        note_index = pitch % 12
        return f"{note_names[note_index]}{octave}"

    def update_and_display(self, tick_count, debug_info, model_events_this_tick):
        # 1. --- Prepare content for all lines ---

        # A. LOGS: Create a log entry for every tick
        user_notes_played_str = ", ".join([self._midi_to_note_name(e['pitch']) for e in debug_info["events_this_tick"]])
        model_notes_played_str = ", ".join([self._midi_to_note_name(e['pitch']) for e in model_events_this_tick if e['type'] == 'note_on'])

        bar = debug_info.get('bar', 0)
        beat = debug_info.get('beat', 0)
        sub_tick = (tick_count % self.ticks_per_beat) + 1
        log_entry = f"[{bar:03d}.{beat}.{sub_tick}]"
        
        if user_notes_played_str:
            log_entry += f" USER: [{user_notes_played_str}]"
        if model_notes_played_str:
            log_entry += f" | MODEL: [{model_notes_played_str}]"
        
        # Add a placeholder for empty ticks to make them less visually jarring
        if not user_notes_played_str and not model_notes_played_str:
            log_entry += " ..."

        self.log_history.append(log_entry)

        # B. STATUS LINE 1: Inputs this tick
        inputs_line = f"INPUTS (current tick): {user_notes_played_str or 'None'}"

        # C. STATUS LINE 2: Last model output (update if inference was triggered)
        if debug_info.get("inference_triggered"):
            future_events = debug_info.get("future_events", [])
            self.last_model_output_str = ", ".join([self._midi_to_note_name(e['pitch']) for e in future_events]) or "None"
            if "last_inference_time" in debug_info:
                self.all_inference_times.append(debug_info['last_inference_time'])

        output_line = f"LAST MODEL OUTPUT:     {self.last_model_output_str}"

        # D. STATUS LINE 3: General status and benchmarking
        bar = debug_info.get('bar', 0)
        beat = debug_info.get('beat', 0)
        status_line = f"TIME: Bar {bar:03d}, Beat {beat} |"
        
        if "warmup_time" in debug_info:
            status_line += f" Warmup: {debug_info['warmup_time']:.3f}s"
        elif "last_inference_time" in debug_info:
            last_time = debug_info['last_inference_time']
            avg_time = debug_info['avg_inference_time']
            count = debug_info['inference_count']
            status_line += f" Round-trip: {last_time:.3f}s (Avg: {avg_time:.3f}s over {count} calls)"
        
        # 2. --- Render the display ---

        if self.is_first_display:
            print("\n" * self.total_managed_lines, end="")
            self.is_first_display = False

        # Move cursor up to the top of the managed area
        print(f"\r\x1b[{self.total_managed_lines}A", end="")

        # Get the last N log lines to display
        display_logs = list(self.log_history)[-self.log_display_count:]
        
        # Print logs
        for i in range(self.log_display_count):
            line_content = display_logs[i] if i < len(display_logs) else ""
            print(f"\x1b[2K{line_content}", flush=True)

        # Print separator and status lines
        try:
            terminal_width = shutil.get_terminal_size().columns
        except OSError:
            terminal_width = 80 # Default width
        print(f"\x1b[2K" + "="*terminal_width, flush=True)
        print(f"\x1b[2K{inputs_line}", flush=True)
        print(f"\x1b[2K{output_line}", flush=True)
        print(f"\x1b[2K{status_line}", flush=True)

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
                f.write("--- StreamMUSE Client Round-Trip Benchmark Log ---\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Total Inferences Logged: {len(self.all_inference_times)}\n")
                f.write(f"Average Round-Trip Time: {avg_time:.4f}s\n")
                f.write(f"Min Round-Trip Time: {min(self.all_inference_times):.4f}s\n")
                f.write(f"Max Round-Trip Time: {max(self.all_inference_times):.4f}s\n")
                f.write("\n--- Raw Data (seconds) ---\n")
                for i, t in enumerate(self.all_inference_times):
                    f.write(f"Inference {i+1}: {t:.6f}\n")
            
            print(f"\nRound-trip log saved to {log_file}")
        except IOError as e:
            print(f"\nError saving log file: {e}")
