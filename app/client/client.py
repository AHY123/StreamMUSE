import sys
import os
import time
import requests
import mido
import pygame.midi
import threading
from queue import Queue
import argparse

# This allows us to import the CLIOutputHandler from app/output_handlers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.output_handlers.cli_output import CLIOutputHandler

# --- MIDI Player for Audio Output ---
class MidiPlayer:
    def __init__(self):
        pygame.midi.init()
        self.output_id = pygame.midi.get_default_output_id()
        if self.output_id == -1:
            print("Warning: No MIDI output device found. Accompaniment will not be played.")
            self.midi_out = None
        else:
            self.midi_out = pygame.midi.Output(self.output_id)
            print(f"MIDI output enabled on device ID {self.output_id}.")

    def note_on(self, pitch, velocity=100):
        if self.midi_out:
            self.midi_out.note_on(pitch, velocity)

    def note_off(self, pitch, velocity=0):
        if self.midi_out:
            self.midi_out.note_off(pitch, velocity)

    def close(self):
        if self.midi_out:
            pygame.midi.quit()

# --- MIDI Input Handler (Runs in a separate thread) ---
def read_midi_input(event_queue: Queue, device_name: str = None):
    try:
        with mido.open_input(device_name) as port:
            print(f"Listening for MIDI input on '{port.name}'...")
            for msg in port:
                if msg.type == 'note_on' and msg.velocity > 0:
                    event = {"key": "MIDI", "pitch": msg.note, "time": time.time()}
                    event_queue.put(event)
    except (OSError, IOError) as e:
        print(f"\nError opening MIDI port: {e}")
        print("Please ensure a MIDI device is connected.")
    except KeyboardInterrupt:
        pass # Allow Ctrl+C to exit gracefully
    finally:
        event_queue.put(None) # Signal the main loop to exit

# --- Conductor Loop (Main application logic) ---
def tick_loop(event_queue: Queue, output_handler: CLIOutputHandler, midi_player: MidiPlayer, server_url: str, tempo: float, ticks_per_beat: int):
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    tick_count = -1
    playback_schedule = {}
    notes_played_since_last_downbeat = []
    
    # --- Benchmarking State (measures client-side round-trip time) ---
    round_trip_times = []
    warmup_completed = False

    while True:
        tick_count += 1
        
        user_notes_this_tick = []
        while not event_queue.empty():
            event = event_queue.get()
            if event is None:
                print("\r\nInput thread exited. Shutting down conductor.")
                return
            user_notes_this_tick.append(event)
        
        if user_notes_this_tick:
            notes_played_since_last_downbeat.extend(user_notes_this_tick)

        is_downbeat_tick = (tick_count % ticks_per_beat) == 0
        debug_info = {"events_this_tick": user_notes_this_tick, "inference_triggered": False}
        
        if is_downbeat_tick and notes_played_since_last_downbeat:
            debug_info["inference_triggered"] = True
            
            # --- Benchmarking & Server Call ---
            start_time = time.perf_counter()
            try:
                response = requests.post(server_url, json=notes_played_since_last_downbeat)
                response.raise_for_status()
                future_events = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Error contacting server: {e}")
                future_events = []
            end_time = time.perf_counter()
            last_round_trip_time = end_time - start_time
            
            if not warmup_completed:
                debug_info["warmup_time"] = last_round_trip_time
                warmup_completed = True
            else:
                round_trip_times.append(last_round_trip_time)
                debug_info["last_inference_time"] = last_round_trip_time # Use same key for CLIOutputHandler
                if round_trip_times:
                    debug_info["avg_inference_time"] = sum(round_trip_times) / len(round_trip_times)
                    debug_info["inference_count"] = len(round_trip_times)
            # --- End Benchmarking ---
            
            for event in future_events:
                play_tick = event["play_at_tick"]
                if play_tick not in playback_schedule:
                    playback_schedule[play_tick] = []
                playback_schedule[play_tick].append({'type': 'note_on', 'pitch': event['pitch']})

                duration_in_seconds = event['duration']
                duration_in_ticks = max(1, round(duration_in_seconds / seconds_per_tick))
                note_off_tick = play_tick + duration_in_ticks
                
                if note_off_tick not in playback_schedule:
                    playback_schedule[note_off_tick] = []
                playback_schedule[note_off_tick].append({'type': 'note_off', 'pitch': event['pitch']})
            
            notes_played_since_last_downbeat = []

        model_events_this_tick = playback_schedule.pop(tick_count, [])
        for event in model_events_this_tick:
            if event['type'] == 'note_on':
                midi_player.note_on(event['pitch'])
            elif event['type'] == 'note_off':
                midi_player.note_off(event['pitch'])

        output_handler.update_and_display(tick_count, debug_info, model_events_this_tick)
        time.sleep(seconds_per_tick)

# --- Main Entry Point ---
def main():
    SERVER_URL = "http://localhost:8000"
    TEMPO = 120
    TICKS_PER_BEAT = 4

    parser = argparse.ArgumentParser(description="StreamMUSE Client: Connects to a StreamMUSE server for real-time music generation.")
    parser.add_argument("--server_url", type=str, default=SERVER_URL, help="URL of the StreamMUSE inference server.")
    parser.add_argument("--tempo", type=float, default=TEMPO, help="Tempo in beats per minute.")
    parser.add_argument("--ticks_per_beat", type=int, default=TICKS_PER_BEAT, help="Subdivisions per beat.")
    args = parser.parse_args()

    # --- Setup Components ---
    event_queue = Queue()
    midi_player = MidiPlayer()
    output_handler = CLIOutputHandler(ticks_per_beat=args.ticks_per_beat)

    # --- Setup Threads ---
    input_thread = threading.Thread(target=read_midi_input, args=(event_queue,), daemon=True)
    conductor_thread = threading.Thread(
        target=tick_loop,
        args=(event_queue, output_handler, midi_player, args.server_url, args.tempo, args.ticks_per_beat),
        daemon=True
    )

    print("Starting StreamMUSE Client...")
    print(f"Connecting to server at {args.server_url}")
    print("\n" * (args.ticks_per_beat + 1)) # Create space for the display

    try:
        input_thread.start()
        conductor_thread.start()
        # The main thread will wait here until the input thread exits (e.g., on error or Ctrl+C)
        input_thread.join()
        conductor_thread.join() # Wait for conductor to finish its last tick
    except KeyboardInterrupt:
        print("\r\nCtrl+C detected. Exiting application.")
    finally:
        output_handler.save_log_on_exit(log_file_prefix="client_round_trip_log")
        midi_player.close()
        print("\r\nApplication has been shut down.")

if __name__ == "__main__":
    main()