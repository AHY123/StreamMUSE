import sys
import os
import time
import requests
import mido
import threading
from queue import Queue
import argparse

# This allows us to import the CLIOutputHandler from app/output_handlers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.output_handlers.cli_output import CLIOutputHandler

def _midi_to_note_name(pitch: int) -> str:
    """
    Converts a MIDI pitch number (0-127) to its scientific pitch notation.
    e.g., 60 -> "C4", 69 -> "A4"
    """
    if not 0 <= pitch <= 127:
        return "N/A"
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note_index = pitch % 12
    return f"{note_names[note_index]}{octave}"

# --- MIDI Player for Audio Output (using mido) ---
class MidiPlayer:
    def __init__(self, port_name: str = None):
        print("\n--- Available MIDI Output Ports ---")
        try:
            output_names = mido.get_output_names()
            if not output_names:
                print("  No MIDI output ports found.")
            else:
                for name in output_names:
                    print(f"  - '{name}'")
        except Exception as e:
            print(f"  Could not list MIDI ports: {e}")
        print("-------------------------------------\n")
        
        try:
            self.port = mido.open_output(port_name)
            print(f"Successfully opened MIDI output port: '{self.port.name}'")
            # Set instrument to Acoustic Grand Piano on channel 0
            self.port.send(mido.Message('program_change', channel=0, program=0))
            print("Set instrument on channel 0 to Acoustic Grand Piano.")
        except (OSError, IOError, mido.MidoError) as e:
            print(f"Warning: Could not open MIDI output port '{port_name}': {e}")
            print("Sound will not be played.")
            self.port = None

    def note_on(self, pitch, velocity=100, channel=0):
        if self.port:
            msg = mido.Message('note_on', note=pitch, velocity=velocity, channel=channel)
            self.port.send(msg)

    def note_off(self, pitch, velocity=0, channel=0):
        if self.port:
            msg = mido.Message('note_off', note=pitch, velocity=velocity, channel=channel)
            self.port.send(msg)

    def close(self):
        if self.port:
            self.port.reset()
            self.port.close()

# --- MIDI Input Handler (Runs in a separate thread) ---
def read_midi_input(event_queue: Queue, device_name: str = None):
    try:
        with mido.open_input(device_name) as port:
            print(f"Listening for MIDI input on '{port.name}'...")
            for msg in port:
                if msg.type == 'note_on' and msg.velocity > 0:
                    event = {"type": "note_on", "pitch": msg.note, "velocity": msg.velocity, "time": time.time()}
                    event_queue.put(event)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    event = {"type": "note_off", "pitch": msg.note, "velocity": 0, "time": time.time()}
                    event_queue.put(event)
    except (OSError, IOError) as e:
        print(f"\nError opening MIDI input port: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        event_queue.put(None)

# --- Inference Worker (Runs in a separate thread) ---
def inference_worker(request_queue: Queue, response_queue: Queue, server_url: str):
    while True:
        request_data = request_queue.get()
        if request_data is None:
            break
        start_time = time.perf_counter()
        try:
            response = requests.post(server_url, json=request_data)
            response.raise_for_status()
            future_events = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error contacting server: {e}")
            future_events = []
        end_time = time.perf_counter()
        round_trip_time = end_time - start_time
        response_queue.put((future_events, round_trip_time))

# --- Conductor Loop (Main application logic) ---
def tick_loop(
    event_queue: Queue, 
    inference_request_queue: Queue, 
    inference_response_queue: Queue,
    output_handler: CLIOutputHandler, 
    midi_player: MidiPlayer, 
    tempo: float, 
    ticks_per_beat: int, 
    beats_per_bar: int, 
    user_input_history: list,
    metronome_enabled: bool
):
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    tick_count = -1
    playback_schedule = {}
    notes_played_since_last_downbeat = []
    
    # --- Benchmarking & Display State ---
    round_trip_times = []
    warmup_completed = False
    last_inference_results = {} # FIX: Persists display data until the next result arrives

    METRONOME_PITCH_FIRST = 76
    METRONOME_PITCH_OTHER = 77
    PERCUSSION_CHANNEL = 9

    while True:
        tick_count += 1
        
        # 1. Harvest user input & PLAY IT (MIDI Thru)
        user_events_this_tick = []
        while not event_queue.empty():
            event = event_queue.get()
            if event is None:
                inference_request_queue.put(None)
                return
            
            user_input_history.append(event)
            user_events_this_tick.append(event)

            # MIDI THRU: Play user's notes immediately on channel 0
            if event['type'] == 'note_on':
                midi_player.note_on(event['pitch'], velocity=event['velocity'], channel=0)
            elif event['type'] == 'note_off':
                midi_player.note_off(event['pitch'], channel=0)
        
        user_notes_for_display = [e for e in user_events_this_tick if e['type'] == 'note_on']
        notes_for_inference = [{"key": "MIDI", "pitch": e['pitch'], "time": e['time']} for e in user_notes_for_display]

        if notes_for_inference:
            notes_played_since_last_downbeat.extend(notes_for_inference)

        # 2. Check for and process completed inference results
        while not inference_response_queue.empty():
            future_events, round_trip_time = inference_response_queue.get()
            
            # Schedule the received notes for playback
            for event in future_events:
                play_tick = event["play_at_tick"]
                if play_tick not in playback_schedule:
                    playback_schedule[play_tick] = []
                playback_schedule[play_tick].append({'type': 'note_on', 'pitch': event['pitch'], 'channel': 0})
                duration_in_seconds = event['duration']
                duration_in_ticks = max(1, round(duration_in_seconds / seconds_per_tick))
                note_off_tick = play_tick + duration_in_ticks
                if note_off_tick not in playback_schedule:
                    playback_schedule[note_off_tick] = []
                playback_schedule[note_off_tick].append({'type': 'note_off', 'pitch': event['pitch'], 'channel': 0})
            
            # FIX: Update the persistent state for display
            current_results = {}
            if not warmup_completed:
                current_results["warmup_time"] = round_trip_time
                warmup_completed = True
            else:
                round_trip_times.append(round_trip_time)
                current_results["last_inference_time"] = round_trip_time
                if round_trip_times:
                    current_results["avg_inference_time"] = sum(round_trip_times) / len(round_trip_times)
                    current_results["inference_count"] = len(round_trip_times)
            current_results["future_events"] = future_events
            last_inference_results = current_results

        # 3. On the downbeat, trigger a new inference if needed
        is_downbeat_tick = (tick_count % ticks_per_beat) == 0
        inference_triggered_this_tick = False
        if is_downbeat_tick and notes_played_since_last_downbeat:
            inference_request_queue.put(notes_played_since_last_downbeat.copy())
            notes_played_since_last_downbeat = []
            inference_triggered_this_tick = True

        # 4. Play any AI notes scheduled for this exact tick
        model_events_this_tick = playback_schedule.pop(tick_count, [])
        for event in model_events_this_tick:
            channel = event.get('channel', 0)
            if event['type'] == 'note_on':
                midi_player.note_on(event['pitch'], channel=channel)
            elif event['type'] == 'note_off':
                midi_player.note_off(event['pitch'], channel=channel)

        # 5. Update the display
        total_ticks_per_bar = ticks_per_beat * beats_per_bar
        bar_count = (tick_count // total_ticks_per_bar) if total_ticks_per_bar > 0 else 0
        beat_in_bar = ((tick_count % total_ticks_per_bar) // ticks_per_beat) if total_ticks_per_bar > 0 else 0
        
        if metronome_enabled and is_downbeat_tick:
            pitch = METRONOME_PITCH_FIRST if beat_in_bar == 0 else METRONOME_PITCH_OTHER
            velocity = 100 if beat_in_bar == 0 else 70
            midi_player.note_on(pitch, velocity=velocity, channel=PERCUSSION_CHANNEL)
            note_off_tick = tick_count + 1
            if note_off_tick not in playback_schedule:
                playback_schedule[note_off_tick] = []
            playback_schedule[note_off_tick].append({'type': 'note_off', 'pitch': pitch, 'channel': PERCUSSION_CHANNEL})

        # FIX: Assemble debug info for display using the persistent state
        debug_info = {
            "events_this_tick": user_notes_for_display, 
            "inference_triggered": inference_triggered_this_tick,
            "bar": bar_count,
            "beat": beat_in_bar + 1
        }
        debug_info.update(last_inference_results)
        output_handler.update_and_display(tick_count, debug_info, model_events_this_tick)
        
        # 6. Wait for the next tick
        time.sleep(seconds_per_tick)

# --- MIDI File Saving ---
def save_midi_log(history: list, ticks_per_beat: int = 480):
    if not history:
        print("\nNo user input to save to MIDI.")
        return
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    history.sort(key=lambda x: x['time'])
    if not history: return
    last_time = history[0]['time']
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
    for event in history:
        delta_seconds = event['time'] - last_time
        delta_ticks = int(mido.second2tick(delta_seconds, ticks_per_beat, mido.bpm2tempo(120)))
        track.append(mido.Message(
            event['type'], note=event['pitch'], velocity=event.get('velocity', 64), time=delta_ticks
        ))
        last_time = event['time']
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"user_input_{timestamp}.mid"
        mid.save(output_path)
        print(f"\nUser input saved to {output_path}")
    except Exception as e:
        print(f"\nError saving MIDI file: {e}")

# --- Main Entry Point ---
def main():
    SERVER_URL = "http://localhost:8000/generate"
    TEMPO = 120.0
    TICKS_PER_BEAT = 4
    BEATS_PER_BAR = 4

    parser = argparse.ArgumentParser(description="StreamMUSE Client")
    parser.add_argument("--server_url", type=str, default=SERVER_URL)
    parser.add_argument("--tempo", type=float, default=TEMPO)
    parser.add_argument("--ticks_per_beat", type=int, default=TICKS_PER_BEAT)
    parser.add_argument("--beats_per_bar", type=int, default=BEATS_PER_BAR)
    parser.add_argument("--log_lines", type=int, default=10)
    parser.add_argument("--metronome", action="store_true", help="Enable an audible MIDI metronome click.")
    parser.add_argument("--midi_output_name", type=str, default=None, help="Specify the MIDI output port name.")
    args = parser.parse_args()

    event_queue = Queue()
    inference_request_queue = Queue()
    inference_response_queue = Queue()
    midi_player = MidiPlayer(port_name=args.midi_output_name)
    output_handler = CLIOutputHandler(ticks_per_beat=args.ticks_per_beat, log_display_count=args.log_lines)
    user_input_history = []

    input_thread = threading.Thread(target=read_midi_input, args=(event_queue,), daemon=True)
    inference_thread = threading.Thread(
        target=inference_worker, 
        args=(inference_request_queue, inference_response_queue, args.server_url), 
        daemon=True
    )
    conductor_thread = threading.Thread(
        target=tick_loop,
        args=(
            event_queue, inference_request_queue, inference_response_queue, 
            output_handler, midi_player, args.tempo, args.ticks_per_beat, 
            args.beats_per_bar, user_input_history, args.metronome
        ),
        daemon=True
    )

    print("Starting StreamMUSE Client...")
    print(f"Connecting to server at {args.server_url}")

    try:
        input_thread.start()
        inference_thread.start()
        conductor_thread.start()
        input_thread.join()
        conductor_thread.join()
        inference_thread.join()
    except KeyboardInterrupt:
        print("\r\nCtrl+C detected. Exiting application.")
    finally:
        output_handler.save_log_on_exit(log_file_prefix="client_round_trip_log")
        save_midi_log(user_input_history)
        midi_player.close()
        print("\r\nApplication has been shut down.")

if __name__ == "__main__":
    main()