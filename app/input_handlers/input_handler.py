"""
This file contains the input handlers for the StreamMUSE client,
including MIDI device input and computer keyboard input.
"""

import time
import mido
from queue import Queue
import heapq 

# --- MIDI Input Handler ---
def read_midi_input(event_queue: Queue, device_name: str = None):
    """
    Worker function for reading MIDI input from a connected device (separate thread).
    """
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
    except (OSError, IOError, mido.MidoError) as e:
        print(f"\nError opening MIDI input port: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        event_queue.put(None)


# --- Keyboard Input Handler ---
KEY_TO_PITCH = {
    # White keys (bottom row)
    'z': 60, 'x': 62, 'c': 64, 'v': 65, 'b': 67, 'n': 69, 'm': 71,
    ',': 72, '.': 74, '/': 76,
    # Black keys (top row)
    's': 61, 'd': 63, 'g': 66, 'h': 68, 'j': 70, 'l': 73, ';': 75,
}
VELOCITY = 100
pressed_keys = set()

def _on_press(key, event_queue):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH and char_key not in pressed_keys:
            pitch = KEY_TO_PITCH[char_key]
            pressed_keys.add(char_key)
            event = {"type": "note_on", "pitch": pitch, "velocity": VELOCITY, "time": time.time()}
            event_queue.put(event)
    except AttributeError:
        pass # Ignore special keys

def _on_release(key, event_queue):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH and char_key in pressed_keys:
            pressed_keys.remove(char_key)
            pitch = KEY_TO_PITCH[char_key]
            event = {"type": "note_off", "pitch": pitch, "velocity": VELOCITY, "time": time.time()}
            event_queue.put(event)
    except AttributeError:
        if key == keyboard.Key.esc:
            # Stop listener
            event_queue.put(None)

def read_keyboard_input(event_queue: Queue):
    from pynput import keyboard
    """
    Worker function for reading computer keyboard input (separate thread).
    Maps keyboard keys to MIDI notes.
    """
    print("Listening for computer keyboard input...")
    print("Mapped keys: 'zxcvbnm,' and 'sdghjl;'")
    print("Press 'ESC' to exit.")
    
    with keyboard.Listener(
            on_press=lambda key: _on_press(key, event_queue),
            on_release=lambda key: _on_release(key, event_queue)) as listener:
        listener.join() 

def read_midi_file_input(event_queue: Queue, midi_file_path:str, tempo: int = 90):
    """
    Plays a sequence of notes with polyphony using a scheduler.

    Args:
        event_queue (Queue): The queue to put MIDI events into.
        notes_to_play (list): A list of tuples, where each tuple is
                              (start_time_in_beats, pitch, duration_in_beats).
        tempo (int): The tempo in beats per minute (BPM).
    """
    beat_duration = 60.0 / tempo
    
    # A priority queue to store future events (note_on, note_off)
    event_schedule = []

    try:
        mid = mido.MidiFile(midi_file_path)
        print(f"Parsing '{midi_file_path}'...")

        absolute_time = 0.0
        # Mido's iteration yields messages with delta times in seconds.
        # We accumulate this to get the absolute time for each event.
        for msg in mid:
            absolute_time += msg.time
            
            # We only care about note_on/note_off messages
            if msg.type == 'note_on' and msg.velocity > 0:
                # Add a 'note_on' event to the schedule
                event = (absolute_time, 'note_on', msg.note, msg.velocity)
                heapq.heappush(event_schedule, event)
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Add a 'note_off' event for the same note
                event = (absolute_time, 'note_off', msg.note, msg.velocity)
                heapq.heappush(event_schedule, event)

    except FileNotFoundError:
        print(f"Error: MIDI file not found at '{midi_file_path}'")
        event_queue.put(None)
        return
    except Exception as e:
        print(f"\nAn error occurred during MIDI file parsing: {e}")
        event_queue.put(None)
        return

    try:
        with mido.open_output(virtual=True) as port:
            print(f"Playing polyphonic notes on '{port.name}' at {tempo} BPM...")
            
            start_of_playback = time.monotonic()
            
            # Loop as long as there are events in the schedule
            while event_schedule:
                # Get the current time elapsed since playback started
                current_time = time.monotonic() - start_of_playback
                
                # Check if the next event's time has been reached
                if event_schedule[0][0] <= current_time:
                    # Get the event from the schedule
                    event_time, event_type, pitch = heapq.heappop(event_schedule)
                    VELOCITY = 100
                    
                    if event_type == "note_on":
                        msg = mido.Message('note_on', note=pitch, velocity=VELOCITY)
                        port.send(msg)
                        # Put the event into the main application queue
                        event_queue.put({"type": "note_on", "pitch": pitch, "velocity": VELOCITY, "time": time.time()})
                    
                    elif event_type == "note_off":
                        msg = mido.Message('note_off', note=pitch, velocity=0)
                        port.send(msg)
                        # Put the event into the main application queue
                        event_queue.put({"type": "note_off", "pitch": pitch, "velocity": 0, "time": time.time()})
                else:
                    # If it's not time for the next event, sleep for a very short duration
                    # to prevent the loop from consuming 100% CPU.
                    time.sleep(0.001)

    except Exception as e:
        print(f"\nAn error occurred during polyphonic playback: {e}")
    finally:
        event_queue.put(None)