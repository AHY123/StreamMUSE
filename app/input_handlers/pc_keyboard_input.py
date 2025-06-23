import sys
import tty
import termios
from queue import Queue

# A mapping from keyboard characters to MIDI note numbers.
KEY_TO_NOTE = {
    'a': 60, 'w': 61, 's': 62, 'e': 63, 'd': 64,
    'f': 65, 't': 66, 'g': 67, 'y': 68, 'h': 69,
    'u': 70, 'j': 71, 'k': 72,
}

def read_keyboard_input(event_queue: Queue):
    """
    A blocking function designed to be run in its own thread.
    It listens for keyboard input and puts structured note events into the
    provided thread-safe queue.
    """
    print("\rListening for keyboard input... Press 'q' or Ctrl+C to quit.")
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            char = sys.stdin.read(1)
            
            # Use Ctrl+C or 'q' to quit
            if char == '\x03' or char.lower() == 'q':
                # Put a "sentinel" value in the queue to signal the app to exit.
                event_queue.put(None) 
                break

            if char in KEY_TO_NOTE:
                note_event = {
                    "type": "note_on",
                    "pitch": KEY_TO_NOTE[char],
                    "key": char
                }
                # Put the structured event into the queue for the metronome to process.
                event_queue.put(note_event)

    finally:
        # Always restore terminal settings on exit.
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\rInput thread stopped.")