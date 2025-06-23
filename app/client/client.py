import requests
from pynput import keyboard
import time

# This should match the server address
SERVER_URL = "http://localhost:8000/generate"

# Maps keyboard keys to MIDI pitch values.
KEY_TO_PITCH = {
    'a': 60, 'w': 61, 's': 62, 'e': 63, 'd': 64, 'f': 65, 't': 66,
    'g': 67, 'y': 68, 'h': 69, 'u': 70, 'j': 71, 'k': 72
}

print("Starting StreamMUSE client...")
print("Press keys (a, w, s, e, d, f, t, g, y, h, u, j, k) to generate music.")
print("Press 'esc' to exit.")

def on_press(key):
    try:
        char_key = key.char
        if char_key in KEY_TO_PITCH:
            pitch = KEY_TO_PITCH[char_key]
            note_event = {
                "key": char_key,
                "pitch": pitch,
                "time": time.time()
            }
            
            print(f"-> Sending note: {note_event['key']} (pitch {note_event['pitch']})")
            
            try:
                # Send the note event to the server
                response = requests.post(SERVER_URL, json=[note_event])
                response.raise_for_status()  # Raise an exception for bad status codes
                
                accompaniment_notes = response.json()
                
                if accompaniment_notes:
                    print(f"<- Received accompaniment: {accompaniment_notes}")
                    # TODO: Add code here to play the received notes using a MIDI library
                    # like pygame.midi or mido.
                else:
                    print("<- Received no accompaniment.")
                    
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with server: {e}")

    except AttributeError:
        # Special keys (like 'esc')
        if key == keyboard.Key.esc:
            print("Exiting client.")
            return False  # Stop listener

# Collect events until released
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()