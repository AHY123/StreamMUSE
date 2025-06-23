import requests
import time
import mido
import mido.backends.rtmidi  # Ensures the rtmidi backend is available

# This should match the server address and may need to be updated
# if your client and server are on different machines.
SERVER_URL = "http://localhost:8000/generate"

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

def main():
    """
    The main function to run the MIDI client.
    """
    port = None
    try:
        # List available MIDI input ports
        input_ports = mido.get_input_names()
        print("Available MIDI input devices:")
        if not input_ports:
            print("  No MIDI devices found. Please connect a MIDI keyboard.")
            return
        
        for i, port_name in enumerate(input_ports):
            print(f"  {i}: {port_name}")
        
        # Open the first available MIDI port
        port_name = input_ports[0]
        port = mido.open_input(port_name)
        print(f"\nListening for MIDI input on '{port_name}'...")
        print("Play notes on your keyboard to generate music.")
        print("Press Ctrl+C to exit.")

        # Loop to process incoming MIDI messages
        for msg in port:
            # We only care about 'note_on' events with velocity > 0
            if msg.type == 'note_on' and msg.velocity > 0:
                note_event = {
                    "key": _midi_to_note_name(msg.note),
                    "pitch": msg.note,
                    "time": time.time()
                }
                
                print(f"-> Sending note: {note_event['key']} (pitch {note_event['pitch']})")
                
                try:
                    # Send the note event to the server
                    response = requests.post(SERVER_URL, json=[note_event])
                    response.raise_for_status()
                    
                    accompaniment_notes = response.json()
                    
                    if accompaniment_notes:
                        notes_str = ", ".join([f"{_midi_to_note_name(n['pitch'])}" for n in accompaniment_notes])
                        print(f"<- Received accompaniment: {notes_str}")
                        # TODO: Add code here to play the received notes.
                    else:
                        print("<- Received no accompaniment.")
                        
                except requests.exceptions.RequestException as e:
                    print(f"\nError communicating with server: {e}")
                    print("Is the server running?")

    except KeyboardInterrupt:
        print("\nExiting client.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if port and not port.closed:
            port.close()
            print("MIDI port closed.")

if __name__ == "__main__":
    main()