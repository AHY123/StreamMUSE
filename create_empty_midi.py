import mido

# Define constants
FILENAME = 'empty_1_minute.mid'
DURATION_SECONDS = 60

# Standard MIDI file parameters
TICKS_PER_BEAT = 480  # Common value for ticks per beat (quarter note)
TEMPO = mido.bpm2tempo(120)  # Standard 120 BPM, returns tempo in microseconds per beat

# --- Create the MIDI File ---

# 1. Create a new MIDI file (type 1)
mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)

# 2. Create a new track
track = mido.MidiTrack()
mid.tracks.append(track)

# 3. Calculate the total duration in MIDI ticks
# The 'time' attribute of a message is a delta time in ticks.
# To make the track last 60 seconds, we add a final 'end_of_track'
# message with a time delta equal to 60 seconds worth of ticks.
duration_ticks = int(mido.second2tick(DURATION_SECONDS, TICKS_PER_BEAT, TEMPO))

# 4. Add an 'end_of_track' meta message to give the track its duration.
# The track is empty of notes, but this message makes it 60 seconds long.
track.append(mido.MetaMessage('end_of_track', time=duration_ticks))

# 5. Save the MIDI file
mid.save(FILENAME)

print(f"✅ Successfully created '{FILENAME}' with a duration of {DURATION_SECONDS} seconds.")