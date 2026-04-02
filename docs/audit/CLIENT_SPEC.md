# CLIENT_SPEC.md — Real-Time Client Specification

> Primary clients: `app/client_lekai.py`, `app/web_client.py`
> Secondary (functional but not demo-primary): `app/client.py`
> Last audited: 2026-03-23

---

## Architecture

The client is a multi-threaded system with:
- **Input Thread**: Reads MIDI/keyboard events, puts into `event_queue`
- **Main Thread**: Tick loop — advances time, collects events, sends inference requests
- **Audio Output**: Scheduled note playback synchronized to tick clock

---

## `app/client_lekai.py` — Lekai-Focused Client

### Entry
```bash
python app/client_lekai.py [options]
```

### CLI Arguments (key subset)
| Argument | Default | Description |
|---|---|---|
| `--server-url` | `http://localhost:8000/generate_accompaniment` | Server endpoint |
| `--tempo` | 120 | BPM |
| `--ticks-per-beat` | 4 | Tick resolution |
| `--use-keyboard-input` | False | Use computer keyboard instead of MIDI |
| `--midi-file-input FILE` | None | Simulate MIDI file input |
| `--midi-file-delay-ticks N` | 8 | Delay before MIDI file starts |
| `--injection-file FILE` | None | MIDI file to inject as prompt |
| `--injection-length N` | 50 | Ticks of injection to use |
| `--midi-file-output FILE` | auto-named | Record session to MIDI |
| `--listening-mode` | depends | Enable adaptive key detection |

### Startup Sequence
1. Parse CLI args → build `StreamMUSEConfig`
2. Open audio output port (MIDI synth)
3. `POST /clear_history` to server
4. If `--injection-file`: call `perform_direct_injection()` → `POST /inject_notes`
5. If listening mode: initialize `PromptLibrary`, start key detection
6. Spawn input thread (MIDI / keyboard / MIDI file)
7. Enter tick loop

### Tick Loop
```
Every tick:
  - Advance system_tick counter
  - Check playback_schedule for notes due at this tick
    → send note_on / note_off to audio_handler
  - Every generation_interval_ticks:
    → Collect melody events since last inference
    → Send POST /generate_accompaniment
    → Receive AccompanimentResponse
    → Add to playback_schedule
    → Log to CLIOutputHandler, JsonLogHandler
```

### Note Scheduling
- Notes are added to `playback_schedule: dict[tick → list[notes]]`
- Tick loop plays all notes at their scheduled tick
- Compensation: requests sent slightly before actual generation tick to account for network latency

### Injection Flow
```python
perform_direct_injection(injection_file, injection_length_ticks, server_url):
  1. Load MIDI file with pretty_midi
  2. Convert to MelodyNoteEvent / AccompanimentNoteEvent lists
  3. POST /inject_notes with {melody_notes, accompaniment_notes, injection_length_ticks}
  4. Assert response.success == True
```

### Listening Mode
- Detects key from accumulated melody using `detect_key_lightweight()`
- Calls `prompt_library.select_prompt(key)` to find matching prompt
- Auto-injects new prompt when key changes significantly
- Can be suppressed with flag (recently added: commit `33094ba`)

---

## `app/web_client.py` — WebSocket Client

### Entry
```bash
python app/web_client.py [options]
```

### Key Differences from client_lekai.py
- Communicates with browser via WebSocket (`websockets` library)
- Drives a web UI (HTML/JS in `app/web_ui/`)
- Browser can send MIDI notes via WebSocket → client → server
- Has MIDI device detection and listing for browser UI
- More complex state machine (1,397 lines)

### WebSocket Protocol
- **Client → Browser**: JSON events (notes, status updates, inference results)
- **Browser → Client**: JSON commands (note_on, note_off, set_tempo, etc.)

### Startup Sequence (inferred)
1. Start WebSocket server on localhost port (default: 8765)
2. Open browser or wait for browser connection
3. Initialize inference loop (same tick-based approach)
4. Forward MIDI/keyboard input to server + browser simultaneously

### Lekai Server Integration
- Commit `6f9308e`: "Integrated lekai server into web_client"
- Uses `/inject_notes` (canonical endpoint)
- Supports listening mode

---

## `app/client.py` — Original Client (Reference)

Same architecture as client_lekai.py but:
- Supports `InferenceEngineStanley` note format (duration-based)
- Deprecated `inject_music_to_server()` function (line 436) — superseded by `perform_direct_injection()`
- `get_injection_status()` function defined but not called internally (used by web_client.py)

---

## Shared Components

### Input Handlers (`app/input_handlers/input_handler.py`)

**`read_midi_input(device_name, event_queue, stop_signal, ...)`**
- Thread worker
- Opens `mido.open_input(device_name)` and polls for messages
- Emits: `{type: "note_on"|"note_off", pitch, velocity, tick}`
- Stops on `stop_signal.is_set()`

**`read_keyboard_input(event_queue, stop_signal, ...)`**
- Thread worker using `pynput.keyboard.Listener`
- `KEY_TO_PITCH`: maps keyboard keys to MIDI pitches (two rows: zxcvbnm... and sdghjl;...)
- Emits same format as MIDI input

**`read_midi_file_input(file_path, event_queue, tick_clock, ...)`**
- Thread worker
- Reads `mido.MidiFile(file_path)`
- Replays events in real-time synchronized to `tick_clock`

### Output Handlers

**`CLIOutputHandler`** (`app/output_handlers/cli_output.py`)
- Persistent terminal UI using ANSI codes
- Shows: pending notes, last generated, round-trip latency, inference time
- Saves CSV/TXT on `close()`

**`AudioOutputHandler`** (`app/output_handlers/audio_output.py`)
- Opens MIDI output port via `mido.open_output()`
- `on(pitch, velocity, program)` → note_on
- `off(pitch, program)` → note_off
- Optional metronome click on beat 1

**`MidiFileHandler`** (`app/output_handlers/midi_file_handler.py`)
- Records session using `pretty_midi`
- Handles both duration notes and event streams
- Multi-instrument via `program` field
- `save(output_path)` to finalize

**`JsonLogHandler`** (`app/output_handlers/json_log_handler.py`)
- Appends each inference call as JSON entry
- `save(output_path)` exports `inferences.json`

---

## Key Utilities

**`app/key_detection.py`**
- `detect_key_lightweight(pitch_histogram)` → key string (e.g., "C_major")
  - Uses pitch class histogram, no external dependency
  - Fast: suitable for real-time use
- `detect_key_music21(notes)` → key string
  - More accurate but slower (calls music21)
  - Fallback/offline use

**`app/prompt_library.py`**
- `PromptLibrary(prompts_dir)` — scans `prompts/` directory organized by key
- `select_prompt(key)` → `(melody_path, accompaniment_path)`
- `load_prompt_notes(melody_path, accompaniment_path)` → note lists for injection

---

## Timing Constants

| Constant | Default | Description |
|---|---|---|
| `ticks_per_beat` | 4 | Ticks per quarter note |
| `tempo` | 120 BPM | Playback speed |
| `generation_interval_ticks` | 2 | How often to request inference |
| `tick_duration_ms` | 60000 / (tempo * ticks_per_beat) | Actual tick duration |

**Do not change `ticks_per_beat`** — it is coupled to the trained model's quantization.
