# PROCESS_FLOW.md — Logic Paths and State Transitions

> Documents the exact control flow as coded, not as documented elsewhere.
> Last audited: 2026-03-23

---

## Flow 1: Session Startup

```
Client Launched
      │
      ▼
Parse CLI Arguments → Build StreamMUSEConfig
      │
      ▼
Open Audio Output Port (mido.open_output)
      │
      ▼
POST /clear_history ──→ Server clears all engine state
      │
      ▼
[If --injection-file]:
  Load MIDI file via pretty_midi
  Convert to MelodyNoteEvent / AccompanimentNoteEvent lists
  POST /inject_notes ──→ Server injects into engine history
      │
      ▼
[If listening mode]:
  Initialize PromptLibrary (scan prompts/ directory by key)
  Start key detection accumulator
      │
      ▼
Spawn Input Thread (MIDI / keyboard / file)
      │
      ▼
Enter Tick Loop (main thread)
```

---

## Flow 2: Tick Loop (Core Real-Time Loop)

```
Tick N
 │
 ├──→ Check playback_schedule[N]
 │         For each note due:
 │           audio_handler.on(pitch, velocity, program)   [note_on]
 │           audio_handler.off(pitch, program)             [note_off]
 │
 ├──→ If N % generation_interval_ticks == 0:
 │         Collect events from event_queue since last inference tick
 │         Package as InferenceRequest
 │         POST /generate_accompaniment (async or sync)
 │              │
 │              ▼
 │         [Server: Flow 3]
 │              │
 │              ▼
 │         Parse AccompanimentResponse
 │         For each accompaniment note:
 │           Add to playback_schedule[note.tick]
 │         Log to CLIOutputHandler, JsonLogHandler
 │
 └──→ Advance to Tick N+1
      Sleep(tick_duration_ms - processing_time)
```

---

## Flow 3: Server Inference Request

```
POST /generate_accompaniment received
      │
      ▼
Record request_arrival_time = time.perf_counter()
      │
      ▼
Validate via Pydantic (InferenceRequest)
      │
      ▼
[Stanley Engine Path]:
  Record preprocess_start_time
  Append melody_notes to melody_history
  Trim history to model_max_seq_len_frames (sliding window)
  Convert history to piano roll tensor (shape: frames × polyphony*3)
  Record inference_start_time
  model.generate(mel_roll, acc_roll, generation_length_frames)
  Record inference_end_time
  Record postprocess_start_time
  decode_output() → duration note dicts
  Apply generation_start_tick offset
  Append to accompaniment_history
  Record response_output_time
  Return AccompanimentResponse

[Lekai Engine Path]:
  Record preprocess_start_time
  Convert melody_notes to event stream (if needed)
  Append to melody_event_history
  _get_mel_pianoroll_for_beat() → tensor per beat
  PianoRollTokenizer.encode() → token IDs
  Record inference_start_time
  PianoLLaMA.generate(tokens, past_key_values)
  Update past_key_values (stateful mode)
  Record inference_end_time
  Record postprocess_start_time
  Decode token IDs → event stream
  Handle note sustain / balance note_on/note_off counts
  Append to accompaniment_history
  Record response_output_time
  Return AccompanimentResponse
```

---

## Flow 4: Music Injection

```
[Canonical Path — client-side]:
Client loads MIDI file via pretty_midi
      │
      ▼
Convert to note event lists
      │
      ▼
POST /inject_notes
      │
      ▼
Server:
  [If Lekai engine]:
    notes_to_events(melody_notes) → event stream
    Set melody_event_history = converted events
  [If Stanley engine]:
    Set melody_history = melody_notes directly
  Set accompaniment_history = accompaniment_notes
  Set injection_length_ticks = request.injection_length_ticks
  Set is_injected = True
      │
      ▼
Return DirectInjectionResponse {success: True, ...}

[Deprecated Path — file-based]:
POST /inject_music {midi_file_path: str (server filesystem path)}
  Server reads file directly
  Same injection logic
  (Do not use for new features)
```

---

## Flow 5: Listening Mode (Adaptive Prompting)

```
Input Thread running
      │
      ▼
Every melody event → add pitch to key_detection_accumulator
      │
      ▼
Every N ticks:
  pitch_histogram = count pitches in last M ticks
  detected_key = detect_key_lightweight(pitch_histogram)
      │
      ▼
If detected_key changed significantly:
  prompt = prompt_library.select_prompt(detected_key)
  [Run Flow 4: Injection with new prompt]
  Log key change to CLI
```

---

## Flow 6: Session Shutdown

```
User presses Ctrl+C / sends stop signal
      │
      ▼
stop_signal.set() → Input thread exits
      │
      ▼
Tick loop exits
      │
      ▼
audio_handler.close() (send all note_offs)
      │
      ▼
midi_file_handler.save(output_path)  → writes .mid file
      │
      ▼
json_log_handler.save(output_path)   → writes inferences.json
      │
      ▼
cli_output_handler.close()           → writes session.csv, session.txt
      │
      ▼
Process exits
```

---

## Error Handling Paths

### Network Error (server unreachable)
```
requests.exceptions.RequestException caught
      │
      ▼
Print warning to console
Continue tick loop (skip this inference cycle)
No crash — system degrades gracefully
```

### Engine Not Loaded (server startup failure)
```
POST /generate_accompaniment
      │
      ▼
inference_engine is None
      │
      ▼
Return HTTP 503 {"detail": "Inference engine not loaded"}
Client receives non-200 → catches exception → continues
```

### Injection Failure
```
POST /inject_notes returns success=False
      │
      ▼
Client logs warning
Session continues without injection
```

### Lekai Engine: Unbalanced Note Events
```
After generation:
  Count note_on/note_off per pitch
  If note_on count != note_off count:
    Add missing note_offs (sustain fix)
    Log [ENGINE DEBUG] Event balance warning
```

---

## State Diagram: Engine History

```
INIT
  │ POST /clear_history
  ▼
EMPTY HISTORY
  │ POST /inject_notes (if injection used)
  ▼
INJECTED STATE
  │ injection_length_ticks set
  │ is_injected = True
  │ POST /generate_accompaniment (first call)
  ▼
ACTIVE GENERATION
  │ History grows each call
  │ Trimmed to model_max_seq_len_frames (sliding window)
  │ (Lekai: past_key_values grows in stateful mode)
  │ POST /clear_history
  ▼
EMPTY HISTORY (reset)
```

---

## Threading Model

```
Main Thread:                    Input Thread:             Server (FastAPI):
─────────────                  ─────────────             ─────────────────
Tick loop                      MIDI / keyboard           Async event loop
  │                            polling                   (uvicorn)
  │                              │                          │
  │  ← event_queue ──────────────┘                          │
  │                                                          │
  │  ─── POST /generate_accompaniment ────────────────────→  │
  │  ←── AccompanimentResponse ───────────────────────────   │
  │                                                          │
  │  → playback_schedule[tick]                               │
  │  → audio_handler.on/off()                                │
  │  → CLIOutputHandler.update()                             │
  │  → JsonLogHandler.append()                               │
```

Note: No explicit thread locks on shared state. Assumes:
- Only one input thread writes to event_queue
- Only main thread reads from event_queue
- Only main thread calls server
- Server is single-process (no concurrent inference calls)
