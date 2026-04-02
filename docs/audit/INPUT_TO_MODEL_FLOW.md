# Input-to-Model Data Flow — StreamMUSE (web_client + Lekai Engine)

> This document traces the exact data path from a physical keypress or MIDI note
> to the token tensor fed into PianoLLaMA. Each layer shows the exact variable names,
> data types, field names, and transformation logic as found in the source code.
>
> **Primary files**: `app/web_client.py`, `app/input_handlers/input_handler.py`,
> `app/server_lekai.py` (or `app/server.py`), `app/inference_engines/transformer_engine_lekai.py`,
> `lekai_model/my_tokenizer.py`, `lekai_model/MidiConverter.py`
>
> Last audited: 2026-03-23

---

## Overview

```
Physical Input
    │
    ▼ Layer 0 — input_handler.py
Event Dict  {"type", "pitch", "velocity", "time"}
    │
    ▼ Layer 1 — web_client.py tick loop
Quantized Event  {"type", "pitch", "tick"}
    │ (accumulated per beat)
    ▼ Layer 2 — web_client.py inference trigger
Request Dict  {"melody_notes": [...], "generation_start_tick": int, ...}
    │
    ▼ Layer 3 — HTTP POST (requests library)
JSON over HTTP
    │
    ▼ Layer 4 — server_lekai.py FastAPI endpoint
Pydantic InferenceRequest → melody_notes_dicts
    │
    ▼ Layer 5 — transformer_engine_lekai.py: _normalize_melody_input()
melody_event_history (tick-adjusted)
    │
    ▼ Layer 6 — MidiConverter.events_to_pianoroll()
np.ndarray  shape (2, 88, 4)  dtype uint8  values {0, 1}
    │
    ▼ Layer 7 — PianoRollTokenizer
Compressed token sequence  shape (K,)  values 0–267
    │
    ▼ Layer 8 — Context assembly
torch.Tensor  shape (1, T)  dtype long  [32-beat history + current]
    │
    ▼ Layer 9 — PianoLLaMA.forward()
Logits → next token → repeat → generated accompaniment tokens
```

---

## Layer 0 — Physical Input Capture

**File**: `app/input_handlers/input_handler.py`

### Keyboard Path (`read_keyboard_input`)

```python
# _on_press() — line 94
char_key = key.char                        # str, e.g. 'z'
pitch = KEY_TO_PITCH[char_key]             # int, e.g. 60 (MIDI note number)
VELOCITY = 100                             # hardcoded constant

event = {
    "type": "note_on",                     # str
    "pitch": pitch,                        # int (0–127)
    "velocity": VELOCITY,                  # int = 100 always
    "time": time.time(),                   # float (wall clock seconds)
}
event_queue.put(event)

# _on_release() — line 107
event = {
    "type": "note_off",
    "pitch": pitch,
    "velocity": 0,
    "time": time.time(),
}
event_queue.put(event)
```

**Key mapping** (line 84-90, partial):
`'z'→60, 'x'→62, 'c'→64, 'v'→65, 'b'→67, 'n'→69, 'm'→71` (C4 major scale)

### MIDI Device Path (`read_midi_input`)

```python
# line 47-62
msg = port.poll()                          # mido.Message

if msg.type == 'note_on' and msg.velocity > 0:
    event = {
        "type": "note_on",
        "pitch": msg.note,                 # int (0–127), real MIDI pitch
        "velocity": msg.velocity,          # int (1–127), real velocity
        "time": time.time(),
    }
    event_queue.put(event)

if msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
    event = {
        "type": "note_off",
        "pitch": msg.note,
        "velocity": 0,
        "time": time.time(),
    }
    event_queue.put(event)
```

**Note**: No `"tick"` field at this layer. Tick is assigned in Layer 1.

---

## Layer 1 — Tick Loop Quantization

**File**: `app/web_client.py`, tick loop approx. lines 714–807

```python
# Every iteration of the main tick loop:
tick_count += 1                                    # int, monotonically increasing
current_tick_ref["current_tick"] = tick_count      # shared with input threads

# Drain event_queue:
while not self.event_queue.empty():
    event = self.event_queue.get_nowait()

    # Tick assignment (line 760):
    event_tick = event.get("tick", tick_count)     # int
    # For keyboard/MIDI: no "tick" field → uses current tick_count
    # For MIDI-file input: "tick" is pre-scheduled → uses that value

    if event["type"] == "note_on":
        quantized_note = {
            "type": "note_on",                     # str
            "pitch": event["pitch"],               # int
            "tick": event_tick,                    # int
            # velocity DROPPED — not included
        }
        notes_for_next_request.append(quantized_note)

    elif event["type"] == "note_off":
        quantized_note = {
            "type": "note_off",                    # str
            "pitch": event["pitch"],               # int
            "tick": event_tick,                    # int
        }
        notes_for_next_request.append(quantized_note)
```

**What's dropped**: `velocity`, `time`
**What's added**: `tick` (from tick_count)
**Accumulator**: `notes_for_next_request: list[dict]` — collects all events until inference trigger

---

## Layer 2 — Inference Trigger

**File**: `app/web_client.py`, approx. lines 1110–1132

```python
# Fires at end of each beat:
if (
    not suppress_inference
    and not is_tick_zero
    and (tick_count % self.config.ticks_per_beat) == (self.config.ticks_per_beat - 1)
):
    # With ticks_per_beat=4, this fires at tick 3, 7, 11, 15, ...

    generation_start_tick = tick_count + 1         # int — the NEXT beat's start

    request_data = {
        "melody_notes": notes_for_next_request,    # list[dict] — ALL accumulated events
        "generation_start_tick": generation_start_tick,    # int
        "generation_length_frames": self.config.generation_length_per_request,  # int
    }

    self.inference_request_queue.put((request_data, request_data.copy()))
    notes_for_next_request = []                    # reset accumulator
```

**Also fires at tick_count == 0** with empty notes (primes the model at session start).

**Accumulation behavior**: `notes_for_next_request` collects events across the ENTIRE interval since the last trigger. With `ticks_per_beat=4`, this is 4 ticks = 1 beat worth of events.

---

## Layer 3 — Network Transmission

**File**: `app/web_client.py`, `_inference_worker` thread, approx. lines 625–668

```python
# Picks up from inference_request_queue:
queue_item = self.inference_request_queue.get(timeout=0.1)
request_data, full_request_dict = queue_item

# Add send timestamp:
client_send_time = time.perf_counter()             # float (high-res seconds)
request_data["client_request_send_time"] = client_send_time

# HTTP POST:
response = requests.post(
    self.config.server_url,                        # e.g. "http://localhost:8988/generate_accompaniment"
    json=request_data,
    timeout=5.0,
)
response_json = response.json()
```

**Final JSON payload on the wire:**
```json
{
    "melody_notes": [
        {"type": "note_on",  "pitch": 60, "tick": 3},
        {"type": "note_off", "pitch": 60, "tick": 5},
        {"type": "note_on",  "pitch": 62, "tick": 6}
    ],
    "generation_start_tick": 8,
    "generation_length_frames": 5,
    "client_request_send_time": 12345.678
}
```

---

## Layer 4 — Server Reception

**File**: `app/server_lekai.py` (or `app/server.py`)

```python
# Pydantic models (server.py:19-34):
class MelodyNoteEvent(BaseModel):
    type: str        # "note_on" or "note_off"
    pitch: int
    tick: int
    # velocity is NOT in the model — stripped earlier, never sent

class InferenceRequest(BaseModel):
    melody_notes: list[MelodyNoteEvent]
    generation_start_tick: int
    client_request_send_time: float
    generation_length_frames: Optional[int] = None

# Endpoint handler:
request_arrival_time = time.perf_counter()

# Convert Pydantic models to plain dicts:
melody_notes_dicts = [note.dict() for note in request.melody_notes]
# Result: [{"type": "note_on", "pitch": 60, "tick": 3}, ...]

# Call engine:
result = inference_engine.generate_accompaniment(
    melody_notes_dicts,
    generation_start_tick=request.generation_start_tick,
)
```

---

## Layer 5 — Engine Normalization

**File**: `app/inference_engines/transformer_engine_lekai.py`

```python
# _normalize_melody_input() — line 161
def _normalize_melody_input(self, melody_notes: list[dict]) -> list[dict]:
    abs_events = []
    for e in melody_notes:
        if e.get("type") not in ("note_on", "note_off"):
            continue                                    # filter invalid
        if "pitch" not in e or "tick" not in e:
            continue                                    # filter incomplete
        abs_e = {
            "type": e["type"],                          # str
            "pitch": int(e["pitch"]),                   # int
            "tick": int(e["tick"]) + self.injection_offset_ticks,  # int, offset-adjusted
        }
        abs_events.append(abs_e)
    return abs_events

# generate_accompaniment() — line 387-388:
abs_events = self._normalize_melody_input(melody_notes)
self.melody_event_history.extend(abs_events)           # list[dict], grows each call
```

**`injection_offset_ticks`**: 0 unless `set_injection_offset()` was called after injection. When injection is used, injected history occupies ticks 0..N, so live notes are offset to avoid collision.

---

## Layer 6 — Pianoroll Construction

**File**: `app/inference_engines/transformer_engine_lekai.py` + `lekai_model/MidiConverter.py`

```python
# _get_mel_pianoroll_for_beat() — line 120
def _get_mel_pianoroll_for_beat(self, beat_start_tick: int, beat_end_tick: int):
    pr = self.midi_converter.events_to_pianoroll(
        events=self.melody_event_history,
        start_tick=beat_start_tick,
        end_tick=beat_end_tick,
        active_pitches=self._active_melody_pitches,    # set[int] — sustaining pitches
    )
    # Update active pitch set from this beat's events (for next beat)
    ...
    return pr   # np.ndarray, shape (2, 88, 4), dtype uint8

# MidiConverter.events_to_pianoroll() — MidiConverter.py line 185
# T = end_tick - start_tick  (= 4 ticks per beat)
pianoroll = np.zeros((2, 88, T), dtype=np.uint8)
# Channel 0: sustain  — 1 where pitch is held at this tick
# Channel 1: onset    — 1 where pitch begins at this tick

# Pitch axis: index = MIDI_pitch - 21  (range 21–108 → index 0–87)
# Time axis: window-relative ticks 0..3
```

**Example** — C4 (MIDI 60) pressed at tick 1, held through tick 3:
```
pitch_index = 60 - 21 = 39

pianoroll[0, 39, 1:4] = 1   # sustain from tick 1 to 3
pianoroll[1, 39, 1]   = 1   # onset at tick 1
```

**Sustain algorithm**:
1. Pitches in `active_pitches` (held from previous beat) seed the entire window with sustain
2. `note_off` events terminate sustain at their tick
3. `note_on` events start sustain at their tick (and mark onset)
4. Events at same tick: `note_off` processed before `note_on`

---

## Layer 7 — Tokenization

**File**: `lekai_model/my_tokenizer.py`

### Step 7a: `image_to_patch_tokens(pianoroll)`

```python
# Input: np.ndarray (2, 88, 4)
# Channel 0 = sustain, Channel 1 = onset

sustain_channel = pianoroll[0]   # (88, 4)
onset_channel   = pianoroll[1]   # (88, 4)

# Ternary encoding: combine into single value per cell
combined = sustain_channel + onset_channel
# Values: 0=silent, 1=sustain_only, 2=onset+sustain

# Encode each pitch row (4 time ticks) as base-3 number:
# token = combined[t=3]*3^0 + combined[t=2]*3^1 + combined[t=1]*3^2 + combined[t=0]*3^3
# Range: 0 to 80  (3^4 - 1 = 80)

tokens = np.dot(combined, [27, 9, 3, 1])   # shape (88,)
# Output shape: (1, 88)
```

### Step 7b: `compress_tokens(tokens, end_marker_id=170)`

Sparse encoding — only transmit non-zero pitch positions:

```python
# For each active pitch:
#   position_marker = (relative_position_from_prev) + 81
#   value = token value (1–80)
# End: end_marker_id (170 for melody, 171 for accompaniment)

# Example output for 2 active pitches at positions 39 and 41:
# [81+39, token_at_39, 81+2, token_at_41, 170]
#  = [120,  tok39,      83,   tok41,       170]

# If no active pitches: [169]  (empty marker)
```

**Token vocabulary:**
```
0–80:     pitch content tokens (ternary-encoded beat values)
81–168:   relative position markers (offset + 81)
169:      empty beat marker
170:      melody end marker
171:      accompaniment end marker
173:      pad marker
255:      bar marker
257:      BOS (beginning of sequence)
258:      PAD
259–263:  time signature tokens
264–266:  BPM tokens
```

---

## Layer 8 — Context Assembly

**File**: `app/inference_engines/transformer_engine_lekai.py`, `generate_accompaniment()` lines 437–520

```python
context_beats = 32                        # lookback window
start_beat = max(0, current_beat - 32)

seq = [
    torch.tensor([257]),                  # BOS
    torch.tensor([259]),                  # time signature token (4/4 = 259)
    torch.tensor([264]),                  # BPM token (120 BPM = 264)
    torch.tensor([173]),                  # pad marker
]

for b in range(start_beat, current_beat):
    if b % 4 == 0:
        seq.append(torch.tensor([255]))   # bar marker (every 4 beats)
        seq.append(torch.tensor([255]))   # bar marker (acc + mel)

    # Accompaniment tokens for beat b (from history)
    acc_tokens = self._get_tokens_for_beat(acc_notes_b, b, end_marker_id=171)
    seq.append(acc_tokens)               # shape (K_acc,)

    # Melody tokens for beat b (from pianoroll)
    mel_pr = self._get_mel_pianoroll_for_beat(beat_start, beat_end)
    mel_tokens = self._get_tokens_for_beat_pianoroll(mel_pr, end_marker_id=170)
    seq.append(mel_tokens)               # shape (K_mel,)

input_ids = torch.cat(seq).unsqueeze(0).to(device)
# Shape: (1, T)  dtype: torch.long
# T ≈ 4 (header) + 32 beats × ~20 tokens/beat ≈ 648 tokens
# Vocab: 0–267 (268 tokens total)
```

**Interleaving order per beat**: `[acc_tokens] [mel_tokens]`
The model sees accompaniment THEN melody for each beat in context.
At generation time, the model generates the NEXT accompaniment tokens given this context.

---

## Layer 9 — Model Forward Pass

**File**: `app/inference_engines/transformer_engine_lekai.py`, `_generate_tokens()` lines 268–327

```python
# Autoregressive generation:
generated_tokens = []

for _ in range(100):                      # max_new_tokens = 100
    outputs = self.model(
        input_ids=current_input,          # shape (1, 1) after first step
        past_key_values=current_past,     # KV cache for efficiency
        use_cache=True,
    )
    # outputs.logits: shape (1, 1, 268)
    next_token_logits = outputs.logits[:, -1, :]   # (1, 268)

    next_token = sample_token(
        next_token_logits,
        temperature=1.2,
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    # shape: (1, 1)  dtype: torch.long  value: 0–267

    generated_tokens.append(next_token.item())

    if next_token.item() in [171, 255]:   # acc end marker or bar token
        break

# Returns: list[int], e.g. [120, 41, 83, 35, 171]
```

**Model**: `PianoLLaMA` — LLaMA architecture adapted for music generation
**Hidden size**: 768, **Heads**: 6, **Max context**: 3500 tokens

---

## Correctness Checkpoints

These are the areas most likely to contain subtle bugs. Verify each when testing:

### CP-1: Tick Assignment Lag
**Location**: `web_client.py:760` — `event_tick = event.get("tick", tick_count)`

For keyboard/MIDI input, events have no `"tick"` field. The tick is assigned as `tick_count` at the moment the tick loop processes the event — NOT when the key was pressed. If the event sits in `event_queue` while the tick loop is sleeping, the assigned tick will be the current tick (potentially later than actual press time).

**Risk**: Notes may be timestamped 0–1 ticks late depending on event queue lag.
**Verify**: Log `time.time()` at key press vs. assigned `tick_count` at processing time.

---

### CP-2: Accumulation Window Boundary
**Location**: `web_client.py:1122` — trigger fires at `tick % ticks_per_beat == ticks_per_beat - 1`

`notes_for_next_request` accumulates events from tick `(N*ticks_per_beat)` through `((N+1)*ticks_per_beat - 1)` — exactly one beat. After trigger, accumulator resets. Notes are sent with the request for the NEXT beat.

**Verify**: If user plays a note at tick 7 (last tick of beat 1), it's sent in the request at tick 7, which asks the model to generate starting at tick 8 (beat 2). Does the model correctly use that note as context?

---

### CP-3: Injection Offset Initialization
**Location**: `transformer_engine_lekai.py` — `self.injection_offset_ticks`

When no injection is used: must be 0. When injection is used: must equal the length of the injected content in ticks.
**Verify**: `clear_history()` resets this to 0. `set_injection_offset()` sets it correctly after injection.

---

### CP-4: Active Pitch Set Correctness
**Location**: `transformer_engine_lekai.py` — `self._active_melody_pitches: set`

This set tracks which MIDI pitches are currently sustaining at the end of each processed beat. It seeds the sustain channel for the next beat's pianoroll.
**Risk**: If a `note_off` is missed (e.g., fast release before quantization), pitch stays in set → phantom sustain.
**Verify**: Print `_active_melody_pitches` after each beat. It should always correctly reflect held keys.

---

### CP-5: generation_start_tick vs. Melody Note Ticks
**Location**: `web_client.py:1122-1124`

`generation_start_tick = tick_count + 1`. The melody notes in the same request have ticks up to `tick_count`. So the model is told: "generate accompaniment starting at tick X+1, given melody notes from the current beat (ticks ≤ X)."
**Verify**: Are the melody note ticks relative to session start or relative to the current request? They are ABSOLUTE (from session tick 0). The engine uses absolute ticks throughout.

---

### CP-6: Velocity Missing from Model Input
**Location**: `web_client.py:764-771` (quantization strips velocity)

Velocity is captured by both keyboard handler (hardcoded 100) and MIDI handler (real value), but is stripped at the quantization step. The pianoroll representation has no velocity channel.
**By design**: The model was trained on velocity-free pianoroll representations.
**Caveat**: MIDI performances with expressive dynamics will lose that information.

---

### CP-7: Beat Interleaving Order
**Location**: `transformer_engine_lekai.py:474-493`

For each beat in context: accompaniment tokens are appended BEFORE melody tokens. This must match the model's training data format exactly.
**Risk**: If training used `[melody, acc]` order but inference uses `[acc, melody]`, the model's attention patterns are wrong.
**Verify**: Cross-check against the training data pipeline in `lekai_model/PianoDataset.py`.
