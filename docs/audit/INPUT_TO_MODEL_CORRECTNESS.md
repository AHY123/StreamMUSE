# Input-to-Model Correctness Analysis

> Deep-dive analysis of whether user notes arrive at the correct position in the
> model's pianoroll representation, from physical keypress to `input_ids` tensor.
> Analysis spans wall time, cycle structure, and data flow.
>
> Code references: `app/web_client.py`, `app/input_handlers/input_handler.py`,
> `app/inference_engines/transformer_engine_lekai.py`, `lekai_model/MidiConverter.py`
>
> Audited: 2026-03-24

---

## 1. Cycle Structure (Wall Time)

At 120 BPM, `ticks_per_beat=4`:
```
seconds_per_tick = (60.0 / 120) / 4 = 0.125 s = 125 ms per tick
```

Each tick is structured in this order (source: `_tick_loop`, lines 714–1134):

```
│ Tick N starts                                          │
│                                                         │
│  1. tick_count += 1                          [0ms]      │
│  2. If tick=0: fire initial inference        [~0ms]     │
│  3. time.sleep(0.1 × 125ms)                 [12.5ms]   │◄─ "early" sleep
│                                                         │
│  4. Drain event_queue ◄─── events read HERE  [~12.5ms] │
│     event_tick = event.get("tick", tick_count)         │
│     → keyboard/MIDI events get tick = N                │
│                                                         │
│  5. Drain inference_response_queue           [~12.5ms] │
│  6. Listening mode, playback scheduling      [~13ms]   │
│                                                         │
│  7. If tick % ticks_per_beat == 3:                      │
│       Fire periodic inference trigger        [~13ms]   │
│                                                         │
│  8. time.sleep(0.9 × 125ms)                 [112.5ms]  │◄─ "late" sleep
│                                                         │
│ Tick N+1 starts                                         │
```

**The event window (step 4) opens at +12.5ms, closes at the start of step 4 on the next tick.**

The "late" sleep (step 8) occupies 90% of each tick. Events that arrive during this window sit in `event_queue` and are **not read until tick N+1's step 4**.

---

## 2. The Tick Assignment Problem

### What the code does

`web_client.py:760`:
```python
event_tick = event.get("tick", tick_count)
```

Keyboard and MIDI device events have **no `"tick"` field** (see `input_handler.py:98-104`). So they always take the fallback: `tick_count` at the moment the loop drains the queue.

### The consequence

A user presses a key at wall-time **W**. The input thread immediately pushes the event to `event_queue`. The main loop reads it at step 4 of whatever tick is active **when the loop gets there**.

| Key pressed during... | Wait in queue | Assigned tick |
|---|---|---|
| Tick N's early sleep (0–12.5ms) | up to 12.5ms | **N** (correct) |
| Tick N's event window (12.5ms) | ~0ms | **N** (correct) |
| Tick N's processing (13–14ms) | next event window | **N+1** (1 tick late) |
| Tick N's late sleep (14–125ms) | up to 112ms | **N+1** (1 tick late) |

**Since the late sleep occupies 90% of each tick, approximately 90% of keypresses will be assigned one tick later than their actual occurrence.**

### Magnitude

At 120 BPM, ticks_per_beat=4:
- 1 tick late = **125 ms offset** in the pianoroll

At 120 BPM, ticks_per_beat=4, 1 beat = 4 ticks = 500 ms. A 1-tick lag places notes ~25% of a beat behind where they actually occurred.

This affects **onset placement** in the pianoroll. A note struck at beat position 0.0 (tick 0) will appear at beat position 0.25 (tick 1) in the model's representation.

### Does it affect correctness of the model's history?

Partially. The model's pianoroll is built from `melody_event_history`, which uses these tick assignments. The RELATIVE ordering of notes is preserved (they're not reordered), but all events are shifted ~1 tick later than reality. Notes at the END of a beat may spill into the NEXT beat's pianoroll.

**Verdict: Systematic 1-tick quantization lag. Not catastrophic but degrades timing accuracy.**

---

## 3. Notes Accumulation and Beat Packaging

### Accumulator lifecycle

`notes_for_next_request: list[dict]` accumulates events between inference triggers. The trigger fires at the **last tick of each beat** (ticks 3, 7, 11, ...):

```python
# web_client.py:1115-1131
if (tick_count % ticks_per_beat) == (ticks_per_beat - 1):
    generation_start_tick = tick_count + 1   # next beat start
    request_data = {
        "melody_notes": notes_for_next_request,
        ...
    }
    self.inference_request_queue.put(...)
    notes_for_next_request = []              # reset
```

### What notes are in each request?

Trace for beat 0 (ticks 0–3):

```
Tick 0:
  ├── INITIAL inference fired FIRST (gen_start=0)
  ├── notes_for_next_request = []   ← cleared
  ├── sleep 12.5ms
  └── Events read → tick assigned 0 → appended to accumulator

Tick 1:
  ├── sleep 12.5ms
  └── Events read → tick assigned 1 → appended

Tick 2:
  └── Events read → tick assigned 2 → appended

Tick 3:
  ├── sleep 12.5ms
  ├── Events read → tick assigned 3 → appended
  └── TRIGGER: notes_for_next_request (ticks 0–3) sent, gen_start=4
      notes_for_next_request = []
```

So the request fired at tick=3 contains events from ticks **0–3** (beat 0's notes), requesting generation starting at tick **4** (beat 1).

This is **internally consistent**: beat 0's melody → generate beat 1's accompaniment. One-beat latency by design.

### Edge case: notes pressed during tick 0's INITIAL trigger

The initial trigger fires at step 2 (before the 12.5ms sleep and before step 4). So:
- Any event in the queue at the exact moment tick_count becomes 0 is NOT captured in the initial request (accumulator was already cleared before step 4 even runs).
- These events are processed at step 4 of tick 0 and correctly go into the NEXT accumulator cycle.

**Verdict: Accumulation logic is correct. Beat boundary packaging is consistent.**

---

## 4. Engine Beat Computation

When the server receives the request with `generation_start_tick=4`:

```python
# transformer_engine_lekai.py:393-406
absolute_generation_start_tick = generation_start_tick + self.injection_offset_ticks
# = 4 + 0 = 4 (when no injection)

current_beat = absolute_generation_start_tick // self.ticks_per_beat
# = 4 // 4 = 1
```

Context rebuild loop (sliding window mode):
```python
context_beats = 32
start_beat = max(0, current_beat - context_beats)  # = max(0, 1-32) = 0
for b in range(0, 1):                               # beat 0 only
    beat_start_tick = 0
    beat_end_tick = 4
    mel_pr_b = _get_mel_pianoroll_for_beat(0, 4)    # ← melody for beat 0
```

The pianoroll for beat 0 is built from `melody_event_history` filtering for `0 <= tick < 4`.

This captures exactly the events sent in this request (ticks 0–3). ✓

**The model then generates acc[beat 1] after seeing mel[beat 0].** The engine's comment confirms this design:
```python
# CRITICAL: The model learns to predict acc[b] AFTER seeing mel[b-1], NOT mel[b]!
```

**Verdict: Beat computation is correct. Pianoroll correctly windows the notes sent.**

---

## 5. Pianoroll Construction Detail

### `events_to_pianoroll` logic (`MidiConverter.py:185-278`)

```python
pianoroll = np.zeros((2, 88, T), dtype=np.uint8)   # T=4 for 1 beat

# Seed sustain from active_pitches (pitches held at start of window)
for p in active_pitches:
    pianoroll[0, pidx(p), 0:T] = 1

# Filter events to window, shift to window-relative ticks
# Sort: note_off before note_on at same tick

for event in sorted_events:
    if note_off:
        pianoroll[0, idx, t:T] = 0    # end sustain from this tick
    if note_on:
        pianoroll[0, idx, t:T] = 1    # start sustain from this tick
        pianoroll[1, idx, t] = 1      # mark onset
```

This is correct for a single beat. Note at tick 0 of the beat gets onset at position 0. Note at tick 2 gets onset at position 2. ✓

### Cross-beat sustain (active_pitches mechanism)

A note pressed in beat 0 and held into beat 1 must appear as **sustain** (channel 0, no onset) at tick 0 of beat 1.

This is handled via `_active_melody_pitches: set` which tracks which pitches are held at the end of each beat and seeds the NEXT beat's pianoroll.

**But there is a serious problem with how this set is managed in sliding window mode.**

---

## 6. BUG: Active Pitch Set Carries Stale State Into Context Rebuild

### The issue

`_active_melody_pitches` is an **instance variable** (set at `__init__`, line 114). It accumulates pitch state across calls to `_get_mel_pianoroll_for_beat()`. The sliding window path (`need_reset=True`) rebuilds the full context from scratch each call, but **never resets `_active_melody_pitches` before the rebuild loop**.

### What happens in practice

**Call 1** (generation_start_tick=4, current_beat=1):
```
Context loop: for b in range(0, 1):  ← beat 0 only
  _get_mel_pianoroll_for_beat(0, 4)
    → active_pitches = {} (initially empty, correct)
    → processes events in ticks 0-3
    → updates _active_melody_pitches = {60} (e.g. C4 held at end of beat 0)
```

**Call 2** (generation_start_tick=8, current_beat=2):
```
Context loop: for b in range(0, 2):  ← beats 0 and 1

  Beat 0: _get_mel_pianoroll_for_beat(0, 4)
    → active_pitches = {60}  ← STALE from end of call 1!
    → seeds pianoroll[0, C4, 0:4] = 1 (C4 appears sustained from tick 0)
    → but maybe C4 was first pressed at tick 2!

  Beat 1: _get_mel_pianoroll_for_beat(4, 8)
    → active_pitches = {updated by beat 0 processing}
```

**Result**: In the second (and all subsequent) calls, the pianoroll for the FIRST beat of the context window is incorrectly seeded with pitch state from the PREVIOUS call's end. Any pitch that was held at the end of call N-1 will appear to have been SUSTAINED FROM THE VERY START of beat 0 in call N's context, even if that pitch was first pressed mid-beat.

### Severity

- **First 32 beats** (before the sliding window starts truncating): `start_beat` is always 0. The stale set is fed into beat 0's pianoroll, which should have an empty initial active set.
- **After beat 32**: The stale set from the previous call's endpoint is fed into the start of a beat that's 32 beats AFTER the previous call started. Notes sustaining at beat 31 are incorrectly seeded into beat 1 (if that's the new `start_beat`).

### What the correct behavior should be

For sliding window (need_reset=True): reset `_active_melody_pitches = set()` **before** the context rebuild loop. The sustain from before the window start is lost (acceptable trade-off). This ensures beat N's pianoroll starts clean.

---

## 7. Secondary Issue: Active Pitch State Diverges Between Context Rebuilds and Real Playback

Each call to `generate_accompaniment` in sliding window mode replays the ENTIRE history from `start_beat` to `current_beat`, calling `_get_mel_pianoroll_for_beat` for each beat. This means `_active_melody_pitches` is modified **32 times per call** as a side effect of context assembly — even for historical beats that don't affect the current generation.

The final state of `_active_melody_pitches` after the loop will be the pitch set at the END of `current_beat - 1`. This is correct for SUBSEQUENT calls. But the side effect means the active set is derived from a full replay, not incremental tracking. If the user adds new notes in the current beat, those are NOT in the history yet (they go into melody_event_history via `_normalize_melody_input` BEFORE the context loop), so they will be included when beat `current_beat - 1` is processed.

Wait — actually the melody notes sent in the current request ARE added to `melody_event_history` at line 388 BEFORE the context loop runs. So when the loop rebuilds the pianoroll for beat `current_beat - 1`, it will see the current request's notes (which belong to beat `current_beat - 1`). This is correct.

But the same notes are also included when the loop later processes beat `current_beat` — no, the loop only goes up to `current_beat - 1`, so that's fine.

**This part is correct.**

---

## 8. Correct Behaviors Verified

| Behavior | Status | Evidence |
|---|---|---|
| note_off processed before note_on at same tick | ✓ Correct | `MidiConverter.py:251` sort key |
| Beat 0 initial request has empty melody | ✓ Expected | Accumulator cleared before event read at tick 0 |
| Beat N's notes sent with gen_start_tick = (N+1)*ticks_per_beat | ✓ Correct | `web_client.py:1121` |
| Engine maps gen_start_tick to correct beat via floor division | ✓ Correct | `engine:406` |
| Pianoroll windows exactly one beat (T=ticks_per_beat=4) | ✓ Correct | `MidiConverter.py:209` |
| Injection offset applied symmetrically (add on input, subtract on output) | ✓ Correct | `engine:183,703` |
| Context ordering: acc BEFORE mel per beat | ✓ Correct | `engine:472` (+ comment at line 431) |
| note_on/note_off balance tracked via _active_acc_pitches for output | ✓ Correct | `engine:118,643` |
| Accumulator reset after each inference trigger | ✓ Correct | `web_client.py:1131` |
| MIDI file events use explicit tick (correct scheduling) | ✓ Correct | `input_handler.py:249` |

---

## 9. Complete Timing Trace: User Presses C4 at Exact Beat Boundary

**Setup**: 120 BPM, ticks_per_beat=4, tick_duration=125ms. User presses C4 (MIDI 60) exactly at wall time 0ms.

```
T=0ms     User presses C4.
           Input thread receives pynput key press.
           event_queue.put({"type":"note_on","pitch":60,"velocity":100,"time":0.0})

T=0ms     tick_count becomes 0.
           Initial inference trigger fires (empty melody).
           tick_loop sleeps 12.5ms.

T=12.5ms  Tick loop wakes, drains event_queue.
           event = {"type":"note_on","pitch":60,...}
           event_tick = event.get("tick", 0) = 0  (tick_count = 0)
           quantized_note = {"type":"note_on","pitch":60,"tick":0}
           notes_for_next_request = [{"type":"note_on","pitch":60,"tick":0}]

T=12.5ms  Tick loop sleeps 112.5ms.

T=125ms   tick_count becomes 1. Sleep 12.5ms.
T=137.5ms Event queue likely empty (no new presses). Sleep 112.5ms.

T=250ms   tick_count becomes 2. Same.

T=375ms   tick_count becomes 3.
           Sleep 12.5ms.
T=387.5ms User releases C4. event_queue.put({"type":"note_off","pitch":60,...})
           Queue drained: event_tick = 3.
           notes_for_next_request = [
             {"type":"note_on",  "pitch":60, "tick":0},
             {"type":"note_off", "pitch":60, "tick":3},
           ]

           INFERENCE TRIGGER fires:
             generation_start_tick = 4
             melody_notes = above list
             notes_for_next_request = []

T=387.5ms HTTP POST sent to server.
```

**Server receives:**
```json
{
  "melody_notes": [
    {"type": "note_on",  "pitch": 60, "tick": 0},
    {"type": "note_off", "pitch": 60, "tick": 3}
  ],
  "generation_start_tick": 4
}
```

**Engine processes:**
```
current_beat = 4 // 4 = 1

Context loop for beat 0 (ticks 0-3):
  events_to_pianoroll(start=0, end=4, active_pitches={})

  Window events: note_on at t=0, note_off at t=3

  pianoroll[0, 39, 0:4] = 1  (sustain for C4, pitch_idx=60-21=39)
  pianoroll[1, 39, 0]   = 1  (onset at tick 0)
  Then: pianoroll[0, 39, 3:4] = 0  (note_off cuts sustain at tick 3)

  Final beat 0 pianoroll:
    sustain: [1, 1, 1, 0]  at pitch 39 (C4)
    onset:   [1, 0, 0, 0]  at pitch 39 (C4)
```

**This is correct.** C4 pressed at tick 0, held for 3 ticks, released before tick 3 completes. The pianoroll faithfully represents this.

---

## 10. Complete Timing Trace: User Presses C4 DURING Late Sleep

**Same setup. User presses C4 at T=50ms (i.e., during tick 0's 90% sleep, which runs from ~12.5ms to ~125ms).**

```
T=0ms     tick_count = 0. Initial inference (empty). Sleep 12.5ms.
T=12.5ms  Queue drained (empty). Sleep 112.5ms.

T=50ms    User presses C4. event_queue.put({note_on, pitch=60})
           Event sits in queue. Tick loop is sleeping.

T=125ms   tick_count = 1. Sleep 12.5ms.
T=137.5ms Queue drained. Event read.
           event_tick = event.get("tick", tick_count) = 1  ← tick 1, not tick 0!
           notes_for_next_request = [{"type":"note_on","pitch":60,"tick":1}]
```

**C4 was pressed at wall time T=50ms but assigned tick=1 (T=125ms wall time).**

This is a **75ms timing error** (0.6 of a tick). The onset in the pianoroll will appear at position 1 within beat 0 instead of position 0.

---

## 11. Summary of Findings

### BUG-1 (HIGH): Active pitch set not reset before sliding window context rebuild

**Location**: `transformer_engine_lekai.py`, before the context rebuild loop (lines 459–493).

**Effect**: In every call after the first, `_active_melody_pitches` carries stale pitch state from the previous call. This seeds incorrect sustain into the FIRST beat of each context rebuild. Notes appear to sustain from earlier than they actually started.

**Fix**: Add `saved_active = set(self._active_melody_pitches)` before the loop, reset to `set()` at start_beat, then restore or recompute as appropriate. Simplest fix: `self._active_melody_pitches = set()` at the start of the context rebuild (before line 459), since events_to_pianoroll will derive the correct state through the replay.

---

### BUG-2 (MEDIUM): Systematic 1-tick quantization lag for keyboard/MIDI input

**Location**: `web_client.py:760` — `event_tick = event.get("tick", tick_count)`.

**Root cause**: The tick loop reads events at step 4 (after the 10% sleep). Any event that arrives during the 90% late sleep is assigned to the NEXT tick. Since the late sleep occupies 90% of each tick's wall time, ~90% of keypresses are assigned 1 tick late.

**Effect**: Note onsets in the pianoroll are ~1 tick (125ms at 120BPM/4tpb) later than actual. Cross-beat notes may shift to the wrong beat.

**Fix options**:
- (A) Record wall time at keypress, convert to tick in the input handler and embed as `"tick"` in the event dict. Requires input handler to have access to the tick clock.
- (B) Assign the tick at the moment the event enters the queue (requires input thread to know current tick). The `current_tick_ref` dict is already shared — input thread could read it.
- (C) Accept the lag (it's consistent and may not affect model quality significantly, given musical expressiveness).

---

### CORRECT: Beat packaging and engine beat mapping

The accumulator → request → engine beat calculation chain is correct. Notes from beat N are sent with `generation_start_tick = (N+1)*ticks_per_beat`, and the engine correctly builds pianoroll for beat N to generate acc for beat N+1.

---

### CORRECT: Pianoroll construction

`events_to_pianoroll` correctly handles:
- Window filtering (absolute tick range)
- Sustain seeding from `active_pitches`
- note_off before note_on at same tick
- Onset marking at exact window-relative position

---

### CORRECT: One-beat generative delay

The model generates acc[beat N+1] after seeing mel[beat N]. This matches the engine's stated design (lines 496–505). The client correctly sends beat N's melody at the end of beat N (trigger at tick 3, 7, ...).

---

## 12. Recommendations for Testing

To verify these findings empirically:

1. **Add tick-assignment logging** in `web_client.py:760`. Log `wall_time_of_key_press` vs `tick_count` to measure actual lag distribution.

2. **Print `_active_melody_pitches` before and after context rebuild loop** in `generate_accompaniment`. Verify it's empty before beat 0 on every call.

3. **Send a known melody** (e.g., C4 pressed exactly at beat boundaries using MIDI file input, which uses pre-scheduled ticks). Verify the pianoroll for each beat via print statements in `_get_mel_pianoroll_for_beat`. MIDI file input uses `"tick"` field explicitly, so BUG-2 does not apply — it's the correct path for deterministic testing.

4. **Inject a known melody** (via `/inject_notes`) and inspect how `melody_event_history` is structured immediately after. Verify ticks are correct and `_active_melody_pitches` initializes to empty.
