# Four Tempo / Recording Fixes (2026-04-08)

Four independent bugs in the real-time path caused the app's actual tempo to drift from its configured BPM, user notes and model accompaniment to sit on different tick bases in `performance.mid`, and brief keypresses to silently disappear from both the recording and the model's view. All four are fixed and verified end-to-end against the fake server with a simulated MIDI device.

## Summary

| # | Bug | File | Symptom |
|---|---|---|---|
| 1 | Tick loop drifts (relative sleeps) and `session_perf_start` sampled before loop | `app/web_client.py`, `app/client_lekai.py` | Real BPM < configured BPM; user-event ticks offset from `tick_count` from the first tick |
| 2 | Model-note ticks overwritten to drifted `tick_count` on the recording path | `app/web_client.py`, `app/client_lekai.py` | Model accompaniment lands on a different tick basis than user melody in `performance.mid` |
| 3 | `MidiFileHandler` was seconds-based (pretty_midi), risking float rounding / tempo-map mismatch | `app/output_handlers/midi_file_handler.py` | Recorded tempo could diverge from app tempo; same-tick blips silently dropped |
| 4 | Same-tick `note_on`/`note_off` pair for a fresh pitch dropped in engine input | `app/inference_engines/transformer_engine_lekai.py` | Brief taps phantom-sustain forever in `_active_melody_pitches`; model never "hears" the release |

---

## Bug 1 — Tick loop drift + misaligned `session_perf_start`

### Root cause
`_tick_loop` in both `web_client.py` and `client_lekai.py` advanced time with two relative sleeps per iteration:
- `time.sleep(seconds_per_tick * 0.1)` before event processing (12.5 ms @ 120 BPM)
- variable work time (event drain, schedule, MIDI writes, inference enqueue/dequeue)
- `time.sleep(seconds_per_tick * 0.9)` at end of tick (112.5 ms)

Each iteration took `0.1·SPT + work + 0.9·SPT > SPT`, so the loop drifted slower than real time. Drift was worst on inference-trigger ticks where the full request pipeline ran between the two sleeps.

Separately, `session_perf_start` was sampled while building `current_tick_ref` during client `__init__`, *before* the tick loop started. By the time `tick_count == 0` was reached, wall time was already past `session_perf_start` by the startup delay. Since `_compute_live_event_tick` in `input_handlers/input_handler.py` uses `(perf_counter - session_perf_start) / seconds_per_tick`, input events got ticks offset from `tick_count` from the very first iteration.

### Fix
In both `_tick_loop` functions:
1. Sample `session_perf_start = time.perf_counter()` on the **first line of the loop function** and write it back into `current_tick_ref["session_perf_start"]` so the input handler shares the same origin.
2. Replace the two relative sleeps with absolute-clock targets:
   - Pre-work sleep: `time.sleep(max(0, session_perf_start + (tick_count + 0.1) * SPT - perf_counter()))`
   - End-of-tick sleep: `time.sleep(max(0, session_perf_start + (tick_count + 1) * SPT - perf_counter()))`
3. Apply the same absolute-clock sleep to the listening-mode branch.

### Result
`tick_count == floor((perf_counter() - session_perf_start) / SPT)` holds throughout the session. Real tempo matches configured tempo within OS sleep granularity.

**Measured empirically (120 BPM configured, 7 s window):**
```
loop span: 56 ticks in 7.0054 s  => 7.9939 ticks/s
effective BPM: 119.908
drift: -0.076 %
```

---

## Bug 2 — Model-note tick overwrite on the recording path

### Root cause
In both clients, every model-generated note had its tick overwritten immediately before being handed to `MidiFileHandler`:
```python
event_for_midi = dict(event)
event_for_midi["tick"] = tick_count   # <-- bug
midi_file_handler.add_model_note(event_for_midi)
```
User notes retained their wall-clock-stamped ticks from `_compute_live_event_tick`, but model notes were overwritten with the drifted logical `tick_count`. Result: in `performance.mid`, user melody and model accompaniment rendered on different tick bases and gradually pulled apart over a session.

The overwrite was a hack to work around loop drift. Once Fix 1 removed the drift, the overwrite became both unnecessary and actively wrong — the server already sends each model note with a correct absolute tick.

### Fix
Deleted the two `event_for_midi["tick"] = tick_count` assignments in `web_client.py` (note_on and note_off call sites) and the two corresponding assignments in `client_lekai.py`. Server-assigned ticks now flow through unchanged to the MIDI writer.

### Result
Model accompaniment is recorded at the tick it was generated for. User and model share the same tick basis in `performance.mid`.

**Verified empirically:** `request_offset_ticks == recorded_offset_ticks == 22` in both sanity and stress test cases — the inference-request batch (user side) and the `performance.mid` recording (mixed user + model) are derived from exactly the same tick origin.

---

## Bug 3 — `MidiFileHandler` rewritten on `mido` for tick-native output

### Root cause
The old handler used `pretty_midi`:
```python
self.midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
# ...
start = tick * seconds_per_tick   # float conversion
```
Every note was stored as float seconds. At save time, pretty_midi wrote a tempo meta and converted seconds → MIDI ticks via that tempo map. Any float rounding, any mismatch between the app's `seconds_per_tick` and the written tempo meta, any DAW re-interpretation of the tempo map risked misaligning the grid. It also forced an earlier "min-1-tick floor" hack (`prev_start + seconds_per_tick`) to be computed in seconds on both the retrigger-close and note_off-close paths — and that fix only covered the file writer, not the model.

### Fix
Full rewrite on `mido`, which writes MIDI natively in integer ticks:
- Store each incoming event as `{"type", "pitch", "tick": int, "velocity"}` in `_user_events` / `_model_events` lists. No seconds anywhere.
- `add_user_note` / `add_model_note` append to the list and handle legacy duration-format by expanding into a paired `(note_on, note_off)`.
- Enforce the min-1-tick invariant at ingest: if a `note_off` would land on the same tick as the matching `note_on`, bump it to `open_tick + 1`. Applies to retrigger-close and to terminal close.
- `finalize` flushes still-open notes at the max observed tick (or `open_tick + 1` if the last event was a note_on at max).
- `save_to_midi` builds a `mido.MidiFile(ticks_per_beat=self.ticks_per_beat)`, one `MidiTrack` per instrument (user = "Guitar", model = "Piano"), sorts events by absolute tick (note_off-before-note_on at same tick for retrigger correctness), converts to delta ticks in-place, writes a single `set_tempo` meta at the head of track 0, and calls `mid.save(...)`.

Public API (`__init__`, `add_user_note`, `add_model_note`, `finalize`, `save_to_midi`) is unchanged so no caller needed to update.

### Result
Every tick the client stamps lands at exactly that tick in the `.mid` file. Changing the tempo meta later shifts absolute seconds without re-quantizing notes. No float conversions at any point.

**Verified:**
- Unit tests cover fresh-tap blip (on@5 → off@6), sustained notes preserved, retrigger close, finalize flush, and tempo meta written correctly.
- End-to-end: `performance.mid` from the virtual-MIDI harness shows `ticks_per_beat=4`, `set_tempo=120.00 BPM`, 9 user note_on + 9 note_off events at the expected integer ticks.

---

## Bug 4 — Same-tick note_on/note_off dropped in model input path

### Root cause
`lekai_model/MidiConverter.py:251` sorts same-tick events with `note_off` before `note_on` to make retriggers work correctly:
```python
evt_sorted.sort(key=lambda e: (int(e.get("tick", 0)), 0 if e.get("type") == "note_off" else 1))
```
Then lines 266-270 guard the note_off with `if p in active:`. For a **fresh** note (not a retrigger), the pitch isn't yet in `active`, so the note_off is a **no-op**. The subsequent note_on then fires: `sustain[t:T] = 1`, `onset[t] = 1`, `active.add(p)`. Net result: the note_off is silently lost. The pitch sustains to window end and propagates into `_active_melody_pitches`, corrupting every subsequent beat's pianoroll carry-in until another release arrives.

The exact same sort-and-replay bug existed in `transformer_engine_lekai.py:_compute_active_melody_pitches_at_tick`, so the sliding-window rebuild's carry-in set was also corrupted.

The off-before-on ordering is deliberately correct for retriggers (note_off-of-old then note_on-of-new for a currently sounding pitch). Flipping the sort would break retriggers. We had to distinguish "fresh blip" from "retrigger".

### Fix
In `_normalize_melody_input` (the single ingest point for live events, before they land in `melody_event_history`), walk the batch in client-send order and detect fresh same-tick pairs:
```python
opened_this_batch: dict[int, int] = {}
for ev in abs_events:
    pitch = ev["pitch"]
    tick = int(ev["tick"])
    if ev["type"] == "note_on":
        opened_this_batch[pitch] = tick
    else:  # note_off
        if opened_this_batch.get(pitch) == tick:
            ev["tick"] = tick + 1   # bump to form a 1-tick blip
        opened_this_batch.pop(pitch, None)
```
A same-tick pair where the note_on appeared first in client order is a fresh tap → bump the note_off by one tick so the two events no longer collide in the downstream sort. A pair where note_off appears first is a retrigger → leave it alone; the existing guard handles it correctly because the pitch *is* in `active` at that moment.

Because every live event flows through `_normalize_melody_input` exactly once before being appended to `self.melody_event_history`, this single fix propagates to:
- the current request's pianoroll (built from the history)
- every future request's sliding-window rebuild (built from the same history)
- `_compute_active_melody_pitches_at_tick` (which replays the same history)

### Result
A brief tap (note_on and note_off at the same integer tick) appears in the model's pianoroll as a 1-tick blip: onset at T, sustain for exactly one tick, released before T+1. `_active_melody_pitches` stays clean. Retriggers are unaffected.

---

## Verification

All four fixes are covered by the test harness at `app/debug/test_four_fixes.py` (13 unit/source-level tests, all passing) plus the end-to-end virtual-MIDI replay at `app/debug/run_lekai_virtual_midi_replay_test.py`.

### End-to-end results (fake server + simulated MIDI device, `muse_client` env)

**Sanity case** (6 notes, min duration 2 ticks):
- `request_offset_ticks == recorded_offset_ticks == 22` (shared tick basis)
- `normalized_request_vs_source.mismatch_count: 0 / 6`
- `normalized_recorded_vs_source.mismatch_count: 0 / 6`

**Stress case** (9 notes, all 1-tick duration packed tight):
- `request_offset_ticks == recorded_offset_ticks == 22` (shared tick basis preserved under load)
- 3 notes shifted by ±1 tick in recording — all on 1-tick-duration notes near tick edges, bounded by MIDI-port + OS-sleep jitter (~12 ms @ 120 BPM). Not loop drift.

**Tempo measurement** (from the client timing log):
```
loop span: 56 ticks in 7.0054 s  => 7.9939 ticks/s
effective BPM: 119.908  (configured 120)
drift: -0.076 %
```

### Files changed

| File | Fix(es) |
|---|---|
| `app/web_client.py` | 1, 2 |
| `app/client_lekai.py` | 1, 2 |
| `app/output_handlers/midi_file_handler.py` | 3 (full rewrite) |
| `app/inference_engines/transformer_engine_lekai.py` | 4 |
| `app/debug/test_four_fixes.py` | New test harness |
| `CLAUDE.md` | Documented `muse_client` vs `muse` conda envs |

### How to re-run

```bash
conda activate muse_client

# Unit + source-level tests
python app/debug/test_four_fixes.py

# Full end-to-end virtual MIDI replay (starts its own fake server on :8010)
python app/debug/run_lekai_virtual_midi_replay_test.py --case all
```

### Out of scope (not addressed by these fixes)

- The ~22-tick startup offset between replay start and client tick 0 — this is client process startup + MIDI port open time, cosmetic, constant per session, and not a tempo or alignment bug.
- Stanley engine (`transformer_engine_stanley.py`) — Fix 4 is Lekai-specific; the Stanley engine uses a different note representation and does not share the pianoroll builder.
- `/inject_music` route — uses a separate MIDI-file loading path with its own `int(round(...))` quantization in `MidiConverter.py`; client-sent ticks never go through it.
