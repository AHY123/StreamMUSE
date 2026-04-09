# Lekai Three Issues Summary

This document summarizes the three major Lekai-path issues that were identified and worked through during debugging:

1. client-side live input tick assignment lag
2. recording-path event reordering in live MIDI input
3. server-side sliding-window rebuild carry-in corruption

For each issue, this note records:

- what the user-visible symptom was
- the real root cause
- how it was confirmed
- why it matters
- what fix was implemented

## Issue 1: Live Input Tick Assignment Lag

### Symptom

Live input events were often landing one tick later than intended in the request stream sent to the Lekai server. In practice this produced:

- shortened notes
- notes starting one tick late or early relative to neighbors
- collapse of very short notes near tick boundaries
- mismatch between a perfectly replayed source melody and the melody reconstructed from server-bound events

This did not affect MIDI-file input in the same way because MIDI-file events already carried explicit ticks.

### Root Cause

In the live-input path, the client loop was assigning ticks too late.

Instead of stamping a live event at the moment it was created, the code was doing:

- input thread enqueues raw event
- main tick loop sleeps for the 10% processing window
- main tick loop drains the queue
- event gets `event.get("tick", tick_count)`

That means events arriving after the dequeue boundary were pushed into the next loop iteration and became one tick late.

### Why It Was Wrong

The event was being quantized against the tick at dequeue time rather than at input time. That is correct for pre-ticked events, but not for live ones.

### How It Was Confirmed

Evidence came from:

- `app/debug/analyze_lekai_client_timing.py`
- opt-in timing logs in `app/lekai_input_debug.py`
- `app/debug/run_lekai_virtual_midi_replay_test.py`

The virtual MIDI replay tests showed:

- before the fix: normalized request notes still had timing distortions
- after the fix: normalized request notes matched source exactly

### Fix Implemented

Live events are now stamped at creation time using a shared session clock:

- `session_perf_start`
- `seconds_per_tick`
- `time.perf_counter()`

This was wired through:

- `app/input_handlers/input_handler.py`
- `app/client_lekai.py`
- `app/web_client.py`

The main loop still keeps `event.get("tick", tick_count)` as a compatibility fallback, but stamped live events now dominate.

### Result

Request-side timing distortions were eliminated in the tested live MIDI replay cases.

## Issue 2: Recording-Path Event Reordering

### Symptom

After fixing live tick stamping, the server-bound request stream became correct, but the saved user MIDI could still be wrong.

The clearest failure mode was:

- request notes matched the source melody
- recorded user MIDI stretched a note to session end
- stress-case replay showed pitch 65 hanging open until finalize

So the remaining bug was no longer in the request path. It was in the recording path.

### Root Cause

`read_midi_input()` was using the shared `event_queue` as both:

- the data channel for note events
- the shutdown/control channel for the MIDI input thread

To check for a stop signal, it would:

- remove the front item from the queue
- if it was not `None`, append it back

That rotates the queue and breaks FIFO ordering. Repeated enough times, it can move a `note_off` ahead of its earlier `note_on`.

### Why It Was Wrong

`MidiFileHandler` assumes it receives a correctly ordered event stream. If a `note_off` arrives before its corresponding `note_on`:

- the `note_off` is ignored because no note is active yet
- the later `note_on` stays active
- `finalize()` closes it at the last known end time

That is exactly how a short note becomes a long stretched note in the saved MIDI.

### How It Was Confirmed

Evidence came from:

- `app/debug/test_recording_order_issue.py`
- `app/debug/run_lekai_virtual_midi_replay_test.py`

The focused tests proved:

- queue rotation could invert `note_on` / `note_off` order
- inverted order alone was sufficient to make `MidiFileHandler` stretch the note

The end-to-end virtual MIDI replay then showed:

- before the fix: request stream correct, recorded MIDI wrong
- after the fix: request and recorded MIDI both matched the source after normalization

### Fix Implemented

The MIDI input thread now uses a dedicated stop signal instead of pushing control through the shared event queue.

This was implemented in:

- `app/input_handlers/input_handler.py`
- `app/client_lekai.py`
- `app/web_client.py`

The key design change is:

- queue = data only
- stop signal = separate thread/event control

### Result

The recording-only mismatch was eliminated in the tested live MIDI replay cases.

## Issue 3: Server-Side Sliding-Window Rebuild Carry-In Corruption

### Symptom

Even when the incoming melody event stream was correct, the Lekai engine could still build an incorrect melody context for the model.

The bug affected the first beat of a sliding-window rebuild:

- a note that really started mid-beat could appear sustained from tick 0
- the onset channel could remain correct
- but the sustain channel could be falsely seeded as already active

This means the model saw a distorted melody prompt even though upstream event formation was already correct.

### Root Cause

In `InferenceEngineLekai`, `_active_melody_pitches` was stored as instance state across calls.

During sliding-window rebuild (`need_reset=True`), the engine replayed history from:

- `start_beat`
- to `current_beat - 1`

but it started that replay with the previous call's `_active_melody_pitches`, not the active set that should be true exactly at the rebuild boundary.

That stale carry-in poisoned the first beat of the rebuild window.

Important detail:

- "first rebuilt beat" means the first beat of the current rebuild window
- not necessarily beat 0 of the whole session
- early in the session, that often is beat 0
- later, when `start_beat > 0`, it is the first beat of the later window

### Why It Was Wrong

The rebuild should start from the active set implied by history at `start_tick = start_beat * ticks_per_beat`, not from session-global state left over from the previous request.

Otherwise the first rebuilt beat can inherit sustain that belongs to the end of the previous call, not the beginning of the current window.

### How It Was Confirmed

This issue was confirmed in stages.

1. Beat-local proof:

- `app/debug/analyze_lekai_engine_context.py`
- later formalized into `app/debug/test_lekai_rebuild_state.py`

These showed that for identical melody history:

- clean carry-in produced sustain like `[0, 0, 1, 1]`
- stale carry-in produced sustain like `[1, 1, 1, 1]`

2. Upstream validation:

The rebuild-state tests also confirmed, for the tested cases, that:

- `_normalize_melody_input()` was correct
- `generate_accompaniment()` stored the expected normalized melody history before rebuild
- same-tick `note_off` / `note_on` ordering remained correct

So the corruption was localized to rebuild, not to upstream history formation.

3. Shifted-window validation:

Expanded tests showed the same failure when `start_beat > 0`, proving this was not just a beat-0 startup artifact.

### Fix Implemented

The fix chosen was exact carry-in recomputation, not a blind reset.

At rebuild start, the engine now recomputes the active melody set exactly at the rebuild boundary from `melody_event_history`:

- only events with `tick < start_tick` are considered
- events are ordered by:
  - tick
  - `note_off` before `note_on` at the same tick
- replay rule:
  - `note_on` adds pitch
  - `note_off` removes pitch

That recomputed set is then used as `_active_melody_pitches` before the rebuild loop starts.

This was implemented in:

- `app/inference_engines/transformer_engine_lekai.py`

### Why Exact Recompute Was Chosen

A simpler reset-to-empty fix would remove stale state, but it would also erase true sustain from notes that legitimately began before the rebuild window.

Exact recomputation preserves:

- removal of stale prior-call carry-in
- true cross-window sustain

### Result

The focused regression suite in `app/debug/test_lekai_rebuild_state.py` now confirms:

- early-session clean/stale runs match on the first rebuilt beat
- later-window clean/stale runs match on the first rebuilt beat
- true cross-window sustain is preserved
- upstream normalization tests still pass

## Final Summary

The three issues were distinct and sequential:

1. live events were stamped too late
2. saved MIDI could still reorder events even after stamping was fixed
3. the server could still rebuild the melody context incorrectly even when the event stream was correct

In short:

- Issue 1 corrupted request timing
- Issue 2 corrupted recorded MIDI ordering
- Issue 3 corrupted model input during server-side rebuild

All three required different fixes:

- event-time stamping at input creation
- separate stop signaling from the shared event queue
- exact rebuild-boundary active-set recomputation from history

## Related Test Files

- `app/debug/analyze_lekai_client_timing.py`
- `app/debug/analyze_lekai_engine_context.py`
- `app/debug/test_recording_order_issue.py`
- `app/debug/test_lekai_rebuild_state.py`
- `app/debug/run_lekai_virtual_midi_replay_test.py`

