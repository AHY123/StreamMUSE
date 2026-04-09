# lessons.md — Hard-Won Lessons and Anti-Patterns

> Derived from git history, commented-out code, debug logs, and code comments.
> Format: Problem: [Observation] → Rule: [Constraint to prevent it]
> Last updated: 2026-04-02

---

## Architecture Lessons

**Problem:** Originally, music injection required a MIDI file to exist on the server filesystem. This tightly coupled server and client deployments.
**Rule:** Injection must be client-side. Client reads the file and sends note lists via `/inject_notes`. Server never reads files. Do not reintroduce file-path-based endpoints.

**Problem:** The server was initially designed only for the Stanley engine. Adding Lekai required branching the internal history management.
**Rule:** Internal history formats differ by engine type. See `server.py:289-299`. Any new server route that touches engine history must handle both formats explicitly.

**Problem:** Melody and accompaniment track interleaving order was incorrect early on (commit `8e8fde1`).
**Rule:** The interleaving order in the model's input tensor is: melody frame, accompaniment frame, melody frame, accompaniment frame... alternating. Verify this assumption against model checkpoint config before changing.

**Problem:** The original `/inject_music` endpoint is deprecated but still present. New developers may use it without realizing it's obsolete.
**Rule:** Use `/inject_notes` (client-side). Do not add new features to `/inject_music`. It exists only for backward compat.

---

## Real-Time Timing Lessons

**Problem:** Early versions did not account for network latency when scheduling accompaniment playback. Notes played late.
**Rule:** Clients must request inference slightly before the actual generation tick (latency compensation). The `assumed_network_latency_ms` field in `InferenceRequest` supports this. Do not remove timing compensation logic.

**Problem:** Tick duration must be calculated precisely. Rounding errors accumulate over long sessions.
**Rule:** `tick_duration_ms = 60000.0 / (tempo * ticks_per_beat)` — use float division. Do not truncate to integer.

**Problem:** In the Lekai clients, live keyboard/MIDI events are assigned
`event.get("tick", tick_count)` only when the tick loop drains the queue after the
initial 10% sleep. Events that arrive later in the tick are pushed into the next
iteration and become 1 tick late.
**Rule:** Do not assume the 10% processing window makes live timing "correct."
If timing correctness matters, assign ticks at enqueue time or derive them from a
shared clock; MIDI-file input with explicit ticks is the deterministic reference path.

**Problem:** In live MIDI replay through `read_midi_input`, even when the client
does capture the event stream, short notes near beat boundaries can collapse into
same-tick `note_off` / `note_on` pairs or lose 1 tick of duration. In one stress
case, the server-bound reconstruction dropped a note entirely while the recorded
user MIDI kept a dangling note alive until finalize.
**Rule:** Treat event-stream correctness and MIDI recording correctness as separate
checks. Compare both the server-bound events and the saved user track against the
same source melody, preferably after normalizing any constant startup offset.

**Problem:** The main source of live-input timing distortion was dequeue-time tick
assignment in the client loop. After moving tick stamping into the input handlers,
the virtual MIDI live replay matched the source melody exactly at the server-bound
event level once constant startup offset was normalized out.
**Rule:** For live inputs, stamp ticks at event creation time using a shared
session clock. Leave dequeue-time `event.get("tick", tick_count)` only as a
compatibility fallback, not as the primary timing mechanism.

**Problem:** A remaining recording-only mismatch can still happen even when the
server-bound event stream is correct. In the current MIDI input handler,
the stop-signal check peeks into `event_queue` by removing and re-appending the
front item, which can reorder live events if the queue is non-empty.
**Rule:** Do not use the shared event queue itself as a control channel for stop
signals in live MIDI input. Queue reordering can break note_on/note_off ordering
for recording even after tick stamping is fixed.

**Problem:** `MidiFileHandler` assumes event order is already valid. If an
out-of-order `note_off` arrives before its earlier `note_on`, the `note_off` is
ignored and the later `note_on` stays active until `finalize()` stretches it to
the session's last known end time.
**Rule:** When debugging MIDI recording, test the recorder with both correctly
ordered and intentionally inverted event streams. A correct request stream does
not prove the saved user MIDI will also be correct.

**Problem:** `performance.mid` is a reconstructed note log, not a lossless raw
event log. If quantized `note_on` and `note_off` land on the same tick, the
resulting zero-length note is dropped by the MIDI writer and may disappear from
the saved file even though the event stream itself was correct.
**Rule:** When validating user-input correctness, treat `performance.mid` as a
record of quantized playable notes. For same-tick edge cases, compare against
the request/event log as well; absence from saved MIDI does not necessarily
mean the underlying event was lost.

**Problem:** End-to-end replay comparisons can look "wrong" if startup offset is
left in place, even when both the request stream and saved MIDI are internally
consistent. In the live Lekai server replay, both logs were shifted by the same
constant offset while still matching each other exactly after normalization.
**Rule:** For live-path correctness checks, compare both raw and offset-normalized
results. Matching normalized request notes and normalized recorded MIDI is strong
evidence that `performance.mid` is consistent with the actual quantized input
stream, even if the session begins with a constant startup shift.

**Problem:** `generation_interval_ticks` is exposed as a runtime knob, but the
actual periodic trigger logic in the Lekai clients still fires once per beat
using `ticks_per_beat - 1`. That means non-default interval settings can be
silently ignored while logs and CLI output imply they are active.
**Rule:** If a timing/interval parameter is user-configurable, the trigger logic,
display text, and saved logs must all use the same source of truth. Do not
surface `generation_interval_ticks` as configurable unless request cadence
actually depends on it.

**Problem:** `inferences.json` currently stores mutable request/response objects.
Later client-side enrichment (for example adding `backup_level` to returned
accompaniment events) can therefore leak into the saved "response" and make it
look like the server returned fields that were really added on the client.
**Rule:** Persist deep-copied snapshots for request/response logs. A saved
inference log should never share object identity with live structures that will
be mutated later in the tick loop.

**Problem:** Failed inference attempts are printed to stdout but are dropped from
the saved inference history because only truthy responses are logged.
**Rule:** If a request is sent, the durable log should record either a success
entry or an explicit failure entry. Missing rows in `inferences.json` should not
be used to imply no request happened.

**Problem:** `tick_history.json` is easy to misread as a musical-tick log, but
its `num_user_notes` field counts events dequeued during that loop iteration,
not events grouped by their stamped musical ticks. This can disagree with the
request event stream while the actual musical data is still correct.
**Rule:** Separate loop-iteration diagnostics from musical-event diagnostics.
Name and document tick-history fields according to what they actually measure.

**Problem:** The recording-path bug was fixed by separating shutdown control from
live MIDI data. Using a dedicated stop event preserves FIFO ordering in the
shared event queue during live MIDI shutdown.
**Rule:** Keep transport/control signals out of the note-event queue. For live
MIDI input, shutdown should use a separate flag (`threading.Event` or equivalent)
so the queue remains data-only and recorder-visible ordering stays intact.

---

## Engine Integration Lessons

**Problem:** Lekai model integration was committed with note "not fully tested" (commit `2039dfb`).
**Rule:** Before demoing with `ENGINE_TYPE=lekai`, run a full end-to-end session test. Known edge cases: note event balance (note_on without note_off), KV cache unbounded growth in stateful mode.

**Problem:** Duration index (not raw tick duration) is stored in notes for the Stanley engine. Raw tick values would break model inference.
**Rule:** Stanley engine note `duration` field is an **index into `DURATION_TEMPLATES`**, not a tick count. Never pass raw tick durations to this engine.

**Problem:** KV cache (`past_key_values`) in Lekai engine grows indefinitely in stateful mode unless `clear_history()` is called.
**Rule:** Always call `POST /clear_history` at the start of each new client session. Web client does this on reconnect.

**Problem:** Lekai melody sustain state (`_active_melody_pitches`) is used as
carry-in for per-beat pianoroll construction. In sliding-window rebuilds, stale
carry-in can corrupt the first rebuilt beat if it is not reset or recomputed for
the rebuild start.
**Rule:** When rebuilding Lekai melody context from history, do not reuse the
previous call's `_active_melody_pitches` blindly. Reset it or recompute the active
set for the window start before constructing the first beat.

**Problem:** Before blaming `_active_melody_pitches`, it is easy to conflate
upstream event normalization with downstream rebuild corruption. The new
`app/debug/test_lekai_rebuild_state.py` checks showed that, for the tested Lekai
path, normalized melody events in `melody_event_history` were already correct
before context rebuild; the divergence appears only when stale active state seeds
the first rebuilt beat.
**Rule:** Separate server-input validation from rebuild-state validation. First
prove what lands in `melody_event_history`, then compare clean-vs-stale rebuilds
using that same normalized history.

**Problem:** The stale-state corruption is not limited to the initial beat-0
warmup case. Expanded rebuild tests showed the same failure when the sliding
window starts later (`start_beat > 0`), meaning stale carry-in can poison the
first beat of any rebuilt window, not just the first beat of the session.
**Rule:** Treat `_active_melody_pitches` as window-boundary state, not
session-global state, during sliding-window rebuilds. If a rebuild starts at
beat N, the active set must be correct for beat N specifically.

**Problem:** A minimal reset at rebuild start would fix stale carry-in but would
also erase true sustain from notes that legitimately began before the rebuild
window. The exact fix needs to preserve real cross-window sustain, not just
remove stale state.
**Rule:** For sliding-window rebuilds, recompute the active melody set at
`start_tick` from `melody_event_history` using the same event ordering semantics
as `MidiConverter.events_to_pianoroll()` (`note_off` before `note_on` at the
same tick). Do not rely on prior-call `_active_melody_pitches`, and do not
replace it with a blind reset if correct carry-in matters.

**Problem:** After debugging multiple interacting Lekai issues, the reasoning can
get fragmented across progress notes, ad hoc tests, and chat context.
**Rule:** Keep a single narrative handoff document when multiple bugs are linked
but distinct. `docs/audit/LEKAI_THREE_ISSUES_SUMMARY.md` now serves as the
canonical summary of the tick-stamping bug, the recording-order bug, and the
server-side rebuild carry-in bug.

---

## MIDI / Audio Lessons

**Problem:** MIDI recording had bugs — notes were not properly finalized at session end (commit `c9b770a`: "Try to fix MIDI recording").
**Rule:** Always call `midi_file_handler.save()` in the shutdown sequence, not during the tick loop. Active notes must be explicitly closed before saving.

**Problem:** Multiple MIDI output ports may exist. Wrong port selection causes silent failure.
**Rule:** Log the selected MIDI output port name at startup. If no sound, first check port selection before debugging inference.

**Problem:** `python-rtmidi` requires hardware MIDI drivers. On machines without MIDI hardware, initialization fails silently.
**Rule:** Check available MIDI devices at startup and print them. Fallback to virtual port (macOS: IAC Driver; Linux: ALSA virtual).

**Problem:** On macOS, `pynput` keyboard listeners may start but still fail to
capture replayed "real user" key events if the Python process is not trusted in
Accessibility settings.
**Rule:** Before relying on `read_keyboard_input` for automated live-replay tests,
verify Accessibility trust for the exact interpreter in use (for example the
`muse_client` conda environment). A zero-note run may indicate OS permission
blocking rather than a StreamMUSE timing bug.

---

## Codebase Structure Lessons

**Problem:** Three client variants exist (client.py, client_lekai.py, web_client.py) with diverging feature sets. Bug fixes made in one are not propagated to others.
**Rule:** For demo purposes, work only in `client_lekai.py` and `web_client.py`. If fixing a bug in shared logic (input_handlers, output_handlers), verify the fix works in both clients.

**Problem:** The `transformers/` directory is a vendored, modified copy of HuggingFace transformers. Installing standard `transformers` from PyPI would shadow the custom version.
**Rule:** Always install via `pip install -e ./transformers/`, not `pip install transformers`. If imports break, check which transformers is on PYTHONPATH.

**Problem:** Debug print statements (`[ENGINE DEBUG]`, `[DEBUG MIDI]`, etc.) are the only observability mechanism. Removing them silently eliminates the ability to diagnose issues.
**Rule:** Do not delete debug print statements unless replacing them with equivalent logging output. They exist because issues have occurred in those paths.

---

## Data Pipeline Lessons

**Problem:** `DURATION_TEMPLATES` and `ticks_per_beat=4` are baked into model checkpoint weights. Changing them requires full retraining.
**Rule:** These constants are frozen. Document them in any config change proposal. Do not "clean up" or "parameterize" them without understanding the retraining cost.

**Problem:** POP909 and ARIA datasets have different track naming conventions. The extractors in `exact/` are dataset-specific.
**Rule:** Do not mix `aria_skyline.py` with POP909 data or vice versa. Use the correct extractor for each dataset.
