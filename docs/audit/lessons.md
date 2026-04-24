# lessons.md — Hard-Won Lessons and Anti-Patterns

> Derived from git history, commented-out code, debug logs, and code comments.
> Format: Problem: [Observation] → Rule: [Constraint to prevent it]
> Last updated: 2026-04-17

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

**Problem:** Late server responses had their events dropped by the client
scheduler because `if note["tick"] >= tick_count:` filtered them out entirely.
For `note_off` events, dropping meant the matching `note_on` stayed open in
`midi_file_handler._open_model` until `finalize()` flushed it at session end,
producing the symptom "accompaniment notes never end in `performance.mid`."
**Rule:** Past-tick `note_off` events must be executed at the current tick
(release-now), not dropped. Dropping a late `note_off` is observationally
indistinguishable from the model failing to end the note. Past-tick `note_on`
events can still be dropped (starting a note in the past is nonsensical), so
the rule is `note_off`-only.

**Problem:** `math.floor(raw_tick + 1e-9)` quantized live events to the previous
tick even when they arrived 99% of the way through it. The `1e-9` epsilon only
covered float-precision rounding, not user intent. Notes struck just before a
beat boundary were recorded on the wrong beat.
**Rule:** For live-input tick quantization, `floor` alone is too aggressive
toward the past. Expose a snap-forward fraction (`snap_forward_fraction`,
default 0.1 = last 10% of the tick snaps to the next tick) so the quantizer
reflects musical intent, not just wall-clock truncation. `0.0` reproduces pure
floor; `0.5` is nearest-tick rounding.

**Problem:** The Lekai clients' `_tick_loop` advanced time with two *relative*
sleeps per iteration — `time.sleep(SPT * 0.1)` pre-work, then variable work,
then `time.sleep(SPT * 0.9)`. Each iteration took `0.1·SPT + work + 0.9·SPT >
SPT`, so the loop drifted slower than real time. At 120 BPM the drift was worst
on inference-trigger ticks where the full request pipeline ran between the two
sleeps. Separately, `session_perf_start` was sampled during client `__init__`
(before the loop started), so `_compute_live_event_tick` was offset from
`tick_count` from the very first iteration.
**Rule:** Use absolute-clock scheduling for periodic timing loops:
`time.sleep(max(0, session_perf_start + (tick_count + 1) * SPT - perf_counter()))`.
Sample `session_perf_start` on the first line of the loop function and write it
back into the shared `current_tick_ref` so input handlers share the same origin.
Relative sleeps are a foot-gun for any loop whose work can take a variable
fraction of the tick.

**Problem:** Model-generated note events had their `tick` overwritten with the
drifted `tick_count` right before being handed to `MidiFileHandler`
(`event_for_midi["tick"] = tick_count`). User notes kept their wall-clock
stamps, so in `performance.mid` the user melody and the model accompaniment
rendered on different tick bases and pulled apart over a session. The overwrite
was a hack to work around loop drift — once the drift was fixed, it became
actively wrong.
**Rule:** Don't rewrite server-assigned ticks on the recording path. The server
sends each generated note with a correct absolute tick; the client's job is to
route it, not mutate it. If you're tempted to "fix" a tick locally to paper
over another bug, fix the other bug instead.

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

**Problem:** `clear_history()` reset `melody_event_history`,
`_active_melody_pitches`, `accompaniment_history`, `past_key_values`, and beat
pointers — but not `_active_acc_pitches`. After a session reset, the first beat
of the next session carried stale acc pitches from the previous session,
emitting phantom `note_off` events for pitches the client had no record of.
**Rule:** When adding new engine state fields, include them in `clear_history()`.
Treat `_active_acc_pitches` and `_active_melody_pitches` symmetrically — both
are window-boundary carry-over state, both must be reset together.

**Problem:** Online sampling parameters were hardcoded in `_generate_tokens`
with values that did NOT match the offline reference used to generate working
samples (`lekai_model/inference_v2.py::batch_generate_all_samples`,
`RT-accompanimentV2/inference_new.py`). The comment on the wrong values even
said "Updated to match inference.py" while the values right below it differed
(1.2 / 0.9 / 1.2 instead of 1.1 / 0.95 / 1.0). Rep penalty was also applied to
the initial context only, never growing with tokens generated in the beat.
**Rule:** Generation hyperparameters should live in one canonical place (engine
constructor or config), not hardcoded in `_generate_tokens`. When a comment
claims "matches X", verify by diffing against X. Repetition penalty must see
the full growing sequence (context + tokens generated this beat) to match
offline `_generate_one_beat` semantics.

**Problem:** The offline pianoroll decoder treats three tokens as end-of-beat
markers for an acc beat: `track_marker_acc / part1_end_marker (171)`,
`empty_marker (169)`, `bar_token_id (255)`. The online `_generate_tokens` was
stopping on only `{171, 255}`, so when the model emitted `169` to signal an
empty beat the loop ran to `max_new_tokens=100`, producing noise tokens that
corrupted following beats.
**Rule:** When porting a generation loop from offline to online, match the
end-marker set exactly. `PianoRollTokenizer.empty_marker` (169) must be a stop
condition for acc-beat generation, not only an in-stream signal.

**Problem:** Sliding-window `context_beats = 32` gave the Lekai model only
~10–15% of its trained context length (`train_cutoff_len = 2048` ≈ 200–300
beats at ~6–10 tokens/beat). Musical coherence suffered on longer sessions.
**Rule:** The sliding-window lookback must cover typical real session length
with margin. 64 beats is adequate for demo-scale sessions (~60 beats observed);
stateful mode (`INFERENCE_MODE=stateful`) remains available for arbitrarily
long sessions via accumulated `past_key_values`.

**Problem:** `lekai_model/MidiConverter.py::events_to_pianoroll` sorts same-tick
events with `note_off` before `note_on` so retriggers work correctly, then
guards the note_off with `if p in active:`. For a **fresh** note where the
pitch isn't yet active, the same-tick note_off becomes a no-op and only the
note_on fires — so brief taps (note_on and note_off at the same integer tick)
phantom-sustain forever in `_active_melody_pitches`, corrupting every
subsequent beat's carry-in. The off-before-on sort is correct for retriggers,
so flipping it is not an option — fresh blips and retriggers must be
distinguished.
**Rule:** Resolve same-tick `note_on`/`note_off` pairs at the single ingest
point (`_normalize_melody_input`) by walking events in client-send order:
track `opened_this_batch[pitch] = tick`; if a later `note_off` lands on the
same tick and pitch, bump its tick to `tick + 1` so the note renders as a
1-tick blip downstream. Do NOT attempt this at the pianoroll builder — it
needs to preserve off-before-on ordering for retriggers.

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

**Problem:** Accompaniment notes were observed "never ending" in
`performance.mid`. The chain was: model emits `note_on` → `note_off` arrives
late in a subsequent server response → client scheduler drops late events via
`if note["tick"] >= tick_count:` → note stays open in
`midi_file_handler._open_model` → `finalize()` at session end sets
`off_tick = max observed tick`. Result: long dangling notes that all terminate
at session end.
**Rule:** When a MIDI writer has a finalize-flush fallback, it will mask
upstream event-loss bugs as "extra-long notes." If many notes in
`performance.mid` end at or near the final tick, suspect dropped `note_off`
events upstream — not model failure to release. The fix is in the scheduler:
release past-tick `note_off` events at `tick_count` rather than dropping them.

**Problem:** `MidiFileHandler` was built on `pretty_midi`, which is
seconds-based: every note stored as float `start`/`end = tick * seconds_per_tick`,
then converted back to MIDI ticks via a tempo meta at save time. Any float
rounding, any mismatch between the app's `seconds_per_tick` and the written
tempo meta, or DAW re-interpretation of the tempo map could misalign the grid.
It also forced a "min-1-tick floor" hack computed in seconds
(`prev_start + seconds_per_tick`) that only covered the file writer, not the
model's view of the same events.
**Rule:** Write MIDI tick-native. `mido.MidiFile(ticks_per_beat=N)` stores
absolute ticks as integers and converts to delta-ticks at write time with no
float arithmetic. Enforce invariants (e.g., min-1-tick duration) at ingest,
not at serialization, so the recorder, the model, and the writer all see the
same integer tick stream.

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
