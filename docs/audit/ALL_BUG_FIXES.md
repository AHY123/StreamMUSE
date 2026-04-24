# StreamMUSE Bug-Fix History (Unified)

This document consolidates all bug-fix work on the `demo_lekai` / `demo_lekai_fixed` branch, organized by session. Each session fixed a distinct cluster of bugs; every entry links to the original per-session summary document for the full root-cause / fix / evidence narrative.

**Scope:** real-time path only (input → server → engine → output). No training, no offline pipeline changes, no schema changes.

## Session index

| Session | Date | Bugs | Per-session doc |
|---|---|---|---|
| 1 | (incremental, March 2026) | 3 | [`LEKAI_THREE_ISSUES_SUMMARY.md`](./LEKAI_THREE_ISSUES_SUMMARY.md) |
| 2 | 2026-04-08 | 4 | [`FOUR_TEMPO_FIXES.md`](./FOUR_TEMPO_FIXES.md) |
| 3 | 2026-04-17 | 7 | [`MODEL_PATH_AUDIT_FIXES.md`](./MODEL_PATH_AUDIT_FIXES.md) |

**Total bugs fixed: 14**, across the Lekai client, server-side engine, recording path, and model generation loop. All fixes preserve the existing model checkpoint (no retraining required).

---

# Session 1 — Lekai Three Issues

**Source:** [`LEKAI_THREE_ISSUES_SUMMARY.md`](./LEKAI_THREE_ISSUES_SUMMARY.md)

Three Lekai-specific bugs identified incrementally during debugging. All three caused timing/ordering divergence somewhere between user input, server-bound events, and saved MIDI.

## 1.1 Live input tick assignment lag

**Symptom:** Live input events landed one tick later than intended in the server-bound request stream. Shortened notes, notes off by one tick, short notes collapsing near tick boundaries. MIDI-file input unaffected (explicit ticks already present).

**Root cause:** The main tick loop stamped live events at *dequeue time* via `event.get("tick", tick_count)` after the 10% processing-window sleep. Events arriving after the dequeue boundary were pushed to the next iteration and became 1 tick late.

**Fix:** Stamp live events at *creation time* using a shared session clock (`session_perf_start`, `seconds_per_tick`, `time.perf_counter()`) in `app/input_handlers/input_handler.py`. Wired through `client_lekai.py` and `web_client.py`. The main-loop `event.get("tick", tick_count)` fallback remains for compatibility but stamped live events dominate.

**Files:** `app/input_handlers/input_handler.py`, `app/client_lekai.py`, `app/web_client.py`.

## 1.2 Recording-path event reordering

**Symptom:** Even with correct server-bound events, `performance.mid` could stretch a note to session end. Stress-case replay showed pitch 65 hanging open until `finalize()`.

**Root cause:** `read_midi_input()` used the shared `event_queue` as both data channel AND stop-signal channel. The stop-check popped and re-appended the front item, rotating the queue. Repeated enough times, this could move a `note_off` ahead of its earlier `note_on`. `MidiFileHandler` then ignored the out-of-order `note_off` (pitch not yet active) and let the later `note_on` run to `finalize()`.

**Fix:** Dedicated `threading.Event` stop signal, separate from the data queue. Queue is now data-only. Implemented in `app/input_handlers/input_handler.py`, `app/client_lekai.py`, `app/web_client.py`.

## 1.3 Server-side sliding-window rebuild carry-in

**Symptom:** Even when the melody event stream was correct, the Lekai engine could build an incorrect melody context for the model. A note that started mid-beat could appear sustained from tick 0 of a rebuilt window.

**Root cause:** `InferenceEngineLekai._active_melody_pitches` is instance state across calls. During sliding-window rebuild (`need_reset=True`), the engine replayed history from `start_beat` to `current_beat - 1` but seeded `_active_melody_pitches` with the *previous call's* state, not the active set implied by history at the rebuild boundary.

**Fix:** At rebuild start, recompute the active melody set exactly at `start_tick = start_beat * ticks_per_beat` from `melody_event_history` (events with `tick < start_tick`, ordered by tick then `note_off`-before-`note_on`). This preserves true cross-window sustain while removing stale prior-call carry-in.

**File:** `app/inference_engines/transformer_engine_lekai.py`.

---

# Session 2 — Four Tempo / Recording Fixes (2026-04-08)

**Source:** [`FOUR_TEMPO_FIXES.md`](./FOUR_TEMPO_FIXES.md)

Four independent bugs in the real-time path that together caused BPM drift, mis-aligned user vs. model tick bases in `performance.mid`, and silent loss of brief keypresses. All verified end-to-end against the fake server with a simulated MIDI device.

## 2.1 Tick loop drift + misaligned `session_perf_start`

**Symptom:** Real BPM < configured BPM; user-event ticks offset from `tick_count` from the first tick of the session.

**Root cause:**
- `_tick_loop` used two *relative* sleeps per iteration: `time.sleep(SPT * 0.1)` pre-work + variable work + `time.sleep(SPT * 0.9)`. Each iteration took `> SPT`, so the loop drifted slower than real time. Drift was worst on inference-trigger ticks where the full request pipeline ran between sleeps.
- `session_perf_start` was sampled during `__init__` (before the loop started), so by `tick_count == 0` wall time was already past it by the startup delay. `_compute_live_event_tick` used `(perf_counter - session_perf_start) / SPT` and produced an immediate offset.

**Fix:** Absolute-clock scheduling. Sample `session_perf_start = time.perf_counter()` at the first line of `_tick_loop` and write it back to `current_tick_ref["session_perf_start"]`. Replace the two relative sleeps with:
```python
pre_work_target  = session_perf_start + (tick_count + 0.1) * SPT
next_tick_target = session_perf_start + (tick_count + 1)   * SPT
time.sleep(max(0, target - perf_counter()))
```

**Result:** `tick_count == floor((perf_counter - session_perf_start) / SPT)` holds throughout the session. Measured drift at 120 BPM / 7s window: **-0.076%** (effective 119.908 BPM).

**Files:** `app/web_client.py`, `app/client_lekai.py`.

## 2.2 Model-note tick overwrite on the recording path

**Symptom:** Model accompaniment landed on a different tick basis than user melody in `performance.mid`; the two pulled apart over a session.

**Root cause:** Before handing each model note to `MidiFileHandler`, both clients did:
```python
event_for_midi["tick"] = tick_count   # overwrites server-assigned tick
```
User notes kept their wall-clock stamps from `_compute_live_event_tick`, model notes got overwritten with drifted logical `tick_count`. Two tick bases, divergent over time. The overwrite was a hack for loop drift — unnecessary and wrong once Fix 2.1 removed the drift.

**Fix:** Deleted the four `event_for_midi["tick"] = tick_count` assignments (2 in `web_client.py`, 2 in `client_lekai.py`). Server-assigned ticks flow through unchanged.

**Result:** `request_offset_ticks == recorded_offset_ticks == 22` in both sanity and stress test cases — user and model share the same tick basis in `performance.mid`.

## 2.3 `MidiFileHandler` rewritten on `mido` for tick-native output

**Symptom:** Recorded tempo could diverge from app tempo (float rounding through pretty_midi's tempo map). Same-tick note_on/note_off blips silently dropped.

**Root cause:** The old handler used `pretty_midi`, seconds-based: `start = tick * seconds_per_tick`, then at save time, `pretty_midi` wrote a tempo meta and converted seconds → ticks via that meta. Any float rounding, tempo-meta mismatch, or DAW re-interpretation could misalign the grid. Also forced a "min-1-tick floor" hack computed in seconds that only covered the writer, not the model.

**Fix:** Full rewrite on `mido`:
- Events stored as `{"type", "pitch", "tick": int, "velocity"}`. No seconds anywhere.
- Min-1-tick invariant enforced at ingest (`note_off` on same tick as `note_on` → bump to `open_tick + 1`).
- `save_to_midi` builds `mido.MidiFile(ticks_per_beat=N)`, one track per instrument (user = "Guitar", model = "Piano"), converts absolute→delta at write time.
- Public API (`__init__`, `add_user_note`, `add_model_note`, `finalize`, `save_to_midi`) unchanged — callers don't change.

**Result:** Every tick the client stamps lands at exactly that tick in the `.mid` file. Tempo-meta changes shift absolute seconds without re-quantizing notes.

**File:** `app/output_handlers/midi_file_handler.py`.

## 2.4 Same-tick `note_on`/`note_off` dropped in engine input

**Symptom:** A brief tap (note_on and note_off at the same integer tick) phantom-sustained forever in `_active_melody_pitches`. Model never "heard" the release, corrupting every subsequent beat's pianoroll carry-in until another release arrived.

**Root cause:** `lekai_model/MidiConverter.py:251` sorts same-tick events with `note_off` before `note_on` (correct for retriggers). Then `if p in active:` guards the `note_off`. For a **fresh** note where the pitch isn't yet in `active`, the `note_off` is a no-op; only the `note_on` fires. `sustain[t:T] = 1, onset[t] = 1, active.add(p)`. The release is silently lost. Same bug also in `_compute_active_melody_pitches_at_tick`. Flipping the sort would break retriggers.

**Fix:** Distinguish fresh blip from retrigger at the ingest point (`_normalize_melody_input`):
```python
opened_this_batch: dict[int, int] = {}
for ev in abs_events:
    if ev["type"] == "note_on":
        opened_this_batch[ev["pitch"]] = int(ev["tick"])
    else:  # note_off
        if opened_this_batch.get(ev["pitch"]) == int(ev["tick"]):
            ev["tick"] = int(ev["tick"]) + 1   # bump fresh-blip off
        opened_this_batch.pop(ev["pitch"], None)
```
Same-tick pair where `note_on` appeared first = fresh tap → bump `note_off` by 1. Pair where `note_off` appeared first = retrigger → leave alone (the existing guard handles it because the pitch *is* in `active` at that moment).

**Result:** A brief tap renders as a 1-tick blip in the model's pianoroll (onset at T, sustain for exactly one tick, released before T+1). Retriggers unaffected.

**File:** `app/inference_engines/transformer_engine_lekai.py`.

### Session 2 verification

- Unit/source tests: `app/debug/test_four_fixes.py` (13 tests, all passing).
- End-to-end: `app/debug/run_lekai_virtual_midi_replay_test.py`.
- Sanity case (6 notes, min 2-tick duration): 0/6 mismatches in normalized request and recorded MIDI; shared tick offset 22.
- Stress case (9 notes, 1-tick each): 0 mismatches in request; 3/9 ±1 tick shifts in recording (OS sleep + MIDI jitter, not loop drift). Shared tick offset 22 preserved.

---

# Session 3 — Model-Path Audit Fixes (2026-04-17)

**Source:** [`MODEL_PATH_AUDIT_FIXES.md`](./MODEL_PATH_AUDIT_FIXES.md)

Full audit of the Lekai online real-time path against the canonical offline reference (`RT-accompanimentV2`, `offline-minimal` branch, cloned at `/tmp/rt-offline/`). Surfaced seven independent bugs across tick quantization, model generation, and client-side event scheduling.

**Offline canonical sampling values** (used across `inference_new.py`, `inference_v2.batch_generate_all_samples`, `lekai_model/inference.batch_generate_50_samples`): `temperature=1.1, top_k=10, top_p=0.95, repetition_penalty=1.0`. End-marker set for an acc beat: `{171 (part1_end), 169 (empty), 255 (bar)}`.

Model architecture configs between offline `RT-accompanimentV2/config.py` and online `lekai_model/config.py` are **identical** — no retraining needed, only generation code changes.

## 3.1 Snap-forward tick quantization

**Symptom:** A note struck 1ms before a tick boundary was floored to the *previous* tick. System felt perpetually "late" — player intended to hit beat N+1 but system recorded beat N.

**Root cause:** `_compute_live_event_tick` at `app/input_handlers/input_handler.py:44` used `math.floor(raw_tick + 1e-9)`. The `1e-9` only covered float precision.

**Fix:** Configurable snap-forward fraction via `timing_context["snap_forward_fraction"]`:
```python
raw_tick = (perf_time - session_perf_start) / seconds_per_tick
snap_fraction = float(timing_context.get("snap_forward_fraction", 0.1))
computed_tick = max(0, int(math.floor(raw_tick + snap_fraction)))
```
Default `0.1` = last 10% of tick (12.5ms at 120 BPM) snaps forward. Exposed as `--snap-forward-fraction` CLI arg in both clients. `0.0` disables; `0.5` = nearest-tick rounding.

**Files:** `app/input_handlers/input_handler.py`, `app/web_client.py` (+ `ClientConfig`), `app/client_lekai.py`.

## 3.2 Sampling parameters match offline

**Symptom:** Online generation quality noticeably worse than offline with the same checkpoint.

**Root cause:** `_generate_tokens` hardcoded `temperature=1.2, top_k=10, top_p=0.9, repetition_penalty=1.2`. The comment above these values said "Updated to match inference.py" — but inference.py and the batch-generation scripts use `1.1 / 10 / 0.95 / 1.0`. Classic drift between intent and implementation.

**Fix:** Set the values to offline canonical. Added `--` commented alternates removed so the constants stand on their own.

**File:** `app/inference_engines/transformer_engine_lekai.py`.

## 3.3 Missing `empty_marker` (169) in end-marker set

**Symptom:** When the model signaled an empty beat with token 169, the generation loop ran to `max_new_tokens=100`, emitting noise tokens that corrupted the subsequent beat's state.

**Root cause:** Online `_generate_tokens` stopped on `{171, 255}` only. Offline `_generate_one_beat` (across `inference_v2.py`, `RT-accompanimentV2/model.py`) stops on `{171, 169, 255}`.

**Fix:** Added `self.tokenizer.empty_marker` to the break condition. Downstream `decompress_tokens` → `patch_tokens_to_image` correctly handles a `[169]`-only beat as an all-zero 88×4 pianoroll.

**File:** `app/inference_engines/transformer_engine_lekai.py`.

## 3.4 Repetition penalty sees the full growing sequence

**Symptom:** Repetition penalty had observably weaker effect online than offline.

**Root cause:** Offline `_generate_one_beat` extends a `generated` tensor with each sampled token and passes it back to `_sample_token`, so rep penalty sees context + every token already generated this beat. Online `_generate_tokens` passed `generated_tokens=input_ids` (the static initial prompt) every iteration — never updated.

**Fix:** Maintain a `full_sequence` tensor in the loop; pass it to `sample_token`; concat each sampled token into it. Matches offline semantics exactly.

**File:** `app/inference_engines/transformer_engine_lekai.py`.

## 3.5 Context window 32 → 64 beats

**Symptom:** Musical coherence degraded on sessions longer than 30-ish beats.

**Root cause:** `context_beats = 32` at `transformer_engine_lekai.py:520`. At ~6–10 tokens per beat, 32 beats ≈ 200–320 tokens — ~10–15% of the model's trained context length (`train_cutoff_len=2048`).

**Fix:** `context_beats = 64`. User confirmed typical sessions reach ~60 beats, so 64 covers full session context without spilling into stateful mode. Stateful mode remains available via `INFERENCE_MODE=stateful` for longer sessions.

**File:** `app/inference_engines/transformer_engine_lekai.py`.

## 3.6 Late `note_off` release-now instead of drop

**Symptom:** Accompaniment notes observed "never ending" in `performance.mid` — many acc notes all terminated exactly at session end.

**Root cause:** `web_client.py:955` scheduler dropped any event whose tick was `< tick_count`:
```python
for note in newly_generated_notes:
    if note["tick"] >= tick_count:
        # schedule
    # else: dropped
```
For a `note_on` this is correct (can't start in the past). For a **`note_off`**, dropping meant the matching `note_on` stayed open in `midi_file_handler._open_model` forever → `finalize()` at session end set `off_tick = max observed tick` → notes pile up at session end.

**Fix:** Split the branch. Past-tick `note_off` events reschedule at `tick_count` (slightly late but released). Past-tick `note_on` events are still dropped.
```python
elif note.get("type") == "note_off":
    late_note = dict(note)
    late_note["tick"] = tick_count
    playback_schedule.setdefault(tick_count, []).append(
        {**late_note, "source": "model"}
    )
```

**File:** `app/web_client.py`. (Same idiom exists in `client_lekai.py`; mirror later if needed.)

## 3.7 `clear_history` resets `_active_acc_pitches`

**Symptom:** After `/clear_history`, the first beat of the next session emitted phantom `note_off` events for pitches the client had no record of.

**Root cause:** `clear_history()` reset `melody_event_history`, `_active_melody_pitches`, `accompaniment_history`, `past_key_values`, and beat pointers — but not `_active_acc_pitches`. Stale acc carry-over persisted across sessions.

**Fix:** One line added: `self._active_acc_pitches = set()`. Symmetric with the existing melody reset.

**File:** `app/inference_engines/transformer_engine_lekai.py`.

### Session 3 verification

- Syntax: `ast.parse` on all 4 modified Python files passes.
- Regression: Session 2's `app/debug/test_four_fixes.py` covers untouched paths (tick loop, mido writer, normalize, model-tick overwrite) — should still pass in `muse_client` env.
- Behavioral: Open `performance.mid` from a Lekai session — acc notes should now end at their generated release ticks, not all pile up at session end.

---

# Cross-session consistency rules (enforced going forward)

Distilled from the 14 fixes above — see `lessons.md` for the full rule list.

1. **Absolute-clock scheduling** for any periodic timing loop. Relative sleeps drift when work is variable.
2. **Stamp live events at creation time**, not dequeue time. Shared session clock via `timing_context`.
3. **Tick-native MIDI I/O** (mido). No float-seconds round-trips.
4. **Same origin for all tick sources.** User input, model events, recording writer, inference engine — all must agree on `session_perf_start` + `seconds_per_tick`.
5. **Resolve same-tick `note_on`/`note_off` collisions at ingest**, not at the pianoroll builder. Walk events in client-send order; distinguish fresh-blip from retrigger.
6. **Past-tick `note_off` events release-now.** Never drop — it will read as "note never ends."
7. **Every stateful field must be in `clear_history()`.** Carry-over state is window-boundary state; both melody and acc sides must reset symmetrically.
8. **Generation hyperparameters live in one canonical place.** Don't hardcode in `_generate_tokens`; cross-check comments against the code they claim to match.
9. **End-marker sets must match training / offline reference.** For Lekai acc generation: `{171, 169, 255}`.
10. **Repetition penalty must see the growing sequence** (context + tokens generated this beat). Passing a static snapshot is a silent no-op within the beat.
11. **Sliding-window lookback must cover typical session length with margin.** 64 beats is adequate for demo (~60-beat sessions); stateful mode for longer.
12. **`performance.mid` is a reconstructed, quantized log.** Compare it against the request stream — absence from saved MIDI ≠ the event was lost.
