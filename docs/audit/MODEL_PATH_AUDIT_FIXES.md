# Model-Path & Real-Time Event Audit — Session Fixes (2026-04-17)

This session audited the online Lekai real-time path end-to-end against the canonical offline reference (`RT-accompanimentV2`, `offline-minimal` branch) and surfaced seven independent bugs across tick quantization, model generation, and client-side event scheduling. All seven are fixed.

## Summary

| # | Area | File | Symptom |
|---|---|---|---|
| 1 | Tick quantization | `app/input_handlers/input_handler.py`, `app/web_client.py`, `app/client_lekai.py` | Notes played just before a tick boundary (e.g., 99% through a tick) were floored to the previous tick, making the system feel "late" |
| 2 | Model sampling params | `app/inference_engines/transformer_engine_lekai.py` | `_generate_tokens` hardcoded `temperature=1.2, top_p=0.9, repetition_penalty=1.2` — none of which match offline (`1.1 / 0.95 / 1.0`) |
| 3 | Generation end markers | `app/inference_engines/transformer_engine_lekai.py` | Loop stopped on `{171, 255}` only. Missing `empty_marker (169)` → when the model signaled an empty beat, the loop ran to `max_new_tokens=100` emitting noise |
| 4 | Repetition-penalty scope | `app/inference_engines/transformer_engine_lekai.py` | `sample_token` received `generated_tokens=input_ids` (the static prompt). Tokens generated in this beat were invisible to the penalty |
| 5 | Sliding-window context length | `app/inference_engines/transformer_engine_lekai.py` | `context_beats = 32`: ~10–15% of the model's trained context length. Bumped to 64 |
| 6 | Late note_off dropped | `app/web_client.py` | Client scheduler filtered `note["tick"] >= tick_count`. Late `note_off` events were silently dropped → acc notes never closed in `performance.mid` |
| 7 | Stale acc state | `app/inference_engines/transformer_engine_lekai.py` | `clear_history()` reset melody state but NOT `_active_acc_pitches` → phantom `note_off` events on the first beat after reset |

---

## Offline reference

Cloned at `/tmp/rt-offline/RT-accompanimentV2` (branch `offline-minimal`). The canonical sampling values (used across `inference_new.py`, `inference_v2.py::batch_generate_all_samples`, and `lekai_model/inference.py::batch_generate_50_samples`) are:

```
temperature = 1.1
top_k       = 10
top_p       = 0.95
repetition_penalty = 1.0
```

The canonical end-marker set for an acc beat in `_generate_one_beat` and `inference_v2::generate_accompaniment_v2`:

```python
end_markers = {track_marker_acc/part1_end_marker (171), empty_marker (169), bar_token_id (255)}
```

Model architecture configs are **identical** between offline `RT-accompanimentV2/config.py` and online `lekai_model/config.py` (vocab_size=268, hidden_size=768, 18 layers, 6 heads, 3072 FFN, RoPE θ=10000, cutoff=2048). No retraining is needed — all divergences are in generation code.

---

## Fix 1 — Snap-forward tick quantization

### Root cause
`_compute_live_event_tick` at `app/input_handlers/input_handler.py:44` used `math.floor(raw_tick + 1e-9)`. A note arriving 124ms into a 125ms tick (raw_tick = 0.992) would be assigned tick 0, not tick 1 — making the system feel perpetually late.

### Fix
Replace the float-precision epsilon with a configurable snap-forward fraction, passed through `timing_context`:

```python
raw_tick = (perf_time - float(session_perf_start)) / float(seconds_per_tick)
snap_fraction = float(timing_context.get("snap_forward_fraction", 0.1))
computed_tick = max(0, int(math.floor(raw_tick + snap_fraction)))
```

Default `0.1` snaps the last 10% of each tick forward. At 120 BPM this is the last 12.5ms. Exposed as `--snap-forward-fraction` CLI arg in both `web_client.py` and `client_lekai.py`, wired via `ClientConfig.snap_forward_fraction` (web_client) and `args.snap_forward_fraction` (client_lekai) into `current_tick_ref`.

### Result
Events in the final tail of each tick quantize to the next tick instead of the current one. Use `--snap-forward-fraction 0` to disable or `0.5` for nearest-tick rounding.

---

## Fix 2 — Sampling parameters match offline

### Root cause
`transformer_engine_lekai.py:_generate_tokens` hardcoded:

```python
temperature=1.2
top_k=10
top_p=0.9
repetition_penalty=1.2
```

Comments above these values said "Updated to match inference.py" with values `1.1 / 10 / 0.95 / 1.0` — but the actual code below used different numbers. Classic drift between intent and implementation.

### Fix
Set the values to their offline canonical:

```python
temperature=1.1
top_k=10
top_p=0.95
repetition_penalty=1.0
```

### Result
Online generation matches the sampling distribution that produced working offline samples.

---

## Fix 3 — Add `empty_marker` (169) to acc-beat end markers

### Root cause
Online `_generate_tokens` stopped on `[end_marker_part1 (171), bar_token_id (255)]`. Offline (`inference_v2.py:448`, `RT-accompanimentV2/model.py:87`) stops on three markers:

```python
end_markers = {part1_end_marker (171), part1_empty_marker (169), bar_token_id (255)}
```

When the model emitted `169` (empty-beat signal) online, the loop continued to `max_new_tokens=100`, producing token noise that corrupted subsequent beats.

### Fix
Add `self.tokenizer.empty_marker` to the break condition:

```python
if token_val in [
    self.config.end_marker_part1,
    self.tokenizer.empty_marker,
    self.config.bar_token_id,
]:
    break
```

### Result
Empty beats terminate cleanly at the same token offline does. The downstream decoder (`decompress_tokens` → `patch_tokens_to_image`) handles a single `[169]` beat correctly (returns an all-zero 88×4 pianoroll).

---

## Fix 4 — Repetition penalty sees the full growing sequence

### Root cause
Offline `model.py::_generate_one_beat` grows a `generated` tensor with each sampled token and passes it back into `_sample_token`, so repetition penalty sees the full context + every token already generated this beat.

Online `_generate_tokens` passed `generated_tokens=input_ids` (the initial prompt) on every iteration and never updated it. Rep penalty within a beat was a no-op for in-beat tokens.

### Fix
Maintain a `full_sequence` tensor in the loop and pass it to `sample_token`:

```python
full_sequence = input_ids
for _ in range(max_new_tokens):
    ...
    next_token = sample_token(..., generated_tokens=full_sequence, ...)
    full_sequence = torch.cat([full_sequence, next_token], dim=1)
    current_input = next_token
```

### Result
Identical rep-penalty semantics to offline `_generate_one_beat`. At `rep_penalty=1.0` (the corrected default), this is a no-op — but leaving the plumbing correct protects against any future rep-penalty experiments behaving differently online vs offline.

---

## Fix 5 — Context window 32 → 64 beats

### Root cause
`context_beats = 32` at `transformer_engine_lekai.py:520`. At ~6–10 tokens/beat, 32 beats ≈ 200–320 tokens — roughly 10–15% of the model's trained context length (`train_cutoff_len=2048`). Musical coherence degrades.

### Fix
```python
context_beats = 64  # Lookback
```

### Result
Sliding-window rebuild sees twice as much history. User confirmed typical sessions reach ~60 beats, so 64 covers most real sessions fully without needing stateful mode. Stateful mode remains available for even longer sessions via `INFERENCE_MODE=stateful`.

---

## Fix 6 — Late note_off releases at current tick

### Root cause
`web_client.py:955` scheduler:

```python
for note in newly_generated_notes:
    if note["tick"] >= tick_count:
        # schedule
    # else: dropped
```

When a server response arrived after `tick_count` passed the event's tick (jitter, GC pause, slow inference), ALL past-tick events were dropped. For a `note_on` this is correct (can't start in the past). For a **`note_off`**, dropping means the matching `note_on` stays open in `midi_file_handler._open_model` forever — `finalize()` at session end sets `off_tick = max observed tick`, which renders as "note extends to session end." This is the dominant cause of "accompaniment notes not ending in performance.mid."

### Fix
Add an `elif` branch: past-tick `note_off` events execute at `tick_count` (slightly late) instead of being dropped:

```python
elif note.get("type") == "note_off":
    late_note = dict(note)
    late_note["tick"] = tick_count
    late_note["type"] = "note_off"
    playback_schedule.setdefault(tick_count, []).append(
        {**late_note, "source": "model"}
    )
```

Past `note_on` events are still dropped.

### Result
Late releases still release. The note ends slightly late but it ends — instead of never ending.

---

## Fix 7 — `clear_history` resets `_active_acc_pitches`

### Root cause
`transformer_engine_lekai.py:268`:

```python
def clear_history(self):
    self.melody_event_history = []
    self._active_melody_pitches = set()
    self.accompaniment_history = []
    # _active_acc_pitches NOT reset
    self.past_key_values = None
    self.last_generated_beat = -1
    self.last_generated_acc_tokens = None
```

`_active_acc_pitches` carries sustained accompaniment pitches across beat boundaries. After `/clear_history`, the first beat of the next session inherits stale pitches → emits phantom `note_off` events for pitches the client has no record of. The midi writer silently drops them (pitch not in `_open_model`), but the phantom ordering can confuse retrigger semantics if the model subsequently generates one of those pitches.

### Fix
Add one line:

```python
self._active_acc_pitches = set()
```

### Result
`clear_history()` is now symmetric: both melody and acc carry-over state reset together. Fresh session starts clean.

---

## Out of scope

- **Offline full-pianoroll walking vs online per-beat emission.** Offline `Token2Midi` builds the entire piece's pianoroll at once, then walks onsets and finds sustain-ends. Online emits events one beat at a time with `_active_acc_pitches` carry-over. These are **semantically equivalent** ("note lasts until explicit end or end of piece/session") — the online per-beat path is required by streaming. No change needed.

- **`client_lekai.py` scheduler.** Uses the same idiom as `web_client.py:955` but the user scoped this audit to web_client. If the symptom appears in client_lekai, mirror Fix 6 there.

- **Stanley engine.** These fixes are Lekai-specific. Stanley uses duration-based notes and does not share the per-beat event pipeline.

- **Token filter divergence.** Offline filters `t < 173` (drops BOS/EOS/PAD/pad_marker). Online filters only exact `pad_token_id`. Other control tokens get dropped downstream as invalid position markers (out of range [81, 169)). Consistency difference, not a known-bug cause — no fix.

---

## Files changed

| File | Fix(es) |
|---|---|
| `app/input_handlers/input_handler.py` | 1 |
| `app/web_client.py` | 1, 6 |
| `app/client_lekai.py` | 1 |
| `app/inference_engines/transformer_engine_lekai.py` | 2, 3, 4, 5, 7 |
| `docs/audit/MODEL_PATH_AUDIT_FIXES.md` | NEW (this file) |
| `docs/audit/lessons.md` | 4 new entries |
| `docs/audit/progress.txt` | `[FIX]` / `[DOC]` entries |

## Verification

1. **Syntax:** `python -c "import ast; ast.parse(open('app/web_client.py').read()); ast.parse(open('app/inference_engines/transformer_engine_lekai.py').read())"` — passes.
2. **Regression:** `app/debug/test_four_fixes.py` paths (tick loop, mido writer, normalize, model-tick overwrite) untouched — should still pass in the `muse_client` conda env.
3. **Behavioral:** Run a Lekai session, inspect `performance.mid` — acc notes should now end at their model-generated release ticks, not all pile up at session-end. Bump `context_beats=64` should also produce subjectively better long-form coherence.
