# INFERENCE_ENGINE_SPEC.md — Inference Engine Specification

> Source files: `app/inference_engines/`
> Last audited: 2026-03-23

---

## Overview

Two inference engines are registered in `app/server.py`. They share the same external API but differ significantly in internal implementation.

| Property | Stanley | Lekai |
|---|---|---|
| Source | `transformer_engine_stanley.py` | `transformer_engine_lekai.py` (55KB) |
| Model Architecture | RoFormer | PianoLLaMA (LLaMA for music) |
| Model Format | PyTorch Lightning `.ckpt` | safetensors `.pt` |
| Note Format | Duration-based dicts | Event stream dicts |
| Context | Sliding window of frames | KV cache (incremental) |
| Status | Stable | Recently integrated, not fully tested |
| Selected via | `ENGINE_TYPE=stanley` | `ENGINE_TYPE=lekai` |

---

## `InferenceEngineStanley`

### Interface

```python
class InferenceEngineStanley:
    def __init__(self, checkpoint_path: str, model_max_seq_len_frames: int = 96,
                 generation_length_frames: int = 20, model_size: str = "0.12B")

    def generate_accompaniment(
        self,
        melody_notes: list[dict],         # [{pitch, tick, duration}, ...]
        generation_start_tick: int,
        generation_length_frames: int = None,
        prompt_length_ticks: int = None
    ) -> list[dict]                        # [{pitch, tick, duration}, ...]

    def clear_history(self) -> None
    def set_injection_offset(self, offset_ticks: int) -> None
```

### State
```python
self.melody_history: list[dict]           # Duration-based notes
self.accompaniment_history: list[dict]    # Duration-based notes
self.model_max_seq_len_frames: int        # Max context window
self.prompt_length_ticks: int             # Current injection offset
```

### Internal Pipeline
```
generate_accompaniment():
  1. Append new melody_notes to melody_history
  2. Trim history to model_max_seq_len_frames (sliding window)
  3. Convert history to piano roll tensor:
     _notes_to_rolls() → shape (frames, max_polyphony * 3)
     Each cell: (program, pitch, duration_idx)
     Padding: 255
  4. Run model forward pass:
     model.generate(mel_roll, acc_roll, generation_length_frames)
  5. Decode output tokens → duration notes
     decode_output() from m2a_transformer_inference.py
  6. Convert absolute ticks based on generation_start_tick
  7. Append new notes to accompaniment_history
  8. Return decoded notes
```

### Note Format (Duration-Based)
```python
{
  "pitch": int,           # MIDI pitch 0-127
  "tick": int,            # Absolute tick (from session start)
  "duration": int         # Index into DURATION_TEMPLATES array
                          # NOT raw tick duration — index only
}
```

### Piano Roll Tensor Format
```
Shape: (num_frames, max_polyphony * 3)
Frame = 1 tick
Each group of 3 values per polyphony slot: [program, pitch, duration_idx]
Padding value: 255 (empty slot)
max_polyphony: 4 (configured at preprocessing time)
```

### Duration Templates
Defined in `preprocess/preprocess_midi2pt_dataset.py` as `DURATION_TEMPLATES`.
These are the only valid note durations. Notes are quantized to nearest template.
**Do not change** — coupled to trained model weights.

---

## `InferenceEngineLekai`

### Interface

```python
class InferenceEngineLekai:
    def __init__(self, checkpoint_path: str, model_size: str = "0.12B",
                 inference_mode: str = "sliding_window")

    def generate_accompaniment(
        self,
        melody_notes: list[dict],         # Event stream: [{type, pitch, tick}, ...]
        generation_start_tick: int,
        generation_length_frames: int = None
    ) -> list[dict]                        # Event stream: [{type, pitch, tick}, ...]

    def clear_history(self) -> None
    def set_injection_offset(self, offset_ticks: int) -> None
    def save_to_midi(self, output_path: str) -> None

    # Internal utilities
    def events_to_notes(self, events: list[dict]) -> list[dict]   # → duration dicts
    def notes_to_events(self, notes: list[dict]) -> list[dict]    # → event stream
    def _get_mel_pianoroll_for_beat(self, beat: int) -> tensor
```

### State
```python
self.melody_event_history: list[dict]     # Event stream
self.accompaniment_history: list[dict]    # (mixed format, see below)
self._active_melody_pitches: dict         # {pitch: tick_started}
self.past_key_values: tuple               # LLaMA KV cache for incremental inference
self.inference_mode: str                  # "sliding_window" or "stateful"
```

### Internal Pipeline
```
generate_accompaniment():
  1. Append new melody events to melody_event_history
  2. Convert melody to pianoroll per beat:
     _get_mel_pianoroll_for_beat() → tensor
  3. Tokenize via PianoRollTokenizer (lekai_model/my_tokenizer.py)
  4. Run PianoLLaMA forward pass:
     - Stateful mode: uses past_key_values for KV caching
     - Sliding window mode: re-encodes context each call
  5. Decode output token IDs → event stream
     generation_utils.py sampling logic
  6. Sustain note handling (DEBUG: Check sustain at beat end)
  7. Filter invalid events (balance note_on/note_off counts)
  8. Append to accompaniment_history
  9. Return event stream
```

### Note Format (Event Stream)
```python
{
  "type": "note_on" | "note_off",
  "pitch": int,           # MIDI pitch 0-127
  "tick": int,            # Absolute tick
  "velocity": int         # Optional, defaults to 100
}
```

### Conversion Between Formats
The server's `/inject_notes` endpoint receives duration-based notes (from `pretty_midi` MIDI file parsing) and must convert to event stream for Lekai engine:
```python
# notes_to_events() in transformer_engine_lekai.py
# Input: [{pitch, tick, duration}, ...]
# Output: [{type: note_on, pitch, tick}, {type: note_off, pitch, tick+duration}, ...]
```

### KV Cache (Stateful Mode)
- `past_key_values` accumulates across inference calls
- Reduces compute for later tokens (incremental generation)
- Cleared by `clear_history()`
- **Risk**: Cache grows unbounded if `clear_history()` not called

---

## Legacy Engines (Archived)

### `transformer_engine.py` — Original Implementation
- Predecessor to Stanley engine
- Not registered in `app/server.py`
- Kept for reference only

### `transformer_engine_midi_input.py` — MIDI Input Variant
- Specialized for direct MIDI file input (not real-time)
- Not registered in `app/server.py`
- Used in isolated test scripts only

---

## Shared Behavior

Both active engines:
1. Are instantiated once at server startup
2. Maintain stateful history across requests (single-client assumption)
3. Support `clear_history()` and `set_injection_offset()`
4. Use the same external note format contract (converted at server boundary)
