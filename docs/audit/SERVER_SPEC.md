# SERVER_SPEC.md — FastAPI Inference Server

> Source of truth: `app/server.py`
> Last audited: 2026-03-23

---

## Startup

**Command:**
```bash
CHECKPOINT_PATH=path/to/model.ckpt uvicorn app.server:app --host 0.0.0.0 --port 8000
```

**Environment Variables:**

| Variable | Default | Valid Values |
|---|---|---|
| `CHECKPOINT_PATH` | (required) | Path to `.ckpt` or `.pt` file |
| `ENGINE_TYPE` | `stanley` | `stanley`, `lekai` |
| `MODEL_MAX_SEQ_LEN_FRAMES` | `96` | Integer |
| `GENERATION_LENGTH_FRAMES` | `20` | Integer |
| `MODEL_SIZE` | `0.12B` | `small`, `0.12B`, `0.25B`, `0.5B` |
| `INFERENCE_MODE` | `sliding_window` | `sliding_window`, `stateful` (Lekai only) |

**Lifespan behavior:** Model is loaded once at server startup via FastAPI `@asynccontextmanager` lifespan handler. Engine is stored in global `inference_engine` variable.

---

## Pydantic Models

### Input

**`MelodyNoteEvent`**
```python
type: str          # "note_on" or "note_off" (for event stream format)
pitch: int         # 0-127
tick: int          # Absolute time in ticks
duration: Optional[int]  # For duration-based format (Stanley engine)
```

**`InferenceRequest`**
```python
melody_notes: list[MelodyNoteEvent]
generation_start_tick: int
client_request_send_time: float
generation_length_frames: Optional[int]     # Override server default
prompt_length_ticks: Optional[int]
inference_interval_ticks: Optional[int]
tempo: Optional[float]
assumed_network_latency_ms: Optional[float]
```

**`DirectInjectionRequest`**
```python
melody_notes: list[MelodyNoteEvent]
accompaniment_notes: list[AccompanimentNoteEvent]
injection_length_ticks: int
```

### Output

**`AccompanimentNoteEvent`**
```python
pitch: int
tick: int
duration: int
program: int    # MIDI instrument number
```

**`Timings`**
```python
request_arrival_time: float       # time.perf_counter() on arrival
response_output_time: float       # time.perf_counter() on response
preprocess_start_time: float
inference_start_time: float
inference_end_time: float
postprocess_start_time: float
```

**`AccompanimentResponse`**
```python
accompaniment: list[AccompanimentNoteEvent]
timings: Timings
generation_start_tick: int
```

**`DirectInjectionResponse`**
```python
success: bool
message: str
melody_notes_injected: int
accompaniment_notes_injected: int
injection_length_ticks: int
```

---

## API Routes

### `POST /generate_accompaniment`
**Purpose:** Primary inference endpoint — returns accompaniment for provided melody.

**Flow:**
1. Record `request_arrival_time = time.perf_counter()`
2. Call `inference_engine.generate_accompaniment(melody_notes, generation_start_tick, ...)`
3. Record timing checkpoints through preprocess → inference → postprocess
4. Return `AccompanimentResponse`

**Error:** Returns HTTP 503 with `{"detail": "..."}` if engine not loaded.

---

### `POST /inject_notes` ← **CANONICAL**
### `POST /D` ← backward-compatible alias
**Purpose:** Inject melody+accompaniment history from client (prime model context).

**Internal logic:**
- If `ENGINE_TYPE=lekai`: converts duration notes to event stream internally before injecting
- If `ENGINE_TYPE=stanley`: injects duration notes directly
- Sets `injection_length_ticks` on engine state
- Sets `is_injected = True`

**Note from code (server.py:289-299):**
```
# IMPORTANT: Internal history formats differ by engine.
# - Lekai engine expects: melody_event_history, accompaniment_history
# - Stanley engine expects: melody_history, accompaniment_history
```

---

### `GET /injection_status`
**Purpose:** Query current injection state.

**Response (dict):**
```python
{
  "is_injected": bool,
  "injection_length_ticks": int,
  "melody_notes": int,
  "accompaniment_notes": int
}
```

---

### `POST /clear_history`
**Purpose:** Reset all engine state (call at session start).

**Effect:**
- Clears `melody_history`, `accompaniment_history`
- Resets injection state
- Zeros accumulated latency counter

**Response:** `{"message": "History cleared successfully"}`

---

### `POST /inject_music` ← **DEPRECATED**
**Purpose:** File-based injection (file must exist on server filesystem).

**Status:** Still functional for backward compat. Do not extend. Replaced by `/inject_notes`.

---

## Engine Architecture

### `InferenceEngineStanley` (`transformer_engine_stanley.py`)
- **Model**: RoFormerSymbolicTransformer (PyTorch Lightning checkpoint)
- **Note Format**: Duration-based `{pitch, tick, duration}` (duration is index into DURATION_TEMPLATES)
- **History**: `melody_history: list[dict]`, `accompaniment_history: list[dict]`
- **Context Window**: `model_max_seq_len_frames` (default: 96 frames = 48 ticks)
- **Generation**: Returns `generation_length_frames` frames per call (default: 20 = 10 ticks)

### `InferenceEngineLekai` (`transformer_engine_lekai.py`)
- **Model**: PianoLLaMA (LLaMA architecture for music, loaded from safetensors)
- **Note Format**: Event stream `{type: note_on|note_off, pitch, tick, velocity}`
- **History**: `melody_event_history: list[dict]`, `accompaniment_history: list[dict]`
- **KV Caching**: Uses `past_key_values` for incremental inference
- **Generation Mode**: Beat-by-beat (`INFERENCE_MODE=sliding_window` or `stateful`)
- **Status**: Added in `demo_lekai` branch; commit notes "not fully tested"

---

## State Management

The server is **stateful**. The inference engine maintains session history between requests. Multiple clients hitting the same server will share and corrupt history. Designed for single-client use.

`POST /clear_history` must be called at the start of each new client session.
