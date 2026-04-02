# PRD.md — Functional Baseline

> This document defines every currently working feature as a hard requirement.
> It establishes the exact current scope. Do not "fix" anything listed here without
> understanding that it is intentionally working as described.
> Last audited: 2026-03-23 (branch: demo_lekai)

---

## Core System: Real-Time Music Accompaniment

### FR-01: Real-Time Accompaniment Generation
**Status**: WORKING
The system generates musical accompaniment in real-time in response to user-played melody input. Generation occurs every N ticks (default: 2 ticks, configurable). Output is scheduled for playback at the correct musical time.

### FR-02: MIDI Device Input
**Status**: WORKING
Users can connect a hardware MIDI device. Notes are read via `mido.open_input()` and fed into the event queue. Available devices are listed at startup.

### FR-03: Computer Keyboard Input
**Status**: WORKING
Users can play melody using the computer keyboard (`--use-keyboard-input` flag). Key mapping is defined in `app/input_handlers/input_handler.py` (`KEY_TO_PITCH` dict). `pynput` handles key events.

### FR-04: MIDI File Simulation Input
**Status**: WORKING
A pre-recorded MIDI file can simulate real-time input (`--midi-file-input` flag). Events are replayed synchronized to the tick clock with optional delay offset.

### FR-05: Real-Time Audio Output
**Status**: WORKING
Generated accompaniment is played back via MIDI output port using `mido.open_output()`. Notes are scheduled on the playback timeline and fired at correct tick times.

### FR-06: MIDI File Recording
**Status**: WORKING (recently fixed)
Sessions can be recorded to a MIDI file (`--midi-file-output` flag). Both melody and accompaniment tracks are saved via `pretty_midi`. Bug fix committed 2026-03-23.

### FR-07: JSON Inference Logging
**Status**: WORKING
Every inference request/response pair is logged to `inferences.json` at session end. Includes full timing data.

### FR-08: CSV/TXT Session Summary
**Status**: WORKING
Round-trip latency, inference time, and timing statistics saved to CSV/TXT at session end by `CLIOutputHandler`.

### FR-09: CLI Real-Time Display
**Status**: WORKING
Terminal UI shows persistent status: pending notes, last generated notes, round-trip latency, inference time, server time. Uses ANSI escape codes for in-place updates.

### FR-10: Music Injection (Prompt Prefill)
**Status**: WORKING
Pre-recorded MIDI files can be injected into the model's history to prime generation with a musical style. Client-side injection via `/inject_notes` endpoint (canonical). File-based `/inject_music` endpoint is deprecated but functional.

### FR-11: Dual Inference Engine Support
**Status**: WORKING
Server supports two inference engines selectable at startup:
- `ENGINE_TYPE=stanley` → `InferenceEngineStanley` (RoFormer-based, duration notes)
- `ENGINE_TYPE=lekai` → `InferenceEngineLekai` (LLaMA-based, event stream)

### FR-12: History Clear
**Status**: WORKING
`POST /clear_history` resets all engine state, injection state, and accumulated latency. Called at session start by client.

### FR-13: Injection Status Query
**Status**: WORKING
`GET /injection_status` returns current injection state: `is_injected`, `injection_length_ticks`, note counts.

### FR-14: Listening Mode (Adaptive Prompt Selection)
**Status**: WORKING (recently added)
Client detects the musical key of played melody and auto-selects a matching prompt from the prompt library. Key detection uses `app/key_detection.py`. Prompt library organized by key in `prompts/` directory. Can be suppressed.

### FR-15: Manual Prompt Selection
**Status**: WORKING (recently added)
Users can manually select a prompt/injection file rather than relying on automatic key detection.

### FR-16: Web Client
**Status**: WORKING (partially tested)
`app/web_client.py` provides a WebSocket-based client for browser UI interaction. Supports the Lekai server variant. Feature parity with client_lekai.py is ongoing.

### FR-17: Performance Benchmarking
**Status**: WORKING
`app/benchmark.py` measures round-trip latency, server processing, inference time, network latency, preprocessing/postprocessing overhead. Results saved as CSV and JSON.

### FR-18: Offline Inference Testing
**Status**: WORKING
`m2a_transformer_inference.py` supports batch offline generation from MIDI file input. Useful for model evaluation without real-time constraints.

---

## Out of Scope (Legacy / Archived)

- Training pipeline (`training_runner.py`) — functional but not demo-critical
- Dataset extraction (`exact/`, `preprocess/`) — not used at inference time
- `transformer_engine.py` / `transformer_engine_midi_input.py` — superseded by Stanley/Lekai engines
- `models/old_m2a_transformer.py`, `old_m2a_nomask_transformer.py` — superseded by `new_m2a_transformer.py`
