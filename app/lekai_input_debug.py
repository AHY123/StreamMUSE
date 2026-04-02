import json
import os
import tempfile
import time


CLIENT_TIMING_DEBUG_ENABLED = os.getenv("STREAMMUSE_DEBUG_CLIENT_TIMING") == "1"
ENGINE_CONTEXT_DEBUG_ENABLED = os.getenv("STREAMMUSE_DEBUG_ENGINE_CONTEXT") == "1"

CLIENT_TIMING_DEBUG_PATH = os.path.join(
    tempfile.gettempdir(), "streammuse_client_timing.jsonl"
)
ENGINE_CONTEXT_DEBUG_PATH = os.path.join(
    tempfile.gettempdir(), "streammuse_engine_context.jsonl"
)


def _append_jsonl(path: str, payload: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never let debug logging break realtime execution.
        pass


def _normalize_event(event: dict | None) -> dict | None:
    if event is None:
        return None
    return {
        "type": event.get("type"),
        "pitch": event.get("pitch"),
        "velocity": event.get("velocity"),
        "tick": event.get("tick"),
        "time": event.get("time"),
    }


def log_client_timing(
    source: str,
    phase: str,
    *,
    tick_count: int | None = None,
    queue_size: int | None = None,
    event: dict | None = None,
    notes: list[dict] | None = None,
    generation_start_tick: int | None = None,
    extra: dict | None = None,
) -> None:
    if not CLIENT_TIMING_DEBUG_ENABLED:
        return

    payload = {
        "wall_time": time.time(),
        "perf_time": time.perf_counter(),
        "source": source,
        "phase": phase,
        "tick_count": tick_count,
        "queue_size": queue_size,
        "event": _normalize_event(event),
        "notes": notes,
        "generation_start_tick": generation_start_tick,
        "extra": extra or {},
    }
    _append_jsonl(CLIENT_TIMING_DEBUG_PATH, payload)


def log_engine_context(
    phase: str,
    *,
    generation_start_tick: int | None = None,
    current_beat: int | None = None,
    start_beat: int | None = None,
    beat_start_tick: int | None = None,
    beat_end_tick: int | None = None,
    need_reset: bool | None = None,
    active_pitches: set | list | tuple | None = None,
    melody_notes: list[dict] | None = None,
    abs_events: list[dict] | None = None,
    extra: dict | None = None,
) -> None:
    if not ENGINE_CONTEXT_DEBUG_ENABLED:
        return

    payload = {
        "wall_time": time.time(),
        "perf_time": time.perf_counter(),
        "phase": phase,
        "generation_start_tick": generation_start_tick,
        "current_beat": current_beat,
        "start_beat": start_beat,
        "beat_start_tick": beat_start_tick,
        "beat_end_tick": beat_end_tick,
        "need_reset": need_reset,
        "active_pitches": sorted(int(p) for p in active_pitches) if active_pitches else [],
        "melody_notes": melody_notes,
        "abs_events": abs_events,
        "extra": extra or {},
    }
    _append_jsonl(ENGINE_CONTEXT_DEBUG_PATH, payload)
