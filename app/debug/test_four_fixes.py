"""
Tests for the four real-time-path fixes. Run in the `muse_client` conda env.

These tests exercise the code paths that do not require the heavy model
dependencies (torch / safetensors / transformers). The engine-side Fix 4
logic is mirrored here as a pure-python reference implementation and
asserted against hand-derived expected outputs that match what the
patched `_normalize_melody_input` should produce.
"""

import os
import sys
import tempfile
import time

import mido
import requests

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from app.output_handlers.midi_file_handler import MidiFileHandler  # noqa: E402


# ---------------------------------------------------------------------------
# Fix 3 — MidiFileHandler (mido tick-native)
# ---------------------------------------------------------------------------

def _load_user_messages(path: str):
    mid = mido.MidiFile(path)
    assert mid.ticks_per_beat == 4, f"expected tpb=4, got {mid.ticks_per_beat}"
    # Track 0 is "Guitar" (user).
    track = mid.tracks[0]
    note_events = []
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type in ("note_on", "note_off"):
            note_events.append((msg.type, msg.note, abs_tick, msg.velocity))
    return note_events


def test_midi_fresh_tap_becomes_one_tick_blip():
    h = MidiFileHandler(tempo=120, ticks_per_beat=4)
    h.add_user_note({"type": "note_on", "pitch": 60, "tick": 5, "velocity": 100})
    h.add_user_note({"type": "note_off", "pitch": 60, "tick": 5})
    with tempfile.TemporaryDirectory() as d:
        h.save_to_midi(d)
        evts = _load_user_messages(os.path.join(d, "performance.mid"))
    # Expected: note_on at 5, note_off at 6 (min-1-tick floor).
    assert ("note_on", 60, 5, 100) in evts, evts
    assert ("note_off", 60, 6, 0) in evts, evts
    print("  OK: fresh tap -> 1-tick blip (on@5, off@6)")


def test_midi_sustained_note_unchanged():
    h = MidiFileHandler(tempo=120, ticks_per_beat=4)
    h.add_user_note({"type": "note_on", "pitch": 62, "tick": 0, "velocity": 100})
    h.add_user_note({"type": "note_off", "pitch": 62, "tick": 8})
    with tempfile.TemporaryDirectory() as d:
        h.save_to_midi(d)
        evts = _load_user_messages(os.path.join(d, "performance.mid"))
    assert ("note_on", 62, 0, 100) in evts, evts
    assert ("note_off", 62, 8, 0) in evts, evts
    print("  OK: sustained note tick=0..8 preserved")


def test_midi_retrigger_same_tick():
    h = MidiFileHandler(tempo=120, ticks_per_beat=4)
    # Retrigger: note_on, then another note_on at the same tick.
    h.add_user_note({"type": "note_on", "pitch": 64, "tick": 3, "velocity": 100})
    h.add_user_note({"type": "note_on", "pitch": 64, "tick": 3, "velocity": 100})
    h.add_user_note({"type": "note_off", "pitch": 64, "tick": 10})
    with tempfile.TemporaryDirectory() as d:
        h.save_to_midi(d)
        evts = _load_user_messages(os.path.join(d, "performance.mid"))
    # Expected: first note_on@3, implicit note_off@4 (min-1-tick close),
    # second note_on@3 (reordered after its preceding off by sort),
    # final note_off@10.
    on_count = sum(1 for e in evts if e[0] == "note_on" and e[1] == 64)
    off_count = sum(1 for e in evts if e[0] == "note_off" and e[1] == 64)
    assert on_count == 2 and off_count == 2, evts
    # Final release is at 10.
    assert max(t for typ, p, t, v in evts if typ == "note_off") == 10
    print("  OK: retrigger on tick 3 produces two notes (both closed)")


def test_midi_finalize_flushes_open_notes():
    h = MidiFileHandler(tempo=120, ticks_per_beat=4)
    h.add_user_note({"type": "note_on", "pitch": 67, "tick": 2, "velocity": 100})
    # No explicit note_off — finalize should emit one.
    h.add_user_note({"type": "note_on", "pitch": 69, "tick": 6, "velocity": 100})
    h.add_user_note({"type": "note_off", "pitch": 69, "tick": 10})
    with tempfile.TemporaryDirectory() as d:
        h.save_to_midi(d)
        evts = _load_user_messages(os.path.join(d, "performance.mid"))
    off_67 = [e for e in evts if e[0] == "note_off" and e[1] == 67]
    assert len(off_67) == 1, evts
    # Close tick must be >= the note_on tick + 1.
    assert off_67[0][2] >= 3
    print("  OK: finalize emits note_off for still-open pitch 67")


def test_midi_tempo_meta_written():
    h = MidiFileHandler(tempo=100, ticks_per_beat=4)
    h.add_user_note({"type": "note_on", "pitch": 60, "tick": 0, "velocity": 100})
    h.add_user_note({"type": "note_off", "pitch": 60, "tick": 4})
    with tempfile.TemporaryDirectory() as d:
        h.save_to_midi(d)
        mid = mido.MidiFile(os.path.join(d, "performance.mid"))
    tempos = [
        m for t in mid.tracks for m in t if m.type == "set_tempo"
    ]
    assert len(tempos) >= 1
    assert abs(mido.tempo2bpm(tempos[0].tempo) - 100) < 0.01
    print("  OK: set_tempo meta == 100 BPM")


# ---------------------------------------------------------------------------
# Fix 4 — Same-tick bump in _normalize_melody_input (reference mirror)
# ---------------------------------------------------------------------------

def _normalize_reference(melody_notes, injection_offset=0, active_pitches=None):
    """Pure-python mirror of the patched _normalize_melody_input. Must stay
    structurally identical to the code in transformer_engine_lekai.py."""
    abs_events = []
    for e in melody_notes:
        if not isinstance(e, dict):
            continue
        if e.get("type") not in {"note_on", "note_off"}:
            continue
        if "pitch" not in e or "tick" not in e:
            continue
        abs_e = e.copy()
        abs_e["tick"] = int(e["tick"]) + injection_offset
        abs_e["pitch"] = int(e["pitch"])
        abs_events.append(abs_e)

    opened_this_batch: dict[int, int] = {}
    for ev in abs_events:
        pitch = ev["pitch"]
        tick = int(ev["tick"])
        if ev["type"] == "note_on":
            opened_this_batch[pitch] = tick
        else:
            if opened_this_batch.get(pitch) == tick:
                ev["tick"] = tick + 1
            opened_this_batch.pop(pitch, None)
    return abs_events


def test_normalize_matches_engine_source():
    """Sanity: assert the engine source contains the bump block we rely on."""
    src_path = os.path.join(
        REPO_ROOT, "app", "inference_engines", "transformer_engine_lekai.py"
    )
    with open(src_path) as f:
        src = f.read()
    assert "opened_this_batch" in src
    assert "ev[\"tick\"] = tick + 1" in src
    print("  OK: engine source contains same-tick bump logic")


def test_normalize_fresh_tap_bumped():
    out = _normalize_reference(
        [
            {"type": "note_on", "pitch": 60, "tick": 3},
            {"type": "note_off", "pitch": 60, "tick": 3},
        ]
    )
    assert out[0]["tick"] == 3 and out[0]["type"] == "note_on"
    assert out[1]["tick"] == 4 and out[1]["type"] == "note_off"
    print("  OK: fresh tap bumped (on@3, off@4)")


def test_normalize_retrigger_untouched():
    out = _normalize_reference(
        [
            {"type": "note_off", "pitch": 60, "tick": 3},
            {"type": "note_on", "pitch": 60, "tick": 3},
        ]
    )
    # note_off was not preceded by an in-batch note_on for pitch 60 at tick 3,
    # so it must NOT be bumped.
    assert out[0]["tick"] == 3 and out[0]["type"] == "note_off"
    assert out[1]["tick"] == 3 and out[1]["type"] == "note_on"
    print("  OK: retrigger (off-then-on) untouched")


def test_normalize_hold_untouched():
    out = _normalize_reference(
        [
            {"type": "note_on", "pitch": 60, "tick": 3},
            {"type": "note_off", "pitch": 60, "tick": 7},
        ]
    )
    assert out[1]["tick"] == 7
    print("  OK: 4-tick hold untouched")


def test_normalize_injection_offset_applied_before_bump():
    out = _normalize_reference(
        [
            {"type": "note_on", "pitch": 60, "tick": 3},
            {"type": "note_off", "pitch": 60, "tick": 3},
        ],
        injection_offset=10,
    )
    assert out[0]["tick"] == 13
    assert out[1]["tick"] == 14  # bumped from 13
    print("  OK: injection_offset applied, then bump")


# ---------------------------------------------------------------------------
# Fix 1 — Absolute-clock loop scheduling (source-level check)
# ---------------------------------------------------------------------------

def test_tick_loop_absolute_clock_source():
    src_path = os.path.join(REPO_ROOT, "app", "web_client.py")
    with open(src_path) as f:
        src = f.read()
    # Anchoring session_perf_start at loop start:
    assert "session_perf_start = time.perf_counter()" in src
    assert 'current_tick_ref["session_perf_start"] = session_perf_start' in src
    # Absolute pre-work and end-of-tick sleeps:
    assert "pre_work_target = session_perf_start + (tick_count + 0.1)" in src
    assert "next_tick_target = session_perf_start + (tick_count + 1)" in src
    # Old relative sleeps gone:
    assert "time.sleep(seconds_per_tick * 0.1)" not in src
    assert "time.sleep(seconds_per_tick * 0.9)" not in src
    print("  OK: tick loop uses absolute-clock scheduling")


# ---------------------------------------------------------------------------
# Fix 2 — Model-note tick overwrite removed
# ---------------------------------------------------------------------------

def test_model_note_tick_not_overwritten():
    src_path = os.path.join(REPO_ROOT, "app", "web_client.py")
    with open(src_path) as f:
        src = f.read()
    # The specific overwrite lines must be gone.
    assert 'event_for_midi["tick"] = tick_count' not in src
    print("  OK: model-note tick overwrite removed")


# ---------------------------------------------------------------------------
# Fake server smoke test (optional — only if server is reachable)
# ---------------------------------------------------------------------------

def test_fake_server_health():
    try:
        r = requests.get("http://localhost:8001/health", timeout=1.5)
        r.raise_for_status()
        print(f"  OK: fake_server /health -> {r.json()}")
    except Exception as e:
        print(f"  SKIP: fake_server not reachable ({e})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Fix 3.1", test_midi_fresh_tap_becomes_one_tick_blip),
    ("Fix 3.2", test_midi_sustained_note_unchanged),
    ("Fix 3.3", test_midi_retrigger_same_tick),
    ("Fix 3.4", test_midi_finalize_flushes_open_notes),
    ("Fix 3.5", test_midi_tempo_meta_written),
    ("Fix 4.0", test_normalize_matches_engine_source),
    ("Fix 4.1", test_normalize_fresh_tap_bumped),
    ("Fix 4.2", test_normalize_retrigger_untouched),
    ("Fix 4.3", test_normalize_hold_untouched),
    ("Fix 4.4", test_normalize_injection_offset_applied_before_bump),
    ("Fix 1  ", test_tick_loop_absolute_clock_source),
    ("Fix 2  ", test_model_note_tick_not_overwritten),
    ("Server ", test_fake_server_health),
]


def main():
    failed = 0
    for label, fn in TESTS:
        print(f"[{label}] {fn.__name__}")
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
