import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import mido
import pretty_midi
import requests


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lekai_input_debug import CLIENT_TIMING_DEBUG_PATH


DEFAULT_SERVER_URL = "http://127.0.0.1:8010/generate_accompaniment"
DEFAULT_CLEAR_URL = "http://127.0.0.1:8010/clear_history"
TEST_SERVER_PATH = ROOT / "app" / "debug" / "lekai_live_test_server.py"
MELODIES_PATH = ROOT / "app" / "debug" / "lekai_live_test_melodies.json"
OUTPUT_DIR = ROOT / "app" / "debug" / "virtual_midi_replay_output"
CLIENT_TIMING_LOG = Path(CLIENT_TIMING_DEBUG_PATH)
VIRTUAL_PORT_NAME = "StreamMUSE Replay Port"


def load_melodies():
    with open(MELODIES_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_midi(notes: list[dict], tempo: float, ticks_per_beat: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0, name="UserMelody")
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=int(note["pitch"]),
                start=float(note["tick"]) * seconds_per_tick,
                end=float(note["tick"] + note["duration"]) * seconds_per_tick,
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(output_path))


def instrument_notes_to_tick_notes(midi_path: Path, instrument_index: int, ticks_per_beat: int):
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    if instrument_index >= len(midi.instruments):
        return []
    instrument = midi.instruments[instrument_index]
    if not midi.get_tempo_changes()[1].size:
        tempo = 120.0
    else:
        tempo = float(midi.get_tempo_changes()[1][0])
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    notes = []
    for note in instrument.notes:
        start_tick = int(round(note.start / seconds_per_tick))
        end_tick = int(round(note.end / seconds_per_tick))
        notes.append(
            {
                "pitch": int(note.pitch),
                "tick": start_tick,
                "duration": max(1, end_tick - start_tick),
            }
        )
    notes.sort(key=lambda n: (n["tick"], n["pitch"], n["duration"]))
    return notes


def remove_stale_logs():
    if CLIENT_TIMING_LOG.exists():
        CLIENT_TIMING_LOG.unlink()


def start_test_server():
    server_log = open(OUTPUT_DIR / "test_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(TEST_SERVER_PATH)],
        cwd=str(ROOT),
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(50):
        try:
            requests.get("http://127.0.0.1:8010/health", timeout=0.2)
            return proc, server_log
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Test server failed to start on 127.0.0.1:8010")


def stop_process(proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def start_client(server_url: str):
    client_log = open(OUTPUT_DIR / "client_lekai.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["STREAMMUSE_DEBUG_CLIENT_TIMING"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "app/client_lekai.py",
            "--server_url",
            server_url,
            "--tempo",
            "120",
            "--ticks_per_beat",
            "4",
            "--generation_interval_ticks",
            "4",
            "--midi_input_name",
            VIRTUAL_PORT_NAME,
        ],
        cwd=str(ROOT),
        stdout=client_log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, client_log


def replay_midi(notes: list[dict], tempo: float, ticks_per_beat: int, output_port, start_delay: float):
    seconds_per_tick = (60.0 / tempo) / ticks_per_beat
    note_events = []
    for note in notes:
        note_events.append((note["tick"], "note_on", int(note["pitch"]), 100))
        note_events.append((note["tick"] + note["duration"], "note_off", int(note["pitch"]), 0))
    note_events.sort(key=lambda item: (item[0], 0 if item[1] == "note_off" else 1, item[2]))

    start_time = time.perf_counter() + start_delay
    for tick, event_type, pitch, velocity in note_events:
        target_time = start_time + tick * seconds_per_tick
        while True:
            remaining = target_time - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.002))
        output_port.send(mido.Message(event_type, note=pitch, velocity=velocity))


def latest_session_dir() -> Path:
    logs_dir = ROOT / "app" / "logs"
    session_dirs = sorted(
        [p for p in logs_dir.iterdir() if p.is_dir() and p.name.startswith("session_")],
        key=lambda p: p.stat().st_mtime,
    )
    if not session_dirs:
        raise RuntimeError("No session log directory was created by client_lekai.py")
    return session_dirs[-1]


def read_client_timing_events():
    if not CLIENT_TIMING_LOG.exists():
        return [], []
    records = []
    with open(CLIENT_TIMING_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    request_notes = []
    for record in records:
        if record.get("phase") in {"request_send_initial", "request_send_periodic"}:
            for note in record.get("notes") or []:
                if note is not None:
                    request_notes.append(
                        {
                            "type": note.get("type"),
                            "pitch": int(note.get("pitch")),
                            "tick": int(note.get("tick")),
                        }
                    )
    request_notes.sort(key=lambda item: (item["tick"], 0 if item["type"] == "note_off" else 1, item["pitch"]))
    return records, request_notes


def events_to_notes(events: list[dict]):
    active = {}
    notes = []
    for event in sorted(events, key=lambda e: (e["tick"], 0 if e["type"] == "note_off" else 1, e["pitch"])):
        key = int(event["pitch"])
        tick = int(event["tick"])
        if event["type"] == "note_on":
            active[key] = tick
        elif event["type"] == "note_off" and key in active:
            start_tick = active.pop(key)
            duration = max(1, tick - start_tick)
            notes.append({"pitch": key, "tick": start_tick, "duration": duration})
    return notes


def compare_notes(source_notes: list[dict], observed_notes: list[dict]):
    source_sorted = sorted(source_notes, key=lambda n: (n["tick"], n["pitch"], n["duration"]))
    observed_sorted = sorted(observed_notes, key=lambda n: (n["tick"], n["pitch"], n["duration"]))
    max_len = max(len(source_sorted), len(observed_sorted))
    mismatches = []
    for idx in range(max_len):
        source = source_sorted[idx] if idx < len(source_sorted) else None
        observed = observed_sorted[idx] if idx < len(observed_sorted) else None
        if source != observed:
            mismatches.append({"index": idx, "source": source, "observed": observed})
    return {
        "source_count": len(source_sorted),
        "observed_count": len(observed_sorted),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def normalize_notes_by_offset(source_notes: list[dict], observed_notes: list[dict]):
    if not source_notes or not observed_notes:
        return None, observed_notes
    offset = observed_notes[0]["tick"] - source_notes[0]["tick"]
    normalized = [dict(note, tick=int(note["tick"]) - int(offset)) for note in observed_notes]
    return offset, normalized


def run_case(
    case_name: str,
    notes: list[dict],
    tempo: float,
    ticks_per_beat: int,
    output_port,
    server_url: str,
    clear_url: str,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    midi_path = OUTPUT_DIR / f"{case_name}.mid"
    write_midi(notes, tempo, ticks_per_beat, midi_path)

    remove_stale_logs()
    clear_response = requests.post(clear_url, timeout=2)
    clear_response.raise_for_status()

    client_proc, client_log = start_client(server_url)
    try:
        time.sleep(2.0)
        replay_midi(notes, tempo, ticks_per_beat, output_port, start_delay=1.0)
        total_ticks = max(note["tick"] + note["duration"] for note in notes)
        time.sleep(((total_ticks + 8) * (60.0 / tempo) / ticks_per_beat))
    finally:
        stop_process(client_proc)
        client_log.close()

    session_dir = latest_session_dir()
    recorded_midi = session_dir / "performance.mid"
    recorded_notes = (
        instrument_notes_to_tick_notes(recorded_midi, instrument_index=0, ticks_per_beat=ticks_per_beat)
        if recorded_midi.exists()
        else []
    )

    _, request_events = read_client_timing_events()
    request_notes = events_to_notes(request_events)
    source_notes = instrument_notes_to_tick_notes(midi_path, instrument_index=0, ticks_per_beat=ticks_per_beat)
    request_offset, normalized_request_notes = normalize_notes_by_offset(source_notes, request_notes)
    recorded_offset, normalized_recorded_notes = normalize_notes_by_offset(source_notes, recorded_notes)

    result = {
        "case_name": case_name,
        "source_midi": str(midi_path),
        "session_dir": str(session_dir),
        "recorded_midi": str(recorded_midi),
        "recorded_midi_exists": recorded_midi.exists(),
        "source_notes": source_notes,
        "request_events": request_events,
        "request_notes": request_notes,
        "request_offset_ticks": request_offset,
        "normalized_request_notes": normalized_request_notes,
        "recorded_notes": recorded_notes,
        "recorded_offset_ticks": recorded_offset,
        "normalized_recorded_notes": normalized_recorded_notes,
        "request_vs_source": compare_notes(source_notes, request_notes),
        "normalized_request_vs_source": compare_notes(source_notes, normalized_request_notes),
        "recorded_vs_source": compare_notes(source_notes, recorded_notes),
        "normalized_recorded_vs_source": compare_notes(source_notes, normalized_recorded_notes),
    }
    result_path = OUTPUT_DIR / f"{case_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run Lekai live virtual MIDI replay tests.")
    parser.add_argument(
        "--case",
        choices=["sanity_case", "stress_case", "all"],
        default="all",
        help="Which test melody to run",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help="Full /generate_accompaniment URL for the target server",
    )
    parser.add_argument(
        "--clear-url",
        default=DEFAULT_CLEAR_URL,
        help="Full /clear_history URL for the target server",
    )
    parser.add_argument(
        "--use-existing-server",
        action="store_true",
        help="Use an already running server instead of starting the local mock server",
    )
    args = parser.parse_args()

    data = load_melodies()
    tempo = float(data["tempo"])
    ticks_per_beat = int(data["ticks_per_beat"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server_proc = None
    server_log = None
    output_port = None
    try:
        if not args.use_existing_server:
            server_proc, server_log = start_test_server()
        output_port = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)
        case_names = ["sanity_case", "stress_case"] if args.case == "all" else [args.case]
        summary = {}
        for case_name in case_names:
            summary[case_name] = run_case(
                case_name,
                data["melodies"][case_name]["notes"],
                tempo,
                ticks_per_beat,
                output_port,
                args.server_url,
                args.clear_url,
            )

        summary_path = OUTPUT_DIR / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary, indent=2))
    finally:
        if output_port is not None:
            output_port.close()
        stop_process(server_proc)
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    main()
