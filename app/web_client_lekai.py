"""
StreamMUSE Web Client Server (Lekai Version)

FastAPI server that:
1. Serves the web UI static files
2. Provides WebSocket endpoint for real-time updates
3. Manages client lifecycle (start/stop/restart)
4. Bridges Lekai client logic to browser

Usage:
    python app/web_client_lekai.py
    # Open http://localhost:8081 in browser
"""

import os
import sys
import asyncio
import threading
import time
import argparse
from queue import Queue
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import json
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.output_handlers.websocket_output import WebSocketOutputHandler
from app.output_handlers.audio_output import AudioOutputHandler
from app.output_handlers.midi_file_handler import MidiFileHandler
from app.output_handlers.json_log_handler import JsonLogHandler
from app.input_handlers.input_handler import (
    read_midi_input,
    read_keyboard_input,
    read_midi_file_input,
)

# Optional dependencies
try:
    from app.key_detection import detect_key_lightweight, detect_key_music21
    from app.prompt_library import PromptLibrary

    LISTENING_MODE_AVAILABLE = True
    print("[INIT] Listening mode dependencies loaded successfully")
except ImportError as e:
    LISTENING_MODE_AVAILABLE = False
    detect_key_lightweight = None
    detect_key_music21 = None
    PromptLibrary = None
    print(f"[INIT] Listening mode dependencies NOT available: {e}")


class ClientConfig(BaseModel):
    """Configuration for Lekai Client"""

    # Network
    server_url: str = "http://localhost:8988/generate_accompaniment"

    # Musical Timing
    tempo: float = 120.0
    ticks_per_beat: int = 4
    beats_per_bar: int = 4
    generation_interval_ticks: int = 4  # Matches client_lekai.py default
    generation_length_per_request: int = 8  # Default from client_lekai logic (frames=config/2 usually)

    # Note Handling
    note_duration_ticks: int = 2  # Matches client_lekai default
    accompaniment_velocity: int = 50
    melody_channel: int = 0
    accompaniment_channel: int = 0

    # MIDI File Input
    midi_file_delay_ticks: int = 0

    # Sampling Parameters (Lekai Specific)
    temperature: float = 1.1
    top_k: int = 10
    top_p: float = 0.95

    # Input/Output
    input_mode: str = "keyboard"
    midi_file_path: Optional[str] = None
    midi_input_name: Optional[str] = None
    midi_output_name: Optional[str] = None
    metronome: bool = True

    # Listening Mode (Optional)
    listening_duration_ticks: int = 0
    listening_mode: str = "auto"
    manual_prompt_path: Optional[str] = None
    key_detection_method: str = "lightweight"
    prompt_dir: Optional[str] = None


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


class ClientManager:
    """Manages the StreamMUSE client lifecycle (Lekai Logic)."""

    GRACE_PERIOD_TICKS = 8

    def __init__(self, ws_handler: WebSocketOutputHandler):
        self.ws_handler = ws_handler
        self.config = ClientConfig()
        self.is_running = False
        self.stop_event = threading.Event()

        self.event_queue: Optional[Queue] = None
        self.inference_request_queue: Optional[Queue] = None
        self.inference_response_queue: Optional[Queue] = None

        self.input_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.tick_thread: Optional[threading.Thread] = None

        self.audio_output_handler: Optional[AudioOutputHandler] = None
        self.all_timing_data = []
        self.tick_history = []

    def start(self, config: Optional[ClientConfig] = None):
        """Start the client with given config."""
        if self.is_running:
            return False

        if config:
            self.config = config

        print(f"Starting Lekai Client with config: {self.config}")

        self.stop_event.clear()
        self.event_queue = Queue()
        self.inference_request_queue = Queue()
        self.inference_response_queue = Queue()
        self.all_timing_data = []
        self.tick_history = []

        # Audio Output (Optional for Web, but good for local monitor)
        try:
            self.audio_output_handler = AudioOutputHandler(
                port_name=self.config.midi_output_name,
                accompaniment_velocity=self.config.accompaniment_velocity,
            )
        except Exception as e:
            print(f"Warning: Could not initialize local audio output: {e}")
            self.audio_output_handler = None

        current_tick_ref = {"current_tick": 0}

        # --- Start Input Thread ---
        if self.config.input_mode == "file" and self.config.midi_file_path:
            self.input_thread = threading.Thread(
                target=read_midi_file_input,
                args=(
                    self.event_queue,
                    self.config.midi_file_path,
                    current_tick_ref,
                    self.config.tempo,
                    self.config.ticks_per_beat,
                    self.config.midi_file_delay_ticks,
                    0,  # start_tick_offset
                    True,  # loop
                    self.config.note_duration_ticks,
                    self.audio_output_handler,
                    self.config.melody_channel,
                ),
                daemon=True,
            )
        elif self.config.input_mode == "midi":
            self.input_thread = threading.Thread(
                target=read_midi_input,
                args=(
                    self.event_queue,
                    self.config.midi_input_name,
                    self.audio_output_handler,
                    self.config.melody_channel,
                ),
                daemon=True,
            )
        else:  # Keyboard
            self.input_thread = threading.Thread(
                target=read_keyboard_input,
                args=(self.event_queue, self.audio_output_handler, self.config.melody_channel),
                daemon=True,
            )

        # --- Start Inference Thread ---
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
        )

        # --- Start Tick Loop ---
        self.tick_thread = threading.Thread(
            target=self.tick_loop,
            args=(current_tick_ref,),
            daemon=True,
        )

        self.input_thread.start()
        self.inference_thread.start()
        self.tick_thread.start()

        self.is_running = True
        self.ws_handler.send_status("running", "Client started")
        self.ws_handler.send_config(self.config.model_dump())

        return True

    def stop(self):
        """Stop the client."""
        if not self.is_running:
            return False

        print("Stopping client...")
        self.stop_event.set()

        # Send poison pills
        if self.event_queue:
            self.event_queue.put(None)
        if self.inference_request_queue:
            self.inference_request_queue.put(None)

        # Wait for join (with timeout)
        if self.input_thread:
            self.input_thread.join(timeout=1.0)
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
        if self.tick_thread:
            self.tick_thread.join(timeout=1.0)

        if self.audio_output_handler:
            self.audio_output_handler.close()
            self.audio_output_handler = None

        self.is_running = False
        self.ws_handler.send_status("stopped", "Client stopped")
        return True

    def restart(self, config: Optional[ClientConfig] = None):
        self.stop()
        time.sleep(0.5)
        return self.start(config)

    def _inference_worker(self):
        """Worker thread for sending requests to server (Adapted for Lekai parameters)."""
        while not self.stop_event.is_set():
            try:
                queue_item = self.inference_request_queue.get(timeout=0.1)
            except Exception:
                continue

            if queue_item is None:
                break

            request_data, full_request_dict = queue_item

            # --- Lekai Specific: Ensure sampling params are present ---
            # (Usually they are put in by tick_loop, but double checking here isn't bad)
            # request_data should already contain text/top_k etc.

            client_send_time = time.perf_counter()
            request_data["client_request_send_time"] = client_send_time
            full_request_dict["client_request_send_time"] = client_send_time

            start_time = client_send_time
            try:
                response = requests.post(self.config.server_url, json=request_data, timeout=5.0)
                response.raise_for_status()
                response_json = response.json()
            except Exception as e:
                print(f"Error contacting server: {e}")
                response_json = None

            end_time = time.perf_counter()
            round_trip_time = end_time - start_time

            self.inference_response_queue.put((response_json, round_trip_time, full_request_dict))

    def tick_loop(self, current_tick_ref: dict):
        """
        Main tick loop (Adapted from client_lekai.py).
        Replaces local print/output with WebSocket broadcasting.
        """
        tempo = self.config.tempo
        ticks_per_beat = self.config.ticks_per_beat
        seconds_per_tick = (60.0 / tempo) / ticks_per_beat

        tick_count = -1
        playback_schedule = {}
        number_of_hit = 0
        total_backup_level = 0

        notes_for_next_request = []
        ticks_per_bar = ticks_per_beat * self.config.beats_per_bar

        # Sampling Params
        sampling_params = {
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
        }

        print(f"Tick Loop Started. Tempo: {tempo}, Interval: {self.config.generation_interval_ticks}")

        # --- Listening Mode Setup (Simplification: using basic logic from web_client if needed, or skipping for now) ---
        # For this iteration, we focus on the core Lekai loop.

        self.ws_handler.send_status("running", "Tick loop running")

        while not self.stop_event.is_set():
            tick_count += 1
            if current_tick_ref:
                current_tick_ref["current_tick"] = tick_count

            # WS Update
            bar_count = tick_count // ticks_per_bar
            beat_in_bar = (tick_count % ticks_per_bar) // ticks_per_beat
            self.ws_handler.send_tick(tick_count, bar_count, beat_in_bar)

            # --- 0. Trigger Inference at tick=0 (Lekai Specific) ---
            is_tick_zero = tick_count == 0
            if is_tick_zero:
                # Immediate initial trigger
                request_data = {"melody_notes": [], "generation_start_tick": 0, **sampling_params}
                print("Triggering initial inference at tick 0")
                self.inference_request_queue.put((request_data, request_data.copy()))

            # --- 1. Process User Input ---
            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                except:
                    break

                if event is None:
                    return

                if event["type"] == "note_on":
                    quantized_note = {
                        "pitch": event["pitch"],
                        "tick": tick_count,  # Current tick logic
                        "duration": self.config.note_duration_ticks,
                    }
                    notes_for_next_request.append(quantized_note)

                    # Send to Frontend
                    self.ws_handler.send_note_on(
                        pitch=event["pitch"],
                        velocity=event.get("velocity", 80),
                        tick=tick_count,
                        duration=self.config.note_duration_ticks,
                        source="user",
                    )
                elif event["type"] == "note_off":
                    self.ws_handler.send_note_off(event["pitch"], tick_count, "user")

            # --- 2. Process Inference Responses ---
            while not self.inference_response_queue.empty():
                try:
                    response_data, round_trip_time, request_data = self.inference_response_queue.get_nowait()
                except:
                    break

                if response_data:
                    # Update Stats
                    timings = response_data.get("timings", {})
                    server_proc = timings.get("response_output_time", 0) - timings.get("request_arrival_time", 0)
                    self.ws_handler.send_stats(
                        round_trip_ms=round_trip_time * 1000,
                        server_process_ms=server_proc * 1000,
                        network_latency_ms=(round_trip_time - server_proc) * 1000,
                    )

                    newly_generated_notes = response_data.get("accompaniment", [])
                    gen_start = request_data.get("generation_start_tick")

                    # Calculate backup level same as client_lekai
                    for n in newly_generated_notes:
                        n["backup_level"] = int(n["tick"] - gen_start) if gen_start is not None else 0

                    # Schedule notes
                    for note in newly_generated_notes:
                        if note["tick"] >= tick_count:
                            if note["tick"] not in playback_schedule:
                                playback_schedule[note["tick"]] = []
                            playback_schedule[note["tick"]].append({**note, "source": "model"})

                            # Notify Frontend of generation (Visualization)
                            self.ws_handler.send_note_on(
                                pitch=note["pitch"],
                                velocity=self.config.accompaniment_velocity,
                                tick=note["tick"],
                                duration=note.get("duration", 4),
                                source="model",
                                backup_level=note.get("backup_level", 0),
                            )

            # --- 3. Trigger Periodic Inference ---
            # Lekai logic: interval check
            is_trigger_tick = (tick_count > 0) and (tick_count % self.config.generation_interval_ticks == 0)

            if is_trigger_tick:
                next_start = tick_count + 1  # Or logic from client_lekai
                # client_lekai logic for trigger:
                # next_interval_start_tick = tick_count + 1 ??
                # Actually client_lekai uses:
                # request_data = { "melody_notes": notes_for_next_request, ... }
                # Let's double check client_lekai Logic.
                # It sends request every interval.

                request_data = {
                    "melody_notes": notes_for_next_request,
                    "generation_start_tick": tick_count,  # Simplification
                    **sampling_params,
                }
                self.inference_request_queue.put((request_data, request_data.copy()))
                notes_for_next_request = []

            # --- 4. Playback / Output ---
            # ... Playback matches web_client logic ...
            scheduled = playback_schedule.pop(tick_count, [])
            for event in scheduled:
                # Play audio locally if handler exists
                if self.audio_output_handler:
                    if event.get("type") == "note_off":
                        self.audio_output_handler.off(event["pitch"])
                    else:
                        self.audio_output_handler.on(event["pitch"], event.get("velocity", 60))

                # WS note_off for model events
                if event.get("type", "note_on") == "note_on":
                    # Schedule note off visual
                    off_tick = tick_count + event.get("duration", 4)
                    if off_tick not in playback_schedule:
                        playback_schedule[off_tick] = []
                    playback_schedule[off_tick].append({**event, "type": "note_off", "source": "model"})
                elif event.get("type") == "note_off":
                    self.ws_handler.send_note_off(event["pitch"], tick_count, "model")

            # Metronome
            if self.config.metronome and self.audio_output_handler:
                if tick_count % self.config.ticks_per_beat == 0:
                    if (tick_count % ticks_per_bar) == 0:
                        self.audio_output_handler.metro_first()
                    else:
                        self.audio_output_handler.metro_other()

            time.sleep(seconds_per_tick)


# --- FastAPI App ---


async def ws_broadcast_loop():
    """Async loop to broadcast messages from queue to WebSocket clients."""
    while True:
        messages = ws_output_handler.get_pending_messages()
        for msg in messages:
            await connection_manager.broadcast(msg)
        await asyncio.sleep(0.01)


@asynccontextmanager
async def lifespan(app: FastAPI):
    broadcast_task = asyncio.create_task(ws_broadcast_loop())
    yield
    broadcast_task.cancel()
    try:
        await broadcast_task
    except asyncio.CancelledError:
        pass
    client_manager.stop()


app_lekai = FastAPI(title="StreamMUSE Lekai Client", lifespan=lifespan)


# Setup connections
connection_manager = ConnectionManager()
ws_output_handler = WebSocketOutputHandler()
client_manager = ClientManager(ws_output_handler)


# Static files - Use web_ui directory
# Get absolute path to web_ui directory
base_dir = os.path.dirname(os.path.abspath(__file__))
web_ui_dir = os.path.join(base_dir, "web_ui")

if os.path.exists(os.path.join(web_ui_dir, "css")):
    app_lekai.mount("/css", StaticFiles(directory=os.path.join(web_ui_dir, "css")), name="css")
if os.path.exists(os.path.join(web_ui_dir, "js")):
    app_lekai.mount("/js", StaticFiles(directory=os.path.join(web_ui_dir, "js")), name="js")
if os.path.exists(os.path.join(web_ui_dir, "assets")):  # Assuming assets might exist
    app_lekai.mount("/assets", StaticFiles(directory=os.path.join(web_ui_dir, "assets")), name="assets")


@app_lekai.get("/")
async def get():
    index_path = os.path.join(web_ui_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Frontend not found at " + web_ui_dir})


@app_lekai.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        # Send initial status
        await websocket.send_text(
            json.dumps(
                {
                    "type": "status",
                    "status": "stopped" if not client_manager.is_running else "running",
                    "message": "Connected to Lekai Client",
                }
            )
        )

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "start":
                # Parse config from message
                config_dict = message.get("config", {})
                config = ClientConfig(**config_dict)
                success = client_manager.start(config)
                if not success:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Failed to start"}))

            elif msg_type == "stop":
                client_manager.stop()

            elif msg_type == "note_on":
                if client_manager.event_queue:
                    client_manager.event_queue.put(
                        {"type": "note_on", "pitch": message.get("pitch"), "velocity": message.get("velocity", 80)}
                    )

            elif msg_type == "note_off":
                if client_manager.event_queue:
                    client_manager.event_queue.put({"type": "note_off", "pitch": message.get("pitch")})

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

        # client_manager.stop() # Optional: stop client on disconnect?


@app_lekai.post("/api/start")
async def start_client(config: Optional[ClientConfig] = None):
    if client_manager.start(config):
        return {"success": True, "message": "Client started"}
    return JSONResponse(status_code=400, content={"success": False, "message": "Client already running"})


@app_lekai.post("/api/stop")
async def stop_client():
    if client_manager.stop():
        return {"success": True, "message": "Client stopped"}
    return JSONResponse(status_code=400, content={"success": False, "message": "Client not running"})


@app_lekai.post("/api/restart")
async def restart_client(config: Optional[ClientConfig] = None):
    if client_manager.restart(config):
        return {"success": True, "message": "Client restarted"}
    return JSONResponse(status_code=500, content={"success": False, "message": "Failed to restart client"})


@app_lekai.get("/api/status")
async def get_status():
    return {"is_running": client_manager.is_running, "config": client_manager.config.model_dump()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamMUSE Web Client Server (Lekai Version)")

    # Connection Args
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:8988/generate_accompaniment",
        help="URL of the StreamMUSE inference server",
    )
    parser.add_argument("--port", type=int, default=8081, help="Port for web UI server")

    # Timing Args
    parser.add_argument("--tempo", type=float, default=120.0, help="Tempo in BPM")
    parser.add_argument("--ticks_per_beat", type=int, default=4, help="Ticks per beat")
    parser.add_argument("--beats_per_bar", type=int, default=4, help="Beats per bar")

    # Generation Args
    parser.add_argument("--generation_interval_ticks", type=int, default=4, help="Ticks between generation requests")
    parser.add_argument("--generation_length_per_request", type=int, default=8, help="Frames to generate per request")
    parser.add_argument(
        "--accompaniment_velocity", type=int, default=50, help="Velocity for accompaniment notes (0-127)"
    )

    # Lekai Specific Args
    parser.add_argument("--temperature", type=float, default=1.1, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Sampling top-k")
    parser.add_argument("--top_p", type=float, default=0.95, help="Sampling top-p")

    # Input/Output Args
    parser.add_argument(
        "--input_mode", type=str, default="keyboard", choices=["keyboard", "midi", "file"], help="Input mode"
    )
    parser.add_argument("--midi_input_name", type=str, default=None, help="Name of MIDI input device")
    parser.add_argument("--midi_output_name", type=str, default=None, help="Name of MIDI output device")
    parser.add_argument("--midi_file_path", type=str, default=None, help="Path to MIDI input file")

    # Listening Mode Args
    parser.add_argument(
        "--listening_duration_ticks", type=int, default=0, help="Duration for listening mode (0 = disabled)"
    )
    parser.add_argument("--prompt_dir", type=str, default=None, help="Directory containing prompt MIDI files")
    parser.add_argument(
        "--key_detection_method",
        type=str,
        default="lightweight",
        choices=["lightweight", "music21"],
        help="Method for key detection",
    )

    args = parser.parse_args()

    # Initialize Config from Args
    client_manager.config = ClientConfig(
        server_url=args.server_url,
        tempo=args.tempo,
        ticks_per_beat=args.ticks_per_beat,
        beats_per_bar=args.beats_per_bar,
        generation_interval_ticks=args.generation_interval_ticks,
        generation_length_per_request=args.generation_length_per_request,
        accompaniment_velocity=args.accompaniment_velocity,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        input_mode=args.input_mode,
        midi_input_name=args.midi_input_name,
        midi_output_name=args.midi_output_name,
        midi_file_path=args.midi_file_path,
        listening_duration_ticks=args.listening_duration_ticks,
        prompt_dir=args.prompt_dir,
        key_detection_method=args.key_detection_method,
    )

    print("Starting StreamMUSE Web Client Server (Lekai)...")
    print(f"Server URL: {args.server_url}")
    print(f"Web UI Port: {args.port}")
    if args.input_mode == "midi":
        print(f"MIDI Input: {args.midi_input_name}")
    print(f"Open http://localhost:{args.port} in your browser")

    uvicorn.run(app_lekai, host="0.0.0.0", port=args.port)
