import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time

# Import the inference engine
from app.inference_engines.transformer_engine import TransformerInferenceEngine

# --- Pydantic Models for API data validation ---
class NoteEvent(BaseModel):
    key: str
    pitch: int
    time: float

class AccompanimentEvent(BaseModel):
    pitch: int
    duration: float
    play_at_tick: int

# --- FastAPI Application ---
app = FastAPI(title="StreamMUSE Inference Server")

# This will be populated at startup
inference_engine: TransformerInferenceEngine = None

@app.on_event("startup")
async def startup_event():
    """
    Load the model when the server starts.
    Reads the checkpoint path from an environment variable.
    """
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path:
        print("Fatal Error: CHECKPOINT_PATH environment variable not set.")
        print("Please run the server like: CHECKPOINT_PATH=path/to/model.ckpt uvicorn ...")
        exit()

    global inference_engine
    try:
        print(f"Loading model from {checkpoint_path}...")
        inference_engine = TransformerInferenceEngine(checkpoint_path=checkpoint_path)
        print("Inference engine loaded successfully.")
    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        # Exit if the model can't be loaded.
        exit()

@app.post("/generate", response_model=list[AccompanimentEvent])
async def generate(note_events: list[NoteEvent]):
    """
    Receives note events from the client and returns generated accompaniment.
    """
    if not inference_engine:
        return {"error": "Inference engine not initialized"}, 503

    # The generate_accompaniment function is stateful and expects a list of dicts.
    user_note_events_dict = [event.dict() for event in note_events]
    
    # The original tick_loop managed a tick_count. For a simple stateless API,
    # we can pass a dummy value. The client is not time-synced.
    current_tick_count = int(time.time())

    accompaniment = inference_engine.generate_accompaniment(
        user_note_events=user_note_events_dict,
        current_tick_count=current_tick_count
    )
    
    return accompaniment