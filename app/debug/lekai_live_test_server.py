import time

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class MelodyNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int


class InferenceRequest(BaseModel):
    melody_notes: list[MelodyNoteEvent]
    generation_start_tick: int
    client_request_send_time: float = 0.0
    generation_length_frames: int | None = None


class AccompanimentNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int


class Timings(BaseModel):
    request_arrival_time: float
    response_output_time: float
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float


class AccompanimentResponse(BaseModel):
    accompaniment: list[AccompanimentNoteEvent]
    timings: Timings
    generation_start_tick: int


app = FastAPI(title="Lekai Live Replay Test Server")


@app.post("/generate_accompaniment", response_model=AccompanimentResponse)
async def generate_accompaniment(request: InferenceRequest):
    request_arrival_time = time.perf_counter()
    preprocess_start_time = request_arrival_time
    inference_start_time = request_arrival_time
    inference_end_time = time.perf_counter()
    postprocess_start_time = inference_end_time
    response_output_time = time.perf_counter()
    return AccompanimentResponse(
        accompaniment=[],
        timings=Timings(
            request_arrival_time=request_arrival_time,
            response_output_time=response_output_time,
            preprocess_start_time=preprocess_start_time,
            inference_start_time=inference_start_time,
            inference_end_time=inference_end_time,
            postprocess_start_time=postprocess_start_time,
        ),
        generation_start_tick=request.generation_start_tick,
    )


@app.post("/clear_history")
async def clear_history():
    return {"message": "History cleared successfully"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
