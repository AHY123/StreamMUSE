# TECH_STACK.md — Verified Dependency Registry

> Verified from `pyproject.toml` and `requirements.txt` on 2026-03-23.
> Only lists dependencies that are actively imported in the codebase.

---

## Runtime: Core ML

| Package | Version | Usage |
|---------|---------|-------|
| `torch` | 2.7.1 | Core tensor operations, model inference |
| `torchaudio` | 2.7.1 | Audio processing utilities |
| `torchvision` | >=0.22.0 | Tensor utilities |
| `pytorch-lightning` | 2.5.1.post0 | Model training framework |
| `transformers` | editable (`./transformers/`) | **Vendored + modified** — custom RoFormer positional encoding |
| `safetensors` | >=0.5.3 | Model weight serialization (Lekai engine) |
| `numpy` | 2.2.6 | Array operations |
| `scipy` | (latest compatible) | Signal processing |

## Runtime: Music / MIDI

| Package | Version | Usage |
|---------|---------|-------|
| `mido` | >=1.3.3 | MIDI file I/O, MIDI port communication |
| `python-rtmidi` | >=1.5.8 | MIDI device access (hardware) |
| `pretty-midi` | >=0.2.10 | MIDI file read/write for recording |
| `music21` | >=9.7.1 | Music theory analysis (key detection fallback) |
| `miditok` | >=3.0.5 | MIDI tokenization (training pipeline) |
| `tokenizers` | >=0.21.1 | HuggingFace tokenizers (Lekai engine) |

## Runtime: Web / Networking

| Package | Version | Usage |
|---------|---------|-------|
| `fastapi` | >=0.116.1 | Inference server framework |
| `uvicorn` | >=0.35.0 | ASGI server for FastAPI |
| `pydantic` | >=2.11.7 | Request/response schema validation |
| `requests` | >=2.32.4 | HTTP client (client → server) |
| `websockets` | >=16.0 | WebSocket support (web_client.py) |

## Runtime: Input / Output

| Package | Version | Usage |
|---------|---------|-------|
| `pynput` | >=1.8.1 | Keyboard input capture |
| `pygame` | (client requirements) | Audio synthesis (client-only) |

## Runtime: Utilities

| Package | Version | Usage |
|---------|---------|-------|
| `pyyaml` | >=6.0.2 | Training config file parsing |
| `tqdm` | >=4.67.1 | Progress bars in preprocessing |
| `joblib` | >=1.5.1 | Parallel processing (dataset extraction) |
| `pandas` | (client requirements) | CSV log analysis |

## Development / Experiment Tracking

| Package | Version | Usage |
|---------|---------|-------|
| `tensorboard` | >=2.19.0 | Training visualization |
| `tensorboardx` | >=2.6.4 | TensorBoard writer |
| `wandb` | >=0.20.1 | Experiment tracking |

---

## Package Manager

- **Tool**: `uv` (fast Python package manager)
- **Command**: `uv run` to install + run
- **PyTorch Index**: `https://download.pytorch.org/whl/cu128` (CUDA 12.8)
- **Python Requirement**: `>=3.10`

## Client-Only Requirements

A separate `app/requirements_client.txt` exists for deployments that only run the client:
```
requests, mido, python-rtmidi, pygame, numpy, pretty_midi, pynput, tqdm, pandas, pyyaml
```

## Critical Notes

1. **`transformers/` is vendored and modified.** It is NOT the standard HuggingFace `transformers` package. It must be installed via `pip install -e ./transformers/`. Do not `pip install transformers` from PyPI — this will shadow the custom version.

2. **CUDA 12.8 required for GPU inference.** The PyTorch build is pinned to `cu128`. Running on CPU is possible but too slow for real-time generation.

3. **`lekai_model/` is a local package** with its own `config.py`, `model.py`, `my_tokenizer.py`. It is imported directly by `transformer_engine_lekai.py`. Not in pyproject.toml — it's part of the source tree.
