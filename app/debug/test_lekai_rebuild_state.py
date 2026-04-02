import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "safetensors" not in sys.modules:
    safetensors_module = types.ModuleType("safetensors")
    safetensors_torch_module = types.ModuleType("safetensors.torch")
    safetensors_torch_module.load_file = lambda checkpoint_path: {}
    safetensors_module.torch = safetensors_torch_module
    sys.modules["safetensors"] = safetensors_module
    sys.modules["safetensors.torch"] = safetensors_torch_module

if "transformers" not in sys.modules:
    transformers_module = types.ModuleType("transformers")

    class DummyLlamaConfig:
        def __init__(self, *args, **kwargs):
            pass

    transformers_module.LlamaConfig = DummyLlamaConfig
    sys.modules["transformers"] = transformers_module

if "lekai_model.config" not in sys.modules:
    config_module = types.ModuleType("lekai_model.config")

    class DummyModelConfig:
        patch_h = 2
        patch_w = 2
        vocab_size = 512
        hidden_size = 16
        num_hidden_layers = 1
        num_attention_heads = 1
        intermediate_size = 32
        max_position_embeddings = 64
        pad_token_id = 99
        bos_token_id = 1
        eos_token_id = 2
        rope_theta = 10000
        dropout = 0.0
        bpm_offset_id = 10
        time_sig_offset_id = 20
        bar_token_id = 30

    config_module.ModelConfig = DummyModelConfig
    sys.modules["lekai_model.config"] = config_module

if "lekai_model.model" not in sys.modules:
    model_module = types.ModuleType("lekai_model.model")

    class DummyPianoLLaMA:
        def __init__(self, *args, **kwargs):
            pass

        def load_state_dict(self, *args, **kwargs):
            return None

        def cuda(self):
            return self

        def eval(self):
            return self

    model_module.PianoLLaMA = DummyPianoLLaMA
    sys.modules["lekai_model.model"] = model_module

if "lekai_model.my_tokenizer" not in sys.modules:
    tokenizer_module = types.ModuleType("lekai_model.my_tokenizer")

    class DummyPianoRollTokenizer:
        def __init__(self, *args, **kwargs):
            self.end_marker_part0 = 170
            self.end_marker_part1 = 171

    tokenizer_module.PianoRollTokenizer = DummyPianoRollTokenizer
    sys.modules["lekai_model.my_tokenizer"] = tokenizer_module

if "lekai_model.PianoDataset" not in sys.modules:
    dataset_module = types.ModuleType("lekai_model.PianoDataset")
    dataset_module.encode_bpm = lambda bpm: int(bpm)
    sys.modules["lekai_model.PianoDataset"] = dataset_module

if "lekai_model.generation_utils" not in sys.modules:
    generation_module = types.ModuleType("lekai_model.generation_utils")
    generation_module.sample_token = lambda *args, **kwargs: torch.tensor([[0]])
    sys.modules["lekai_model.generation_utils"] = generation_module

from app.inference_engines.transformer_engine_lekai import InferenceEngineLekai
from lekai_model.MidiConverter import MidiConverter


class _DummyConfig:
    bos_token_id = 1
    bpm_offset_id = 10
    time_sig_offset_id = 20
    bar_token_id = 30
    pad_token_id = 99
    end_marker_part0 = 170
    end_marker_part1 = 171


class _DummyTokenizer:
    end_marker_part0 = 170
    end_marker_part1 = 171


def make_lightweight_engine(injection_offset=0):
    engine = InferenceEngineLekai.__new__(InferenceEngineLekai)
    engine.config = _DummyConfig()
    engine.tokenizer = _DummyTokenizer()
    engine.ticks_per_beat = 4
    engine.midi_converter = MidiConverter(ticks_per_beat=4)
    engine.melody_event_history = []
    engine.accompaniment_history = []
    engine.counter = 0
    engine.injection_offset_ticks = injection_offset
    engine.inference_mode = "sliding_window"
    engine.past_key_values = None
    engine.last_generated_beat = -1
    engine.last_generated_acc_tokens = None
    engine._active_melody_pitches = set()
    engine._active_acc_pitches = set()
    return engine


def pitch_channels(pr, pitch):
    pitch_idx = int(pitch) - 21
    return {
        "sustain": pr[0, pitch_idx, :].astype(int).tolist(),
        "onset": pr[1, pitch_idx, :].astype(int).tolist(),
    }


def install_rebuild_stubs(engine):
    engine.captured_melody_history = None
    engine.captured_beat_ranges = []
    engine.captured_mel_prs = []

    def fake_get_tokens_for_beat(self, notes, beat_idx, end_marker_id):
        return torch.tensor([101], dtype=torch.long)

    def fake_get_tokens_for_beat_pianoroll(self, pianoroll, end_marker_id):
        self.captured_mel_prs.append(np.array(pianoroll, copy=True))
        return torch.tensor([102], dtype=torch.long)

    def fake_generate_tokens(self, input_ids, past_key_values=None):
        return [], None

    original_get_mel = engine._get_mel_pianoroll_for_beat

    def wrapped_get_mel(self, beat_start_tick, beat_end_tick):
        if self.captured_melody_history is None:
            self.captured_melody_history = [dict(e) for e in self.melody_event_history]
        self.captured_beat_ranges.append((int(beat_start_tick), int(beat_end_tick)))
        return original_get_mel(beat_start_tick, beat_end_tick)

    engine._get_tokens_for_beat = types.MethodType(fake_get_tokens_for_beat, engine)
    engine._get_tokens_for_beat_pianoroll = types.MethodType(
        fake_get_tokens_for_beat_pianoroll, engine
    )
    engine._generate_tokens = types.MethodType(fake_generate_tokens, engine)
    engine._get_mel_pianoroll_for_beat = types.MethodType(wrapped_get_mel, engine)
    return engine


class LekaiRebuildStateTests(unittest.TestCase):
    def test_normalize_melody_input_preserves_events_and_applies_offset(self):
        engine = make_lightweight_engine(injection_offset=8)
        melody_notes = [
            {"type": "note_off", "pitch": "60", "tick": "4"},
            {"type": "note_on", "pitch": 62, "tick": 4},
            {"type": "ignore_me", "pitch": 70, "tick": 5},
            {"type": "note_off", "pitch": 62, "tick": 7},
        ]

        normalized = engine._normalize_melody_input(melody_notes, generation_start_tick=12)

        self.assertEqual(
            normalized,
            [
                {"type": "note_off", "pitch": 60, "tick": 12},
                {"type": "note_on", "pitch": 62, "tick": 12},
                {"type": "note_off", "pitch": 62, "tick": 15},
            ],
        )

    def test_generate_accompaniment_uses_expected_normalized_history_before_rebuild(self):
        engine = install_rebuild_stubs(make_lightweight_engine(injection_offset=0))
        melody_notes = [
            {"type": "note_on", "pitch": 60, "tick": 0},
            {"type": "note_off", "pitch": 60, "tick": 2},
            {"type": "note_on", "pitch": 62, "tick": 2},
            {"type": "note_off", "pitch": 62, "tick": 4},
        ]

        generated_events, *_ = engine.generate_accompaniment(
            melody_notes=melody_notes,
            generation_start_tick=4,
            bpm=120,
            time_sig=(4, 4),
        )

        self.assertEqual(generated_events, [])
        self.assertEqual(
            engine.captured_melody_history,
            [
                {"type": "note_on", "pitch": 60, "tick": 0},
                {"type": "note_off", "pitch": 60, "tick": 2},
                {"type": "note_on", "pitch": 62, "tick": 2},
                {"type": "note_off", "pitch": 62, "tick": 4},
            ],
        )
        self.assertEqual(engine.melody_event_history, engine.captured_melody_history)
        self.assertEqual(engine.captured_beat_ranges, [(0, 4)])

    def test_generate_accompaniment_applies_injection_offset_before_rebuild(self):
        engine = install_rebuild_stubs(make_lightweight_engine(injection_offset=16))
        melody_notes = [
            {"type": "note_on", "pitch": 65, "tick": 0},
            {"type": "note_off", "pitch": 65, "tick": 2},
            {"type": "note_on", "pitch": 67, "tick": 2},
            {"type": "note_off", "pitch": 67, "tick": 4},
        ]

        generated_events, *_ = engine.generate_accompaniment(
            melody_notes=melody_notes,
            generation_start_tick=20,
            bpm=120,
            time_sig=(4, 4),
        )

        self.assertEqual(generated_events, [])
        self.assertEqual(
            engine.captured_melody_history,
            [
                {"type": "note_on", "pitch": 65, "tick": 16},
                {"type": "note_off", "pitch": 65, "tick": 18},
                {"type": "note_on", "pitch": 67, "tick": 18},
                {"type": "note_off", "pitch": 67, "tick": 20},
            ],
        )
        self.assertEqual(engine.melody_event_history, engine.captured_melody_history)
        self.assertEqual(len(engine.captured_beat_ranges), 9)
        self.assertEqual(engine.captured_beat_ranges[0], (0, 4))
        self.assertEqual(engine.captured_beat_ranges[-1], (32, 36))

    def test_same_tick_note_off_before_note_on_is_preserved_for_rebuild(self):
        engine = make_lightweight_engine()
        engine.melody_event_history = [
            {"type": "note_on", "pitch": 60, "tick": 2},
            {"type": "note_off", "pitch": 60, "tick": 4},
            {"type": "note_on", "pitch": 60, "tick": 4},
        ]
        engine._active_melody_pitches = {60}

        pr = engine._get_mel_pianoroll_for_beat(4, 8)

        self.assertEqual(
            pitch_channels(pr, 60),
            {
                "sustain": [1, 1, 1, 1],
                "onset": [1, 0, 0, 0],
            },
        )

    def test_compute_active_melody_pitches_at_tick_preserves_true_cross_window_sustain(self):
        engine = make_lightweight_engine()
        engine.melody_event_history = [
            {"type": "note_on", "pitch": 60, "tick": 2},
            {"type": "note_off", "pitch": 60, "tick": 6},
            {"type": "note_on", "pitch": 64, "tick": 6},
            {"type": "note_off", "pitch": 64, "tick": 8},
        ]

        self.assertEqual(engine._compute_active_melody_pitches_at_tick(0), set())
        self.assertEqual(engine._compute_active_melody_pitches_at_tick(4), {60})
        self.assertEqual(engine._compute_active_melody_pitches_at_tick(6), {60})
        self.assertEqual(engine._compute_active_melody_pitches_at_tick(8), {64})
        self.assertEqual(engine._compute_active_melody_pitches_at_tick(9), set())

    def test_generate_accompaniment_rebuild_boundary_recomputes_clean_carry_in(self):
        melody_history = [
            {"type": "note_on", "pitch": 60, "tick": 2},
            {"type": "note_off", "pitch": 60, "tick": 4},
            {"type": "note_on", "pitch": 62, "tick": 6},
            {"type": "note_off", "pitch": 62, "tick": 8},
        ]

        clean_engine = install_rebuild_stubs(make_lightweight_engine())
        clean_engine.melody_event_history = [dict(e) for e in melody_history]

        stale_engine = install_rebuild_stubs(make_lightweight_engine())
        stale_engine.melody_event_history = [dict(e) for e in melody_history]
        stale_engine._active_melody_pitches = {60}

        clean_result, *_ = clean_engine.generate_accompaniment(
            melody_notes=[],
            generation_start_tick=8,
            bpm=120,
            time_sig=(4, 4),
        )
        stale_result, *_ = stale_engine.generate_accompaniment(
            melody_notes=[],
            generation_start_tick=8,
            bpm=120,
            time_sig=(4, 4),
        )

        self.assertEqual(clean_result, [])
        self.assertEqual(stale_result, [])
        self.assertEqual(clean_engine.melody_event_history, stale_engine.melody_event_history)
        self.assertEqual(clean_engine.captured_melody_history, stale_engine.captured_melody_history)
        self.assertEqual(clean_engine.captured_beat_ranges, [(0, 4), (4, 8)])
        self.assertEqual(stale_engine.captured_beat_ranges, [(0, 4), (4, 8)])
        self.assertEqual(len(clean_engine.captured_mel_prs), 2)
        self.assertEqual(len(stale_engine.captured_mel_prs), 2)

        clean_first = clean_engine.captured_mel_prs[0]
        stale_first = stale_engine.captured_mel_prs[0]
        clean_second = clean_engine.captured_mel_prs[1]
        stale_second = stale_engine.captured_mel_prs[1]

        self.assertEqual(clean_first.tolist(), stale_first.tolist())
        self.assertEqual(clean_second.tolist(), stale_second.tolist())
        self.assertEqual(pitch_channels(clean_first, 60)["sustain"], [0, 0, 1, 1])
        self.assertEqual(pitch_channels(stale_first, 60)["sustain"], [0, 0, 1, 1])

    def test_generate_accompaniment_rebuild_boundary_after_window_shift_recomputes_clean_carry_in(self):
        melody_history = [
            {"type": "note_on", "pitch": 60, "tick": 6},
            {"type": "note_off", "pitch": 60, "tick": 8},
            {"type": "note_on", "pitch": 62, "tick": 10},
            {"type": "note_off", "pitch": 62, "tick": 12},
        ]

        clean_engine = install_rebuild_stubs(make_lightweight_engine())
        clean_engine.melody_event_history = [dict(e) for e in melody_history]

        stale_engine = install_rebuild_stubs(make_lightweight_engine())
        stale_engine.melody_event_history = [dict(e) for e in melody_history]
        stale_engine._active_melody_pitches = {60}

        clean_result, *_ = clean_engine.generate_accompaniment(
            melody_notes=[],
            generation_start_tick=132,
            bpm=120,
            time_sig=(4, 4),
        )
        stale_result, *_ = stale_engine.generate_accompaniment(
            melody_notes=[],
            generation_start_tick=132,
            bpm=120,
            time_sig=(4, 4),
        )

        self.assertEqual(clean_result, [])
        self.assertEqual(stale_result, [])
        self.assertEqual(clean_engine.melody_event_history, stale_engine.melody_event_history)
        self.assertEqual(clean_engine.captured_melody_history, stale_engine.captured_melody_history)
        self.assertEqual(clean_engine.captured_beat_ranges[0], (4, 8))
        self.assertEqual(stale_engine.captured_beat_ranges[0], (4, 8))
        self.assertEqual(len(clean_engine.captured_mel_prs), 32)
        self.assertEqual(len(stale_engine.captured_mel_prs), 32)

        clean_first = clean_engine.captured_mel_prs[0]
        stale_first = stale_engine.captured_mel_prs[0]

        self.assertEqual(clean_first.tolist(), stale_first.tolist())
        self.assertEqual(pitch_channels(clean_first, 60)["sustain"], [0, 0, 1, 1])
        self.assertEqual(pitch_channels(stale_first, 60)["sustain"], [0, 0, 1, 1])

        for clean_pr, stale_pr in zip(
            clean_engine.captured_mel_prs[1:], stale_engine.captured_mel_prs[1:]
        ):
            self.assertEqual(clean_pr.tolist(), stale_pr.tolist())

    def test_generate_accompaniment_rebuild_boundary_preserves_true_cross_window_sustain(self):
        melody_history = [
            {"type": "note_on", "pitch": 60, "tick": 2},
            {"type": "note_off", "pitch": 60, "tick": 6},
            {"type": "note_on", "pitch": 64, "tick": 6},
            {"type": "note_off", "pitch": 64, "tick": 8},
        ]

        clean_engine = install_rebuild_stubs(make_lightweight_engine())
        clean_engine.melody_event_history = [dict(e) for e in melody_history]

        stale_engine = install_rebuild_stubs(make_lightweight_engine())
        stale_engine.melody_event_history = [dict(e) for e in melody_history]
        stale_engine._active_melody_pitches = {72}

        clean_result, *_ = clean_engine.generate_accompaniment(
            melody_notes=[],
            generation_start_tick=132,
            bpm=120,
            time_sig=(4, 4),
        )
        stale_result, *_ = stale_engine.generate_accompaniment(
            melody_notes=[],
            generation_start_tick=132,
            bpm=120,
            time_sig=(4, 4),
        )

        self.assertEqual(clean_result, [])
        self.assertEqual(stale_result, [])
        self.assertEqual(clean_engine.captured_beat_ranges[0], (4, 8))
        self.assertEqual(stale_engine.captured_beat_ranges[0], (4, 8))

        clean_first = clean_engine.captured_mel_prs[0]
        stale_first = stale_engine.captured_mel_prs[0]

        self.assertEqual(clean_first.tolist(), stale_first.tolist())
        self.assertEqual(
            pitch_channels(clean_first, 60),
            {
                "sustain": [1, 1, 0, 0],
                "onset": [0, 0, 0, 0],
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
