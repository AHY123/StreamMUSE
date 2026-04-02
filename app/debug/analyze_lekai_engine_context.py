import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lekai_model.MidiConverter import MidiConverter


def summarize_pitch(pr, pitch):
    pitch_idx = pitch - 21
    return {
        "sustain": pr[0, pitch_idx, :].astype(int).tolist(),
        "onset": pr[1, pitch_idx, :].astype(int).tolist(),
    }


class MinimalLekaiMelodyState:
    def __init__(self):
        self.midi_converter = MidiConverter(ticks_per_beat=4)
        self.melody_event_history = [
            {"type": "note_on", "pitch": 60, "tick": 2},
            {"type": "note_off", "pitch": 60, "tick": 4},
        ]
        self._active_melody_pitches = set()

    def get_mel_pianoroll_for_beat(self, beat_start_tick: int, beat_end_tick: int):
        pr = self.midi_converter.events_to_pianoroll(
            self.melody_event_history,
            start_tick=beat_start_tick,
            end_tick=beat_end_tick,
            active_pitches=self._active_melody_pitches,
        )
        beat_events = [
            e
            for e in self.melody_event_history
            if beat_start_tick <= e.get("tick", -1) < beat_end_tick
        ]
        beat_events.sort(
            key=lambda e: (
                int(e.get("tick", 0)),
                0 if e.get("type") == "note_off" else 1,
            )
        )
        for e in beat_events:
            pitch = int(e["pitch"])
            if e["type"] == "note_on":
                self._active_melody_pitches.add(pitch)
            elif e["type"] == "note_off":
                self._active_melody_pitches.discard(pitch)
        return pr


def main():
    print("Lekai engine context analysis")

    clean_engine = MinimalLekaiMelodyState()
    first_pass = clean_engine.get_mel_pianoroll_for_beat(0, 4)
    print("Clean first pass beat 0:")
    print(summarize_pitch(first_pass, 60))
    print({"active_after_first_pass": sorted(clean_engine._active_melody_pitches)})
    print()

    stale_engine = MinimalLekaiMelodyState()
    stale_engine._active_melody_pitches = {60}
    stale_pass = stale_engine.get_mel_pianoroll_for_beat(0, 4)
    print("Rebuild beat 0 with stale carry-in {60}:")
    print(summarize_pitch(stale_pass, 60))
    print({"active_after_stale_pass": sorted(stale_engine._active_melody_pitches)})
    print()

    print("Difference in sustain ticks:")
    print(
        {
            "clean_sustain_total": int(np.sum(first_pass[0] > 0)),
            "stale_sustain_total": int(np.sum(stale_pass[0] > 0)),
        }
    )


if __name__ == "__main__":
    main()
