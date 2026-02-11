#!/usr/bin/env python3
"""Detect the musical key of MIDI files using music21."""

from __future__ import annotations

import argparse
from pathlib import Path

from music21 import converter


def detect_key(midi_path: Path):
    score = converter.parse(str(midi_path))
    key = score.analyze("key")
    tonic = key.tonic.name
    mode = key.mode
    confidence = getattr(key, "correlationCoefficient", 0.0)
    tonic = tonic.replace("-", "b")  # Replace '-' with 'b' for flats
    return tonic, mode, confidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Key detection for MIDI files using music21"
    )
    parser.add_argument("midi", nargs="+", type=Path, help="MIDI file(s) to analyze")
    args = parser.parse_args()

    for midi_path in args.midi:
        try:
            tonic, mode, confidence = detect_key(midi_path)
            print(f"{midi_path}: {tonic} {mode} (confidence {confidence:.3f})")
        except Exception as exc:
            print(f"{midi_path}: ERROR {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
