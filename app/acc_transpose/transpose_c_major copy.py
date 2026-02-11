#!/usr/bin/env python3
"""Transpose a C-major MIDI file into multiple target keys."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pretty_midi

KEY_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def transpose_pm(pm: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
    """Return a deep-copied PrettyMIDI transposed by ``semitones``."""

    transpose = pretty_midi.PrettyMIDI()
    transpose.instruments = []

    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for note in inst.notes:
            new_pitch = note.pitch + semitones
            if not 0 <= new_pitch <= 127:
                raise ValueError(
                    f"Transposition of {semitones} semitones drives note {note.pitch} out of MIDI range"
                )
            new_inst.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=new_pitch,
                    start=note.start,
                    end=note.end,
                )
            )
        for bend in inst.pitch_bends:
            new_inst.pitch_bends.append(pretty_midi.PitchBend(pitch=bend.pitch, time=bend.time))
        for control in inst.control_changes:
            new_inst.control_changes.append(
                pretty_midi.ControlChange(number=control.number, value=control.value, time=control.time)
            )
        transpose.instruments.append(new_inst)

    transpose.resolution = pm.resolution
    transpose.time_signature_changes = list(pm.time_signature_changes)
    transpose.key_signature_changes = list(pm.key_signature_changes)
    transpose.lyrics = list(pm.lyrics)
    transpose.metadata = pm.metadata
    return transpose


def parse_keys(values: Iterable[str]) -> list[str]:
    parsed = []
    for key in values:
        normalized = key.strip().replace('m', '').upper()
        if normalized not in KEY_TO_PC:
            raise ValueError(f"Unknown key name: {key}")
        parsed.append(normalized)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transpose a C-major MIDI into one or more target major keys."
    )
    parser.add_argument("midi", type=Path, help="Input MIDI file (assumed C major).")
    parser.add_argument(
        "--keys",
        nargs="+",
        required=True,
        help="Target keys (e.g., G D A F). Enharmonics like Bb/C# are accepted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save transposed files (defaults to the input directory).",
    )
    args = parser.parse_args()

    input_path = args.midi.expanduser().resolve()
    output_dir = (args.output_dir or input_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(str(input_path))

    stem = input_path.stem
    for key in parse_keys(args.keys):
        semitones = (KEY_TO_PC[key] - KEY_TO_PC['C']) % 12
        transposed = transpose_pm(pm, semitones)
        out_path = output_dir / f"{stem}_to_{key.replace('#', 'sharp').replace('B', 'B').replace('b', 'flat')}.mid"
        transposed.write(str(out_path))
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
