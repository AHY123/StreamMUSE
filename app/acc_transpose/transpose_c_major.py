#!/usr/bin/env python3
"""Transpose a C-major MIDI file into user-specified major or minor keys."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import copy
import pretty_midi

NOTE_BASE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


@dataclass(frozen=True)
class TargetKey:
    """Normalized representation of a requested key."""

    tonic: str  # e.g., C, F#, Bb
    mode: str  # "major" or "minor"
    original: str

    @property
    def semitone_shift(self) -> int:
        return (tonic_to_pc(self.tonic) - NOTE_BASE["C"]) % 12

    def filename_stub(self) -> str:
        tonic_stub = self.tonic.replace("#", "sharp").replace("b", "flat")
        return f"{tonic_stub}_{self.mode}"

    def display_name(self) -> str:
        return f"{self.tonic} {self.mode}"


def transpose_pm(pm: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
    """Return a deep copy with pitches shifted by ``semitones`` only."""
    # print(semitones)

    transposed = copy.deepcopy(pm)
    for inst in transposed.instruments:
        for note in inst.notes:
            new_pitch = note.pitch + semitones
            if not 0 <= new_pitch <= 127:
                raise ValueError(f"Transposition of {semitones} semitones drives note {note.pitch} out of MIDI range")
            note.pitch = new_pitch
    return transposed


def tonic_to_pc(tonic: str) -> int:
    letter = tonic[0]
    if letter not in NOTE_BASE:
        raise ValueError(f"Unsupported tonic letter: {tonic}")
    semitone = NOTE_BASE[letter]
    for accidental in tonic[1:]:
        if accidental == "#":
            semitone += 1
        elif accidental in {"b", "B"}:
            semitone -= 1
        else:
            raise ValueError(f"Unsupported accidental '{accidental}' in tonic '{tonic}'")
    return semitone % 12


def parse_keys(values: Iterable[str]) -> list[TargetKey]:
    parsed: list[TargetKey] = []
    for raw in values:
        if not raw or not raw.strip():
            raise ValueError("Key names must be non-empty")
        normalized = _normalize_key_name(raw)

        letter_part = normalized[0]
        if letter_part.upper() not in NOTE_BASE:
            raise ValueError(f"Unknown key letter in '{raw}'")
        idx = 1
        accidental_part: list[str] = []
        while idx < len(normalized) and normalized[idx] in {"#", "b", "B"}:
            accidental = "#" if normalized[idx] == "#" else "b"
            accidental_part.append(accidental)
            idx += 1
        tonic = letter_part.upper() + "".join(accidental_part)

        mode_fragment = re.sub(r"[\-_.\s]+", "", normalized[idx:])
        if not mode_fragment:
            mode = "major"
        else:
            lowered = mode_fragment.lower()
            if lowered in {"m", "min", "minor"}:
                mode = "major" if mode_fragment == "M" else "minor"
            elif lowered in {"maj", "major"}:
                mode = "major"
            else:
                raise ValueError(f"Unrecognized mode in key '{raw}'")

        parsed.append(TargetKey(tonic=tonic, mode=mode, original=raw.strip()))
    return parsed


def _normalize_key_name(raw: str) -> str:
    text = raw.strip()
    text = text.replace("♭", "b").replace("♯", "#")
    text = re.sub(r"(?i)flat", "b", text)
    text = re.sub(r"(?i)sharp", "#", text)
    return text


def transpose_midi_file(midi_path: Path | str, keys: Iterable[str], output_dir: Path | None = None) -> list[Path]:
    """Transpose the given MIDI file (assumed C major) into the requested keys.

    Args:
        midi_path: Path or string to the input MIDI file.
        keys: Iterable of key strings (e.g., ["G", "Am", "F#"]).
        output_dir: Optional directory to write transposed files. Defaults to input file's directory.

    Returns:
        List of pathlib.Path objects for the written transposed MIDI files.
    """
    midi_path = Path(midi_path).expanduser().resolve()
    out_dir = (Path(output_dir) if output_dir is not None else midi_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    stem = midi_path.stem
    written: list[Path] = []

    for key in parse_keys(keys):
        transposed = transpose_pm(pm, key.semitone_shift)
        out_path = out_dir / f"{stem}_to_{key.filename_stub()}.mid"
        transposed.write(str(out_path))
        written.append(out_path)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transpose a C-major MIDI into one or more target major or minor keys."
    )
    parser.add_argument("midi", type=Path, help="Input MIDI file (assumed C major).")
    parser.add_argument(
        "--keys",
        nargs="+",
        required=True,
        help="Target keys (e.g., G D Am F#m). Enharmonics like Bb/C# are accepted.",
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
        transposed = transpose_pm(pm, key.semitone_shift)
        out_path = output_dir / f"{stem}_to_{key.filename_stub()}.mid"
        transposed.write(str(out_path))
        print(f"Wrote {out_path} ({key.display_name()})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
