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
        """Calculate the chromatic shift needed to move from C tonic to the target tonic."""
        return (tonic_to_pc(self.tonic) - NOTE_BASE["C"]) % 12

    def filename_stub(self) -> str:
        """Generate a clean stub for the output filename."""
        tonic_stub = self.tonic.replace("#", "sharp").replace("b", "flat")
        return f"{tonic_stub}_{self.mode}"

    def display_name(self) -> str:
        """Return the formatted key name."""
        return f"{self.tonic} {self.mode}"


def tonic_to_pc(tonic: str) -> int:
    """Convert tonic name (e.g., 'G#', 'Bb') to pitch class (0-11)."""
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


def transpose_pm(pm: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
    """
    Return a deep copy with all pitches shifted by 'semitones' (chromatic shift).
    This handles the change of the tonic (e.g., C -> G).
    """
    # print(f"Applying chromatic tonic shift: {semitones} semitones")

    transposed = copy.deepcopy(pm)
    for inst in transposed.instruments:
        for note in inst.notes:
            new_pitch = note.pitch + semitones
            if not 0 <= new_pitch <= 127:
                raise ValueError(f"Transposition of {semitones} semitones drives note {note.pitch} out of MIDI range")
            note.pitch = new_pitch
    return transposed


def adjust_mode_if_minor(
    pm: pretty_midi.PrettyMIDI,
    target_key: TargetKey,
    semitones_applied: int,
) -> pretty_midi.PrettyMIDI:
    # ... (代码省略)

    if target_key.mode == "major":
        return pm

    # 修正：只降 III 和 VI 级 (E, A)，VII 级保持不变 (B)，以形成和声小调。
    original_c_major_pcs_to_lower = {4, 9}  # E, A

    # 检查点 2: 打印 pcs_to_lower 集合 (现在是 {4, 9})
    print(f"--- DEBUG: C Major PCs to lower (Harmonic Minor): {original_c_major_pcs_to_lower}")

    adjusted_pm = copy.deepcopy(pm)

    # ... (其余逻辑不变)

    notes_adjusted = 0

    for inst in adjusted_pm.instruments:
        for note in inst.notes:
            original_c_major_pc = (note.pitch - semitones_applied) % 12

            if original_c_major_pc in original_c_major_pcs_to_lower:
                note.pitch -= 1
                notes_adjusted += 1
                # 💥 检查点 3: 打印被调整的音符信息
                # print(f"--- DEBUG: Adjusted Note: Original PC {original_c_major_pc}, New Pitch {note.pitch}")

    print(f"--- DEBUG: Total Notes Adjusted: {notes_adjusted}")

    # 💥 检查点 4: 如果 notes_adjusted 为 0，说明逻辑失败
    if target_key.mode == "minor" and notes_adjusted == 0:
        print("--- CRITICAL ERROR: Mode adjustment for minor key failed to modify any notes. ---")

    return adjusted_pm

# def adjust_mode_if_minor(
#     pm: pretty_midi.PrettyMIDI,
#     target_key: TargetKey,
#     tonic_pc: int,
# ) -> pretty_midi.PrettyMIDI:
#     """
#     If the target mode is minor, adjust the 3rd, 6th, and 7th scale degrees
#     (relative to the new tonic) to create the natural minor sound.

#     This function performs the mechanical Major-to-Minor mode conversion.
#     It assumes 'pm' is already chromatically transposed to the target tonic.
#     """
#     if target_key.mode == "major":
#         # No mode adjustment needed for major keys
#         return pm

#     # print(f"Applying mode adjustment for {target_key.display_name()} (Natural Minor)")

#     # 1. Identify the pitch classes that need to be lowered by one semitone.
#     # These are the 3rd, 6th, and 7th degrees of the Major scale built on the tonic.
#     # Major scale degrees relative to tonic:
#     # I=0, II=2, III=4 (needs lowering), IV=5, V=7, VI=9 (needs lowering), VII=11 (needs lowering)
#     major_semitone_offsets_to_lower = {4, 9, 11}

#     # Calculate the exact pitch classes (0-11) that need to be lowered for the target tonic
#     pcs_to_lower = {(tonic_pc + offset) % 12 for offset in major_semitone_offsets_to_lower}
#     # print(f"Pitch classes to lower for mode adjustment: {pcs_to_lower}")
#     # Deep copy before modification (using copy of the already chromatically transposed MIDI)
#     adjusted_pm = copy.deepcopy(pm)

#     # 2. Apply the mode adjustment
#     for inst in adjusted_pm.instruments:
#         for note in inst.notes:
#             note_pc = note.pitch % 12

#             if note_pc in pcs_to_lower:
#                 # Lower the pitch by one semitone (e.g., E -> Eb, A -> Ab, B -> Bb)
#                 note.pitch -= 1
#                 if not 0 <= note.pitch <= 127:
#                     # This check is usually redundant if the chromatic shift was successful
#                     raise ValueError(f"Mode adjustment drove note {note.pitch + 1} out of MIDI range.")

#     return adjusted_pm


def parse_keys(values: Iterable[str]) -> list[TargetKey]:
    """Parse input key strings (e.g., 'G', 'Am', 'F#m') into TargetKey objects."""
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
    """Replace unicode and 'sharp'/'flat' with standard '#'/'b'."""
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

    for target_key in parse_keys(keys):
        # 1. Chromatic Tonic Transposition (C Major -> Target Tonic Major)
        transposed_pm = transpose_pm(pm, target_key.semitone_shift)

        # 2. Mode Adjustment (Target Tonic Major -> Target Tonic Minor, if requested)
        final_pm = adjust_mode_if_minor(transposed_pm, target_key, target_key.semitone_shift)

        out_path = out_dir / f"{stem}_to_{target_key.filename_stub()}.mid"
        final_pm.write(str(out_path))
        written.append(out_path)

    return written


def main() -> int:
    """Main function to parse arguments and execute transposition."""
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
    for target_key in parse_keys(args.keys):
        # 1. Chromatic Tonic Transposition
        transposed_pm = transpose_pm(pm, target_key.semitone_shift)

        # 2. Mode Adjustment
        final_pm = adjust_mode_if_minor(transposed_pm, target_key, tonic_to_pc(target_key.tonic))

        out_path = output_dir / f"{stem}_to_{target_key.filename_stub()}.mid"
        final_pm.write(str(out_path))
        print(f"Wrote {out_path} ({target_key.display_name()})")

    return 0


if __name__ == "__main__":
    # Note: This block raises SystemExit(main()) only when run as a script.
    # For interactive use (like in a notebook or terminal), you would call
    # transpose_midi_file directly after defining the functions.
    raise SystemExit(main())
