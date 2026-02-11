# ...existing code...
#!/usr/bin/env python3
"""Notebook -> script: run key detection and transpose steps that were in the notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

melody_file_path = "input/mel/001.mid"
accompaniment_file_path = "acc-poly-pattern_resampled/c_major_poly_stride.mid"


def _display_name_from_outpath(p: Path) -> str:
    """Derive a human-readable key display from the transposed filename."""
    # Expect filename like: <stem>_to_{tonic_stub}_{mode}.mid
    stem = p.stem
    if "_to_" not in stem:
        return stem
    after = stem.split("_to_", 1)[1]
    # split last '_' to separate tonic_stub and mode
    if "_" in after:
        tonic_stub, mode = after.rsplit("_", 1)
    else:
        tonic_stub, mode = after, ""
    tonic = tonic_stub.replace("sharp", "#").replace("flat", "b")
    mode_display = mode if mode else "major"
    return f"{tonic} {mode_display}"


def transpose_in_memory_v1(midi_path: Path | str, keys: Iterable[str]):
    """Use app.acc_transpose.transpose_c_major.transpose_midi_file — write to a temp dir, run detect_key on each file, delete files."""
    from app.acc_transpose.transpose_c_major import transpose_midi_file
    from app.acc_transpose.music21_key_detect import detect_key
    import tempfile
    import os
    from pathlib import Path

    results: list[tuple[Path, str]] = []
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        written = transpose_midi_file(midi_path, keys, output_dir=out_dir)
        print(f"[v1] Written {len(written)} files to temporary dir; detecting keys:")
        for p in written:
            try:
                detected = detect_key(str(p))
            except Exception as e:
                print(f"[v1] Failed to detect key for {p.name}: {e}")
                detected = None
            display = _display_name_from_outpath(p)
            print(f"[v1] {p.name} -> transposed: {display} | detected: {detected}")
            results.append((p, detected))
            try:
                os.unlink(p)
            except Exception:
                pass
    return results


def transpose_in_memory_v2(midi_path: Path | str, keys: Iterable[str]):
    """Use app.acc_transpose.transpose_c_major2.transpose_midi_file — write to a temp dir, run detect_key on each file, delete files."""
    from app.acc_transpose.transpose_c_major2 import transpose_midi_file
    from app.acc_transpose.music21_key_detect import detect_key
    import tempfile
    import os
    from pathlib import Path

    results: list[tuple[Path, str]] = []
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        written = transpose_midi_file(midi_path, keys, output_dir=out_dir)
        print(f"[v2] Written {len(written)} files to temporary dir; detecting keys:")
        for p in written:
            try:
                detected = detect_key(str(p))
            except Exception as e:
                print(f"[v2] Failed to detect key for {p.name}: {e}")
                detected = None
            display = _display_name_from_outpath(p)
            print(f"[v2] {p.name} -> transposed: {display} | detected: {detected}")
            results.append((p, detected))
            try:
                os.unlink(p)
            except Exception:
                pass
    return results


def main() -> int:
    # NOTE: Do not import low-level transpose helpers here per requirement; local imports are used below.

    key = [
        # 升号调和自然调 (Sharps and Naturals)
        "C",
        "Am",
        "G",
        "Em",
        "D",
        "Bm",
        "A",
        "F#m",
        "E",
        "C#m",
        "B",
        "G#m",
        "F#",
        "D#m",
        "C#",
        "A#m",
        # 降号调 (Flats)
        "F",
        "Dm",
        "Bb",
        "Gm",
        "Eb",
        "Cm",
        "Ab",
        "Fm",
        "Db",
        "Bbm",
        "Gb",
        "Ebm",
        "Cb",
        "Abm",
    ]

    # -- Transposition via transpose_midi_file only (writes to temp dir, detect_key run on files) --
    print("\nRunning transpositions using transpose_midi_file (temp files written, detected, then removed):")
    res1 = transpose_in_memory_v1(accompaniment_file_path, key)
    res2 = transpose_in_memory_v2(accompaniment_file_path, key)

    # Summarize counts so it's clear which function produced which results
    print(
        f"\nSummary: transpose_c_major produced {len(res1)} transpositions; transpose_c_major2 produced {len(res2)} transpositions."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ...existing code...
