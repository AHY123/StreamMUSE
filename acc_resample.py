#!/usr/bin/env python3
"""
acc_resample.py

将 MIDI 文件的 ticks_per_beat (TPB) 重采样到新的值。
支持单文件和目录（可选递归）处理，保留相对目录结构或在同目录下添加后缀。
"""

import argparse
import sys
from pathlib import Path
import mido


def resample_midi_file(input_path, output_path, new_tpb):
    """将 MIDI 文件的 ticks_per_beat 转换为 new_tpb 并保存。

    input_path: path-like to source .mid file
    output_path: path-like for destination .mid file (will be created/overwritten)
    new_tpb: int ticks per beat for output
    """
    original_midi = mido.MidiFile(input_path)
    original_tpb = original_midi.ticks_per_beat

    if original_tpb == new_tpb:
        # 如果不需要转换，直接保存（复制语义）
        original_midi.save(output_path)
        print(f"Skipped conversion (same TPB={original_tpb}) for: {input_path} -> copied to {output_path}")
        return

    conversion_factor = float(new_tpb) / float(original_tpb)

    new_midi = mido.MidiFile()
    new_midi.ticks_per_beat = int(new_tpb)

    for track in original_midi.tracks:
        new_track = mido.MidiTrack()
        current_absolute_time_in_old_tpb = 0

        # 1. 第一遍：计算并存储所有事件在 NEW_TPB 下的绝对时间
        events_with_new_abs_time = []
        for msg in track:
            # 累加旧的 delta time 得到旧的绝对时间
            current_absolute_time_in_old_tpb += msg.time

            # 将旧的绝对时间转换为新的绝对时间 (取整)
            new_absolute_time = int(current_absolute_time_in_old_tpb * conversion_factor)

            # 存储消息和新的绝对时间。使用 copy 以避免修改原消息对象
            try:
                copy_msg = msg.copy(time=0)
            except Exception:
                # 部分消息类型可能不支持 copy(); 回退为浅拷贝
                import copy as _copy

                copy_msg = _copy.copy(msg)
                copy_msg.time = 0

            events_with_new_abs_time.append((new_absolute_time, copy_msg))

        # 2. 第二遍：根据新的绝对时间计算新的 delta time
        last_absolute_time = 0
        for new_abs_time, msg in events_with_new_abs_time:
            new_delta_time = new_abs_time - last_absolute_time
            # 确保非负
            if new_delta_time < 0:
                new_delta_time = 0
            msg.time = new_delta_time
            new_track.append(msg)
            last_absolute_time = new_abs_time

        new_midi.tracks.append(new_track)

    # 确保输出父目录存在
    out_parent = Path(output_path).parent
    out_parent.mkdir(parents=True, exist_ok=True)

    new_midi.save(output_path)
    print(f"Resampled {input_path} (TPB {original_tpb}) -> {output_path} (TPB {new_tpb})")


def process_file(src_path: Path, dest_path: Path, new_tpb: int, overwrite: bool = False):
    if dest_path.exists() and not overwrite:
        print(f"Destination exists, skipping (use --overwrite to force): {dest_path}")
        return
    resample_midi_file(str(src_path), str(dest_path), new_tpb)


def process_directory(
    src_dir: Path,
    dest_dir: Path,
    new_tpb: int,
    suffix: str = "_resampled",
    recursive: bool = False,
    overwrite: bool = False,
):
    """Process all .mid/.midi files in src_dir. Place outputs in dest_dir preserving relative structure.

    - If dest_dir is same as src_dir, outputs will be created next to sources using suffix.
    """
    pattern = "**/*.mid" if recursive else "*.mid"
    files = list(src_dir.glob(pattern))
    # Also include .midi
    midi_pattern = "**/*.midi" if recursive else "*.midi"
    files += list(src_dir.glob(midi_pattern))

    if not files:
        print(f"No .mid/.midi files found in {src_dir} (recursive={recursive})")
        return

    for file in files:
        # compute relative path to preserve structure
        rel = file.relative_to(src_dir)
        if dest_dir.resolve() == src_dir.resolve():
            # Same directory: append suffix before extension
            stem = file.stem + suffix
            out_file = dest_dir / file.with_name(stem + file.suffix).name
        else:
            out_file_parent = dest_dir / rel.parent
            out_file_parent.mkdir(parents=True, exist_ok=True)
            out_file = out_file_parent / (file.stem + suffix + file.suffix)

        process_file(file, out_file, new_tpb, overwrite=overwrite)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Resample MIDI files' ticks_per_beat (TPB). Supports single files or directories."
    )
    p.add_argument("input", help="Input file or directory")
    p.add_argument(
        "-o",
        "--output",
        help="Output file or directory. If omitted: for file input a sibling file with suffix is created; for dir input a new directory '<input>_resampled' is created.",
    )
    p.add_argument("-t", "--tpb", type=int, required=True, help="New ticks per beat (integer).")
    p.add_argument(
        "-r", "--recursive", action="store_true", help="When input is a directory, recurse into subdirectories"
    )
    p.add_argument("-w", "--overwrite", action="store_true", help="Overwrite existing output files")
    p.add_argument(
        "-s",
        "--suffix",
        default="",
        help="Suffix to add to filenames when writing next to sources (default: _resampled)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.input)
    new_tpb = args.tpb

    if not src.exists():
        print(f"Input does not exist: {src}")
        sys.exit(2)

    # Determine output behavior
    if args.output:
        out = Path(args.output)
    else:
        if src.is_file():
            # place next to input with suffix
            out = src.parent
        else:
            # create a sibling directory named <src>_resampled
            out = src.with_name(src.name + "_resampled")

    if src.is_file():
        # single file
        if out.exists() and out.is_dir():
            # user provided a directory -> create file inside
            out_file = out / (src.stem + args.suffix + src.suffix)
        elif out.exists() and out.is_file():
            out_file = out
        else:
            # out does not exist: decide whether it's intended as file or directory
            if str(out).endswith((".mid", ".midi")):
                out_file = out
            else:
                # treat as directory (create if needed)
                out.mkdir(parents=True, exist_ok=True)
                out_file = out / (src.stem + args.suffix + src.suffix)

        process_file(src, out_file, new_tpb, overwrite=args.overwrite)

    else:
        # directory
        if out.exists() and out.is_file():
            print(f"Output must be a directory when input is a directory: {out}")
            sys.exit(2)
        # if output doesn't exist, create it
        out.mkdir(parents=True, exist_ok=True)
        process_directory(src, out, new_tpb, suffix=args.suffix, recursive=args.recursive, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
