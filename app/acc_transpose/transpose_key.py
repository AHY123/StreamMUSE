#!/usr/bin/env python3
"""Transpose a MIDI file from a source key/mode to a target key/mode.
Can be run as a script or imported as a module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Literal, Tuple, Union

import copy
import pretty_midi

# --- 核心常量 ---
# 映射主音到音高类 (Pitch Class)
KEY_TO_PC: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}

# 小调相对于其相对大调主音的偏移量（通常为 -3）
MINOR_OFFSET: int = -3

# 类型别名
ModeType = Literal["major", "minor"]
KeyTonic = Literal[
    "C", "C#", "DB", "D", "D#", "EB", "E", "FB", "F", "E#", "F#", "GB", "G", "G#", "AB", "A", "A#", "BB", "B", "CB"
]
ParsedKey = Tuple[KeyTonic, ModeType]


# --- 1. 核心转调函数 ---
def transpose_pm(pm: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
    """对 PrettyMIDI 对象进行平移。"""

    transposed = copy.deepcopy(pm)
    for inst in transposed.instruments:
        for note in inst.notes:
            new_pitch = note.pitch + semitones
            if not 0 <= new_pitch <= 127:
                raise ValueError(f"Transposition of {semitones} semitones drives note {note.pitch} out of MIDI range")
            note.pitch = new_pitch
    return transposed


# --- 2. 调性分析/计算函数 (作为可导入的实用工具) ---
def get_semitones_for_transposition(
    source_tonic: KeyTonic, source_mode: ModeType, target_tonic: KeyTonic, target_mode: ModeType
) -> int:
    """
    计算从源调性到目标调性所需的半音数。

    此函数是**您的主要需求**，它接收调性主音和调式。

    注意：在 MIDI 级别，只进行音高平移，不改变大小调结构。
    """

    def get_absolute_pc(tonic: str, mode: str) -> int:
        if tonic not in KEY_TO_PC:
            raise ValueError(f"Unknown tonic name: {tonic}")

        pc = KEY_TO_PC[tonic]
        # 如果是小调，将其视为从其相对大调主音开始转调
        if mode.lower() == "minor":
            # 这里的逻辑是：小调的“转调点”相对于大调调式向后平移 3 个半音
            # C大调（PC 0）到A小调（PC 9）。如果从C大调转到A小调，需要 +9
            # 但如果从C小调（PC 0）转到A小调（PC 9），只需 +9
            # 为了让大小调转调的步数一致，我们将所有调性统一到一个虚拟的“大调”系统下计算
            # A小调 (PC 9) 的相对大调是 C 大调 (PC 0)，差异是 -9 或 +3
            pc = (pc - MINOR_OFFSET) % 12

        return pc

    source_abs_pc = get_absolute_pc(source_tonic, source_mode)
    target_abs_pc = get_absolute_pc(target_tonic, target_mode)

    semitones = (target_abs_pc - source_abs_pc) % 12

    return semitones


def parse_keys(values: Iterable[str]) -> list[ParsedKey]:
    """解析键名字符串，返回 (主音, 调式) 元组列表。"""
    parsed = []
    for key in values:
        normalized = key.strip().upper().replace("♭", "B").replace("FLAT", "B").replace("SHARP", "#")

        mode: ModeType = "major"

        # 识别小调标记
        if normalized.endswith("M") or normalized.endswith("MIN"):
            mode = "minor"
            if normalized.endswith("MIN"):
                normalized = normalized[:-3]
            else:  # Ends with 'M'
                normalized = normalized[:-1]

        # 移除任何剩余的大调标记
        normalized = normalized.replace("MAJ", "").replace("MAJOR", "")

        tonic = normalized.strip()

        if tonic not in KEY_TO_PC:
            raise ValueError(f"Unknown key tonic: {tonic}")

        parsed.append((tonic, mode))
    return parsed


def transpose_midi_file(
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    source_key_str: str,
    target_key_str: str,
) -> None:
    """
    加载 MIDI 文件，解析源和目标调性，计算转调半音数，执行转调并保存结果。

    Args:
        input_path: 输入 MIDI 文件的路径。
        output_path: 输出转调后 MIDI 文件的路径。
        source_key_str: 源调性的字符串表示 (例如: 'C', 'Am', 'EbMaj')。
        target_key_str: 目标调性的字符串表示 (例如: 'G', 'Dm', 'Bb').

    Raises:
        FileNotFoundError: 如果输入文件不存在。
        ValueError: 如果键名未知或转调超出 MIDI 范围。

    """
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)
    # --- a. 解析源和目标调性 ---
    # parse_keys 返回一个列表，但这里只处理单个源和单个目标
    try:
        source_tonic, source_mode = parse_keys([source_key_str])[0]
        target_tonic, target_mode = parse_keys([target_key_str])[0]
    except (ValueError, IndexError) as e:
        # 重新抛出解析错误，提供上下文
        raise ValueError(f"Key parsing error for '{source_key_str}' or '{target_key_str}': {e}")

    # --- b. 加载 MIDI ---
    if not input_path.exists():
        raise FileNotFoundError(f"Input MIDI file not found: {input_path}")

    pm = pretty_midi.PrettyMIDI(str(input_path))

    # --- c. 计算半音数 ---
    semitones = get_semitones_for_transposition(source_tonic, source_mode, target_tonic, target_mode)

    # --- d. 执行转调 ---
    transposed_pm = transpose_pm(pm, semitones)

    # --- e. 保存结果 ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transposed_pm.write(str(output_path))

    return output_path


# --- 3. 脚本执行主函数 (仅在作为脚本运行时调用) ---
def main() -> int:
    """命令行接口 (CLI) 主函数。"""
    parser = argparse.ArgumentParser(
        description="Transpose a MIDI file from a source key/mode to one or more target keys/modes."
    )
    parser.add_argument("midi", type=Path, help="Input MIDI file.")
    parser.add_argument(
        "--source-key",
        required=True,
        help="Source key (e.g., C, Am, EbMaj). Default mode is Major if not specified.",
    )
    parser.add_argument(
        "--target-keys",
        nargs="+",
        required=True,
        help="Target keys (e.g., G, Dm, B♭).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save transposed files (defaults to the input directory).",
    )
    args = parser.parse_args()

    # 设置路径和加载MIDI
    input_path = args.midi.expanduser().resolve()
    output_dir = (args.output_dir or input_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(str(input_path))

    # 解析源调性 (只取第一个)
    source_tonic, source_mode = parse_keys([args.source_key])[0]

    # 循环处理目标调性并转调
    stem = input_path.stem
    for target_tonic, target_mode in parse_keys(args.target_keys):
        # 计算转调半音数 (使用可导入函数)
        semitones = get_semitones_for_transposition(source_tonic, source_mode, target_tonic, target_mode)

        transposed = transpose_pm(pm, semitones)

        # 构建文件名
        mode_suffix = "m" if target_mode == "minor" else ""
        key_for_filename = f"{target_tonic}{mode_suffix}".replace("#", "sharp").replace("B", "B")
        out_path = output_dir / f"{stem}_to_{key_for_filename}.mid"

        transposed.write(str(out_path))
        print(f"Wrote {out_path}")

    return 0


# --- 4. 脚本/模块的入口点 ---
if __name__ == "__main__":
    # 当文件作为脚本直接运行时，执行 main() 函数
    raise SystemExit(main())
