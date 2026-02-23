"""Check filesystem paths for all metric types using `get_path`.

Usage examples:
  python -m scripts.run_all_types --key interval1_gen3_prompt_128_gen_576
  python -m scripts.run_all_types --key interval1_gen3_prompt_128_gen_576 --base_dir result/results-experiments2-local

The script iterates through `RESULT`, `EXP_RAW`, `BATCH`, `NLL` types and
prints the resolved Path and whether it exists / is a file / is a directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
from typing import Iterable

from scripts.dir_stats import get_path, RESULT, EXP_RAW, NLL


def iterate_types() -> Iterable[tuple[str, str]]:
    """Yield (category, type_name) pairs in a predictable order."""
    for t in sorted(RESULT):
        yield ("RESULT", t)
    for t in sorted(EXP_RAW):
        yield ("EXP_RAW", t)
    # for t in sorted(BATCH):
        # yield ("BATCH", t)
    for t in sorted(NLL):
        yield ("NLL", t)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check paths for all metric types")
    parser.add_argument("--key", required=True, help="Key to look up (same as in your examples)")
    parser.add_argument("--base_dir", default=None, help="Optional base_dir forwarded to get_path")
    args = parser.parse_args()

    key = args.key
    base_dir = args.base_dir

    print(f"Checking key: {key!r} (base_dir={base_dir!r})")
    print("---")

    for category, type_name in iterate_types():
        try:
            p = get_path(key, type_name, base_dir)
        except Exception as e:
            print(f"{category} {type_name} -> ERROR: {e}")
            continue

        # Ensure we have a Path for checks
        p_path = Path(p)

        exists = p_path.exists()
        is_file = p_path.is_file()
        is_dir = p_path.is_dir()

        print(f"{category} {type_name} -> {p_path}")
        print(f"{''} exists={exists} is_file={is_file} is_dir={is_dir}")
        print()


if __name__ == "__main__":
    main()
