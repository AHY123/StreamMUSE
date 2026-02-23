"""Export aggregated statistics from result JSON files into CSV.

CSV layout:
- first column: `key` (filename without .json)
- remaining columns: for each requested `type` and each statistic name,
  header format is `{type}-{stat}` (e.g. `pitch_jsd-mean`).

This script scans a `base_dir` for `*.json` result files (only top-level
files), parses per-type values using `scripts.json_processors.parse_by_type`,
computes statistics via `scripts.aggreate.compute_stats`, and writes an
aggregated CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List

from scripts.dir_stats import RESULT, EXP_RAW, NLL
from scripts.json_processors import parse_by_type
from scripts.aggreate import compute_stats
from scripts.dir_stats import get_path


DEFAULT_STATS = [
    # "count",
    # "sum",
    # "min",
    # "max",
    "mean",
    # "variance_pop",
    # "variance_samp",
    # "stdev_pop",
    "stdev_samp",
    # "std",
    # "p25",
    # "p50",
    # "p75",
    # "iqr",
]


def find_result_files(base_dir: Path) -> List[Path]:
    """Return list of top-level JSON files inside `base_dir`.

    Only files directly under `base_dir` are considered (no recursion).
    """
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    return sorted([p for p in base_dir.iterdir() if p.is_file() and p.suffix == ".json"])


def build_headers(types: Iterable[str], stats: Iterable[str]) -> List[str]:
    headers = ["key"]
    for t in types:
        for s in stats:
            headers.append(f"{t}-{s}")
    return headers


def row_for_key(key: str, base_dir: Path, types: Iterable[str], stats: Iterable[str]) -> List[str]:
    """Compute a CSV row for `key`.

    For each `type` resolve the appropriate path using `get_path(key, type, base_dir)`
    and then call `parse_by_type`. This allows support for RESULT, EXP_RAW and
    NLL types (e.g. hit_rate) when `base_dir` is provided.
    """
    row = [key]
    for t in types:
        try:
            target_path = get_path(key, t, base_dir)
            if t == "nll" or t == "nll_weighted":
                ...
            items = parse_by_type(key, t, target_path)
        except Exception as exc:  # defensive: parser may raise on malformed JSON or missing files
            items = []
            print(f"Warning: parse failure for key={key} type={t}: {exc}", file=sys.stderr) 
            # it's only used in results becuase of the name of the file in of nll is not standard
            # but it will still work
        stats_map = compute_stats(items)
        for s in stats:
            # normalize missing -> empty string to keep CSV clean
            v = stats_map.get(s)
            if v is None:
                row.append("")
            else:
                row.append(str(v))
    return row


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export aggregated result statistics to CSV")
    parser.add_argument("--base_dir", required=True, help="Directory containing result JSON files")
    parser.add_argument("--out", required=True, help="Output CSV file path")
    parser.add_argument(
        "--types",
        default=",".join(sorted(RESULT)+sorted(EXP_RAW)+sorted(NLL)),
        help="Comma-separated list of types to include (default: all RESULT types)",
    )
    parser.add_argument(
        "--stats",
        default=",".join(DEFAULT_STATS),
        help="Comma-separated list of statistic names to compute (default: common set)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print header and summary only, do not write CSV")
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir)
    out_path = Path(args.out)
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    stats = [s.strip() for s in args.stats.split(",") if s.strip()]

    files = find_result_files(base_dir)
    if not files:
        print(f"No result JSON files found in {base_dir}", file=sys.stderr)
        return 1

    headers = build_headers(types, stats)
    if args.dry_run:
        print(",".join(headers))
        print(f"Found {len(files)} result files; first 5: {[p.name for p in files[:5]]}")
        return 0

    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for p in files:
            key = p.stem
            row = row_for_key(key, base_dir, types, stats)
            writer.writerow(row)

    print(f"Wrote CSV to {out_path} ({len(files)} rows)")
    # python3 -m scripts.export_results_csv --base_dir result/results-experiments2-local --out tmp/2-local.csv
    # python3 -m scripts.export_results_csv --base_dir result/results-experiments2-local_server --out tmp/2-local_server.csv
    # python3 -m scripts.export_results_csv --base_dir result/results-experiments2-remote --out tmp/2-remote.csv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
