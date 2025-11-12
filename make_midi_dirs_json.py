#!/usr/bin/env python3
"""
Scan specified experiment directories for .mid/.pt files and write a JSON manifest.

Usage examples:
  python make_midi_dirs_json.py --roots experiments1 experiments2-local experiments2-local_server experiments2-remote experiments3-local --out records/midi_dirs.json

This writes a JSON mapping per provided root name with found files and directories.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


def find_midi_and_pt_files(root: Path, exts: List[str]) -> Dict[str, List[str]]:
    """Walk `root` and return a mapping: directory -> list of matching filenames (relative to repo root)."""
    out: Dict[str, List[str]] = {}
    if not root.exists():
        return out

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        matches = [f for f in filenames if any(f.lower().endswith(ext) for ext in exts)]
        if matches:
            # store paths relative to current working directory
            rel_dir = os.path.relpath(dirpath, os.getcwd())
            out[rel_dir] = sorted(matches)
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate JSON manifest of .mid/.pt files under given roots')
    parser.add_argument('--roots', '-r', nargs='+', default=[
        'experiments1', 'experiments2-local', 'experiments2-local_server', 'experiments2-remote', 'experiments3-local'
    ], help='Root directories to scan (relative or absolute)')
    parser.add_argument('--out', '-o', default='records/midi_dirs.json', help='Output JSON path')
    parser.add_argument('--exts', type=str, default='mid,pt', help='Comma-separated file extensions to include (no dot). Default: mid,pt')
    parser.add_argument('--min-per-dir', type=int, default=0, help='Only include directories with at least this many matching files')
    args = parser.parse_args(argv)

    exts = ['.' + e.strip().lower() for e in args.exts.split(',') if e.strip()]

    result: Dict[str, Dict] = {}
    for r in args.roots:
        root_path = Path(os.path.expanduser(r))
        if not root_path.exists():
            result[r] = {'error': f'root not found: {str(root_path)}'}
            continue
        found = find_midi_and_pt_files(root_path, exts)
        # Filter by min-per-dir if requested
        if args.min_per_dir > 0:
            found = {d: files for d, files in found.items() if len(files) >= args.min_per_dir}
        # Flatten file list to full relative paths for convenience
        flat_files = []
        for d, files in found.items():
            for f in files:
                flat_files.append(os.path.join(d, f))
        result[r] = {
            'root': str(os.path.abspath(root_path)),
            'total_dirs': len(found),
            'total_files': len(flat_files),
            'dirs': found,
            'files': sorted(flat_files),
        }

    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'Wrote manifest to {out_path} (roots scanned: {len(args.roots)})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
