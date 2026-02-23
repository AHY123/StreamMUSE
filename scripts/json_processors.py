"""JSON parsing helpers for different metric types.

This module provides small, well-documented parsers for the same metric
categories used by `dir_stats`:
- RESULT: single JSON file -> returns parsed dict
- NLL: single JSON-like file -> returns parsed dict
- EXP_RAW: directory of JSON files -> returns list[dict]
- BATCH: directory of JSON files -> returns list[dict]

The main entrypoint is `parse_by_type(key, type, base_dir=None)` which
uses `dir_stats.get_path` to resolve a path and then dispatches to the
appropriate parser. Callers can also call per-type helpers directly if
they have an explicit path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from .dir_stats import (
    TypeFromResult,
    TypeFromExpRaw,
    TypeFromNll,
    Type,
    EXP_RAW,
    NLL,
    RESULT,
)


def _load_json_file(p: Path) -> dict:
    """Load and return JSON content from a file path.

    Raises FileNotFoundError / JSONDecodeError to the caller.
    """
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_result_value(item: dict, type_str: str):
    """Extract a single value from a `detail` item according to `type_str`.

    Returns None if the requested value can't be found on the item.
    """
    # direct top-level fields
    if type_str in item:
        return item.get(type_str)

    # nested mappings
    if type_str in ("consonant_ratio", "unsupported_ratio"):
        # harmonicity contains consonant/dissonant/unsupported
        h = item.get("harmonicity") if isinstance(item, dict) else None
        if isinstance(h, dict):
            return h.get(type_str)
        return None

    if type_str == "prompt_generated_txt_mean_distance":
        # prefer prompt_generated_continuation_polydis -> txt_mean_distance
        pg = item.get("prompt_generated_continuation_polydis")
        if isinstance(pg, dict) and "txt_mean_distance" in pg:
            return pg.get("txt_mean_distance")
        # fallback to prompt_polydis
        pp = item.get("prompt_polydis")
        if isinstance(pp, dict) and "txt_mean_distance" in pp:
            return pp.get("txt_mean_distance")
        return None

    # frechet_music_distance may be top-level or nested under other keys
    if type_str == "frechet_music_distance":
        return item.get("frechet_music_distance")

    # unknown type -> None
    return None


def parse_result_file(key: str, type: TypeFromResult, path: Union[str, Path]) -> List:
    """Parse a single RESULT JSON file and return a list of extracted values.

    For each example in `details` (must be a list) extract the value
    corresponding to `type` using `_extract_result_value`. If `details`
    is missing or not a list, returns an empty list.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    data = _load_json_file(p)
    if not isinstance(data, dict):
        return []

    details = data.get("details")
    if not isinstance(details, list):
        return []

    tstr = str(type)
    out = []
    for item in details:
        if not isinstance(item, dict):
            out.append(None)
            continue
        out.append(_extract_result_value(item, tstr))
    return out


def parse_nll_file(key: str, type: TypeFromNll, path: Union[str, Path]) -> List[dict]:
    """Parse an NLL result file (JSON) and return the object.

    Parameters:
    - key: the lookup key
    - type: the metric type
    - path: path to the NLL JSON file
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    data = _load_json_file(p)
    # Support extracting numeric lists from common NLL file shapes. Use a
    # helper so callers (and tests) can operate on structured data as well.
    return _extract_nll_values_from_data(data, type)


def _extract_nll_values_from_data(data: object, type: TypeFromNll) -> List[float]:
    """Extract numeric NLL values from a loaded JSON object.

    The function accepts a few common shapes produced by NLL runs:
    - mapping of sample name -> {total_nll, avg_nll, total_tokens}
    - list of such per-sample dicts
    - a single dict containing the fields directly

    For `type == 'nll'` we return per-sample `avg_nll` values (falling back
    to `total_nll/total_tokens` if `avg_nll` is missing). For
    `type == 'nll_weighted'` we return per-sample `total_nll` (so downstream
    code can aggregate weighted sums). Non-numeric or missing fields are
    skipped.
    """
    tstr = str(type)
    out: List[float] = []

    def _process_record(rec: object):
        # rec should be a dict with possible keys
        if not isinstance(rec, dict):
            return
        # prefer explicit avg_nll / total_nll / total_tokens
        avg = rec.get("avg_nll")
        total = rec.get("total_nll")
        tokens = rec.get("total_tokens")

        # helpers to coerce
        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if tstr == "nll":
            v = _as_float(avg)
            if v is None and total is not None and tokens is not None:
                tval = _as_float(total)
                tok = _as_float(tokens)
                if tval is not None and tok not in (None, 0):
                    v = tval / tok
            if v is not None:
                out.append(v)
            return

        if tstr == "nll_weighted":
            # For weighted NLL we collect total_nll and total_tokens so
            # downstream we can compute the global weighted mean (sum(total_nll)/sum(total_tokens)).
            v_total = _as_float(total)
            v_tokens = _as_float(tokens)
            if v_total is None:
                # try to reconstruct from avg * tokens
                if avg is not None and v_tokens not in (None, 0):
                    a = _as_float(avg)
                    if a is not None:
                        v_total = a * v_tokens
            # attach totals as a pair by encoding as tuple-like list [total, tokens]
            # We'll accumulate these outside when data is processed.
            if v_total is not None and v_tokens not in (None, 0):
                out.append((v_total, v_tokens))
            return

        # unknown -> nothing
        return

    # If data is a dict mapping sample -> metrics, iterate values
    if isinstance(data, dict):
        # If dict looks like a single-record (contains total_nll/avg_nll),
        # treat it as one record; otherwise iterate its values.
        if any(k in data for k in ("avg_nll", "total_nll", "total_tokens")):
            _process_record(data)
        else:
            for v in data.values():
                _process_record(v)
        # If requesting nll_weighted, we may have appended (total, tokens)
        if tstr == "nll_weighted":
            # accumulate totals to compute global weighted mean
            sum_nll = 0.0
            sum_tokens = 0.0
            for item in out:
                if isinstance(item, tuple) and len(item) == 2:
                    try:
                        sum_nll += float(item[0])
                        sum_tokens += float(item[1])
                    except Exception:
                        continue
            if sum_tokens > 0:
                return [sum_nll / sum_tokens]
            # fallback empty
            return []
        return out

    # If data is a list of records
    if isinstance(data, list):
        for item in data:
            _process_record(item)
        if tstr == "nll_weighted":
            sum_nll = 0.0
            sum_tokens = 0.0
            for item in out:
                if isinstance(item, tuple) and len(item) == 2:
                    try:
                        sum_nll += float(item[0])
                        sum_tokens += float(item[1])
                    except Exception:
                        continue
            if sum_tokens > 0:
                return [sum_nll / sum_tokens]
            return []
        return out

    # otherwise empty
    return out


def parse_json_dir(path: Union[str, Path]) -> List[dict]:
    """Load all `.json` files in a directory and return list of parsed objects.

    Parameters:
    - key: the lookup key
    - type: the metric type
    - path: directory path containing JSON files

    If directory does not exist, returns an empty list. Files are read in
    sorted name order to make results deterministic.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    results: List[dict] = []
    for f in sorted(p.glob("*.json")):
        try:
            results.append(_load_json_file(f))
        except Exception:
            # keep going on malformed files but do not stop the whole parse
            continue
    return results


def _extract_tick_history_from_data(data: dict, type: TypeFromExpRaw) -> List:
    """Extract tick-history list from a parsed JSON object.

    The function tries several common field names and shapes and returns
    a list of tick entries if found; otherwise returns an empty list.

    This is intentionally permissive — downstream metric-specific code
    will interpret each tick entry.
    """

    # Map tick entries to numeric values according to `type` so downstream
    # `compute_stats` can operate directly. The mapping covers common
    # EXP_RAW types such as `hit_rate` and `backup_level`.
    tstr = str(type)

    def _map_tick_to_value(tick) -> object:
        # If tick is not a dict, return it directly if numeric-like
        if not isinstance(tick, dict):
            return tick

        if tstr == "hit_rate":
            # convert boolean `is_hit` to 1/0
            val = tick.get("is_hit")
            if isinstance(val, bool):
                return 1 if val else 0
            try:
                return 1 if int(val) else 0
            except Exception:
                return 0

        if tstr == "backup_level":
            return tick.get("backup_level")

        if tstr == "hit_rate_weighted":
            # use `backup_level` to compute a simple weight: (32 - backup_level)/32
            backup_level = tick.get("backup_level")
            try:
                weight = (32 - float(backup_level)) / 32
            except Exception:
                weight = 0.0
            is_hit = tick.get("is_hit")
            if isinstance(is_hit, bool):
                return weight if is_hit else 0
            try:
                return weight if int(is_hit) else 0
            except Exception:
                return 0

        # Fallback: if the tick dict contains the requested key, return it
        if tstr in tick:
            return tick.get(tstr)

        # Unknown mapping -> None
        return None

    return [_map_tick_to_value(t) for t in data]


def _extract_tick_history_from_file(path: Union[str, Path], type: TypeFromExpRaw) -> List:
    """Load JSON file and extract its tick-history list (or empty list).

    Raises FileNotFoundError / JSONDecodeError to caller as appropriate.
    """
    p = Path(path)
    data = _load_json_file(p)
    # print(f"path: {path},{_extract_tick_history_from_data(data, type)}")  # --- IGNORE ---
    return _extract_tick_history_from_data(data, type)


def parse_exp_raw_dir(key: str, type: TypeFromExpRaw, path: Union[str, Path]) -> List:
    """Parse an EXP_RAW `batch_run` directory and return concatenated tick items.

    For each JSON file in `path`, extract its tick history and concatenate
    all tick entries into a single list which is returned. This allows
    downstream code to compute metrics across the entire batch run.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []

    # Prefer explicit `tick_history.json` files (possibly nested). If none
    # are found, fall back to any top-level `*.json` files in the directory.
    ticks: List = []
    files = sorted(p.glob("**/tick_history.json"))
    
    for f in files:
        try:
            part = _extract_tick_history_from_file(f, type)
        except Exception:
            # skip malformed files but continue processing the directory
            continue
        if part:
            ticks.extend(part)

    # Map tick entries to numeric values according to `type` so downstream
    # `compute_stats` can operate directly. The mapping covers common
    # EXP_RAW types such as `hit_rate` and `backup_level`.
    tstr = str(type)

    def _map_tick_to_value(tick) -> object:
        # If tick is not a dict, return it directly if numeric-like
        if not isinstance(tick, dict):
            return tick

        if tstr == "hit_rate":
            # convert boolean `is_hit` to 1/0
            val = tick.get("is_hit")
            if isinstance(val, bool):
                return 1 if val else 0
            try:
                return 1 if int(val) else 0
            except Exception:
                return 0

        if tstr == "backup_level":
            return tick.get("backup_level")

        if tstr == "hit_rate_weighted":
            # use `hit_weight` if present, else fallback to `is_hit` as 1/0
            backup_level = tick.get("backup_level")
            weight = (32 - backup_level) / 32
            is_hit = tick.get("is_hit")
            if isinstance(is_hit, bool):
                return weight if is_hit else 0
            try:
                return weight if int(is_hit) else 0
            except Exception:
                return 0
        # Fallback: if the tick dict contains the requested key, return it
        if tstr in tick:
            return tick.get(tstr)

        # Unknown mapping -> None
        return None

    mapped = [_map_tick_to_value(t) for t in ticks]
    return mapped


# def parse_batch_run_dir(key: str, type: TypeFromBatchRun, path: Union[str, Path]) -> List[dict]:
#     """Parse a BATCH `batch_run` directory and return a list of parsed JSONs."""
#     return parse_json_dir(path)


def parse_by_type(key: str, type: Type, path: Union[str, Path]) -> List[dict]:
    """Parse data for (key, type) using the provided `path`.

    The caller supplies the resolved `path` (no `base_dir` lookup here).

        Returns:
        - a list of parsed JSON objects. For RESULT/NLL the list contains one dict;
            for EXP_RAW/BATCH it contains all parsed dicts from the directory.

    Raises FileNotFoundError if a required single-file is missing.
    """

    t = str(type)
    if t in RESULT:
        return parse_result_file(key, type, path)
    if t in NLL:
        return parse_nll_file(key, type, path)
    if t in EXP_RAW:
        return parse_exp_raw_dir(key, type, path)
    # if t in BATCH:
    #     return parse_batch_run_dir(key, type, path)

    raise ValueError(f"Unrecognized type: {type}")
