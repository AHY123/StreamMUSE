from typing import Literal
from pathlib import Path
from typing import Union, List
import os

TypeFromResult = Literal[
    "pitch_jsd",
    "onset_jsd",
    "consonant_ratio",
    "unsupported_ratio",
    "prompt_generated_txt_mean_distance",
    "frechet_music_distance",
]

TypeFromExpRaw = Literal[
    "hit_rate",
    "hit_rate_weighted",
    "backup_level",
]

# TypeFromBatchRun = Literal["backup_level",]

TypeFromNll = Literal[
    "nll",
    "nll_weighted",
]

# Type = TypeFromResult | TypeFromExpRaw | TypeFromBatchRun | TypeFromNll
Type = Union[TypeFromResult, TypeFromExpRaw, TypeFromNll]

RESULT = {
    "pitch_jsd",
    "onset_jsd",
    "consonant_ratio",
    "unsupported_ratio",
    "prompt_generated_txt_mean_distance",
    "frechet_music_distance",
}
EXP_RAW = {
    "hit_rate",
    "hit_rate_weighted",
    "backup_level",
}
# BATCH = {
#     "backup_level",
# }
NLL = {
    "nll",
    "nll_weighted",
}


def get_dir_from_result(key: str, type: TypeFromResult, base_dir: Union[str, Path] = "result") -> Path:
    """Build the file path for a result metric.

    Parameters:
    - key: metric key/name (file name without suffix)
    - type: one of `TypeFromResult` literal values
    - base_dir: base directory or path (defaults to `'result'`)

    Returns:
    - Path to the JSON file for the metric (adds `.json` suffix).
    """
    return Path(base_dir) / (key + ".json")


def get_dir_from_exp_raw(key: str, type: TypeFromExpRaw, base_dir: Union[str, Path] = "records") -> Path:
    """Build the path to an experiment-raw `batch_run` directory.

    Parameters:
    - key: a compound key expected to contain underscores; the function
        splits it at the 2nd underscore to derive path components.
    - type: one of `TypeFromExpRaw` literal values (kept for signature
        compatibility but not used in the resulting path)
    - base_dir: base directory or path (defaults to `'records'`)

    Returns:
    - Path to the `batch_run` directory for the provided key.
    """
    from .utils import split_at_nth_underscore

    pre_key, post_key = split_at_nth_underscore(key, 2)
    pre_key = pre_key.replace("_gen", "_gen_frame_")
    pre_key = pre_key.replace("interval", "interval_")
    return Path(base_dir) / "raw" / "realtime" / "baseline" / pre_key / post_key / "batch_run"


# def get_dir_from_batch_run(key: str, type: TypeFromBatchRun, base_dir: Union[str, Path] = "records") -> Path:
#     """Build the path to batch-run related data.

#     Parameters:
#     - key: a compound key; split at the 2nd underscore to derive path components
#     - type: one of `TypeFromBatchRun` (kept for signature compatibility)
#     - base_dir: base directory or path (defaults to `'records'`)

#     Returns:
#     - Path to the `batch_run` directory corresponding to `key`.
#     """
#     from .utils import split_at_nth_underscore

#     pre_key, post_key = split_at_nth_underscore(key, 2)
#     pre_key = pre_key.replace("_gen", "_gen_frame_")
#     pre_key = pre_key.replace("interval", "interval_")
#     return Path(base_dir) / "raw" / "realtime" / "baseline" / pre_key / post_key / "batch_run"


def get_dir_from_nll(key: str, type: TypeFromNll, base_dir: Union[str, Path] = "records") -> Path:
    """Locate the NLL result JSON file for a given key.

    Parameters:
    - key: search key used to find a unique NLL run name
    - type: one of `TypeFromNll` (kept for signature compatibility)
    - base_dir: base directory or path containing an `nll_runs` directory

    Returns:
    - Path to the matching NLL JSON file inside `nll_runs`.

    Notes:
    - The function lists entries under `<base_dir>/nll_runs` and then
        resolves a unique name using `find_unique_path_with_target` from
        `.utils`.
    """
    from .utils import find_unique_path_with_target, split_at_nth_underscore
    if "offline" in key:
        nll_runs_dir = Path(base_dir) / "nll_runs"
        nll_list = os.listdir(nll_runs_dir)
        # print(nll_list)
        nll_name = find_unique_path_with_target(nll_list, "generated_without_prompt")
        # print(nll_name,"offline")
        return nll_runs_dir / nll_name
    else:
        pre_key, post_key = split_at_nth_underscore(key, 2)
        pre_key = pre_key.replace("_gen", "_gen_frame_")
        pre_key = pre_key.replace("interval", "interval_")
        key = pre_key + "_" + post_key
        nll_runs_dir = Path(base_dir) / "nll_runs"
        nll_list = os.listdir(nll_runs_dir)
        nll_name = find_unique_path_with_target(nll_list, key)
        return nll_runs_dir / nll_name


def get_keys_from_dir(path: Union[str, Path]) -> List[str]:
    """List candidate keys (directory names) inside `path`.

    Returns only the immediate child directory names (not files). If the
    path does not exist, returns an empty list.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return []
    keys: List[str] = []
    for child in p.iterdir():
        # Exclude directories whose name starts with "offline" (legacy/aux data)
        if child.is_dir() and not child.name.startswith("offline"):
            keys.append(child.name)
    return sorted(keys)


def get_path(key: str, type: Type, base_dir: Union[str, Path] | None = None) -> str | Path:
    """For a directory `key`, build a mapping from each key -> target Path.

    The mapping uses helper functions above depending on which literal
    `type` belongs to. If `base_dir` is provided it will be forwarded to
    the helper; otherwise each helper uses its default base (e.g. `result` or
    `records`). Raises ValueError if `type` is not recognized.

    This function was previously named `get_dir` and was renamed to
    `get_path` for clarity.
    """

    t = str(type)

    if t in RESULT:

        def mapper(k: str) -> Path:
            return get_dir_from_result(k, t, base_dir or "result")  # type: ignore
    elif t in EXP_RAW:

        def mapper(k: str) -> Path:
            return get_dir_from_exp_raw(k, t, base_dir or "records")  # type: ignore
    # elif t in BATCH:

    #     def mapper(k: str) -> Path:
    #         return get_dir_from_batch_run(k, t, base_dir or "records")  # type: ignore
    elif t in NLL:

        def mapper(k: str) -> Path:
            return get_dir_from_nll(k, t, base_dir or "records")  # type: ignore
    else:
        raise ValueError(f"Unrecognized type: {type}")
    return mapper(key)
