"""Small statistics helpers used by scripts.

Provides a single entrypoint `compute_stats(values)` which returns a dict
with count, sum, min, max, mean, variance (pop/sample), stdev (pop/sample),
and percentiles (p25,p50,p75) plus iqr. Designed to be dependency-free
(only stdlib `statistics`).
"""

from __future__ import annotations

from typing import Iterable, List, Dict
import statistics


def _percentile_sorted(sorted_vals: List[float], p: float) -> float:
    """Compute percentile p (0-100) on already-sorted list using linear interpolation.

    If list is empty returns 0.0. If single item, returns that item.
    """
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    # Convert p to fractional rank between 0 and n-1
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def compute_stats(values: Iterable[float]) -> Dict[str, float]:
    """Return a dictionary of common descriptive statistics for `values`.

    Keys returned include:
      - count, sum, min, max
      - mean
      - variance_pop, variance_samp
      - stdev_pop, stdev_samp
      - p25, p50, p75, iqr

    All numeric outputs are floats (or 0.0 when not applicable).
    """
    vals = [float(x) for x in values]
    out: Dict[str, float] = {}
    if not vals:
        # consistent shape when no data
        out.update(
            {
                "count": 0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "variance_pop": 0.0,
                "variance_samp": 0.0,
                "stdev_pop": 0.0,
                "stdev_samp": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "iqr": 0.0,
            }
        )
        return out

    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    s = sum(vals_sorted)
    mn = float(vals_sorted[0])
    mx = float(vals_sorted[-1])
    mean_v = float(statistics.mean(vals_sorted))
    # population (p) and sample (s) variance
    try:
        var_pop = float(statistics.pvariance(vals_sorted))
    except Exception:
        var_pop = 0.0
    try:
        var_samp = float(statistics.variance(vals_sorted)) if n > 1 else 0.0
    except Exception:
        var_samp = 0.0
    try:
        stdev_pop = float(statistics.pstdev(vals_sorted))
    except Exception:
        stdev_pop = 0.0
    try:
        stdev_samp = float(statistics.stdev(vals_sorted)) if n > 1 else 0.0
    except Exception:
        stdev_samp = 0.0

    p25 = _percentile_sorted(vals_sorted, 25.0)
    p50 = _percentile_sorted(vals_sorted, 50.0)
    p75 = _percentile_sorted(vals_sorted, 75.0)
    iqr = float(p75 - p25)

    out.update(
        {
            "count": n,
            "sum": float(s),
            "min": mn,
            "max": mx,
            "mean": mean_v,
            "variance_pop": var_pop,
            "variance_samp": var_samp,
            "stdev_pop": stdev_pop,
            "stdev_samp": stdev_samp,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "iqr": iqr,
        }
    )
    return out
