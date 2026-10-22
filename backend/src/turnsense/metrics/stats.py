"""Percentiles and confidence intervals used everywhere results are reported."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Two-sided 95% Student-t critical values, df = 1..30 (beyond that use the normal value).
_T95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]  # fmt: skip


def t_critical_95(df: int) -> float:
    if df < 1:
        return math.nan
    return _T95[df - 1] if df <= len(_T95) else 1.96


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy default). None for an empty sample."""
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def percentiles(
    values: Sequence[float], qs: Sequence[float] = (50, 95, 99), min_samples: int = 1
) -> dict[float, float | None]:
    if len(values) < max(min_samples, 1):
        return {q: None for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {q: float(np.percentile(arr, q)) for q in qs}


def mean_ci95(
    values: Sequence[float | None],
) -> tuple[float | None, float | None, float | None, int]:
    """Mean with a t-based 95% CI over non-null values: (mean, low, high, n)."""
    clean = [float(v) for v in values if v is not None and not math.isnan(v)]
    n = len(clean)
    if n == 0:
        return None, None, None, 0
    mean = float(np.mean(clean))
    if n < 2:
        return mean, None, None, n
    sd = float(np.std(clean, ddof=1))
    half = t_critical_95(n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half, n


def bootstrap_ci95(
    values: Sequence[float], stat=np.mean, n_boot: int = 2000, seed: int = 0
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI — for per-event metrics pooled across trials."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.apply_along_axis(stat, 1, arr[idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
