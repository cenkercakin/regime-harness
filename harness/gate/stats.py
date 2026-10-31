"""Multiple-testing statistics: Deflated Sharpe Ratio and Probability of Backtest Overfitting.

References
----------
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest Overfitting".
All Sharpe ratios here are per-period (not annualised), as the formulas require.
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected best Sharpe among `n_trials` zero-skill trials whose Sharpes have variance `sr_variance`."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    z1 = _N.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float) -> float:
    """P(true Sharpe > sr_benchmark) given an observed per-period `sr` over `n_obs` periods."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if n_obs < 2 or denom <= 0:
        return float("nan")
    return _N.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom))


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, sr_variance: float) -> tuple[float, float]:
    """Return (DSR, SR*) where SR* is the expected max Sharpe from pure luck."""
    sr_star = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe(sr, sr_star, n_obs, skew, kurt), sr_star


def pooled_sr_variance(groups: list[tuple[int, float, float]]) -> float:
    """Combine (n, mean, variance) of Sharpe estimates from several trial batches."""
    groups = [g for g in groups if g[0] > 0]
    n = sum(g[0] for g in groups)
    if n < 2:
        return 0.0
    mean = sum(g[0] * g[1] for g in groups) / n
    within = sum((g[0] - 1) * g[2] for g in groups if g[0] > 1)
    between = sum(g[0] * (g[1] - mean) ** 2 for g in groups)
    return (within + between) / (n - 1)


def pbo_cscv(returns: np.ndarray, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting via combinatorially symmetric cross-validation.

    `returns`: T x N matrix, one column per strategy variant, same dates.
    Splits rows into `n_blocks` contiguous blocks; for every half/half split, picks
    the best in-sample variant and records where it ranks out-of-sample.
    PBO = share of splits where the in-sample winner is below the OOS median.
    """
    M = np.asarray(returns, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError("PBO needs at least 2 variants")
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks must be even and >= 4")
    M = np.nan_to_num(M[T % n_blocks:])
    blocks = M.reshape(n_blocks, -1, N)
    L = blocks.shape[1]
    s1 = blocks.sum(axis=1)                 # n_blocks x N
    s2 = (blocks ** 2).sum(axis=1)

    combos = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    mask = np.zeros((len(combos), n_blocks))
    np.put_along_axis(mask, combos, 1.0, axis=1)

    def sr(mask_):
        cnt = mask_.sum(axis=1, keepdims=True) * L
        m1 = mask_ @ s1 / cnt
        var = (mask_ @ s2 - cnt * m1 ** 2) / (cnt - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(var > 0, m1 / np.sqrt(var), 0.0)
        return out

    is_sr, oos_sr = sr(mask), sr(1.0 - mask)
    best = is_sr.argmax(axis=1)
    best_oos = oos_sr[np.arange(len(combos)), best]
    rank = (oos_sr < best_oos[:, None]).sum(axis=1) + 0.5 * ((oos_sr == best_oos[:, None]).sum(axis=1) - 1) + 1
    omega = rank / (N + 1)
    logits = np.log(omega / (1.0 - omega))
    # Degradation: how much of the IS edge survives OOS
    slope = float(np.polyfit(is_sr[np.arange(len(combos)), best], best_oos, 1)[0]) if len(combos) > 2 else float("nan")
    return {
        "pbo": float((logits <= 0).mean()),
        "n_splits": int(len(combos)),
        "median_logit": float(np.median(logits)),
        "is_oos_slope": slope,
    }
