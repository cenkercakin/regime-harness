from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(r: pd.Series | np.ndarray, periods_per_year: float | None = None) -> float:
    """Per-period Sharpe (mean/std, ddof=1); annualised if periods_per_year is given."""
    a = np.asarray(r, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return float("nan")
    sd = a.std(ddof=1)
    if sd == 0:
        return 0.0
    s = a.mean() / sd
    return float(s * np.sqrt(periods_per_year)) if periods_per_year else float(s)


def cagr(r: pd.Series, periods_per_year: float) -> float:
    a = np.asarray(r, dtype=float)
    a = a[~np.isnan(a)]
    if not len(a):
        return float("nan")
    growth = float(np.prod(1.0 + a))
    return growth ** (periods_per_year / len(a)) - 1.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def moments(r: pd.Series) -> tuple[float, float]:
    """(skewness, kurtosis) with kurtosis non-excess (normal = 3), as DSR expects."""
    s = pd.Series(r).dropna()
    return float(s.skew()), float(s.kurt() + 3.0)


def summary(r: pd.Series, periods_per_year: float) -> dict:
    sk, ku = moments(r)
    return {
        "sharpe_ann": sharpe(r, periods_per_year),
        "sharpe_pp": sharpe(r),
        "cagr": cagr(r, periods_per_year),
        "max_dd": max_drawdown(r),
        "skew": sk,
        "kurt": ku,
        "n_obs": int(pd.Series(r).notna().sum()),
    }
