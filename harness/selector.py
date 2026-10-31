"""Monthly strategy selector: the "look at many, pivot to the best" layer.

Treats each strategy's daily net return series as an asset and decides, at each
month-end, how much capital each strategy gets for the next month:

* hard  - 100% to the strategy with the best trailing Sharpe
* tilt  - weights proportional to positive trailing Sharpe (all <= 0 -> equal weight)
* blend - equal weight in every live strategy (the bar the other two must beat)

Constraints: crypto strategies together never exceed `crypto_cap`. Switching
strategies costs the per-side cost of the venue being bought or sold, and moves
into or out of crypto also pay the TRY<->USD spread placeholder. Fills happen at
the next close via the same backtest engine, so the timing contract is shared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import backtest
from .engine.returns import month_end_dates

MODES = ("hard", "tilt", "blend")


def trailing_sharpe(S: pd.DataFrame, d, months: int, min_obs: int, rf: pd.Series | None = None) -> pd.Series:
    """Sharpe of returns ABOVE cash. Without subtracting `rf`, a T-bill fund's tiny
    volatility gives it the highest Sharpe of all and the selector just sits in cash."""
    X = S if rf is None else S.sub(rf.reindex(S.index).fillna(0.0), axis=0)
    win = X.loc[(X.index > d - pd.DateOffset(months=months)) & (X.index <= d)]
    ok = win.notna().sum() >= min_obs
    sd = win.std(ddof=1)
    sr = win.mean() / sd.replace(0, np.nan)
    return sr[ok].dropna()


def _cap(w: pd.Series, crypto: set, cap: float, mode: str, sr: pd.Series) -> pd.Series:
    c = [s for s in w.index if s in crypto]
    total = w[c].sum() if c else 0.0
    if total <= cap + 1e-12:
        return w
    w = w.copy()
    w[c] *= cap / total
    excess = 1.0 - w.sum()
    others = [s for s in sr.index if s not in crypto]
    if not others:
        return w  # nothing to move into: the excess stays in cash
    if mode == "hard":
        w[sr[others].idxmax()] = w.get(sr[others].idxmax(), 0.0) + excess
    else:
        base = w[[s for s in others if s in w.index]]
        if base.sum() > 0:
            w[base.index] += excess * base / base.sum()
        else:
            w[others] = w.reindex(others).fillna(0.0) + excess / len(others)
    return w


def targets(S: pd.DataFrame, mode: str, months: int = 12, crypto: set = frozenset(), crypto_cap: float = 0.25, min_frac: float = 0.8, rf: pd.Series | None = None) -> pd.DataFrame:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rows = {}
    for d in month_end_dates(S.index):
        if d - pd.DateOffset(months=months) < S.index[0]:
            continue
        span = ((S.index > d - pd.DateOffset(months=months)) & (S.index <= d)).sum()
        sr = trailing_sharpe(S, d, months, int(span * min_frac), rf)
        if sr.empty:
            continue
        if mode == "hard":
            w = pd.Series({sr.idxmax(): 1.0})
        elif mode == "tilt":
            pos = sr.clip(lower=0)
            w = pos / pos.sum() if pos.sum() > 0 else pd.Series(1.0 / len(sr), index=sr.index)
        else:
            w = pd.Series(1.0 / len(sr), index=sr.index)
        rows[d] = _cap(w.reindex(sr.index).fillna(0.0), set(crypto), crypto_cap, mode, sr).to_dict()
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=S.columns).fillna(0.0)


def switching_costs(columns, crypto: set, etf_bps: float = 10.0, crypto_bps: float = 15.0, fx_spread_bps: float = 30.0) -> backtest.CostModel:
    """Costs for moving capital between strategies (FX spread is a placeholder, not a quote)."""
    return backtest.CostModel(
        default_bps=etf_bps,
        bps={s: crypto_bps for s in columns if s in crypto},
        fx_spread_bps=fx_spread_bps,
        fx_symbols=frozenset(s for s in columns if s in crypto),
    )


def run(S: pd.DataFrame, mode: str, months: int, crypto: set = frozenset(), costs: backtest.CostModel | None = None, crypto_cap: float = 0.25, rf: pd.Series | None = None) -> backtest.BacktestResult:
    T = targets(S, mode, months, crypto, crypto_cap, rf=rf)
    return backtest.run(T, S, costs or switching_costs(S.columns, set(crypto)), lag=1)
