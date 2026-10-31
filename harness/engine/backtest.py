"""Drift-aware daily backtest with execution lag and per-symbol costs.

Timing contract (the part most backtests get wrong)
---------------------------------------------------
* A strategy produces a *target* row on date t using data up to and including t.
* The order executes at the close of t + `lag` bars (lag >= 1). `lag=1` = next
  close for ETFs and crypto; use `lag=2` for TEFAS-style forward pricing, where
  an order placed today fills at a NAV published later.
* Between rebalances, weights drift with prices (no free daily rebalancing).
* Cost on a fill = sum(|target - drifted weight| * per-side cost).
* Weight not invested sits in cash earning 0; hold a cash ETF (BIL) explicitly
  if you want T-bill yield.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Per-side cost in basis points of traded notional.

    `fx_spread_bps` is added for symbols in `fx_symbols`: assets bought with TRY
    that must be converted (BIST, TEFAS from a USD base). Placeholder until
    Midas/bank spreads are taken from published fee pages.
    """

    default_bps: float = 10.0
    bps: dict = field(default_factory=dict)
    fx_spread_bps: float = 0.0
    fx_symbols: frozenset = frozenset()

    def per_side(self, symbols) -> np.ndarray:
        return np.array(
            [self.bps.get(s, self.default_bps) + (self.fx_spread_bps if s in self.fx_symbols else 0.0) for s in symbols]
        ) / 1e4

    def scaled(self, k: float) -> "CostModel":
        return CostModel(self.default_bps * k, {s: v * k for s, v in self.bps.items()}, self.fx_spread_bps * k, self.fx_symbols)


CRYPTO_COSTS = CostModel(default_bps=15.0)          # 0.10% fee + 0.05% slippage per side
ETF_COSTS = CostModel(default_bps=10.0)             # placeholder until Midas fee page is checked


@dataclass
class BacktestResult:
    returns: pd.Series      # daily net portfolio return
    gross: pd.Series        # before costs
    turnover: pd.Series     # one-sided traded fraction per day
    weights: pd.DataFrame   # end-of-day weights after any fill


class TimingError(ValueError):
    pass


def run(targets: pd.DataFrame, returns: pd.DataFrame, costs: CostModel, lag: int = 1) -> BacktestResult:
    """Simulate `targets` (rows = rebalance dates, NaN rows ignored) over daily `returns`."""
    if lag < 1:
        raise TimingError("lag must be >= 1: a signal computed at close t cannot also fill at close t")
    targets = targets.reindex(columns=returns.columns).dropna(how="all")
    if targets.empty:
        raise ValueError("no target rows")
    if not targets.index.isin(returns.index).all():
        raise TimingError("target dates must be trading dates in the returns index")
    tw = targets.fillna(0.0).to_numpy(dtype=float)
    if (tw < -1e-12).any():
        raise ValueError("Phase 2 engine is long-only")
    if (tw.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("target weights sum above 1 (no leverage)")

    dates = returns.index
    pos = {d: i for i, d in enumerate(dates)}
    fills = {}
    for d, row in zip(targets.index, tw):
        k = pos[d] + lag
        if k < len(dates):
            fills[k] = row  # later signal for the same fill day overwrites earlier

    R = returns.to_numpy(dtype=float)
    avail = ~np.isnan(R)
    R = np.nan_to_num(R)
    c = costs.per_side(returns.columns)
    n, m = R.shape
    w = np.zeros(m)
    net, gross, turn = np.zeros(n), np.zeros(n), np.zeros(n)
    W = np.zeros((n, m))
    for t in range(n):
        g = float(w @ R[t])
        if w.any():
            w = w * (1.0 + R[t]) / (1.0 + g)
        gross[t] = g
        cost = 0.0
        if t in fills:
            tgt = fills[t]
            if (tgt[~avail[t]] > 0).any():
                bad = list(returns.columns[(tgt > 0) & ~avail[t]])
                raise TimingError(f"{dates[t].date()}: target holds {bad} with no price that day")
            trade = np.abs(tgt - w)
            turn[t] = trade.sum()
            cost = float(trade @ c)
            w = tgt.copy()
        net[t] = (1.0 + g) * (1.0 - cost) - 1.0
        W[t] = w
    idx = dates
    return BacktestResult(
        returns=pd.Series(net, index=idx),
        gross=pd.Series(gross, index=idx),
        turnover=pd.Series(turn, index=idx),
        weights=pd.DataFrame(W, index=idx, columns=returns.columns),
    )
