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
    # Optional time-varying cost (dates x symbols, bps per side), e.g. cheaper while a
    # stock is in the most-liquid group. Overrides `bps`/`default_bps` where present.
    bps_frame: pd.DataFrame | None = field(default=None, compare=False, hash=False)

    def per_side(self, symbols) -> np.ndarray:
        return np.array(
            [self.bps.get(s, self.default_bps) + (self.fx_spread_bps if s in self.fx_symbols else 0.0) for s in symbols]
        ) / 1e4

    def scaled(self, k: float) -> "CostModel":
        return CostModel(self.default_bps * k, {s: v * k for s, v in self.bps.items()}, self.fx_spread_bps * k, self.fx_symbols,
                         None if self.bps_frame is None else self.bps_frame * k)


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


@dataclass(frozen=True)
class ShortRules:
    """Short selling, when allowed.

    * `shortable`: bool frame (dates x symbols). False on a date means no short may
      be held that day: a short held into a False day is force-covered at that close.
      This is how SPK short-selling bans and universe membership are applied.
    * `borrow_bps_annual`: stock-lending fee on the short notional, accrued daily.
    * Gross exposure (|long| + |short|) is capped at `max_gross`; net exposure must
      stay within [0, 1] (no net-short books, no leverage on the net).
    """

    shortable: pd.DataFrame
    borrow_bps_annual: float = 500.0
    periods_per_year: float = 252.0
    max_gross: float = 2.0


def run(targets: pd.DataFrame, returns: pd.DataFrame, costs: CostModel, lag: int = 1, shorts: ShortRules | None = None,
        unpriced: str = "raise") -> BacktestResult:
    """Simulate `targets` (rows = rebalance dates, NaN rows ignored) over daily `returns`.

    `unpriced`: what to do when a fill wants to trade a symbol with no price that day.
    "raise" (default; ETFs, crypto) treats it as a bug. "hold" (single stocks, which get
    suspended) leaves that position as it is until the stock trades again."""
    if lag < 1:
        raise TimingError("lag must be >= 1: a signal computed at close t cannot also fill at close t")
    targets = targets.reindex(columns=returns.columns).dropna(how="all")
    if targets.empty:
        raise ValueError("no target rows")
    if not targets.index.isin(returns.index).all():
        raise TimingError("target dates must be trading dates in the returns index")
    tw = targets.fillna(0.0).to_numpy(dtype=float)
    if shorts is None:
        if (tw < -1e-12).any():
            raise ValueError("short positions need ShortRules (engine is long-only by default)")
        if (tw.sum(axis=1) > 1 + 1e-9).any():
            raise ValueError("target weights sum above 1 (no leverage)")
    else:
        net_exp, gross_exp = tw.sum(axis=1), np.abs(tw).sum(axis=1)
        if (net_exp > 1 + 1e-9).any() or (net_exp < -1e-9).any():
            raise ValueError("net exposure must stay within [0, 1]")
        if (gross_exp > shorts.max_gross + 1e-9).any():
            raise ValueError(f"gross exposure above {shorts.max_gross}")

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
    C = None
    if costs.bps_frame is not None:
        C = costs.bps_frame.reindex(index=returns.index, columns=returns.columns).ffill().to_numpy(dtype=float) / 1e4
    n, m = R.shape
    can_short = None
    if shorts is not None:
        can_short = shorts.shortable.reindex(index=returns.index, columns=returns.columns).fillna(False).to_numpy(dtype=bool)
        borrow = shorts.borrow_bps_annual / 1e4 / shorts.periods_per_year
    w = np.zeros(m)
    net, gross, turn = np.zeros(n), np.zeros(n), np.zeros(n)
    W = np.zeros((n, m))
    for t in range(n):
        cost = 0.0
        if can_short is not None and (w < 0).any():
            cost += float(-w[w < 0].sum()) * borrow   # fee on the short actually held during the day
        g = float(w @ R[t])
        if w.any():
            w = w * (1.0 + R[t]) / (1.0 + g)
        gross[t] = g
        tgt = fills.get(t)
        if can_short is not None:
            banned = (w < 0) & ~can_short[t]
            if banned.any():                       # force-cover shorts that became illegal today
                tgt = (w if tgt is None else tgt).copy()
                tgt[banned] = 0.0
            if tgt is not None and ((tgt < 0) & ~can_short[t]).any():
                tgt = tgt.copy()
                tgt[(tgt < 0) & ~can_short[t]] = 0.0   # a new short on an unshortable name is not opened
        if tgt is not None:
            stuck = ~avail[t] & (np.abs(tgt - w) > 1e-12)
            if stuck.any() and unpriced == "hold":
                wanted = tgt.copy()
                tgt = tgt.copy()
                tgt[stuck] = w[stuck]
                if t + 1 < n and (t + 1) not in fills:
                    fills[t + 1] = wanted               # retry the original order next day
            elif (np.abs(tgt[~avail[t]]) > 0).any():
                bad = list(returns.columns[(np.abs(tgt) > 0) & ~avail[t]])
                raise TimingError(f"{dates[t].date()}: target holds {bad} with no price that day")
            trade = np.abs(tgt - w)
            turn[t] = trade.sum()
            ct = c if C is None else np.where(np.isnan(C[t]), c, C[t])
            cost += float(trade @ ct)
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
