"""Spot grid trading on intraday OHLC bars (pre-registered).

Mechanics
---------
* Start (and every re-centre): rebalance to 50% coin / 50% cash at a known price,
  then set the centre C to that price. Levels are geometric: L(k) = C * (1 + spacing)^k
  for k = -half .. +half (`levels` = 2 * half).
* Fixed coin quantity per level, u = coin / half, so selling `half` levels up sells
  everything and buying `half` levels down uses (slightly less than) all the cash.
* Position index p (0 at the centre). Active orders: buy u at L(p-1), sell u at L(p+1).
  A buy at L(p-1) moves p down by one; a sell at L(p+1) moves it up by one.

Conservative fill rules (no price data can show whether a limit order really filled)
--------------------------------------------------------------------------------------
* Trade-through: a buy fills only if the bar's low is at least `trade_through` below
  the level; a sell only if the high is at least that much above it. A touch is not a fill.
* Fills happen at the level price, never at a better gap price.
* If one bar crosses levels on both sides, only the side nearer the bar's open fills;
  the order of moves inside the bar is unknown, so no round trip is credited.
* A breakout re-centre is decided at a bar's close and executed at the next bar's open.
* Fees are charged on every fill and every rebalance, on traded notional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RECENTRE = ("monthly", "breakout")


@dataclass(frozen=True)
class GridSpec:
    spacing: float
    levels: int = 20
    recentre: str = "monthly"
    fee: float = 0.001
    trade_through: float = 0.0005

    @property
    def name(self) -> str:
        return f"grid_{self.spacing * 100:g}pct_{self.recentre}"


@dataclass
class GridResult:
    equity: pd.Series        # portfolio value at each bar close (starts at 1.0)
    daily: pd.Series         # daily returns (UTC days)
    fills: int
    recentres: int
    fees_paid: float
    at_bottom: float         # share of bars pinned at the lowest level (fully in coin)
    at_top: float            # share of bars pinned at the highest level (fully in cash)
    coin_share: pd.Series    # coin share of value, daily


def simulate(bars: pd.DataFrame, spec: GridSpec) -> GridResult:
    if spec.recentre not in RECENTRE:
        raise ValueError(f"recentre must be one of {RECENTRE}")
    if spec.levels % 2 or spec.levels < 2:
        raise ValueError("levels must be an even number >= 2")
    bars = bars.sort_index()
    o, h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    ts = bars.index
    month = (ts.year * 12 + ts.month).to_numpy()
    n, half, g, fee, tt = len(bars), spec.levels // 2, 1.0 + spec.spacing, spec.fee, spec.trade_through
    powers = g ** np.arange(-half, half + 1)

    cash, coin, fees_paid = 1.0, 0.0, 0.0
    fills = recentres = bottom = top = 0
    eq = np.empty(n)
    share = np.empty(n)
    levels = u = None
    p = 0
    pending = False

    def rebalance(price):
        nonlocal cash, coin, fees_paid
        value = cash + coin * price
        target = value / 2.0 / price
        trade = target - coin
        f = abs(trade) * price * fee
        cash -= trade * price + f
        coin = target
        fees_paid += f

    for t in range(n):
        new_month = t > 0 and month[t] != month[t - 1]
        if t == 0 or pending or (spec.recentre == "monthly" and new_month):
            rebalance(o[t])
            levels = o[t] * powers
            u = coin / half
            p = 0
            if t > 0:
                recentres += 1
            pending = False

        down = p > -half and lo[t] <= levels[p - 1 + half] * (1 - tt)
        up = p < half and h[t] >= levels[p + 1 + half] * (1 + tt)
        if down and up:
            down, up = ((o[t] - lo[t]) <= (h[t] - o[t])), ((o[t] - lo[t]) > (h[t] - o[t]))
        if down:
            while p > -half and lo[t] <= levels[p - 1 + half] * (1 - tt):
                price = levels[p - 1 + half]
                qty = min(u, cash / (price * (1 + fee)))
                if qty <= 0:
                    break
                coin += qty
                cash -= qty * price * (1 + fee)
                fees_paid += qty * price * fee
                p -= 1
                fills += 1
        elif up:
            while p < half and h[t] >= levels[p + 1 + half] * (1 + tt):
                price = levels[p + 1 + half]
                qty = min(u, coin)
                if qty <= 0:
                    break
                coin -= qty
                cash += qty * price * (1 - fee)
                fees_paid += qty * price * fee
                p += 1
                fills += 1

        if spec.recentre == "breakout" and t + 1 < n:
            if (p == -half and c[t] < levels[0] / g) or (p == half and c[t] > levels[-1] * g):
                pending = True
        bottom += p == -half
        top += p == half
        eq[t] = cash + coin * c[t]
        share[t] = coin * c[t] / eq[t] if eq[t] > 0 else 0.0

    equity = pd.Series(eq, index=ts)
    day = ts.normalize()
    d_eq = equity.groupby(day).last()
    daily = d_eq.pct_change()
    daily.iloc[0] = d_eq.iloc[0] / 1.0 - 1.0
    return GridResult(
        equity=equity, daily=daily, fills=fills, recentres=recentres, fees_paid=fees_paid,
        at_bottom=bottom / n, at_top=top / n, coin_share=pd.Series(share, index=ts).groupby(day).last(),
    )


def resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Coarser OHLC bars (e.g. '1h', '1D') from finer ones, for the resolution check."""
    out = bars.resample(rule, label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


SPECS = [
    GridSpec(0.01, 20, "monthly"),
    GridSpec(0.01, 20, "breakout"),
    GridSpec(0.025, 20, "monthly"),
    GridSpec(0.025, 20, "breakout"),
]
SOURCE = "Grid trading as practised by exchange grid bots; economically a threshold-rebalancing rule (Perold & Sharpe 1988, Dynamic strategies for asset allocation)"
MECHANISM = "if price oscillates within a range, buying each dip and selling each bounce harvests volatility; loses when price trends out of the range"
