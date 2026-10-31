"""Time-series momentum across asset classes (Moskowitz, Ooi & Pedersen 2012 style, long-only).

Each month-end, every risky ETF whose trailing total return over `lookback`
months beats cash (BIL) over the same window keeps its equal-weight slot;
otherwise that slot moves to BIL. Signal uses closes up to the month-end only;
the engine fills at the next close.
"""

from __future__ import annotations

import pandas as pd

from ..engine.returns import month_end_dates

RISKY = ["VT", "SPY", "EFA", "EEM", "TLT", "GLD"]
CASH = "BIL"


def targets(returns: pd.DataFrame, lookback_months: int, risky=RISKY, cash=CASH) -> pd.DataFrame:
    idx = (1.0 + returns.fillna(0.0)).cumprod()
    idx = idx.where(returns.notna().cumsum() > 0)  # NaN before a symbol exists
    me = month_end_dates(returns.index)
    rows = {}
    for i, d in enumerate(me):
        if i < lookback_months:
            continue
        past = me[i - lookback_months]
        live = [s for s in risky if pd.notna(idx.at[past, s]) and pd.notna(idx.at[d, s])]
        if not live or pd.isna(idx.at[past, cash]):
            continue
        cash_ret = idx.at[d, cash] / idx.at[past, cash] - 1.0
        w = {s: 0.0 for s in returns.columns}
        slot = 1.0 / len(live)
        for s in live:
            if idx.at[d, s] / idx.at[past, s] - 1.0 > cash_ret:
                w[s] += slot
            else:
                w[cash] += slot
        rows[d] = w
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=returns.columns).fillna(0.0)


def equal_weight_targets(returns: pd.DataFrame, risky=RISKY) -> pd.DataFrame:
    """Benchmark: equal weight in whichever risky ETFs exist, rebalanced monthly."""
    me = month_end_dates(returns.index)
    rows = {}
    for d in me:
        live = [s for s in risky if pd.notna(returns.at[d, s])]
        if live:
            rows[d] = {s: (1.0 / len(live) if s in live else 0.0) for s in returns.columns}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=returns.columns).fillna(0.0)
