"""Daily total-return matrices built from raw snapshots.

Yahoo's stored `close` is split-adjusted but not dividend-adjusted, so the total
return on day t is (close_t + dividend_t) / close_{t-1} - 1. Crypto has no
dividends. Everything is in USD in Phase 2 (TRY assets arrive in Phase 5 and
will be converted with the fx source).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..store import SnapshotStore


def total_returns(bars: pd.DataFrame, symbols: list[str] | None = None) -> pd.DataFrame:
    """Wide frame: index = date, columns = symbol, values = daily total return.

    NaN before a symbol's first bar. Gaps inside a series (missing days) give one
    return spanning the gap, never a fabricated zero.
    """
    df = bars if symbols is None else bars[bars["symbol"].isin(symbols)]
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date")
        close = g["close"].to_numpy(dtype=float)
        div = g["dividends"].to_numpy(dtype=float) if "dividends" in g else np.zeros(len(g))
        r = np.full(len(g), np.nan)
        r[1:] = (close[1:] + np.nan_to_num(div[1:])) / close[:-1] - 1.0
        out[sym] = pd.Series(r, index=pd.to_datetime(g["date"]).values)
    wide = pd.DataFrame(out).sort_index()
    wide.index.name = "date"
    if symbols is not None:
        wide = wide.reindex(columns=symbols)
    return wide


def load_returns(store: SnapshotStore, source: str, symbols: list[str] | None = None, as_of: str | None = None) -> pd.DataFrame:
    return total_returns(store.load(source, as_of=as_of), symbols)


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in `index`."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)
