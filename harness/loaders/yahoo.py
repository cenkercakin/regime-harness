"""Daily bars from Yahoo Finance via yfinance (unofficial, free, can break).

We store raw split-adjusted OHLC plus dividend and split events, not Yahoo's
dividend-adjusted close. Adjusted series are rebuilt from the raw data in code,
so a backtest never depends on how Yahoo chose to adjust on a given day.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}


def _download(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),  # exclusive
        auto_adjust=False,
        actions=True,
        progress=False,
        group_by="ticker",
        threads=False,
        multi_level_index=True,
    )


def fetch_daily(tickers: list[str], start: date, end: date, downloader=_download) -> pd.DataFrame:
    """Daily bars for `tickers` from `start` up to but excluding `end`."""
    raw = downloader(tickers, start, end)
    frames = []
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            sub = raw[t].copy()
        else:
            sub = raw.copy()
        sub = sub.rename(columns=COLUMN_MAP)
        sub = sub[[c for c in COLUMN_MAP.values() if c in sub.columns]]
        sub = sub.dropna(subset=["close"])
        if sub.empty:
            continue
        sub.index = pd.to_datetime(sub.index).tz_localize(None) if getattr(sub.index, "tz", None) else pd.to_datetime(sub.index)
        sub = sub.reset_index(names="date")
        sub["symbol"] = t
        for c in ("dividends", "splits"):
            if c not in sub:
                sub[c] = 0.0
            sub[c] = sub[c].fillna(0.0)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", *COLUMN_MAP.values()])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] < end]
    cols = ["symbol", "date", "open", "high", "low", "close", "volume", "dividends", "splits"]
    return df[cols].reset_index(drop=True)
