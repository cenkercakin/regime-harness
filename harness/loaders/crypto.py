"""Crypto candles from an exchange's public endpoints via ccxt (no account).

Daily candles are stored with plain dates; intraday candles (e.g. 5m) with UTC timestamps.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone

import pandas as pd

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "6h": 21_600_000, "1d": 86_400_000}
DAY_MS = TF_MS["1d"]


def _make_exchange(exchange_id: str):
    import ccxt

    return getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 30_000})


def _fetch_page(ex, sym, timeframe, since, limit, retries=6):
    """One request, retried with backoff on network errors and rate limits."""
    for k in range(retries):
        try:
            return ex.fetch_ohlcv(sym, timeframe=timeframe, since=since, limit=limit)
        except Exception as exc:
            name = type(exc).__name__
            if k == retries - 1 or not any(w in name for w in ("Network", "RateLimit", "Timeout", "DDoS", "Unavailable")):
                raise
            time.sleep(2 ** k)


def fetch_daily(
    symbols: list[str],
    start: date,
    end: date,
    exchange_id: str = "coinbaseexchange",
    exchange=None,
    page_limit: int = 300,
    max_pages: int | None = None,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """OHLCV for `symbols` from `start` up to but excluding `end` (UTC midnight).

    `end` is normally the snapshot date, so no still-forming candle is ever stored.
    """
    ex = exchange or _make_exchange(exchange_id)
    tf = TF_MS[timeframe]
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp() * 1000)
    if max_pages is None:
        max_pages = (end_ms - start_ms) // (tf * page_limit) + 50
    rows = []
    for sym in symbols:
        since, pages = start_ms, 0
        while since < end_ms and pages < max_pages:
            batch = _fetch_page(ex, sym, timeframe, since, page_limit)
            pages += 1
            batch = [c for c in batch if since <= c[0] < end_ms]
            if not batch:
                # No data at `since` (coin listed later, or an exchange outage): jump ahead.
                since += page_limit * tf
                continue
            rows.extend((sym, *c) for c in batch)
            since = batch[-1][0] + tf
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    stamp = pd.to_datetime(df["ts"], unit="ms")
    df["date"] = stamp.dt.date if timeframe == "1d" else stamp
    df = df.drop(columns="ts").drop_duplicates(["symbol", "date"], keep="last")
    return df[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
