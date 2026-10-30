"""Daily crypto candles from an exchange's public endpoints via ccxt (no account)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

DAY_MS = 86_400_000


def _make_exchange(exchange_id: str):
    import ccxt

    return getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 30_000})


def fetch_daily(
    symbols: list[str],
    start: date,
    end: date,
    exchange_id: str = "coinbaseexchange",
    exchange=None,
    page_limit: int = 300,
    max_pages: int = 100,
) -> pd.DataFrame:
    """Daily OHLCV for `symbols` from `start` up to but excluding `end` (UTC days).

    `end` is normally the snapshot date, so today's still-forming candle is never stored.
    """
    ex = exchange or _make_exchange(exchange_id)
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    for sym in symbols:
        since, pages = start_ms, 0
        while since < end_ms and pages < max_pages:
            batch = ex.fetch_ohlcv(sym, timeframe="1d", since=since, limit=page_limit)
            pages += 1
            batch = [c for c in batch if since <= c[0] < end_ms]
            if not batch:
                # Exchange may have no data yet at `since` (coin listed later): jump ahead.
                since += page_limit * DAY_MS
                continue
            rows.extend((sym, *c) for c in batch)
            since = batch[-1][0] + DAY_MS
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df.drop(columns="ts").assign(date=pd.Series(dtype="object"))
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.date
    df = df.drop(columns="ts").drop_duplicates(["symbol", "date"], keep="last")
    return df[["symbol", "date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
