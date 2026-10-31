"""One pull = fetch every source, health-check it, store the delta as today's snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from . import health
from .loaders import crypto, yahoo
from .store import SnapshotStore


def _crypto_fetch(cfg, start, end):
    return crypto.fetch_daily(cfg["symbols"], start, end, exchange_id=cfg["exchange"], timeframe=cfg.get("timeframe", "1d"))


def _yahoo_fetch(cfg, start, end):
    """Large universes go in batches with a pause, to stay under Yahoo's rate limits."""
    import time

    import pandas as pd

    syms, size = cfg["symbols"], cfg.get("batch_size", 0)
    if not size or len(syms) <= size:
        return yahoo.fetch_daily(syms, start, end)
    parts = []
    for i in range(0, len(syms), size):
        parts.append(yahoo.fetch_daily(syms[i:i + size], start, end))
        time.sleep(cfg.get("batch_pause_s", 2))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume", "dividends", "splits"])


DEFAULT_FETCHERS = {"crypto": _crypto_fetch, "yahoo": _yahoo_fetch}


@dataclass
class PullResult:
    source: str
    written: bool = False
    skipped: bool = False      # today's snapshot already exists (e.g. a manual run got there first)
    entry: dict | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def pull_source(store: SnapshotStore, name: str, cfg: dict, snapshot_date: date, fetchers=None) -> PullResult:
    fetchers = fetchers or DEFAULT_FETCHERS
    res = PullResult(source=name)
    snap_id = snapshot_date.isoformat()
    if snap_id in store.snapshot_ids(name):
        res.skipped = True
        res.warnings.append(f"snapshot {snap_id} already exists; nothing to do")
        return res

    start = date.fromisoformat(cfg["start"])
    entries = store.manifest(name)["snapshots"]
    max_dates = [e["max_date"] for e in entries if e.get("max_date")]
    if cfg.get("lookback_days") and max_dates:
        start = max(start, date.fromisoformat(max(max_dates)[:10]) - timedelta(days=cfg["lookback_days"]))

    try:
        df = fetchers[cfg["loader"]](cfg, start, snapshot_date)
    except Exception as exc:  # network, vendor format change, rate limit
        res.errors.append(f"fetch failed: {type(exc).__name__}: {exc}")
        return res

    expected = cfg["symbols"]
    if cfg.get("max_missing_frac"):
        df, expected, notes = _quarantine(df, expected, cfg["max_missing_frac"])
        if expected is None:
            res.errors.extend(notes)
            return res
        res.warnings.extend(notes)
    rep = health.check(
        df,
        snapshot_date,
        expected_symbols=expected,
        max_stale_days=cfg["max_stale_days"],
        max_abs_daily_return=cfg.get("max_abs_daily_return", 0.5),
    )
    res.errors, res.warnings = rep.errors, res.warnings + rep.warnings
    if not rep.ok:
        return res
    notes = res.warnings if len(res.warnings) <= 40 else res.warnings[:40] + [f"... and {len(res.warnings) - 40} more warnings"]
    res.entry = store.write(name, df, snap_id, notes=notes)
    res.written = True
    return res


def _quarantine(df, expected, max_missing_frac):
    """For large universes (hundreds of stocks), one bad or missing ticker must not
    block the rest. Drop rows with impossible prices; tolerate a few missing tickers.
    Returns (clean_df, symbols_to_expect, notes); symbols is None if too many are missing."""
    notes = []
    if len(df):
        bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])
        if bad.any():
            notes.append(f"dropped {int(bad.sum())} rows with impossible prices across {df.loc[bad, 'symbol'].nunique()} symbols")
            df = df[~bad]
    got = set(df["symbol"].unique()) if len(df) else set()
    missing = [s for s in expected if s not in got]
    if len(missing) > max_missing_frac * len(expected):
        return df, None, [f"{len(missing)} of {len(expected)} symbols missing (limit {max_missing_frac:.0%}), e.g. {missing[:10]}"]
    if missing:
        notes.append(f"{len(missing)} symbols returned no data: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    return df, [s for s in expected if s in got], notes
