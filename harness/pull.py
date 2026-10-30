"""One pull = fetch every source, health-check it, store the delta as today's snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from . import health
from .loaders import crypto, yahoo
from .store import SnapshotStore


def _crypto_fetch(cfg, start, end):
    return crypto.fetch_daily(cfg["symbols"], start, end, exchange_id=cfg["exchange"])


def _yahoo_fetch(cfg, start, end):
    return yahoo.fetch_daily(cfg["symbols"], start, end)


DEFAULT_FETCHERS = {"crypto": _crypto_fetch, "yahoo": _yahoo_fetch}


@dataclass
class PullResult:
    source: str
    written: bool = False
    entry: dict | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def pull_source(store: SnapshotStore, name: str, cfg: dict, snapshot_date: date, fetchers=None) -> PullResult:
    fetchers = fetchers or DEFAULT_FETCHERS
    res = PullResult(source=name)
    snap_id = snapshot_date.isoformat()
    if snap_id in store.snapshot_ids(name):
        res.errors.append(f"snapshot {snap_id} already exists")
        return res

    start = date.fromisoformat(cfg["start"])
    entries = store.manifest(name)["snapshots"]
    max_dates = [e["max_date"] for e in entries if e.get("max_date")]
    if cfg.get("lookback_days") and max_dates:
        start = max(start, date.fromisoformat(max(max_dates)) - timedelta(days=cfg["lookback_days"]))

    try:
        df = fetchers[cfg["loader"]](cfg, start, snapshot_date)
    except Exception as exc:  # network, vendor format change, rate limit
        res.errors.append(f"fetch failed: {type(exc).__name__}: {exc}")
        return res

    rep = health.check(
        df,
        snapshot_date,
        expected_symbols=cfg["symbols"],
        max_stale_days=cfg["max_stale_days"],
        max_abs_daily_return=cfg.get("max_abs_daily_return", 0.5),
    )
    res.errors, res.warnings = rep.errors, rep.warnings
    if not rep.ok:
        return res
    res.entry = store.write(name, df, snap_id, notes=rep.warnings)
    res.written = True
    return res
