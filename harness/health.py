"""Health checks run on every pull before it is stored.

Errors stop the source from being written (a bad pull must not enter the store).
Warnings are written into the snapshot's manifest entry so they stay visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

PRICE_COLS = ("open", "high", "low", "close")


@dataclass
class HealthReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check(
    df: pd.DataFrame,
    snapshot_date: date,
    expected_symbols: list[str],
    max_stale_days: int,
    max_abs_daily_return: float = 0.5,
) -> HealthReport:
    r = HealthReport()
    if df.empty:
        r.errors.append("pull returned no rows")
        return r

    dates = pd.to_datetime(df["date"]).dt.date
    if df.duplicated(["symbol", "date"]).any():
        r.errors.append("duplicate (symbol, date) rows")
    if (dates >= snapshot_date).any():
        r.errors.append("rows dated on or after the snapshot date (incomplete bar leaked in)")

    for c in PRICE_COLS:
        if c in df and (df[c] <= 0).any():
            r.errors.append(f"non-positive values in {c}")
    if {"high", "low"} <= set(df.columns) and (df["high"] < df["low"]).any():
        r.errors.append("high below low on some rows")

    got = set(df["symbol"].unique())
    for s in expected_symbols:
        if s not in got:
            r.errors.append(f"symbol {s} missing from pull")

    for sym, g in df.assign(_d=dates).groupby("symbol"):
        last = g["_d"].max()
        stale = (snapshot_date - last).days
        if stale > max_stale_days:
            r.warnings.append(f"{sym}: last bar {last} is {stale} days old")
        closes = g.sort_values("_d")["close"]
        rets = closes.pct_change().abs()
        jumps = rets[rets > max_abs_daily_return]
        if len(jumps):
            r.warnings.append(f"{sym}: {len(jumps)} daily moves above {max_abs_daily_return:.0%} (check splits)")
        if closes.isna().any():
            r.warnings.append(f"{sym}: {int(closes.isna().sum())} missing closes")
    return r
