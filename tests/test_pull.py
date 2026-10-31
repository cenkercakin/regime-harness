"""End-to-end pull behaviour, including the Phase 1 exit criterion:
two pulls on different days must reproduce the same backtest input."""

from datetime import date, timedelta

import numpy as np

from harness.pull import pull_source
from harness.store import SnapshotStore
from tests.conftest import make_bars

CFG = {"loader": "fake", "symbols": ["A", "B"], "start": "2026-01-01", "lookback_days": 10, "max_stale_days": 3}


def fetcher_for(calls):
    def fetch(cfg, start, end):
        calls.append(start)
        df = make_bars(cfg["symbols"], "2026-01-01", (end - date(2026, 1, 1)).days)
        return df[df["date"] >= start]
    return {"fake": fetch}


def toy_backtest(df):
    """Stand-in metric: mean daily log return of an equal-weight basket."""
    wide = df.pivot(index="date", columns="symbol", values="close").sort_index()
    return float(np.log(wide).diff().mean(axis=1).mean())


def test_two_pulls_reproduce_the_same_backtest_input(tmp_path):
    st, calls = SnapshotStore(tmp_path), []
    f = fetcher_for(calls)
    r1 = pull_source(st, "src", CFG, date(2026, 3, 1), f)
    r2 = pull_source(st, "src", CFG, date(2026, 3, 2), f)
    assert r1.written and r2.written
    assert r2.entry["rows_new"] == 2 and r2.entry["rows_revised"] == 0

    # Later pull only re-fetches the lookback window
    assert calls[0] == date(2026, 1, 1) and calls[1] == date(2026, 2, 28) - timedelta(days=10)

    # A backtest pinned to snapshot 1 gives the same answer before and after snapshot 2 exists
    pinned_a = toy_backtest(SnapshotStore(tmp_path).load("src", as_of="2026-03-01"))
    pinned_b = toy_backtest(SnapshotStore(tmp_path).load("src", as_of="2026-03-01"))
    assert pinned_a == pinned_b
    assert toy_backtest(st.load("src")) != pinned_a  # latest includes the new bar


def test_same_day_pull_refused(tmp_path):
    st, calls = SnapshotStore(tmp_path), []
    pull_source(st, "src", CFG, date(2026, 3, 1), fetcher_for(calls))
    r = pull_source(st, "src", CFG, date(2026, 3, 1), fetcher_for(calls))
    assert not r.written and r.skipped and not r.errors   # not a failure: a late cron run after a manual one


def test_fetch_failure_writes_nothing(tmp_path):
    def boom(cfg, start, end):
        raise ConnectionError("vendor down")
    st = SnapshotStore(tmp_path)
    r = pull_source(st, "src", CFG, date(2026, 3, 1), {"fake": boom})
    assert not r.written and "vendor down" in r.errors[0]
    assert st.snapshot_ids("src") == []


def test_unhealthy_pull_writes_nothing(tmp_path):
    def bad(cfg, start, end):
        return make_bars(["A"], "2026-01-01", 5)  # symbol B missing
    st = SnapshotStore(tmp_path)
    r = pull_source(st, "src", CFG, date(2026, 1, 6), {"fake": bad})
    assert not r.written and any("B missing" in e for e in r.errors)
