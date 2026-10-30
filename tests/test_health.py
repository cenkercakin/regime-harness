from datetime import date

from harness.health import check


def test_clean_pull_passes(bars):
    df = bars(["A"], "2026-01-01", 30)
    r = check(df, date(2026, 1, 31), ["A"], max_stale_days=2)
    assert r.ok and not r.warnings


def test_incomplete_bar_is_an_error(bars):
    df = bars(["A"], "2026-01-01", 31)  # last bar dated on the snapshot day
    r = check(df, date(2026, 1, 31), ["A"], max_stale_days=2)
    assert not r.ok and any("incomplete" in e for e in r.errors)


def test_bad_prices_and_missing_symbols(bars):
    df = bars(["A"], "2026-01-01", 10)
    df.loc[3, "close"] = 0
    r = check(df, date(2026, 1, 11), ["A", "B"], max_stale_days=2)
    assert any("non-positive" in e for e in r.errors)
    assert any("B missing" in e for e in r.errors)


def test_stale_and_jump_warnings(bars):
    df = bars(["A"], "2026-01-01", 10)
    df.loc[5:, "close"] *= 3  # looks like an unhandled split
    r = check(df, date(2026, 1, 25), ["A"], max_stale_days=2)
    assert r.ok
    assert any("days old" in w for w in r.warnings)
    assert any("daily moves" in w for w in r.warnings)


def test_empty_pull(bars):
    r = check(bars(["A"], "2026-01-01", 0), date(2026, 1, 2), ["A"], max_stale_days=2)
    assert not r.ok
