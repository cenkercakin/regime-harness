import numpy as np
import pandas as pd
import pytest

from harness.strategies.grid import GridSpec, resample, simulate


def bars_from(close, start="2026-01-01", wiggle=0.0005):
    idx = pd.date_range(start, periods=len(close), freq="5min")
    c = np.asarray(close, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * (1 + wiggle), "low": np.minimum(o, c) * (1 - wiggle), "close": c}, index=idx)


N = 288 * 120   # 120 days of 5-minute bars


def test_sideways_market_is_profitable():
    t = np.arange(N)
    close = 100 * (1 + 0.05 * np.sin(2 * np.pi * t / (288 * 3)))
    r = simulate(bars_from(close), GridSpec(0.01, 20, "breakout"))
    assert r.fills > 200
    assert r.equity.iloc[-1] > 1.05                       # ends where it started, but harvested the swings


def test_downtrend_loses_and_pins_to_the_bottom():
    close = np.linspace(100, 40, N)
    r = simulate(bars_from(close), GridSpec(0.01, 20, "monthly"))
    assert r.equity.iloc[-1] < 0.5 + 0.5 * 0.4             # worse than just holding 50/50
    assert r.at_bottom > 0.3


def test_uptrend_sells_out_and_lags():
    close = np.linspace(100, 250, N)
    r = simulate(bars_from(close), GridSpec(0.01, 20, "monthly"))
    assert r.equity.iloc[-1] < 0.5 + 0.5 * 2.5             # worse than holding 50/50
    assert r.at_top > 0.3


def test_one_round_trip_exact_accounting():
    g, fee, tt = 1.01, 0.001, 0.0005
    idx = pd.date_range("2026-01-01", periods=3, freq="5min")
    b = pd.DataFrame({
        "open": [100.0, 100.0, 99.5],
        "high": [100.0, 100.0, 100.1],
        "low": [100.0, 98.9, 99.5],          # 98.9 <= 99.0099 * 0.9995 -> buy at L(-1)
        "close": [100.0, 99.5, 100.0],
    }, index=idx)
    r = simulate(b, GridSpec(0.01, 20, "monthly", fee=fee, trade_through=tt))
    # setup: buy 0.5 worth at 100 with fee
    coin = 0.5 / 100
    cash = 1.0 - 0.5 - 0.5 * fee
    u = coin / 10
    buy = 100 / g
    coin += u
    cash -= u * buy * (1 + fee)
    sell = 100.0                          # back to L(0): high 100.1 >= 100 * 1.0005
    coin -= u
    cash += u * sell * (1 - fee)
    assert r.fills == 2
    assert r.equity.iloc[-1] == pytest.approx(cash + coin * 100.0, rel=1e-12)


def test_touch_is_not_a_fill():
    idx = pd.date_range("2026-01-01", periods=2, freq="5min")
    level = 100 / 1.01
    b = pd.DataFrame({"open": [100.0, 100.0], "high": [100.0, 100.0], "low": [100.0, level], "close": [100.0, 100.0]}, index=idx)
    assert simulate(b, GridSpec(0.01)).fills == 0


def test_bar_crossing_both_sides_fills_only_one_side():
    idx = pd.date_range("2026-01-01", periods=2, freq="5min")
    b = pd.DataFrame({"open": [100.0, 100.0], "high": [100.0, 103.5], "low": [100.0, 99.0 * 0.98], "close": [100.0, 100.0]}, index=idx)
    r = simulate(b, GridSpec(0.01))
    # open is nearer the low: only the two buy levels fill (99.01, 98.03); no sells in the same bar
    assert r.fills == 2
    assert r.coin_share.iloc[-1] > 0.5


def test_no_lookahead():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, N)))
    b = bars_from(close)
    cut = b.index[N // 2]
    b2 = b.copy()
    later = b2.index > cut
    b2.loc[later, ["open", "high", "low", "close"]] *= rng.uniform(0.7, 1.3, (later.sum(), 1))
    for spec in (GridSpec(0.01, 20, "monthly"), GridSpec(0.025, 20, "breakout")):
        e1, e2 = simulate(b, spec).equity, simulate(b2, spec).equity
        pd.testing.assert_series_equal(e1.loc[:cut], e2.loc[:cut])


def test_monthly_recentre_count_and_resample():
    close = np.full(288 * 95, 100.0)        # Jan 1 -> early April
    b = bars_from(close)
    r = simulate(b, GridSpec(0.01, 20, "monthly"))
    assert r.recentres == 3
    h = resample(b, "1h")
    assert len(h) == len(b) // 12 and h["high"].max() == pytest.approx(b["high"].max())


def test_bad_spec_refused():
    b = bars_from(np.full(10, 100.0))
    with pytest.raises(ValueError):
        simulate(b, GridSpec(0.01, 20, "weekly"))
    with pytest.raises(ValueError):
        simulate(b, GridSpec(0.01, 7))


def test_fee_share_measured_against_value_at_the_time():
    b = bars_from(np.full(20, 100.0))
    r = simulate(b, GridSpec(0.01, 20, "monthly", fee=0.001))
    assert r.fee_frac == pytest.approx(0.5 * 0.001)     # only the opening 50% buy was charged
