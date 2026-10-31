import numpy as np
import pandas as pd
import pytest

from harness.engine import backtest
from harness.engine.backtest import CostModel, TimingError
from harness.engine.metrics import max_drawdown, sharpe
from harness.engine.returns import month_end_dates, total_returns

FREE = CostModel(default_bps=0.0)


def rets(values, cols=("A", "B")):
    idx = pd.bdate_range("2026-01-01", periods=len(values))
    return pd.DataFrame(values, index=idx, columns=list(cols), dtype=float)


def test_total_return_includes_dividend_and_keeps_gaps():
    bars = pd.DataFrame({
        "symbol": ["X"] * 3, "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]).date,
        "close": [100.0, 99.0, 101.0], "dividends": [0.0, 2.0, 0.0],
    })
    r = total_returns(bars)["X"]
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.01)          # (99 + 2) / 100 - 1
    assert r.iloc[2] == pytest.approx(101 / 99 - 1)


def test_signal_fills_next_close_not_same_close():
    R = rets([[0.0, 0], [0.10, 0], [0.10, 0], [0.0, 0]])
    tg = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[R.index[0]])
    out = backtest.run(tg, R, FREE, lag=1).returns
    # Signal on day 0 -> fill at close of day 1 -> first exposure is day 2
    assert out.iloc[1] == 0.0 and out.iloc[2] == pytest.approx(0.10)


def test_perfect_foresight_of_same_day_is_useless():
    rng = np.random.default_rng(1)
    R = rets(rng.normal(0, 0.01, size=(500, 2)))
    # "Cheating" signal: on each day hold A only if A goes up THAT day
    tg = pd.DataFrame({"A": (R["A"] > 0).astype(float), "B": 0.0}, index=R.index)
    out = backtest.run(tg, R, FREE, lag=1).returns
    assert abs(sharpe(out, 252)) < 1.5   # with lag the leak is gone; same-day fill would give Sharpe > 20


def test_lag_zero_refused():
    R = rets([[0.0, 0], [0.1, 0]])
    with pytest.raises(TimingError):
        backtest.run(pd.DataFrame({"A": [1.0]}, index=[R.index[0]]), R, FREE, lag=0)


def test_costs_charged_on_turnover_and_scale():
    R = rets([[0.0, 0]] * 4)
    tg = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[R.index[0], R.index[1]])
    res = backtest.run(tg, R, CostModel(default_bps=10.0), lag=1)
    assert res.turnover.iloc[1] == pytest.approx(1.0)     # cash -> A
    assert res.turnover.iloc[2] == pytest.approx(2.0)     # A -> B: sell 1, buy 1
    assert res.returns.iloc[2] == pytest.approx(-0.002)
    res2 = backtest.run(tg, R, CostModel(default_bps=10.0).scaled(2), lag=1)
    assert res2.returns.iloc[2] == pytest.approx(-0.004)


def test_fx_spread_applies_only_to_try_assets():
    c = CostModel(default_bps=10, fx_spread_bps=50, fx_symbols=frozenset({"THYAO.IS"}))
    assert list(c.per_side(["SPY", "THYAO.IS"])) == pytest.approx([0.001, 0.006])


def test_weights_drift_between_rebalances():
    R = rets([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    tg = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[R.index[0]])
    w = backtest.run(tg, R, FREE, lag=1).weights
    assert w.iloc[2]["A"] == pytest.approx(2 / 3)  # A doubled, no free rebalance back to 50/50


def test_holding_unpriced_asset_refused():
    R = rets([[0.0, np.nan], [0.0, np.nan], [0.0, 0.0]])
    with pytest.raises(TimingError):
        backtest.run(pd.DataFrame({"A": [0.0], "B": [1.0]}, index=[R.index[0]]), R, FREE, lag=1)


def test_leverage_and_shorts_refused():
    R = rets([[0.0, 0]] * 3)
    with pytest.raises(ValueError):
        backtest.run(pd.DataFrame({"A": [0.8], "B": [0.8]}, index=[R.index[0]]), R, FREE)
    with pytest.raises(ValueError):
        backtest.run(pd.DataFrame({"A": [-0.5], "B": [0.5]}, index=[R.index[0]]), R, FREE)


def test_metrics_and_month_ends():
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    idx = pd.bdate_range("2026-01-01", "2026-03-31")
    me = month_end_dates(idx)
    assert [d.day for d in me] == [30, 27, 31]


def _shortable(R, value=True):
    return pd.DataFrame(value, index=R.index, columns=R.columns)


def test_short_profits_when_asset_falls_and_pays_borrow():
    R = rets([[0.0, 0.0], [0.0, 0.0], [-0.10, 0.0], [0.0, 0.0]])
    tg = pd.DataFrame({"A": [-0.5], "B": [0.5]}, index=[R.index[0]])
    rules = backtest.ShortRules(_shortable(R), borrow_bps_annual=252 * 10, periods_per_year=252)   # 10 bps per day
    out = backtest.run(tg, R, FREE, lag=1, shorts=rules).returns
    assert out.iloc[2] == pytest.approx((1 + 0.05) * (1 - 0.0005) - 1)   # +5% on half short, less one day's borrow


def test_ban_forces_cover_and_blocks_new_shorts():
    R = rets([[0.0, 0.0]] * 6)
    sh = _shortable(R)
    sh.iloc[3:, 0] = False                              # ban on A from day 3
    tg = pd.DataFrame({"A": [-0.5, -0.5], "B": [0.5, 0.5]}, index=[R.index[0], R.index[3]])
    w = backtest.run(tg, R, FREE, lag=1, shorts=backtest.ShortRules(sh, 0.0)).weights
    assert w.iloc[2]["A"] == pytest.approx(-0.5)
    assert w.iloc[3]["A"] == 0.0 and w.iloc[5]["A"] == 0.0   # covered, and not re-opened during the ban


def test_short_limits():
    R = rets([[0.0, 0.0]] * 3)
    rules = backtest.ShortRules(_shortable(R))
    with pytest.raises(ValueError):          # gross 2.4 > 2
        backtest.run(pd.DataFrame({"A": [-1.2], "B": [1.2]}, index=[R.index[0]]), R, FREE, shorts=rules)
    with pytest.raises(ValueError):          # net short
        backtest.run(pd.DataFrame({"A": [-0.8], "B": [0.2]}, index=[R.index[0]]), R, FREE, shorts=rules)
    with pytest.raises(ValueError):          # shorts without rules
        backtest.run(pd.DataFrame({"A": [-0.5], "B": [0.5]}, index=[R.index[0]]), R, FREE)


def test_suspended_stock_is_held_not_traded_in_hold_mode():
    R = rets([[0.0, 0.0], [np.nan, 0.0], [0.05, 0.0], [0.0, 0.0]])
    tg = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[R.index[0]])
    first = backtest.run(tg, R, FREE, lag=1, unpriced="hold")      # buy A on day 1: A suspended
    assert first.weights.iloc[1]["A"] == 0.0                       # could not buy on the suspended day
    assert first.weights.iloc[2]["A"] == pytest.approx(1.0)        # bought the next day it traded
