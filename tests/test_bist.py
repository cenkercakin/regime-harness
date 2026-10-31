import numpy as np
import pandas as pd
import pytest

from harness.engine import backtest
from harness.strategies import bist
from harness.strategies.bist import BIST_LIBRARY

IDX = pd.bdate_range("2018-01-01", "2021-12-31")
SYMS = [f"S{i:02d}.IS" for i in range(40)]


def raw(seed=0):
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0005, 0.012, len(IDX))
    r = pd.DataFrame(common[:, None] + rng.normal(0, 0.015, (len(IDX), len(SYMS))), index=IDX, columns=SYMS)
    r.iloc[0] = np.nan
    r.iloc[:300, -5:] = np.nan                               # late listings
    turnover = pd.DataFrame(rng.lognormal(15, 1, (len(IDX), len(SYMS))), index=IDX, columns=SYMS)
    turnover *= np.linspace(3, 0.5, len(SYMS))               # liquidity ranking
    index_try = pd.Series(common, index=IDX)
    usdtry = pd.Series(np.exp(np.cumsum(rng.normal(0.0008, 0.005, len(IDX)))) * 5, index=IDX)
    return r, turnover, index_try, usdtry


def data(seed=0):
    return bist.build(*raw(seed))


def test_usd_conversion():
    r = pd.Series([np.nan, 0.10], index=IDX[:2])
    fx = pd.Series([10.0, 11.0], index=IDX[:2])
    assert bist.to_usd(r, fx).iloc[1] == pytest.approx(1.10 * 10 / 11 - 1)   # +10% in TRY, TRY fell 10%


def test_jump_is_neutralised_and_stock_excluded():
    r, t, i, fx = raw()
    r.iloc[500, 0] = -0.5                                     # unadjusted bonus issue
    bd = bist.build(r, t, i, fx)
    assert bd.flagged.iloc[500, 0]
    assert abs(bd.R.iloc[500, 0]) < 0.1
    d = IDX[500]
    me = [k for k in bd.members if k >= d][0]
    assert SYMS[0] not in bd.members[me]


def test_universe_uses_history_and_liquidity():
    bd = data()
    assert bd.members[sorted(bd.members)[5]] == []             # under a year of history: nobody qualifies
    d = sorted(bd.members)[14]
    u = bd.members[d]
    assert all(s not in u for s in SYMS[-5:])                 # not enough history yet
    assert u[0] == SYMS[0] or SYMS[0] in u[:5]                # most liquid near the top


def test_shortable_mask_respects_bans(tmp_path):
    bd = data()
    m = bist.shortable_mask(bd)
    ban = bist.ban_days(m.index)
    assert not m.loc[ban.to_numpy()].any().any()
    assert m.loc[~ban.to_numpy()].sum(axis=1).max() <= 30


@pytest.mark.parametrize("spec", BIST_LIBRARY, ids=[s.name for s in BIST_LIBRARY])
def test_bist_no_lookahead(spec):
    r, t, i, fx = raw(1)
    cut = IDX[len(IDX) * 2 // 3]
    bd1 = bist.build(r, t, i, fx)
    rng = np.random.default_rng(3)
    later = r.index > cut
    r2, t2, i2 = r.copy(), t.copy(), i.copy()
    r2.loc[later] = r2.loc[later] * rng.uniform(0.5, 1.5, r2.loc[later].shape)
    t2.loc[later] = t2.loc[later] * rng.uniform(0.2, 5, t2.loc[later].shape)
    i2.loc[later] = i2.loc[later] * -1
    bd2 = bist.build(r2, t2, i2, fx)
    T1, T2 = spec.fn(bd1), spec.fn(bd2)
    assert len(T1.loc[:cut]) > 0
    pd.testing.assert_frame_equal(T1.loc[:cut], T2.loc[:cut], check_dtype=False)


@pytest.mark.parametrize("spec", BIST_LIBRARY, ids=[s.name for s in BIST_LIBRARY])
def test_bist_runs_in_engine_with_short_rules(spec):
    bd = data(2)
    T = spec.fn(bd)
    rules = backtest.ShortRules(bist.shortable_mask(bd), borrow_bps_annual=500)
    costs = backtest.CostModel(20.0, bps_frame=bist.cost_frame(bd))
    res = backtest.run(T, bd.R, costs, lag=1, shorts=rules)
    assert np.isfinite(res.returns).all()
    if not (T < 0).any().any():
        assert (res.weights >= -1e-12).all().all()
