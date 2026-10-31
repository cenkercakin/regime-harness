"""Every library strategy must be causal, long-only and runnable through the engine."""

import numpy as np
import pandas as pd
import pytest

from harness.engine import backtest
from harness.strategies.library import CASH, COINS, LIBRARY, RISKY, MarketData

ETF_IDX = pd.bdate_range("2010-01-01", "2013-12-31")
CRYPTO_IDX = pd.date_range("2018-01-01", "2020-12-31", freq="D")


def etf_md(seed=0, idx=ETF_IDX):
    rng = np.random.default_rng(seed)
    syms = RISKY + [CASH]
    R = pd.DataFrame(rng.normal(0.0003, 0.011, (len(idx), len(syms))), index=idx, columns=syms)
    R[CASH] = 0.0001
    R.iloc[0] = np.nan
    close = 100 * (1 + R.fillna(0)).cumprod()
    span = pd.DataFrame(rng.uniform(0.002, 0.02, close.shape), index=idx, columns=syms)
    low = close * (1 - span * rng.uniform(0, 1, close.shape))
    high = low + close * span
    return MarketData(R, high, low, close)


def crypto_md(seed=0, idx=CRYPTO_IDX):
    rng = np.random.default_rng(seed)
    R = pd.DataFrame(rng.normal(0.001, 0.04, (len(idx), 2)), index=idx, columns=COINS)
    R.iloc[0] = np.nan
    return MarketData(R)


def perturb_after(md, cut, seed):
    """Replace everything after `cut` with unrelated data. Rows on or before `cut` must not change."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("R", "high", "low", "close"):
        df = getattr(md, name)
        if df is None:
            out[name] = None
            continue
        df = df.copy()
        later = df.index > cut
        df.loc[later] = df.loc[later].to_numpy() * rng.uniform(0.5, 1.5, df.loc[later].shape)
        out[name] = df
    return MarketData(**out)


@pytest.mark.parametrize("spec", LIBRARY, ids=[s.name for s in LIBRARY])
def test_no_lookahead(spec):
    md = etf_md() if spec.venue == "us_etfs" else crypto_md()
    cut = md.R.index[len(md.R) * 2 // 3]
    T1 = spec.fn(md)
    T2 = spec.fn(perturb_after(md, cut, seed=7))
    assert len(T1.loc[:cut]) > 0, "strategy produced no decisions before the cut"
    pd.testing.assert_frame_equal(T1.loc[:cut], T2.loc[:cut], check_dtype=False)


@pytest.mark.parametrize("spec", LIBRARY, ids=[s.name for s in LIBRARY])
def test_long_only_and_runs_in_engine(spec):
    md = etf_md(1) if spec.venue == "us_etfs" else crypto_md(1)
    T = spec.fn(md)
    vals = T.to_numpy(dtype=float)
    assert (vals >= -1e-12).all()
    assert (vals.sum(axis=1) <= 1 + 1e-9).all()
    res = backtest.run(T, md.R, backtest.CostModel(10.0), lag=1)
    assert np.isfinite(res.returns).all()


def test_library_names_unique_and_documented():
    names = [s.name for s in LIBRARY]
    assert len(names) == len(set(names))
    assert all(s.source and s.mechanism for s in LIBRARY)
    assert len(LIBRARY) >= 30
