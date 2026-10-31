import numpy as np
import pandas as pd
import pytest

from harness import selector
from harness.engine.metrics import sharpe

IDX = pd.bdate_range("2010-01-01", "2019-12-31")


def alternating_library(seed=0):
    """Two strategies that each dominate for alternating 2-year stretches, plus noise ones."""
    rng = np.random.default_rng(seed)
    regime = ((IDX.year - 2010) // 2) % 2
    a = np.where(regime == 0, 0.0012, -0.0004) + rng.normal(0, 0.006, len(IDX))
    b = np.where(regime == 1, 0.0012, -0.0004) + rng.normal(0, 0.006, len(IDX))
    cols = {"A": a, "B": b}
    for i in range(3):
        cols[f"n{i}"] = rng.normal(0.0001, 0.006, len(IDX))
    return pd.DataFrame(cols, index=IDX)


def test_selector_catches_persistent_leadership():
    S = alternating_library()
    T = selector.targets(S, "hard", 3)
    leader = np.where(((T.index.year - 2010) // 2) % 2 == 0, "A", "B")
    held = T.idxmax(axis=1).to_numpy()
    # Chance would be 1/5. A 3-month window is noisy, so noise strategies win some months by luck.
    assert (held == leader).mean() > 0.5
    hard = selector.run(S, "hard", 3).returns
    blend = selector.run(S, "blend", 3).returns
    assert sharpe(hard, 252) > sharpe(blend, 252)


def test_selector_cannot_beat_blend_on_pure_noise():
    rng = np.random.default_rng(1)
    S = pd.DataFrame(rng.normal(0.0002, 0.008, (len(IDX), 8)), index=IDX, columns=[f"s{i}" for i in range(8)])
    hard = selector.run(S, "hard", 6).returns
    blend = selector.run(S, "blend", 6).returns
    assert sharpe(hard, 252) < sharpe(blend, 252) + 0.3


def test_crypto_cap_is_enforced():
    S = alternating_library()
    T = selector.targets(S, "hard", 3, crypto={"A"}, crypto_cap=0.25)
    assert (T["A"] <= 0.25 + 1e-9).all()
    assert np.allclose(T.sum(axis=1), 1.0)
    T2 = selector.targets(S, "tilt", 3, crypto={"A", "B"}, crypto_cap=0.25)
    assert ((T2["A"] + T2["B"]) <= 0.25 + 1e-9).all()


def test_selector_decision_uses_only_past():
    S = alternating_library()
    cut = pd.Timestamp("2015-06-30")
    T1 = selector.targets(S, "tilt", 6)
    S2 = S.copy()
    S2.loc[S2.index > cut] *= -3
    T2 = selector.targets(S2, "tilt", 6)
    pd.testing.assert_frame_equal(T1.loc[:cut], T2.loc[:cut])


def test_switching_into_crypto_pays_fx_spread():
    c = selector.switching_costs(["etf_strat", "coin_strat"], {"coin_strat"}, etf_bps=10, crypto_bps=15, fx_spread_bps=30)
    assert list(c.per_side(["etf_strat", "coin_strat"])) == pytest.approx([0.001, 0.0045])


def test_unknown_mode_refused():
    with pytest.raises(ValueError):
        selector.targets(alternating_library(), "best_guess", 6)


def test_cash_does_not_win_on_raw_sharpe_when_rf_given():
    """A T-bill fund has a huge raw Sharpe (tiny volatility). Above cash, its Sharpe is ~0."""
    rng = np.random.default_rng(2)
    rf = pd.Series(0.0001, index=IDX)
    S = pd.DataFrame({
        "cash": rf + rng.normal(0, 0.00002, len(IDX)),
        "equity": 0.0005 + rng.normal(0, 0.01, len(IDX)),
    }, index=IDX)
    raw = selector.targets(S, "hard", 12).idxmax(axis=1)
    above = selector.targets(S, "hard", 12, rf=rf).idxmax(axis=1)
    assert (raw == "cash").mean() > 0.9          # the v1 bug
    assert (above == "equity").mean() > 0.7      # fixed
