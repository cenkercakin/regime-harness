"""Gate calibration: it must reject luck and accept a real (planted) edge."""

import numpy as np
import pandas as pd
import pytest

from harness.gate import VaultError, research_gate, stats, vault_test
from harness.gate.splits import purged_kfold, walk_forward
from harness.ledger import TrialLedger

IDX = pd.bdate_range("2008-01-01", "2026-10-30")


def noise_family(n_var, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0002, 0.01, size=(len(IDX), n_var)), index=IDX, columns=[f"v{i}" for i in range(n_var)])


def planted_family(seed=0):
    """An asset with slow regimes in its drift. Trend variants exploit it; random variants don't."""
    rng = np.random.default_rng(seed)
    regime = np.sign(np.sin(np.arange(len(IDX)) / 90.0))
    asset = pd.Series(0.0025 * regime + rng.normal(0, 0.01, len(IDX)), index=IDX)
    cols = {}
    for L in (20, 40, 60, 80):
        on = (asset.rolling(L).mean() > 0).shift(2).fillna(False)  # signal lagged like the engine
        cols[f"trend_{L}"] = asset.where(on, 0.0)
    for i in range(4):
        on = pd.Series(rng.random(len(IDX)) < 0.5, index=IDX)
        cols[f"rand_{i}"] = asset.where(on, 0.0)
    return pd.DataFrame(cols), asset


def test_expected_max_sharpe_grows_with_trials():
    v = 0.01 ** 2
    assert stats.expected_max_sharpe(1, v) == 0
    assert stats.expected_max_sharpe(10, v) < stats.expected_max_sharpe(1000, v)


def test_pooled_variance_matches_direct():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 50), rng.normal(0.5, 2, 30)
    pooled = stats.pooled_sr_variance([(50, a.mean(), a.var(ddof=1)), (30, b.mean(), b.var(ddof=1))])
    assert pooled == pytest.approx(np.concatenate([a, b]).var(ddof=1))


def test_pbo_noise_is_about_half_and_planted_edge_is_low():
    assert 0.3 < stats.pbo_cscv(noise_family(50).to_numpy(), 8)["pbo"] < 0.7
    fam, _ = planted_family()
    assert stats.pbo_cscv(fam.to_numpy(), 8)["pbo"] < 0.2


def test_purged_kfold_has_gaps_and_no_overlap():
    for tr, te in purged_kfold(1000, k=5, purge=21, embargo=21):
        assert not set(tr) & set(te)
        lo, hi = te.min(), te.max()
        assert not any(lo - 21 <= i < lo for i in tr)
        assert not any(hi < i <= hi + 21 for i in tr)


def test_walk_forward_only_uses_past_to_select():
    fam = noise_family(3)
    oos, log = walk_forward(fam, 3, 6)
    assert oos.index.min() >= fam.index[0] + pd.DateOffset(years=3)
    assert all(row["test_start"] > row["train_start"] for row in log)
    assert oos.index.is_monotonic_increasing and not oos.index.duplicated().any()


def test_gate_rejects_200_random_variants(tmp_path):
    led = TrialLedger(tmp_path / "t.jsonl")
    fam = noise_family(200, seed=3)
    bench = pd.Series(np.random.default_rng(9).normal(0.0002, 0.01, len(IDX)), index=IDX)
    rep = research_gate("rand", "calibration", fam, fam - 0.0001, bench, led, "2026-10-31", 252)
    checks = {c.name: c for c in rep.checks}
    assert rep.verdict == "fail"
    assert not checks["deflated_sharpe_active"].passed
    assert checks["pbo"].value > 0.2
    assert led.count("calibration") == 200


def test_gate_accepts_planted_edge(tmp_path):
    led = TrialLedger(tmp_path / "t.jsonl")
    fam, asset = planted_family()
    rep = research_gate("planted", "test", fam, fam - 0.00005, asset, led, "2026-10-31", 252)
    assert rep.verdict == "pass", rep.table()
    assert rep.selected.startswith("trend_")


def test_more_prior_trials_raise_the_bar(tmp_path):
    fam, asset = planted_family()
    led1, led2 = TrialLedger(tmp_path / "a.jsonl"), TrialLedger(tmp_path / "b.jsonl")
    for _ in range(5):
        led2.append({"spec_id": "old", "venue": "test", "snapshot_id": "x", "params": {}, "n_variants": 1000,
                     "metrics": {"variant_sr_mean": 0.0, "variant_sr_var": 0.002}, "verdict": "fail"})
    d1 = {c.name: c.value for c in research_gate("p", "test", fam, fam, asset, led1, "s", 252).checks}["deflated_sharpe_active"]
    d2 = {c.name: c.value for c in research_gate("p", "test", fam, fam, asset, led2, "s", 252).checks}["deflated_sharpe_active"]
    assert d2 < d1


def test_vault_is_one_shot_and_survivors_only(tmp_path):
    led = TrialLedger(tmp_path / "t.jsonl")
    fam = noise_family(5)
    bench = fam.mean(axis=1)
    research_gate("bad", "x", fam, fam, bench, led, "s", 252)
    with pytest.raises(VaultError, match="no passing"):
        vault_test("bad", "x", fam, bench, led, "s", 252)

    fam2, asset = planted_family()
    research_gate("good", "y", fam2, fam2, asset, led, "s", 252)
    vault_test("good", "y", fam2, asset, led, "s", 252)
    with pytest.raises(VaultError, match="already used"):
        vault_test("good", "y", fam2, asset, led, "s", 252)
    assert led.verify() == []


def test_research_never_sees_the_vault(tmp_path):
    led = TrialLedger(tmp_path / "t.jsonl")
    fam = noise_family(3)
    fam.loc[fam.index >= fam.index[-1] - pd.DateOffset(months=18), "v0"] = 0.05  # huge edge only in the vault
    rep = research_gate("peek", "z", fam, fam, fam.mean(axis=1) * 0, led, "s", 252)
    assert rep.wf_oos.index.max() < fam.index[-1] - pd.DateOffset(months=18)


def test_market_exposure_is_not_mistaken_for_skill(tmp_path):
    """Random part-time holdings of a strongly rising asset have a high Sharpe but no skill."""
    rng = np.random.default_rng(5)
    asset = pd.Series(0.0008 + rng.normal(0, 0.01, len(IDX)), index=IDX)  # ~Sharpe 1.3 drift
    fam = pd.DataFrame({f"r{i}": asset.where(rng.random(len(IDX)) < 0.8, 0.0) for i in range(100)})
    rep = research_gate("exposure", "cal", fam, fam, asset, TrialLedger(tmp_path / "t.jsonl"), "s", 252)
    checks = {c.name: c for c in rep.checks}
    assert not checks["deflated_sharpe_active"].passed
    assert rep.verdict == "fail"
