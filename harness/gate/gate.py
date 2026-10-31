"""The validation gate: every check a strategy family must pass before paper trading.

Usage: compute the full-sample daily returns of every variant in a family
(causal signals, realistic costs), plus the same at doubled costs, then call
`research_gate`. It evaluates only data before the vault cutoff and logs one
ledger row whatever the verdict. `vault_test` is a separate, one-time call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..engine.metrics import max_drawdown, moments, sharpe
from ..ledger import TrialLedger
from . import splits, stats


@dataclass(frozen=True)
class Thresholds:
    dsr_min: float = 0.95
    pbo_max: float = 0.20
    vault_months: int = 18
    wf_train_years: float = 3.0
    wf_test_months: int = 6
    cv_folds: int = 5
    purge: int = 21
    embargo: int = 21
    pbo_blocks: int = 16


@dataclass
class Check:
    name: str
    value: float
    rule: str
    passed: bool


@dataclass
class GateReport:
    spec_id: str
    venue: str
    verdict: str
    selected: str
    checks: list[Check] = field(default_factory=list)
    n_trials: int = 0
    wf_oos: pd.Series | None = None
    wf_log: list[dict] = field(default_factory=list)

    def table(self) -> str:
        lines = [f"{self.spec_id} [{self.venue}] -> {self.verdict.upper()}  (selected: {self.selected}, N={self.n_trials})"]
        for c in self.checks:
            lines.append(f"  {'PASS' if c.passed else 'FAIL'}  {c.name:<28} {c.value:>10.4f}   {c.rule}")
        return "\n".join(lines)


def _clean(d: dict) -> dict:
    """NaN/inf are not valid JSON; record them as null so the ledger write never fails."""
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def vault_cutoff(index: pd.DatetimeIndex, months: int) -> pd.Timestamp:
    return index[-1] - pd.DateOffset(months=months)


def _excess_ann(r: pd.Series, bench: pd.Series, ppy: float) -> float:
    b = bench.reindex(r.index).fillna(0.0)
    return float((r - b).mean() * ppy)


def research_gate(
    spec_id: str,
    venue: str,
    variants: pd.DataFrame,
    variants_2x_cost: pd.DataFrame,
    benchmark: pd.Series,
    ledger: TrialLedger,
    snapshot_id: str,
    periods_per_year: float,
    params: dict | None = None,
    th: Thresholds = Thresholds(),
    n_trials_floor: int = 0,
    sr_variance_floor: float = 0.0,
) -> GateReport:
    """`n_trials_floor` / `sr_variance_floor`: when a whole pre-registered library is
    evaluated, every member is judged against the library's full trial count and
    Sharpe dispersion, so the order of evaluation cannot make the bar easier."""
    ppy = periods_per_year
    cutoff = vault_cutoff(variants.index, th.vault_months)
    V = variants.loc[variants.index < cutoff]
    V2 = variants_2x_cost.loc[variants_2x_cost.index < cutoff]
    B = benchmark.loc[benchmark.index < cutoff]
    n_var = V.shape[1]
    checks: list[Check] = []

    # Skill is measured on ACTIVE returns (strategy minus benchmark). Total-return
    # Sharpe rewards plain market exposure: random part-time holdings of rising
    # assets have a positive Sharpe with zero skill. Selection, DSR and PBO all use A.
    Bv = B.reindex(V.index).fillna(0.0)
    A = V.sub(Bv, axis=0)
    A2 = V2.sub(Bv, axis=0)

    # 1. Walk-forward out-of-sample excess return after costs, and at doubled costs
    wf_a, wf_log = splits.walk_forward(A, th.wf_train_years, th.wf_test_months)
    wf2_a, _ = splits.walk_forward(A2, th.wf_train_years, th.wf_test_months)
    ex, ex2 = float(wf_a.mean() * ppy), float(wf2_a.mean() * ppy)
    checks.append(Check("wf_excess_ann", ex, "> 0 vs benchmark, after costs", ex > 0))
    checks.append(Check("wf_excess_ann_2x_cost", ex2, "> 0 with costs doubled", ex2 > 0))
    wf = wf_a + Bv.reindex(wf_a.index)
    mdd, mdd_b = max_drawdown(wf), max_drawdown(Bv.reindex(wf.index))
    checks.append(Check("wf_max_dd", mdd, f">= benchmark {mdd_b:.4f}", mdd >= mdd_b))

    # 2. Purged K-fold: selection on purged train folds, scored on held-out folds
    cv_a = splits.cv_oos(A, th.cv_folds, th.purge, th.embargo)
    cv_ex = float(cv_a.mean() * ppy)
    checks.append(Check("cv_excess_ann", cv_ex, "> 0 vs benchmark", cv_ex > 0))

    # 3. Deflated Sharpe of the full-sample winner's active return, deflated by every trial on this venue
    sr_by_var = A.apply(lambda s: sharpe(s.to_numpy()))
    srs = sr_by_var.to_numpy(dtype=float)
    srs = srs[~np.isnan(srs)]
    selected = str(sr_by_var.idxmax())
    n_trials = max(ledger.count(venue) + n_var, n_trials_floor)
    var = stats.pooled_sr_variance(ledger.sr_batches(venue) + [(len(srs), float(srs.mean()), float(srs.var(ddof=1)) if len(srs) > 1 else 0.0)])
    var = max(var, sr_variance_floor)
    best = A[selected]
    sk, ku = moments(best)
    dsr, sr_star = stats.deflated_sharpe(sharpe(best.to_numpy()), int(best.notna().sum()), sk, ku, n_trials, var)
    checks.append(Check("deflated_sharpe_active", dsr, f"> {th.dsr_min} (SR* {sr_star * np.sqrt(ppy):.2f} ann.)", dsr > th.dsr_min))

    # 4. PBO across the family
    if n_var >= 2:
        p = stats.pbo_cscv(A.to_numpy(), th.pbo_blocks)
        checks.append(Check("pbo", p["pbo"], f"< {th.pbo_max}", p["pbo"] < th.pbo_max))
        pbo_val = p["pbo"]
    else:
        pbo_val = float("nan")

    verdict = "pass" if all(c.passed for c in checks) else "fail"
    ledger.append(
        {
            "spec_id": spec_id,
            "venue": venue,
            "snapshot_id": snapshot_id,
            "stage": "research",
            "n_variants": n_var,
            "params": params or {},
            "selected": selected,
            "research_end": str(cutoff.date()),
            "metrics": _clean({
                "wf_excess_ann": ex,
                "wf_excess_ann_2x": ex2,
                "wf_sharpe_ann": sharpe(wf.to_numpy(), ppy),
                "cv_excess_ann": cv_ex,
                "dsr": dsr,
                "sr_star_ann": float(sr_star * np.sqrt(ppy)),
                "pbo": None if np.isnan(pbo_val) else pbo_val,
                "variant_sr_mean": float(srs.mean()),
                "variant_sr_var": float(srs.var(ddof=1)) if len(srs) > 1 else 0.0,
                "n_trials": n_trials,
            }),
            "verdict": verdict,
        }
    )
    return GateReport(spec_id, venue, verdict, selected, checks, n_trials, wf, wf_log)


class VaultError(Exception):
    pass


def vault_test(
    spec_id: str,
    venue: str,
    variants: pd.DataFrame,
    benchmark: pd.Series,
    ledger: TrialLedger,
    snapshot_id: str,
    periods_per_year: float,
    th: Thresholds = Thresholds(),
) -> GateReport:
    """One-time test on the locked last `vault_months`. A failure is permanent."""
    rows = [r for r in ledger.rows() if r["spec_id"] == spec_id]
    if any(r.get("stage") == "vault" for r in rows):
        raise VaultError(f"{spec_id} already used its vault test; it cannot be retried or retuned")
    research = [r for r in rows if r.get("stage") == "research"]
    if not research or research[-1]["verdict"] != "pass":
        raise VaultError(f"{spec_id} has no passing research gate; the vault is only for survivors")
    selected = research[-1]["selected"]
    cutoff = vault_cutoff(variants.index, th.vault_months)
    r = variants.loc[variants.index >= cutoff, selected]
    b = benchmark.loc[benchmark.index >= cutoff]
    ex = _excess_ann(r, b, periods_per_year)
    sr = sharpe(r.to_numpy(), periods_per_year)
    checks = [Check("vault_excess_ann", ex, "> 0 vs benchmark", ex > 0), Check("vault_sharpe_ann", sr, "> 0", sr > 0)]
    verdict = "pass" if all(c.passed for c in checks) else "fail"
    ledger.append(
        {
            "spec_id": spec_id,
            "venue": venue,
            "snapshot_id": snapshot_id,
            "stage": "vault",
            "n_variants": 0,
            "params": {"selected": selected},
            "selected": selected,
            "metrics": _clean({"vault_excess_ann": ex, "vault_sharpe_ann": sr}),
            "verdict": verdict,
        }
    )
    return GateReport(spec_id, venue, verdict, selected, checks, ledger.count(venue))
