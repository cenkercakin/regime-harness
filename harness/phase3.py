"""Phase 3 run: whole library through the gate, then the monthly selector on top.

Two experiments, as pre-registered:
  A. ETF library only, ETF trading calendar, full history (Jul 2009 onward).
  B. ETF + crypto library on a 7-day calendar from 2017 (crypto history is shorter).
Every strategy and every selector family is logged to the ledger. The vault
(last 18 months) is never read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import selector
from .engine import backtest
from .engine.metrics import max_drawdown, sharpe, summary
from .engine.returns import load_returns
from .gate import research_gate
from .gate.gate import vault_cutoff
from .gate import stats
from .ledger import TrialLedger
from .store import SnapshotStore
from .strategies import tsmom
from .strategies.library import CASH, COINS, LIBRARY, RISKY, MarketData

ETF_START = pd.Timestamp("2009-08-03")
MIX_START = pd.Timestamp("2017-01-02")
WINDOWS = (6, 12)
COSTS = {"us_etfs": backtest.ETF_COSTS, "crypto": backtest.CRYPTO_COSTS}


@dataclass
class Phase3Result:
    snapshot: str
    library: pd.DataFrame = field(default_factory=pd.DataFrame)
    selectors: dict = field(default_factory=dict)
    library_pbo: dict = field(default_factory=dict)
    yearly: dict = field(default_factory=dict)


def _wide(bars: pd.DataFrame, col: str) -> pd.DataFrame:
    w = bars.pivot(index="date", columns="symbol", values=col)
    w.index = pd.to_datetime(w.index)
    return w.sort_index()


def market_data(store: SnapshotStore, snap: str) -> tuple[MarketData, MarketData]:
    etf_syms = RISKY + [CASH]
    bars = store.load("us_etfs", as_of=snap)
    R = load_returns(store, "us_etfs", etf_syms, snap)
    etf = MarketData(R, _wide(bars, "high")[etf_syms], _wide(bars, "low")[etf_syms], _wide(bars, "close")[etf_syms])
    C = load_returns(store, "crypto", COINS, store.snapshot_ids("crypto")[-1] if snap not in store.snapshot_ids("crypto") else snap)
    return etf, MarketData(C)


def _strategy_returns(spec, md, cost_mult=1.0) -> pd.Series:
    T = spec.fn(md)
    return backtest.run(T, md.R, COSTS[spec.venue].scaled(cost_mult), lag=1).returns


def _benchmarks(etf: MarketData, crypto: MarketData) -> tuple[pd.Series, pd.Series]:
    b_etf = backtest.run(tsmom.equal_weight_targets(etf.R), etf.R, backtest.ETF_COSTS, lag=1).returns
    from .engine.returns import month_end_dates
    me = month_end_dates(crypto.R.index)
    T = pd.DataFrame({c: [0.5] * len(me) for c in COINS}, index=me)
    first = crypto.R["ETH/USD"].first_valid_index()
    T.loc[T.index <= first, "ETH/USD"] = 0.0
    T.loc[T.index <= first, "BTC/USD"] = 1.0
    b_cry = backtest.run(T, crypto.R, backtest.CRYPTO_COSTS, lag=1).returns
    return b_etf, b_cry


def _row(name, spec_family, venue, r, bench, ppy, rep):
    cut = vault_cutoff(r.index, 18)
    rr, bb = r.loc[r.index < cut], bench.reindex(r.index).fillna(0.0).loc[r.index < cut]
    s = summary(rr, ppy)
    checks = {c.name: c for c in rep.checks}
    return {
        "strategy": name, "family": spec_family, "venue": venue,
        "cagr": s["cagr"], "sharpe": s["sharpe_ann"], "max_dd": s["max_dd"],
        "active_ann": float((rr - bb).mean() * ppy),
        "wf_excess": checks["wf_excess_ann"].value,
        "wf_excess_2x": checks["wf_excess_ann_2x_cost"].value,
        "cv_excess": checks["cv_excess_ann"].value,
        "dsr": checks["deflated_sharpe_active"].value,
        "dd_ok": checks["wf_max_dd"].passed,
        "verdict": rep.verdict,
        "failed": ", ".join(c.name for c in rep.checks if not c.passed),
    }


def run(store: SnapshotStore, ledger: TrialLedger, snap: str | None = None, gate_library: bool = True, selector_version: str = "v2") -> Phase3Result:
    """`gate_library=False` recomputes the library returns without logging them again
    (re-gating would double-count those trials)."""
    snap = snap or store.snapshot_ids("us_etfs")[-1]
    etf, cry = market_data(store, snap)
    b_etf, b_cry = _benchmarks(etf, cry)
    res = Phase3Result(snapshot=snap)

    # ---- 1. every strategy, 1x and 2x costs
    rets, rets2 = {}, {}
    for spec in LIBRARY:
        md = etf if spec.venue == "us_etfs" else cry
        start = ETF_START if spec.venue == "us_etfs" else MIX_START
        rets[spec.name] = _strategy_returns(spec, md).loc[start:]
        rets2[spec.name] = _strategy_returns(spec, md, 2.0).loc[start:]

    # ---- 2. gate each strategy, judged against the whole venue library's trial count and dispersion
    rows = []
    for venue, bench, ppy in ((("us_etfs", b_etf, 252), ("crypto", b_cry, 365)) if gate_library else ()):
        specs = [s for s in LIBRARY if s.venue == venue]
        cut = vault_cutoff(rets[specs[0].name].index, 18)
        b = bench.reindex(rets[specs[0].name].index).fillna(0.0)
        active_sr = np.array([sharpe((rets[s.name] - b).loc[: cut - pd.Timedelta(days=1)].to_numpy()) for s in specs])
        var_floor = float(np.nanvar(active_sr, ddof=1))
        n_floor = ledger.count(venue) + len(specs)
        for s in specs:
            V = rets[s.name].to_frame(s.name)
            V2 = rets2[s.name].to_frame(s.name)
            rep = research_gate(
                f"lib-{s.name}-v1", venue, V, V2, b, ledger, snap, ppy,
                params={"family": s.family, "source": s.source, "library": "phase3-v1"},
                n_trials_floor=n_floor, sr_variance_floor=var_floor,
            )
            rows.append(_row(s.name, s.family, venue, rets[s.name], b, ppy, rep))
    res.library = pd.DataFrame(rows)

    # ---- 3. library-level overfitting: if you just picked the best-looking ETF strategy, how often is it a dud?
    etf_names = [s.name for s in LIBRARY if s.venue == "us_etfs"]
    A = pd.DataFrame({n: rets[n] for n in etf_names})
    cut = vault_cutoff(A.index, 18)
    act = A.sub(b_etf.reindex(A.index).fillna(0.0), axis=0).loc[A.index < cut]
    res.library_pbo = stats.pbo_cscv(act.to_numpy(), 16)

    # ---- 4. selectors
    crypto_names = [s.name for s in LIBRARY if s.venue == "crypto"]
    S_etf = A
    S2_etf = pd.DataFrame({n: rets2[n] for n in etf_names})
    union = S_etf.index.union(pd.DataFrame({n: rets[n] for n in crypto_names}).index)
    union = union[union >= MIX_START]
    S_mix = pd.concat([S_etf.reindex(union).fillna(0.0), pd.DataFrame({n: rets[n] for n in crypto_names}).reindex(union).fillna(0.0)], axis=1)
    S2_mix = pd.concat([S2_etf.reindex(union).fillna(0.0), pd.DataFrame({n: rets2[n] for n in crypto_names}).reindex(union).fillna(0.0)], axis=1)

    rf_all = etf.R[CASH]
    for label, S, S2, ppy, crypto_set, market in (
        ("selector_etf", S_etf, S2_etf, 252, set(), b_etf),
        ("selector_mixed", S_mix, S2_mix, 365, set(crypto_names), None),
    ):
        costs = selector.switching_costs(S.columns, crypto_set)
        rf = rf_all.reindex(S.index).fillna(0.0)
        blend = selector.run(S, "blend", 12, crypto_set, costs, rf=rf)
        fam, fam2, info = {}, {}, {}
        for mode in ("hard", "tilt"):
            for m in WINDOWS:
                k = f"{mode}_{m}m"
                r = selector.run(S, mode, m, crypto_set, costs, rf=rf)
                r2 = selector.run(S2, mode, m, crypto_set, costs.scaled(2.0), rf=rf)
                fam[k], fam2[k] = r.returns, r2.returns
                Tk = selector.targets(S, mode, m, crypto_set, rf=rf)
                picks = Tk.idxmax(axis=1)
                info[k] = {
                    "switches": int((picks != picks.shift()).sum() - 1),
                    "switch_cost_ann": float((r.gross - r.returns).mean() * ppy),
                    "top_picks": picks.value_counts().head(5).to_dict(),
                    "avg_crypto": float(Tk[list(crypto_set)].sum(axis=1).mean()) if crypto_set else 0.0,
                }
        # Compare only once every version is invested: first 12-month decision, filled next close
        first_decision = selector.targets(S, "blend", max(WINDOWS), crypto_set, rf=rf).index[0]
        start = S.index[S.index.get_loc(first_decision) + 2]
        V = pd.DataFrame(fam).loc[start:]
        V2 = pd.DataFrame(fam2).loc[start:]
        bench = blend.returns.loc[start:]
        rep = research_gate(
            f"{label}-{selector_version}", label, V, V2, bench, ledger, snap, ppy,
            params={"modes": ["hard", "tilt"], "windows_months": list(WINDOWS), "benchmark": "equal-weight blend of the same library", "crypto_cap": 0.25,
                    "ranking": "trailing Sharpe of returns above BIL", "fix": "v1 ranked on raw Sharpe (picked cash) and scored warm-up months"},
        )
        cutv = vault_cutoff(V.index, 18)
        desc = {}
        series = dict(V)
        series["blend"] = bench
        if market is not None:
            series["market_equal_weight"] = market.reindex(V.index).fillna(0.0)
        else:
            series["market_btc_eth_5050"] = b_cry.reindex(V.index).fillna(0.0)
            series["market_etf_equal_weight"] = b_etf.reindex(V.index).fillna(0.0)
        for k, r in series.items():
            rr = r.loc[r.index < cutv]
            s = summary(rr, ppy)
            desc[k] = {"cagr": s["cagr"], "sharpe": s["sharpe_ann"], "max_dd": s["max_dd"]}
        res.selectors[label] = {"report": rep, "describe": pd.DataFrame(desc).T, "info": info, "period": (str(V.index[0].date()), str((cutv - pd.Timedelta(days=1)).date()))}
        yearly = {k: r.loc[r.index < cutv].groupby(r.loc[r.index < cutv].index.year).apply(lambda x: float((1 + x).prod() - 1)) for k, r in series.items()}
        res.yearly[label] = pd.DataFrame(yearly)
    return res


def _pct(x):
    return "" if pd.isna(x) else f"{x * 100:.1f}%"


def to_markdown(res: Phase3Result) -> str:
    L = res.library.copy()
    out = [f"# Phase 3 results (snapshot {res.snapshot})", ""]
    if not L.empty:
        passed = L[L.verdict == "pass"]
        out.append(f"**Library: {len(L)} strategies, {len(passed)} passed the gate.** Vault (last 18 months) untouched.")
        out.append("")
    for venue, title in () if L.empty else (("us_etfs", "ETF strategies (vs equal-weight of 6 ETFs, Aug 2009 – Apr 2025)"), ("crypto", "Crypto strategies (vs 50/50 BTC/ETH, Jan 2017 – Apr 2025)")):
        sub = L[L.venue == venue].sort_values("active_ann", ascending=False)
        out += [f"## {title}", "", "| Strategy | Family | CAGR | Sharpe | Max DD | vs bench /yr | Walk-fwd excess | Deflated Sharpe | Verdict | Failed checks |", "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            out.append(f"| {r.strategy} | {r.family} | {_pct(r.cagr)} | {r.sharpe:.2f} | {_pct(r.max_dd)} | {_pct(r.active_ann)} | {_pct(r.wf_excess)} | {r.dsr:.2f} | {r.verdict.upper()} | {r.failed} |")
        out.append("")
    p = res.library_pbo
    out += [f"**Library-level overfitting:** picking the best-looking ETF strategy in-sample, PBO = {p['pbo']:.2f} (median logit {p['median_logit']:.2f}).", ""]
    for label, d in res.selectors.items():
        rep = d["report"]
        out += [f"## {label} ({d['period'][0]} – {d['period'][1]})", "", "```", rep.table(), "```", "", "| Version | CAGR | Sharpe | Max DD |", "|---|---|---|---|"]
        for k, r in d["describe"].iterrows():
            out.append(f"| {k} | {_pct(r.cagr)} | {r.sharpe:.2f} | {_pct(r.max_dd)} |")
        out += ["", "| Version | Switches | Switching cost /yr | Avg crypto weight | Most-held strategies |", "|---|---|---|---|---|"]
        for k, i in d["info"].items():
            tops = ", ".join(f"{n} ({c})" for n, c in i["top_picks"].items())
            out.append(f"| {k} | {i['switches']} | {_pct(i['switch_cost_ann'])} | {_pct(i['avg_crypto'])} | {tops} |")
        out += ["", "Calendar-year returns:", "", res.yearly[label].map(lambda x: f"{x * 100:.1f}%").to_markdown(), ""]
    return "\n".join(out)
