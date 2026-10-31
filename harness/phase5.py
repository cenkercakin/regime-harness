"""Phase 5 run: BIST stock library through the gate (benchmark: XU100 in USD).

All results carry the survivorship warning: the ticker list holds only companies
listed today, so history before today is optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import backtest
from .engine.metrics import sharpe, summary
from .engine.returns import month_end_dates, total_returns
from .gate import research_gate
from .gate.gate import vault_cutoff
from .ledger import TrialLedger
from .store import SnapshotStore
from .strategies import bist
from .strategies.bist import BIST_LIBRARY

START = pd.Timestamp("2011-03-01")
PPY = 252
BORROW_BPS = 500.0    # placeholder stock-lending fee, % a year on short notional


def _wide(bars, col):
    w = bars.pivot(index="date", columns="symbol", values=col)
    w.index = pd.to_datetime(w.index)
    return w.sort_index()


def load(store: SnapshotStore, snap: str | None = None):
    snap = snap or store.snapshot_ids("bist")[-1]
    bars = store.load("bist", as_of=snap)
    r_try = total_returns(bars)
    close, vol = _wide(bars, "close"), _wide(bars, "volume")
    ib = store.load("bist_index", as_of=snap)
    idx_try = total_returns(ib)["XU100.IS"]
    # Trading calendar = days the XU100 traded (drops stray vendor rows on holidays)
    days = idx_try.dropna().index
    r_try = r_try.loc[r_try.index.isin(days)]
    turnover_days = r_try.index
    turnover = (close * vol).reindex(index=turnover_days, columns=r_try.columns)
    fx_bars = store.load("fx")
    fx = _wide(fx_bars, "close")["USDTRY=X"]
    bd = bist.build(r_try, turnover, idx_try.reindex(r_try.index), fx)
    return snap, bd, idx_try, r_try


def run(store: SnapshotStore, ledger: TrialLedger, snap: str | None = None) -> dict:
    snap, bd, idx_try, r_try = load(store, snap)
    rules = backtest.ShortRules(bist.shortable_mask(bd), borrow_bps_annual=BORROW_BPS, periods_per_year=PPY)
    rules2 = backtest.ShortRules(rules.shortable, borrow_bps_annual=2 * BORROW_BPS, periods_per_year=PPY)
    costs = backtest.CostModel(20.0, bps_frame=bist.cost_frame(bd))
    bench = bd.index_R.fillna(0.0).loc[START:]

    # equal-weight top-100, monthly, for context (not a gate benchmark)
    ew_rows = {d: {s: 1.0 / len(bd.universe(d)) for s in bd.universe(d)} for d in month_end_dates(bd.R.index) if len(bd.universe(d)) >= 10}
    ew_T = pd.DataFrame.from_dict(ew_rows, orient="index").reindex(columns=bd.R.columns).fillna(0.0)
    ew = backtest.run(ew_T, bd.R, costs, lag=1, unpriced="hold").returns.loc[START:]

    rets, rets2, info = {}, {}, {}
    for spec in BIST_LIBRARY:
        T = spec.fn(bd)
        res = backtest.run(T, bd.R, costs, lag=1, shorts=rules, unpriced="hold")
        res2 = backtest.run(T, bd.R, costs.scaled(2.0), lag=1, shorts=rules2, unpriced="hold")
        rets[spec.name], rets2[spec.name] = res.returns.loc[START:], res2.returns.loc[START:]
        W = res.weights.loc[START:]
        wanted_short = (T < 0).any(axis=1)
        info[spec.name] = {
            "avg_holdings": float((W.abs() > 1e-9).sum(axis=1).mean()),
            "turnover_yr": float(res.turnover.loc[START:].sum() / (len(W) / PPY)),
            "cost_drag_yr": float((res.gross - res.returns).loc[START:].mean() * PPY),
            "avg_short": float(-W.clip(upper=0).sum(axis=1).mean()),
            "uses_shorts": bool(wanted_short.any()),
        }

    cut = vault_cutoff(bench.index, 18)
    act_sr = np.array([sharpe((rets[n] - bench).loc[bench.index < cut].to_numpy()) for n in rets])
    var_floor = float(np.nanvar(act_sr, ddof=1))
    n_floor = ledger.count("bist") + len(BIST_LIBRARY)
    rows = []
    for spec in BIST_LIBRARY:
        V, V2 = rets[spec.name].to_frame(spec.name), rets2[spec.name].to_frame(spec.name)
        rep = research_gate(
            f"lib-{spec.name}-v1", "bist", V, V2, bench, ledger, snap, PPY,
            params={"family": spec.family, "source": spec.source, "library": "phase5-v1", "borrow_bps": BORROW_BPS,
                    "warning": "survivorship-biased universe (currently listed stocks only)"},
            n_trials_floor=n_floor, sr_variance_floor=var_floor,
        )
        r = rets[spec.name].loc[rets[spec.name].index < cut]
        s = summary(r, PPY)
        ch = {c.name: c for c in rep.checks}
        rows.append({
            "strategy": spec.name, "family": spec.family, "cagr": s["cagr"], "sharpe": s["sharpe_ann"], "max_dd": s["max_dd"],
            "active_ann": float((r - bench.loc[r.index]).mean() * PPY),
            "wf_excess": ch["wf_excess_ann"].value, "dsr": ch["deflated_sharpe_active"].value,
            "verdict": rep.verdict, "failed": ", ".join(c.name for c in rep.checks if not c.passed), **info[spec.name],
        })

    research = lambda x: x.loc[x.index < cut]
    ctx = {"XU100 (USD, price index)": summary(research(bench), PPY), "Equal-weight top 100 (USD)": summary(research(ew), PPY)}
    yearly = pd.DataFrame({k: research(v).groupby(research(v).index.year).apply(lambda x: float((1 + x).prod() - 1))
                           for k, v in {**{n: rets[n] for n in rets}, "XU100_usd": bench, "equal_weight_top100": ew}.items()})
    sizes = pd.Series({d: len(m) for d, m in bd.members.items()})
    notes = {
        "snapshot": snap,
        "period": (str(START.date()), str((cut - pd.Timedelta(days=1)).date())),
        "stocks_with_data": int(r_try.notna().any().sum()),
        "jumps_neutralised": int(bd.flagged.sum().sum()),
        "stocks_with_jumps": int(bd.flagged.any().sum()),
        "universe_size_median": float(sizes[sizes.index >= START].median()),
        "ban_days_in_research": int(bist.ban_days(bench.index)[bench.index < cut].sum()),
        "index_jumps_neutralised": int((idx_try.abs() > bist.JUMP_LIMIT).sum()),
    }
    return {"table": pd.DataFrame(rows), "context": ctx, "yearly": yearly, "notes": notes}


def _p(x):
    return "" if pd.isna(x) else f"{x * 100:.1f}%"


def to_markdown(out: dict) -> str:
    n, L = out["notes"], out["table"].sort_values("active_ann", ascending=False)
    lines = [f"# Phase 5 results: BIST stocks (snapshot {n['snapshot']})", "",
             f"**{int((L.verdict == 'pass').sum())} of {len(L)} strategies passed the gate.** Research period {n['period'][0]} – {n['period'][1]}; vault untouched.", "",
             "> **Survivorship warning:** the universe is built from companies listed today. Companies that failed or delisted are missing, so every number below is optimistic.", "",
             "## Data", "",
             f"- Stocks with data: {n['stocks_with_data']}; median monthly universe: {n['universe_size_median']:.0f} stocks",
             f"- Daily moves over 25% neutralised as corporate-action errors: {n['jumps_neutralised']} (in {n['stocks_with_jumps']} stocks); index jumps neutralised: {n['index_jumps_neutralised']}",
             f"- Short-ban days inside the research period: {n['ban_days_in_research']} (shorts force-covered)", "",
             "## Benchmarks", "", "| | CAGR | Sharpe | Max DD |", "|---|---|---|---|"]
    for k, s in out["context"].items():
        lines.append(f"| {k} | {_p(s['cagr'])} | {s['sharpe_ann']:.2f} | {_p(s['max_dd'])} |")
    lines += ["", "## Strategies (vs XU100 in USD)", "",
              "| Strategy | Family | CAGR | Sharpe | Max DD | vs XU100 /yr | Walk-fwd excess | Deflated Sharpe | Turnover /yr | Cost drag /yr | Avg short | Verdict | Failed checks |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in L.iterrows():
        lines.append(f"| {r.strategy} | {r.family} | {_p(r.cagr)} | {r.sharpe:.2f} | {_p(r.max_dd)} | {_p(r.active_ann)} | {_p(r.wf_excess)} | {r.dsr:.2f} | {r.turnover_yr:.1f}x | {_p(r.cost_drag_yr)} | {_p(r.avg_short)} | {r.verdict.upper()} | {r.failed} |")
    lines += ["", "## Calendar-year returns (USD)", "", out["yearly"].map(lambda x: f"{x * 100:.1f}%").to_markdown(), ""]
    return "\n".join(lines)
