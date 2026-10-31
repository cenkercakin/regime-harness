"""Grid trading run: 4 pre-registered variants x BTC, ETH through the gate.

Benchmark: 50% coin / 50% cash, rebalanced monthly (a grid is roughly half invested,
so this asks whether the grid machinery beats plain rebalancing). Buy-and-hold is
shown for context. Fees 0.10% per side; the gate's doubled-cost check covers 0.20%.
Trials count towards the "crypto" venue. Resolution check: the same grids on hourly
and daily bars built from the 5-minute data (diagnostic only, not logged).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import backtest
from .engine.metrics import sharpe, summary
from .engine.returns import month_end_dates
from .gate import research_gate
from .gate.gate import vault_cutoff
from .ledger import TrialLedger
from .store import SnapshotStore
from .strategies.grid import MECHANISM, SOURCE, SPECS, GridSpec, resample, simulate

PPY = 365
COINS = ["BTC/USD", "ETH/USD"]
FEE = 0.001


def bars(store: SnapshotStore, sym: str, snap: str | None = None) -> pd.DataFrame:
    df = store.load("crypto_5m", as_of=snap)
    b = df[df["symbol"] == sym].set_index("date")[["open", "high", "low", "close"]]
    b.index = pd.to_datetime(b.index)
    return b.sort_index()


def _benchmarks(b: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = b["close"].groupby(b.index.normalize()).last()
    R = close.pct_change().to_frame("coin")
    me = month_end_dates(R.index)
    T = pd.DataFrame({"coin": 0.5}, index=me)
    fifty = backtest.run(T, R, backtest.CostModel(FEE * 1e4), lag=1).returns
    return fifty, R["coin"].fillna(0.0)


def run(store: SnapshotStore, ledger: TrialLedger, snap: str | None = None) -> dict:
    snap = snap or store.snapshot_ids("crypto_5m")[-1]
    results, gate_reports, resolution = {}, {}, []
    fam, fam2, benches, holds, info = {}, {}, {}, {}, {}
    for sym in COINS:
        b = bars(store, sym, snap)
        fifty, hold = _benchmarks(b)
        benches[sym], holds[sym] = fifty, hold
        V, V2 = {}, {}
        for spec in SPECS:
            r = simulate(b, spec)
            r2 = simulate(b, GridSpec(spec.spacing, spec.levels, spec.recentre, fee=2 * FEE, trade_through=spec.trade_through))
            V[spec.name], V2[spec.name] = r.daily, r2.daily
            years = len(r.daily) / PPY
            info[(sym, spec.name)] = {
                "fills_yr": r.fills / years, "recentres": r.recentres, "fee_drag_yr": r.fees_paid / r.equity.mean() / years,
                "at_bottom": r.at_bottom, "at_top": r.at_top, "avg_coin_share": float(r.coin_share.mean()),
            }
            for label, rule in (("5m", None), ("1h", "1h"), ("1d", "1D")):
                rb = r if rule is None else simulate(resample(b, rule), spec)
                cut = vault_cutoff(rb.daily.index, 18)
                s = summary(rb.daily.loc[rb.daily.index < cut], PPY)
                resolution.append({"coin": sym, "variant": spec.name, "bars": label, "cagr": s["cagr"], "sharpe": s["sharpe_ann"],
                                   "fills_yr": rb.fills / (len(rb.daily) / PPY)})
        idx = fifty.index
        fam[sym] = pd.DataFrame(V).reindex(idx).fillna(0.0)
        fam2[sym] = pd.DataFrame(V2).reindex(idx).fillna(0.0)

    # every variant judged against all 8 grid trials on the crypto venue
    act = []
    for sym in COINS:
        cut = vault_cutoff(fam[sym].index, 18)
        A = fam[sym].sub(benches[sym], axis=0).loc[fam[sym].index < cut]
        act += [sharpe(A[c].to_numpy()) for c in A]
    var_floor = float(np.nanvar(act, ddof=1))
    n_floor = ledger.count("crypto") + len(COINS) * len(SPECS)

    for sym in COINS:
        coin = sym.split("/")[0].lower()
        rep = research_gate(
            f"grid-{coin}-v1", "crypto", fam[sym], fam2[sym], benches[sym], ledger, snap, PPY,
            params={"variants": [s.name for s in SPECS], "levels": 20, "fee_per_side": FEE, "trade_through": 0.0005,
                    "bars": "5m", "benchmark": "50/50 coin/cash rebalanced monthly", "source": SOURCE, "mechanism": MECHANISM},
            n_trials_floor=n_floor, sr_variance_floor=var_floor,
        )
        gate_reports[sym] = rep
        cut = vault_cutoff(fam[sym].index, 18)
        rows = {}
        for k in list(fam[sym].columns):
            r = fam[sym][k].loc[fam[sym].index < cut]
            s = summary(r, PPY)
            rows[k] = {"cagr": s["cagr"], "sharpe": s["sharpe_ann"], "max_dd": s["max_dd"],
                       "vs_5050": float((r - benches[sym].loc[r.index]).mean() * PPY), **info[(sym, k)]}
        for k, r in (("50/50 rebalanced", benches[sym]), ("buy and hold", holds[sym])):
            s = summary(r.loc[r.index < cut], PPY)
            rows[k] = {"cagr": s["cagr"], "sharpe": s["sharpe_ann"], "max_dd": s["max_dd"]}
        y = {k: fam[sym][k] for k in fam[sym]} | {"50/50": benches[sym], "hold": holds[sym]}
        yearly = pd.DataFrame({k: v.loc[v.index < cut].groupby(v.loc[v.index < cut].index.year).apply(lambda x: float((1 + x).prod() - 1)) for k, v in y.items()})
        results[sym] = {"table": pd.DataFrame(rows).T, "yearly": yearly,
                        "period": (str(fam[sym].index[0].date()), str((cut - pd.Timedelta(days=1)).date()))}
    return {"snapshot": snap, "results": results, "gate": gate_reports, "resolution": pd.DataFrame(resolution)}


def _p(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def to_markdown(out: dict) -> str:
    lines = [f"# Grid trading results (5-minute bars, snapshot {out['snapshot']})", "",
             "Spot grids, 20 geometric levels, 0.10% fee per side, fills only on a 0.05% trade-through. "
             "Benchmark: 50/50 coin/cash rebalanced monthly. Vault (last 18 months) untouched.", ""]
    for sym, res in out["results"].items():
        T = res["table"]
        lines += [f"## {sym} ({res['period'][0]} – {res['period'][1]})", "", "```", out["gate"][sym].table(), "```", "",
                  "| Version | CAGR | Sharpe | Max DD | vs 50/50 /yr | Fills /yr | Fee drag /yr | Time fully in coin | Time fully in cash | Re-centres |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in T.iterrows():
            extra = "" if pd.isna(r.get("fills_yr")) else f" {_p(r.vs_5050)} | {r.fills_yr:,.0f} | {_p(r.fee_drag_yr)} | {_p(r.at_bottom)} | {_p(r.at_top)} | {int(r.recentres)} |"
            if not extra:
                extra = " | | | | | |"
            lines.append(f"| {k} | {_p(r.cagr)} | {r.sharpe:.2f} | {_p(r.max_dd)} |{extra}")
        lines += ["", "Calendar-year returns:", "", res["yearly"].map(lambda x: f"{x * 100:.1f}%").to_markdown(), ""]
    R = out["resolution"]
    lines += ["## Resolution check: same grids on coarser bars (research period)", "",
              "| Coin | Variant | 5m CAGR | 1h CAGR | 1d CAGR | 5m fills /yr | 1h fills /yr | 1d fills /yr |", "|---|---|---|---|---|---|---|---|"]
    for (coin, var), g in R.groupby(["coin", "variant"], sort=False):
        g = g.set_index("bars")
        lines.append(f"| {coin} | {var} | {_p(g.at['5m', 'cagr'])} | {_p(g.at['1h', 'cagr'])} | {_p(g.at['1d', 'cagr'])} | "
                     f"{g.at['5m', 'fills_yr']:,.0f} | {g.at['1h', 'fills_yr']:,.0f} | {g.at['1d', 'fills_yr']:,.0f} |")
    return "\n".join(lines) + "\n"
