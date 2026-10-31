"""Research runs: build a variant family, backtest it, send it through the gate."""

from __future__ import annotations

import pandas as pd

from .engine import backtest
from .engine.returns import load_returns
from .gate import research_gate, vault_test
from .ledger import TrialLedger
from .store import SnapshotStore
from .strategies import random_entry, tsmom

ETF_SYMBOLS = tsmom.RISKY + [tsmom.CASH]
PPY_ETF = 252


def _family(returns: pd.DataFrame, target_sets: dict[str, pd.DataFrame], costs: backtest.CostModel, lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    net, net2 = {}, {}
    for name, tg in target_sets.items():
        net[name] = backtest.run(tg, returns, costs, lag).returns
        net2[name] = backtest.run(tg, returns, costs.scaled(2.0), lag).returns
    return pd.DataFrame(net), pd.DataFrame(net2)


def _common_start(returns: pd.DataFrame) -> pd.DataFrame:
    """Start once every symbol has a price, so all variants and the benchmark see the same universe."""
    first = returns.apply(lambda s: s.first_valid_index()).max()
    return returns.loc[returns.index > first]


def etf_inputs(store: SnapshotStore, as_of: str | None):
    R = _common_start(load_returns(store, "us_etfs", ETF_SYMBOLS, as_of))
    bench = backtest.run(tsmom.equal_weight_targets(R), R, backtest.ETF_COSTS, lag=1).returns
    return R, bench


def run_tsmom(store: SnapshotStore, ledger: TrialLedger, as_of: str | None = None, lookbacks=(3, 6, 9, 12), vault: bool = False):
    snap = as_of or store.snapshot_ids("us_etfs")[-1]
    R, bench = etf_inputs(store, snap)
    tsets = {f"tsmom_{L}m": tsmom.targets(R, L) for L in lookbacks}
    V, V2 = _family(R, tsets, backtest.ETF_COSTS, lag=1)
    # Drop the warm-up year so the longest lookback is not penalised for sitting in cash
    start = V.index[0] + pd.DateOffset(months=max(lookbacks))
    V, V2, bench = V.loc[start:], V2.loc[start:], bench.loc[start:]
    if vault:
        return vault_test("tsmom-etf-v1", "us_etfs", V, bench, ledger, snap, PPY_ETF)
    return research_gate(
        "tsmom-etf-v1", "us_etfs", V, V2, bench, ledger, snap, PPY_ETF,
        params={"lookbacks_months": list(lookbacks), "rebalance": "month-end", "lag": 1, "costs_bps": backtest.ETF_COSTS.default_bps},
    )


def run_calibration(store: SnapshotStore, ledger: TrialLedger, as_of: str | None = None, n: int = 200):
    snap = as_of or store.snapshot_ids("us_etfs")[-1]
    R, bench = etf_inputs(store, snap)
    tsets = {f"rand_{i:03d}": random_entry.targets(R, seed=i, risky=tsmom.RISKY) for i in range(n)}
    V, V2 = _family(R, tsets, backtest.ETF_COSTS, lag=1)
    return research_gate(
        f"calibration-random-{n}", "calibration", V, V2, bench, ledger, snap, PPY_ETF,
        params={"n_variants": n, "p_in": 0.5, "note": "no-skill control; must fail"},
    )
