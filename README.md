# regime-harness

Zero-cost research harness for a regime-rotation strategy across crypto, US-listed ETFs,
BIST and TEFAS. Its job is to **reject** strategies cheaply and honestly; only survivors
reach paper trading.

- **Phase 1 (done):** data layer, snapshot store, trial ledger, daily pull.
- **Phase 2 (done):** backtest engine and validation gate.
- Next: regime layer (Phase 3), LLM idea loop (Phase 4), BIST/TEFAS sleeves (Phase 5).

## What's here

| Path | Purpose |
| --- | --- |
| `harness/store.py` | Append-only snapshot store. Each pull saves only new or revised rows; `load(source, as_of=...)` rebuilds data exactly as known on that day. SHA-256 per file. |
| `harness/ledger.py` | Hash-chained trial ledger. Every backtest family, pass or fail, is a row; the summed `n_variants` is N for the Deflated Sharpe Ratio. Editing or deleting a row is detected. |
| `harness/health.py` | Checks each pull before storing: incomplete bars, bad prices, missing symbols (errors); staleness, suspicious jumps (warnings). |
| `harness/loaders/` | Crypto via ccxt (Coinbase public data), ETFs/gold/cash/USDTRY via yfinance. |
| `harness/engine/` | Total returns from raw closes + dividends; drift-aware backtest with execution lag (1 = next close, 2 = TEFAS-style next NAV); per-symbol costs with a TRY→USD spread placeholder; metrics. |
| `harness/gate/` | Walk-forward, purged/embargoed K-fold, Deflated Sharpe (N from the ledger), PBO via CSCV, one-shot 18-month vault. |
| `harness/strategies/` | `tsmom` (time-series momentum), `random_entry` (no-skill control). |
| `.github/workflows/daily-pull.yml` | Runs tests, pulls, verifies and commits a snapshot every day at 02:17 UTC. |

## Commands

```bash
pip install -r requirements.txt
pytest                                   # no network needed
python -m harness pull                   # today's snapshot (UTC)
python -m harness verify                 # hashes + ledger chain
python -m harness status                 # rows, last bar, revisions per source
python -m harness research calibration   # 200 random variants; must FAIL
python -m harness research tsmom         # TSMOM family through the gate
python -m harness research tsmom --vault # one-time vault test, survivors only
python -m harness ledger                 # trial counts per venue
```

## The gate

A strategy family passes only if **every** check passes on data before the vault cutoff:

| Check | Rule |
| --- | --- |
| Walk-forward excess return (3y train / 6m test, re-selected each step) | > 0 a year vs benchmark, after costs |
| Same, with costs doubled | > 0 |
| Walk-forward max drawdown | no worse than the benchmark's |
| Purged K-fold excess (5 folds, 21-day purge and embargo) | > 0 |
| Deflated Sharpe of active returns | > 0.95, N = all trials ever logged on that venue |
| Probability of Backtest Overfitting | < 0.20 |
| Vault (last 18 months, one attempt ever) | excess > 0 and Sharpe > 0 |

Skill is measured on **active** returns (strategy minus benchmark). Total-return Sharpe
rewards plain market exposure: random part-time holdings of rising assets score a positive
Sharpe with zero skill. The first calibration run in the ledger used total returns and is
kept there on purpose. The ledger is append-only, and that run belongs to the `calibration`
venue, so it doesn't raise the bar for real strategies.

## First results (snapshot 2026-10-30, research period Jun 2009 – Apr 2025)

| Family | Verdict | Notes |
| --- | --- | --- |
| `calibration-random-200` | FAIL (as required) | DSR 0.00, PBO 0.44 |
| `tsmom-etf-v1` (3/6/9/12-month lookbacks) | FAIL | Return and Sharpe below equal-weight; better drawdown only |

TSMOM variants vs the equal-weight benchmark of VT, SPY, EFA, EEM, TLT and GLD, before the vault:

| | CAGR | Sharpe | Max drawdown |
| --- | --- | --- | --- |
| 3m | 4.6% | 0.59 | −15.6% |
| 12m | 4.6% | 0.56 | −13.2% |
| Equal-weight benchmark | 8.3% | 0.71 | −25.5% |

## Rules this code enforces

- Snapshots are never overwritten; a same-day re-pull is refused.
- Today's still-forming bar is never stored, and a signal at close t cannot fill before close t+1.
- Rows a vendor drops later (delisted coins) stay in history.
- Weights drift between rebalances; there is no free daily rebalancing.
- Research code never sees the last 18 months; the vault test can run once per spec, and only after a research pass.
- The ledger has no delete or edit function, and `verify` catches manual edits.

## Known limits

- Costs are placeholders (10 bps per side for ETFs, 15 bps for crypto) until Midas's published fees and FX spread are checked.
- Cash earns nothing unless a strategy holds BIL explicitly.
- PBO is coarse for small families (4 variants); treat it as a red flag, not a precise probability.
- yfinance is unofficial and can break or be rate-limited.
- GitHub pauses scheduled workflows on repos with no activity for 60 days; the daily commits count as activity.
