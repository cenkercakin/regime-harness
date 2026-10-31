# regime-harness

Zero-cost research harness for a regime-rotation strategy across crypto, US-listed ETFs,
BIST and TEFAS. Its job is to **reject** strategies cheaply and honestly; only survivors
reach paper trading.

- **Phase 1 (done):** data layer, snapshot store, trial ledger, daily pull.
- **Phase 2 (done):** backtest engine and validation gate.
- **Phase 3 (done):** 31-strategy library + monthly selector (hard switch / soft tilt / blend). Results: `reports/phase3-2026-10-31.md`.
- **Phase 5 (done):** BIST data (630 stocks, XU100), short selling with SPK ban calendar, 11 stock strategies incl. correlation-breakdown pairs. Results: `reports/phase5-bist-2026-10-31.md`.
- Next: LLM idea loop (Phase 4), TEFAS sleeve.

## What's here

| Path | Purpose |
| --- | --- |
| `harness/store.py` | Append-only snapshot store. Each pull saves only new or revised rows; `load(source, as_of=...)` rebuilds data exactly as known on that day. SHA-256 per file. |
| `harness/ledger.py` | Hash-chained trial ledger. Every backtest family, pass or fail, is a row; the summed `n_variants` is N for the Deflated Sharpe Ratio. Editing or deleting a row is detected. |
| `harness/health.py` | Checks each pull before storing: incomplete bars, bad prices, missing symbols (errors); staleness, suspicious jumps (warnings). |
| `harness/loaders/` | Crypto via ccxt (Coinbase public data), ETFs/gold/cash/USDTRY via yfinance. |
| `harness/engine/` | Total returns from raw closes + dividends; drift-aware backtest with execution lag (1 = next close, 2 = TEFAS-style next NAV); per-symbol costs with a TRY→USD spread placeholder; metrics. |
| `harness/gate/` | Walk-forward, purged/embargoed K-fold, Deflated Sharpe (N from the ledger), PBO via CSCV, one-shot 18-month vault. |
| `harness/strategies/library.py` | 31 pre-registered strategies (trend, momentum, mean reversion, risk, regime, seasonality, defensive, crypto), each with source and mechanism. |
| `harness/selector.py` | Monthly selector: ranks strategies by trailing Sharpe above cash; hard switch, soft tilt or blend; crypto capped at 25%; switching costs incl. FX placeholder. |
| `harness/phase3.py` | Runs the library and selectors through the gate and writes the report. |
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
python -m harness research library --selectors-only  # selectors only; strategies already logged
python -m harness research bist          # BIST stock library through the gate
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

- **BIST is survivorship-biased:** `config/bist_tickers.txt` lists only companies listed today, so delisted failures are missing and BIST results are optimistic.
- XU100 from Yahoo is a price index (no dividends), so "vs XU100" overstates stock strategies by roughly the dividend yield; the report also shows an equal-weight top-100 benchmark.
- Short-selling bans in `config/short_bans.csv` are researched from SPK/KAP news with sources; shorts are force-covered on ban days (the real rule may only block new shorts). Borrow fee is a 5% a year placeholder.

- Costs are placeholders (10 bps per side for ETFs, 15 bps for crypto) until Midas's published fees and FX spread are checked.
- Cash earns nothing unless a strategy holds BIL explicitly.
- PBO is coarse for small families (4 variants); treat it as a red flag, not a precise probability.
- yfinance is unofficial and can break or be rate-limited.
- GitHub pauses scheduled workflows on repos with no activity for 60 days; the daily commits count as activity.
