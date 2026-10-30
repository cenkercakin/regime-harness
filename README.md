# regime-harness

Zero-cost research harness for a regime-rotation strategy across crypto, US-listed ETFs,
BIST and TEFAS. **Phase 1 = data layer + trial ledger only.** No strategies yet, by design:
the store and ledger must be trustworthy before anything is tested on them.

## What's here

| Path | Purpose |
| --- | --- |
| `harness/store.py` | Append-only snapshot store. Each pull saves only new or revised rows; `load(source, as_of=...)` rebuilds data exactly as known on that day. SHA-256 per file. |
| `harness/ledger.py` | Hash-chained trial ledger. Every backtest, pass or fail, is a row; the count is N for the Deflated Sharpe Ratio. Editing or deleting a row is detected. |
| `harness/health.py` | Checks each pull before storing: incomplete bars, bad prices, missing symbols (errors); staleness, suspicious jumps (warnings). |
| `harness/loaders/` | Crypto via ccxt (Coinbase public data), ETFs/gold/cash/USDTRY via yfinance. |
| `harness/config.py` | The Phase 1 universe. |
| `.github/workflows/daily-pull.yml` | Runs tests, pulls, verifies and commits a snapshot every day at 02:17 UTC. |

## Setup (once)

1. Create a **public** GitHub repo (free unlimited Actions minutes), e.g. `regime-harness`.
2. Unzip this folder and push it:
   ```bash
   cd regime-harness
   git init && git add . && git commit -m "Phase 1: data layer and trial ledger"
   git branch -M main
   git remote add origin https://github.com/<you>/regime-harness.git
   git push -u origin main
   ```
3. In the repo: **Actions** tab → enable workflows → open `daily-pull` → **Run workflow** to do the first pull now.
4. Check the run log: every source should print `[ok]`. The first run pulls full history; later runs pull only recent days.

## Local use

```bash
pip install -r requirements.txt
pytest                      # 28 tests, no network needed
python -m harness pull      # today's snapshot (UTC)
python -m harness verify    # hashes + ledger chain
python -m harness status    # rows, last bar, revisions per source
```

## Rules this code enforces

- Snapshots are never overwritten; a same-day re-pull is refused.
- Today's still-forming bar is never stored (a common lookahead leak).
- Rows a vendor drops later (delisted coins) stay in history.
- Yahoo's dividend-adjusted close is not stored; raw OHLC + dividends + splits are, so adjustments are reproducible in code.
- The ledger has no delete or edit function, and `verify` catches manual edits.

## Known limits

- yfinance is unofficial and can break or be rate-limited; a failed source fails the run (red X) but the other sources are still committed.
- GitHub pauses scheduled workflows on repos with no activity for 60 days; the daily commits count as activity.
- BIST and TEFAS loaders come in Phase 5.
