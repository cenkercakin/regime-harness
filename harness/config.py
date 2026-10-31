"""Phase 1 universe. Every source here is free and needs no account or API key.

Notes on choices
----------------
* Crypto uses Coinbase's public market-data endpoints. GitHub Actions runners are
  in the US, where Binance.com refuses connections, so Binance is not the default.
* US-listed ETFs match what Midas can buy. BIL stands in for USD cash (T-bills),
  GLD for gold. VT is the global-equity benchmark; SPY/EFA/EEM let the regime
  layer see US vs developed vs emerging separately.
* `lookback_days` is how far back each later pull re-fetches. Yahoo is re-pulled
  in full so split rewrites of history are caught as revisions.
"""

from pathlib import Path

_TICKERS = Path(__file__).resolve().parent.parent / "config" / "bist_tickers.txt"


def bist_symbols() -> list[str]:
    lines = _TICKERS.read_text().splitlines() if _TICKERS.exists() else []
    return [f"{t.strip()}.IS" for t in lines if t.strip() and not t.startswith("#")]


SOURCES = {
    "crypto": {
        "loader": "crypto",
        "exchange": "coinbaseexchange",
        "symbols": ["BTC/USD", "ETH/USD"],
        "start": "2016-01-01",
        "lookback_days": 10,
        "max_stale_days": 2,
        "max_abs_daily_return": 0.5,
    },
    "us_etfs": {
        "loader": "yahoo",
        "symbols": ["VT", "SPY", "EFA", "EEM", "TLT", "GLD", "BIL"],
        "start": "2008-01-01",
        "lookback_days": None,
        "max_stale_days": 5,
        "max_abs_daily_return": 0.25,
    },
    "fx": {
        "loader": "yahoo",
        "symbols": ["USDTRY=X"],
        "start": "2008-01-01",
        "lookback_days": None,
        "max_stale_days": 5,
        "max_abs_daily_return": 0.25,
    },
    "crypto_5m": {
        # 5-minute candles for intraday strategies (grid trading). ~105k bars per coin per year.
        "loader": "crypto",
        "exchange": "coinbaseexchange",
        "timeframe": "5m",
        "symbols": ["BTC/USD", "ETH/USD"],
        "start": "2017-01-01",
        "lookback_days": 2,
        "max_stale_days": 2,
        "max_abs_daily_return": 0.5,
    },
    "bist": {
        # All currently listed BIST stocks. Survivorship-biased: delisted names are absent.
        "loader": "yahoo",
        "symbols": bist_symbols(),
        "start": "2010-01-01",
        "lookback_days": None,      # full re-pull: Yahoo rewrites history after splits / bonus issues
        "max_stale_days": 7,
        "max_abs_daily_return": 0.4,
        "max_missing_frac": 0.10,
        "batch_size": 50,
        "batch_pause_s": 2,
    },
    "bist_index": {
        "loader": "yahoo",
        "symbols": ["XU100.IS", "XU030.IS"],
        "start": "2010-01-01",
        "lookback_days": None,
        "max_stale_days": 7,
        "max_abs_daily_return": 0.25,
    },
}
