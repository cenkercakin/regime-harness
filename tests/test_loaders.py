from datetime import date, datetime, timezone

import pandas as pd

from harness.loaders import crypto, yahoo

DAY = 86_400_000


def ms(d):
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


class FakeExchange:
    """Serves daily candles from `listed` onwards, at most `limit` per call, like an exchange."""

    def __init__(self, listed, upto):
        self.first, self.last, self.calls = ms(listed), ms(upto), 0

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        t = max(since, self.first)
        out = []
        while t <= self.last and len(out) < limit:
            out.append([t, 1.0, 2.0, 0.5, 1.5, 10.0])
            t += DAY
        return out


def test_crypto_paginates_skips_prelisting_and_excludes_today():
    ex = FakeExchange(listed=date(2026, 3, 1), upto=date(2026, 10, 31))
    df = crypto.fetch_daily(["BTC/USD"], date(2025, 1, 1), date(2026, 10, 31), exchange=ex, page_limit=50)
    assert df["date"].min() == date(2026, 3, 1)
    assert df["date"].max() == date(2026, 10, 30)  # snapshot day's forming candle excluded
    assert not df.duplicated(["symbol", "date"]).any()
    assert len(df) == (date(2026, 10, 31) - date(2026, 3, 1)).days
    assert ex.calls > 1


def test_yahoo_parses_multiindex_and_drops_end_day():
    idx = pd.date_range("2026-10-26", "2026-10-30", freq="B")
    cols = pd.MultiIndex.from_product([["SPY", "GLD"], ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw.loc[idx[-1], ("GLD", "Close")] = float("nan")  # missing bar dropped

    df = yahoo.fetch_daily(["SPY", "GLD", "MISSING"], date(2026, 10, 1), date(2026, 10, 30), downloader=lambda *a: raw)
    assert set(df["symbol"]) == {"SPY", "GLD"}
    assert df["date"].max() == date(2026, 10, 29)
    assert list(df.columns) == ["symbol", "date", "open", "high", "low", "close", "volume", "dividends", "splits"]
    assert "Adj Close" not in df.columns


class FakeIntraday:
    def __init__(self, tf_ms, first, last):
        self.tf, self.first, self.last, self.calls = tf_ms, first, last, 0

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        if self.calls == 2:
            raise type("NetworkError", (Exception,), {})("flaky")   # one transient failure
        t = max(since, self.first)
        out = []
        while t <= self.last and len(out) < limit:
            out.append([t, 1.0, 2.0, 0.5, 1.5, 10.0])
            t += self.tf
        return out


def test_crypto_5m_timestamps_pagination_and_retry(monkeypatch):
    monkeypatch.setattr(crypto.time, "sleep", lambda s: None)
    ex = FakeIntraday(300_000, ms(date(2026, 10, 1)), ms(date(2026, 10, 3)))
    df = crypto.fetch_daily(["BTC/USD"], date(2026, 10, 1), date(2026, 10, 3), exchange=ex, page_limit=100, timeframe="5m")
    assert len(df) == 2 * 288                                   # two full days of 5-minute bars, end day excluded
    assert pd.Timestamp(df["date"].iloc[1]) - pd.Timestamp(df["date"].iloc[0]) == pd.Timedelta(minutes=5)
    assert not df.duplicated(["symbol", "date"]).any()
