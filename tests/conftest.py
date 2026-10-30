import numpy as np
import pandas as pd
import pytest


def make_bars(symbols, start, days, seed=0):
    rows = []
    for s in symbols:
        # One stream per symbol, so pulling more days never changes earlier bars
        rng = np.random.default_rng([seed, *map(ord, s)])
        px = 100.0
        for d in pd.date_range(start, periods=days, freq="D"):
            px *= float(np.exp(rng.normal(0, 0.01)))
            rows.append({"symbol": s, "date": d.date(), "open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1000.0})
    return pd.DataFrame(rows)


@pytest.fixture
def bars():
    return make_bars
