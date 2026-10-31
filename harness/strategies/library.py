"""Pre-registered strategy library (Phase 3).

Every strategy is fixed here BEFORE any result is seen: one parameter set, a
published source, and the mechanism that should make it work. No tuning after
results; a changed parameter is a new strategy and a new trial.

Contract for every strategy function
------------------------------------
* Input: `MarketData` for one calendar (ETF trading days, or crypto's 7-day week).
* Output: target weights, rows = decision dates, columns = symbols, long-only, sum <= 1.
* A row dated t may use data up to and including the close of t; the engine
  fills it at close t+1. `tests/test_library.py` checks this for every strategy
  by scrambling all data after a date and confirming earlier rows don't change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..engine.returns import month_end_dates
from . import tsmom

RISKY = ["VT", "SPY", "EFA", "EEM", "TLT", "GLD"]
CASH = "BIL"
COINS = ["BTC/USD", "ETH/USD"]


@dataclass
class MarketData:
    R: pd.DataFrame                    # daily total returns, wide
    high: pd.DataFrame | None = None   # raw daily high/low/close (for bar-shape signals)
    low: pd.DataFrame | None = None
    close: pd.DataFrame | None = None

    @property
    def px(self) -> pd.DataFrame:
        """Total-return index (starts at 1), NaN before each symbol's first bar."""
        idx = (1.0 + self.R.fillna(0.0)).cumprod()
        return idx.where(self.R.notna().cumsum() > 0)


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    venue: str          # "us_etfs" or "crypto"
    fn: Callable[[MarketData], pd.DataFrame]
    source: str
    mechanism: str


# ---------------------------------------------------------------- helpers
def _frame(rows: dict, cols) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(cols), dtype=float)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(cols)).fillna(0.0).sort_index()


def _changes_only(T: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where the target changes, so daily signals don't force daily rebalancing."""
    if T.empty:
        return T
    diff = T.diff().abs().sum(axis=1)
    diff.iloc[0] = 1.0
    return T[diff > 1e-12]


def _past(px: pd.DataFrame, d, months: int) -> pd.Series:
    """Index level `months` before d (last available on or before that date)."""
    sub = px.loc[: d - pd.DateOffset(months=months)]
    return sub.iloc[-1] if len(sub) else pd.Series(np.nan, index=px.columns)


def _ret(px: pd.DataFrame, d, months: int) -> pd.Series:
    return px.loc[d] / _past(px, d, months) - 1.0


def _week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).max().values).sort_values()


def _live(px: pd.DataFrame, d, syms) -> list[str]:
    return [s for s in syms if pd.notna(px.at[d, s])]


def _monthly(md: MarketData, rule) -> pd.DataFrame:
    px = md.px
    rows = {}
    for d in month_end_dates(md.R.index):
        w = rule(px, d)
        if w:
            rows[d] = w
    return _frame(rows, md.R.columns)


def _slots(px, d, keep: dict[str, bool], cash=CASH) -> dict:
    """Equal slot per live symbol; a symbol failing its filter hands its slot to cash."""
    live = [s for s in keep if pd.notna(px.at[d, s])]
    if not live or pd.isna(px.at[d, cash]):
        return {}
    w = {cash: 0.0}
    for s in live:
        if keep[s]:
            w[s] = w.get(s, 0.0) + 1.0 / len(live)
        else:
            w[cash] += 1.0 / len(live)
    return w


def _warm(px, d, months) -> bool:
    return len(px.loc[: d - pd.DateOffset(months=months)]) > 0


# ---------------------------------------------------------------- trend (time series)
def tsmom_12m(md):
    return tsmom.targets(md.R, 12, RISKY, CASH)


def tsmom_multi(md):
    def rule(px, d):
        if not _warm(px, d, 12):
            return {}
        score = {s: 0.0 for s in RISKY}
        for m in (1, 3, 6, 12):
            r = _ret(px, d, m)
            for s in RISKY:
                score[s] += 0.25 * float(r[s] > r[CASH])
        live = _live(px, d, RISKY)
        w = {CASH: 0.0}
        for s in live:
            w[s] = score[s] / len(live)
            w[CASH] += (1.0 - score[s]) / len(live)
        return w
    return _monthly(md, rule)


def _sma_filter(md, n, every="month"):
    px = md.px
    above = px > px.rolling(n, min_periods=n).mean()
    dates = month_end_dates(md.R.index) if every == "month" else md.R.index
    rows = {}
    for d in dates:
        if px.loc[:d].shape[0] < n:
            continue
        rows[d] = _slots(px, d, {s: bool(above.at[d, s]) for s in RISKY})
    rows = {d: w for d, w in rows.items() if w}
    T = _frame(rows, md.R.columns)
    return T if every == "month" else _changes_only(T)


def sma200_monthly(md):
    return _sma_filter(md, 200, "month")


def sma200_daily(md):
    return _sma_filter(md, 200, "day")


def golden_cross(md):
    px = md.px
    up = px.rolling(50, min_periods=50).mean() > px.rolling(200, min_periods=200).mean()
    rows = {}
    for d in md.R.index[200:]:
        w = _slots(px, d, {s: bool(up.at[d, s]) for s in RISKY})
        if w:
            rows[d] = w
    return _changes_only(_frame(rows, md.R.columns))


def donchian(md):
    px = md.px
    hi = px.rolling(100, min_periods=100).max()
    lo = px.rolling(50, min_periods=50).min()
    state = {s: False for s in RISKY}
    rows = {}
    for d in md.R.index[100:]:
        for s in RISKY:
            if pd.isna(px.at[d, s]):
                continue
            if not state[s] and px.at[d, s] >= hi.at[d, s]:
                state[s] = True
            elif state[s] and px.at[d, s] <= lo.at[d, s]:
                state[s] = False
        w = _slots(px, d, dict(state))
        if w:
            rows[d] = w
    return _changes_only(_frame(rows, md.R.columns))


# ---------------------------------------------------------------- cross-sectional / dual momentum
def _top_k(md, k, months, skip=0):
    def rule(px, d):
        if not _warm(px, d, months):
            return {}
        r = px.loc[d] if skip == 0 else _past(px, d, skip)
        r = r / _past(px, d, months) - 1.0
        r = r[RISKY].dropna().sort_values(ascending=False)
        return {s: 1.0 / k for s in r.index[:k]}
    return _monthly(md, rule)


def xs_top2_6m(md):
    return _top_k(md, 2, 6)


def xs_top3_12_1(md):
    return _top_k(md, 3, 12, skip=1)


def gem(md):
    def rule(px, d):
        if not _warm(px, d, 12):
            return {}
        r = _ret(px, d, 12)
        if r["SPY"] > r[CASH]:
            return {"SPY": 1.0} if r["SPY"] >= r["EFA"] else {"EFA": 1.0}
        return {"TLT": 1.0}
    return _monthly(md, rule)


def accel_dual_mom(md):
    def rule(px, d):
        if not _warm(px, d, 6):
            return {}
        sc = sum(_ret(px, d, m) for m in (1, 3, 6)) / 3.0
        best = "SPY" if sc["SPY"] >= sc["EFA"] else "EFA"
        return {best: 1.0} if sc[best] > 0 else {"TLT": 1.0}
    return _monthly(md, rule)


def vaa(md):
    off, dfn = ["SPY", "EFA", "EEM", "TLT"], ["GLD", CASH]

    def rule(px, d):
        if not _warm(px, d, 12):
            return {}
        r1, r3, r6, r12 = (_ret(px, d, m) for m in (1, 3, 6, 12))
        score = 12 * r1 + 4 * r3 + 2 * r6 + r12
        if (score[off] > 0).all():
            return {score[off].idxmax(): 1.0}
        return {score[dfn].idxmax(): 1.0}
    return _monthly(md, rule)


# ---------------------------------------------------------------- mean reversion
def _rsi(px: pd.Series, n: int) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rsi2_spy(md):
    px = md.px["SPY"]
    rsi, sma = _rsi(px, 2), px.rolling(200, min_periods=200).mean()
    inpos, rows = False, {}
    for d in md.R.index[200:]:
        if not inpos and px[d] > sma[d] and rsi[d] < 10:
            inpos = True
        elif inpos and rsi[d] > 70:
            inpos = False
        rows[d] = {"SPY": 1.0} if inpos else {CASH: 1.0}
    return _changes_only(_frame(rows, md.R.columns))


def weekly_reversal(md):
    px = md.px
    rows = {}
    we = _week_ends(md.R.index)
    for i in range(1, len(we)):
        d, prev = we[i], we[i - 1]
        r = (px.loc[d, RISKY] / px.loc[prev, RISKY] - 1.0).dropna().sort_values()
        if len(r) >= 2:
            rows[d] = {s: 0.5 for s in r.index[:2]}
    return _frame(rows, md.R.columns)


def ibs_spy(md):
    if md.high is None:
        return pd.DataFrame(columns=md.R.columns, dtype=float)
    h, l, c = md.high["SPY"], md.low["SPY"], md.close["SPY"]
    ibs = (c - l) / (h - l).replace(0, np.nan)
    rows = {d: ({"SPY": 1.0} if ibs[d] < 0.2 else {CASH: 1.0}) for d in md.R.index if pd.notna(ibs.get(d))}
    return _changes_only(_frame(rows, md.R.columns))


# ---------------------------------------------------------------- risk-based allocation
def _inv_vol(md, syms, lookback=60):
    vol = md.R[syms].rolling(lookback, min_periods=lookback).std()

    def rule(px, d):
        v = vol.loc[d].dropna()
        v = v[v > 0]
        if v.empty:
            return {}
        w = (1 / v) / (1 / v).sum()
        return w.to_dict()
    return _monthly(md, rule)


def inverse_vol(md):
    return _inv_vol(md, RISKY)


def risk_parity_3(md):
    return _inv_vol(md, ["SPY", "TLT", "GLD"])


def min_variance(md):
    R = md.R[RISKY]

    def rule(px, d):
        win = R.loc[:d].tail(120).dropna(axis=1)
        if len(win) < 120 or win.shape[1] < 2:
            return {}
        cov = win.cov().to_numpy() + np.eye(win.shape[1]) * 1e-8
        act = list(range(win.shape[1]))
        for _ in range(win.shape[1]):
            sub = cov[np.ix_(act, act)]
            w = np.linalg.solve(sub, np.ones(len(act)))
            w = w / w.sum()
            if (w >= 0).all():
                break
            act = [a for a, x in zip(act, w) if x > 0]
        return {win.columns[a]: float(x) for a, x in zip(act, w)}
    return _monthly(md, rule)


def vol_target_spy(md):
    vol = md.R["SPY"].rolling(20, min_periods=20).std() * np.sqrt(252)
    rows = {}
    for d in _week_ends(md.R.index):
        if pd.notna(vol.get(d)):
            w = float(min(1.0, 0.10 / vol[d]))
            rows[d] = {"SPY": w, CASH: 1.0 - w}
    return _frame(rows, md.R.columns)


def sixty_forty(md):
    return _monthly(md, lambda px, d: {"SPY": 0.6, "TLT": 0.4})


def permanent(md):
    return _monthly(md, lambda px, d: {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, CASH: 0.25})


# ---------------------------------------------------------------- correlation / regime
def stock_bond_corr(md):
    corr = md.R["SPY"].rolling(60, min_periods=60).corr(md.R["TLT"])

    def rule(px, d):
        c = corr.get(d)
        if pd.isna(c):
            return {}
        return {"SPY": 0.4, "GLD": 0.3, CASH: 0.3} if c > 0 else {"SPY": 0.6, "TLT": 0.4}
    return _monthly(md, rule)


def vol_regime(md):
    vol = md.R["SPY"].rolling(20, min_periods=20).std()
    med = vol.rolling(252, min_periods=252).median()

    def rule(px, d):
        if pd.isna(med.get(d)):
            return {}
        if vol[d] > med[d]:
            return {"GLD": 0.5, CASH: 0.5}
        live = _live(px, d, RISKY)
        return {s: 1.0 / len(live) for s in live}
    return _monthly(md, rule)


def drawdown_brake(md):
    ew = md.R[RISKY].mean(axis=1, skipna=True).fillna(0.0)
    eq = (1 + ew).cumprod()
    dd = eq / eq.rolling(126, min_periods=1).max() - 1.0
    me = set(month_end_dates(md.R.index))
    px = md.px
    out, rows = False, {}
    for d in md.R.index[126:]:
        changed = False
        if not out and dd[d] < -0.10:
            out, changed = True, True
        elif out and dd[d] > -0.05:
            out, changed = False, True
        if changed or d in me:
            if out:
                rows[d] = {CASH: 1.0}
            else:
                live = _live(px, d, RISKY)
                rows[d] = {s: 1.0 / len(live) for s in live}
    return _frame(rows, md.R.columns)


# ---------------------------------------------------------------- seasonality
def turn_of_month(md):
    idx = md.R.index
    ym = idx.year * 12 + idx.month
    pos_in_month = pd.Series(1, index=idx).groupby(ym).cumsum().to_numpy()
    size = pd.Series(1, index=idx).groupby(ym).transform("sum").to_numpy()
    rows = {}
    for i, d in enumerate(idx):
        if pos_in_month[i] == size[i] - 1:      # 2nd-last day -> fill at last day's close
            rows[d] = {"SPY": 1.0}
        elif pos_in_month[i] == 3:              # 3rd day -> fill at 4th day's close
            rows[d] = {CASH: 1.0}
    return _frame(rows, md.R.columns)


def sell_in_may(md):
    rows = {}
    for d in month_end_dates(md.R.index):
        if d.month == 10:
            rows[d] = {"SPY": 1.0}
        elif d.month == 4:
            rows[d] = {CASH: 1.0}
    return _frame(rows, md.R.columns)


# ---------------------------------------------------------------- defensive
def gold_cash(md):
    return _monthly(md, lambda px, d: {"GLD": 0.5, CASH: 0.5})


def cash_bil(md):
    return _monthly(md, lambda px, d: {CASH: 1.0})


# ---------------------------------------------------------------- crypto (7-day calendar, cash = 0% stablecoin)
def btc_sma200(md):
    px = md.px["BTC/USD"]
    sma = px.rolling(200, min_periods=200).mean()
    rows = {d: {"BTC/USD": 1.0 if px[d] > sma[d] else 0.0} for d in md.R.index[200:] if pd.notna(sma[d])}
    return _changes_only(_frame(rows, md.R.columns))


def crypto_tsmom_3m(md):
    def rule(px, d):
        if not _warm(px, d, 3):
            return {}
        r = _ret(px, d, 3)
        live = _live(px, d, COINS)
        w = {s: 1.0 / len(live) for s in live if r[s] > 0}
        return w or {"BTC/USD": 0.0}
    return _monthly(md, rule)


def crypto_xs_1m(md):
    def rule(px, d):
        if not _warm(px, d, 1):
            return {}
        r = _ret(px, d, 1)[COINS].dropna()
        if r.empty:
            return {}
        best = r.idxmax()
        return {best: 1.0} if r[best] > 0 else {"BTC/USD": 0.0}
    return _monthly(md, rule)


def btc_vol_target(md):
    vol = md.R["BTC/USD"].rolling(30, min_periods=30).std() * np.sqrt(365)
    rows = {}
    for d in _week_ends(md.R.index):
        if pd.notna(vol.get(d)) and vol[d] > 0:
            rows[d] = {"BTC/USD": float(min(1.0, 0.5 / vol[d]))}
    return _frame(rows, md.R.columns)


# ---------------------------------------------------------------- registry
LIBRARY: list[Spec] = [
    Spec("tsmom_12m", "trend", "us_etfs", tsmom_12m, "Moskowitz, Ooi & Pedersen (2012), Time series momentum", "slow information diffusion and herding make 12-month trends persist"),
    Spec("tsmom_multi", "trend", "us_etfs", tsmom_multi, "Hurst, Ooi & Pedersen (2017), A century of evidence on trend-following", "averaging 1/3/6/12-month signals diversifies across trend speeds"),
    Spec("sma200_monthly", "trend", "us_etfs", sma200_monthly, "Faber (2007), A quantitative approach to tactical asset allocation", "price below its 10-month average flags a downtrend; exiting avoids the deep part of bear markets"),
    Spec("sma200_daily", "trend", "us_etfs", sma200_daily, "Faber (2007), daily-checked variant", "same filter, reacts faster at higher cost"),
    Spec("golden_cross", "trend", "us_etfs", golden_cross, "Brock, Lakonishok & LeBaron (1992), Simple technical trading rules", "short average above long average captures medium-term trend"),
    Spec("donchian", "trend", "us_etfs", donchian, "Donchian channel / Turtle rules (Faith 2007)", "new 100-day highs signal breakouts; 50-day lows end them"),
    Spec("xs_top2_6m", "momentum", "us_etfs", xs_top2_6m, "Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013)", "assets that outperformed peers keep outperforming for months"),
    Spec("xs_top3_12_1", "momentum", "us_etfs", xs_top3_12_1, "Asness, Moskowitz & Pedersen (2013), Value and momentum everywhere", "12-1 momentum skips the last month to avoid short-term reversal"),
    Spec("gem", "momentum", "us_etfs", gem, "Antonacci (2014), Dual Momentum Investing", "relative momentum picks the stronger equity market; absolute momentum exits to bonds in bear markets"),
    Spec("accel_dual_mom", "momentum", "us_etfs", accel_dual_mom, "EngineeredPortfolio (2018), Accelerating Dual Momentum", "faster 1/3/6-month blend reacts sooner than 12-month dual momentum"),
    Spec("vaa", "momentum", "us_etfs", vaa, "Keller & Keuning (2017), Breadth momentum and vigilant asset allocation", "any weak offensive asset signals breadth deterioration; go defensive early"),
    Spec("rsi2_spy", "mean_reversion", "us_etfs", rsi2_spy, "Connors & Alvarez (2008), Short Term Trading Strategies That Work", "sharp 2-day oversold dips in an uptrend tend to bounce"),
    Spec("weekly_reversal", "mean_reversion", "us_etfs", weekly_reversal, "Lehmann (1990), Fads, martingales and market efficiency", "last week's losers rebound as liquidity-driven selling reverses"),
    Spec("ibs_spy", "mean_reversion", "us_etfs", ibs_spy, "Pagonidis (2013), The IBS effect", "closes near the day's low tend to be followed by up days"),
    Spec("inverse_vol", "risk", "us_etfs", inverse_vol, "Maillard, Roncalli & Teiletche (2010), naive risk parity", "equalising risk contribution avoids concentration in the most volatile asset"),
    Spec("risk_parity_3", "risk", "us_etfs", risk_parity_3, "Dalio (All Weather) / Qian (2005), Risk parity", "stocks, bonds and gold hedge different economic regimes"),
    Spec("min_variance", "risk", "us_etfs", min_variance, "Clarke, de Silva & Thorley (2006), Minimum-variance portfolios", "low-risk portfolios have historically delivered similar returns with less risk"),
    Spec("vol_target_spy", "risk", "us_etfs", vol_target_spy, "Moreira & Muir (2017), Volatility-managed portfolios", "high recent volatility predicts high future volatility but not higher returns, so cut exposure"),
    Spec("sixty_forty", "risk", "us_etfs", sixty_forty, "Classic balanced portfolio", "stock-bond diversification; a reference point every other strategy should beat"),
    Spec("permanent", "risk", "us_etfs", permanent, "Browne (1987), The Permanent Portfolio", "one asset for each regime: growth, deflation, inflation, recession"),
    Spec("stock_bond_corr", "regime", "us_etfs", stock_bond_corr, "Ilmanen (2003); Campbell, Sunderam & Viceira (2017)", "when stocks and bonds move together, bonds stop hedging; swap them for gold and cash"),
    Spec("vol_regime", "regime", "us_etfs", vol_regime, "Ang & Bekaert (2002), regime switches in volatility", "high-volatility regimes have worse risk-adjusted equity returns"),
    Spec("drawdown_brake", "regime", "us_etfs", drawdown_brake, "Grossman & Zhou (1993), Optimal investment strategies for controlling drawdowns", "cutting exposure after a 10% fall limits the tail of large losses"),
    Spec("turn_of_month", "seasonality", "us_etfs", turn_of_month, "Lakonishok & Smidt (1988); Ariel (1987)", "month-end pension and payroll flows push prices up around the turn of the month"),
    Spec("sell_in_may", "seasonality", "us_etfs", sell_in_may, "Bouman & Jacobsen (2002), The Halloween indicator", "November-April has historically outperformed May-October"),
    Spec("gold_cash", "defensive", "us_etfs", gold_cash, "Baur & Lucey (2010), Is gold a hedge or a safe haven?", "gold and T-bills hold value in equity crashes"),
    Spec("cash_bil", "defensive", "us_etfs", cash_bil, "T-bill anchor", "zero-risk option the selector can always fall back to"),
    Spec("btc_sma200", "crypto_trend", "crypto", btc_sma200, "Detzel et al. (2021), Learning and predictability via technical analysis: Bitcoin", "crypto trends are strong and long; a 200-day filter avoids most of the 70-80% crashes"),
    Spec("crypto_tsmom_3m", "crypto_trend", "crypto", crypto_tsmom_3m, "Liu & Tsyvinski (2021), Risks and returns of cryptocurrency", "crypto shows strong time-series momentum at 1-3 month horizons"),
    Spec("crypto_xs_1m", "crypto_momentum", "crypto", crypto_xs_1m, "Liu, Tsyvinski & Wu (2022), Common risk factors in cryptocurrency", "the stronger coin over the last month tends to keep leading"),
    Spec("btc_vol_target", "crypto_risk", "crypto", btc_vol_target, "Moreira & Muir (2017) applied to bitcoin", "scaling to 50% annual volatility cuts exposure in crypto's violent phases"),
]
