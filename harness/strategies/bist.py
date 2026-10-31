"""BIST stock strategies (Phase 5), pre-registered.

Data rules (applied before any strategy sees the data)
------------------------------------------------------
* Returns are converted to USD daily with the stored USD/TRY rate.
* BIST caps daily moves at about 10%, so a daily move above 25% in either direction is
  treated as a corporate-action data error (an unadjusted bonus issue or split): that
  day's return is set to 0 and the stock is excluded from the universe for that month
  and the next.
* Universe each month = the 100 stocks with the highest median daily TRY turnover
  over the last 63 trading days, among stocks with at least 252 days of history.
  Only data up to the decision date is used. The top 30 of those are the shortable set
  (a proxy for BIST30), except on SPK short-ban days.
* SURVIVORSHIP: the ticker list contains only companies listed today. Results over
  earlier years are optimistic, because companies that later failed are missing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.returns import month_end_dates
from .library import Spec, _changes_only, _week_ends

BAN_FILE = Path(__file__).resolve().parents[2] / "config" / "short_bans.csv"
JUMP_LIMIT = 0.25


@dataclass
class BistData:
    R: pd.DataFrame                      # daily USD total returns of stocks (cleaned)
    turnover: pd.DataFrame               # daily TRY turnover (close x volume)
    index_R: pd.Series                   # XU100 daily USD return (price index)
    members: dict = field(default_factory=dict)   # month-end -> list of top-100 symbols, ranked
    flagged: pd.DataFrame | None = None  # True where a jump was neutralised

    @property
    def px(self) -> pd.DataFrame:
        idx = (1.0 + self.R.fillna(0.0)).cumprod()
        return idx.where(self.R.notna().cumsum() > 0)

    def universe(self, d, n: int = 100) -> list[str]:
        """Members chosen at the latest month-end on or before d."""
        keys = [k for k in self.members if k <= d]
        return self.members[keys[-1]][:n] if keys else []


def ban_days(index: pd.DatetimeIndex, path: Path = BAN_FILE) -> pd.Series:
    banned = pd.Series(False, index=index)
    if not path.exists():
        return banned
    with open(path) as f:
        for row in csv.DictReader(f):
            banned |= (index >= pd.Timestamp(row["start"])) & (index <= pd.Timestamp(row["end"]))
    return banned


def to_usd(r_try: pd.DataFrame | pd.Series, usdtry: pd.Series) -> pd.DataFrame | pd.Series:
    fx = usdtry.reindex(r_try.index.union(usdtry.index)).sort_index().ffill().reindex(r_try.index)
    fx_ratio = fx.shift(1) / fx
    if isinstance(r_try, pd.DataFrame):
        return (1.0 + r_try).mul(fx_ratio, axis=0) - 1.0
    return (1.0 + r_try) * fx_ratio - 1.0


def build(r_try: pd.DataFrame, turnover: pd.DataFrame, index_try: pd.Series, usdtry: pd.Series,
          n_universe: int = 100, min_history: int = 252, lookback: int = 63) -> BistData:
    jumps = r_try.abs() > JUMP_LIMIT
    clean = r_try.mask(jumps, 0.0)
    R = to_usd(clean, usdtry)
    R = R.where(r_try.notna())
    idx_r = to_usd(index_try.mask(index_try.abs() > JUMP_LIMIT, 0.0), usdtry)  # e.g. the July 2020 /100 redenomination

    hist = r_try.notna().cumsum()
    med_turn = turnover.rolling(lookback, min_periods=lookback // 2).median()
    jump_month = jumps.groupby([jumps.index.year, jumps.index.month]).transform("any")
    members = {}
    me = month_end_dates(r_try.index)
    for i, d in enumerate(me):
        prev = me[i - 1] if i else None
        bad = set(jumps.columns[jump_month.loc[d]])
        if prev is not None:
            bad |= set(jumps.columns[jump_month.loc[prev]])
        ok = (hist.loc[d] >= min_history) & r_try.loc[:d].iloc[-1].notna()
        cand = med_turn.loc[d][ok].drop(labels=list(bad), errors="ignore").dropna()
        members[d] = list(cand.sort_values(ascending=False).index[:n_universe])
    return BistData(R=R, turnover=turnover, index_R=idx_r, members=members, flagged=jumps)


def shortable_mask(bd: BistData, top: int = 30) -> pd.DataFrame:
    """True where a stock may be held short: in the latest top-30 and not on a ban day."""
    mask = pd.DataFrame(False, index=bd.R.index, columns=bd.R.columns)
    me = sorted(bd.members)
    for i, d in enumerate(me):
        end = me[i + 1] if i + 1 < len(me) else bd.R.index[-1] + pd.Timedelta(days=1)
        rows = (bd.R.index > d) & (bd.R.index <= end)
        cols = [c for c in bd.members[d][:top] if c in mask.columns]
        mask.loc[rows, cols] = True
    banned = ban_days(bd.R.index)
    mask.loc[banned.to_numpy()] = False
    return mask


def cost_frame(bd: BistData, top_bps: float = 10.0, other_bps: float = 20.0, top: int = 30) -> pd.DataFrame:
    """Per-side cost: cheaper while a stock is in the most-liquid group (spread + slippage; commission 0)."""
    C = pd.DataFrame(other_bps, index=bd.R.index, columns=bd.R.columns)
    me = sorted(bd.members)
    for i, d in enumerate(me):
        end = me[i + 1] if i + 1 < len(me) else bd.R.index[-1] + pd.Timedelta(days=1)
        rows = (bd.R.index > d) & (bd.R.index <= end)
        C.loc[rows, [c for c in bd.members[d][:top] if c in C.columns]] = top_bps
    return C


# ---------------------------------------------------------------- helpers
def _equal(names, total=1.0) -> dict:
    return {s: total / len(names) for s in names} if names else {}


def _monthly(bd: BistData, rule) -> pd.DataFrame:
    rows = {}
    for d in month_end_dates(bd.R.index):
        u = bd.universe(d)
        if len(u) < 10:
            continue
        w = rule(d, u)
        if w is not None:
            rows[d] = w
    return _frame(rows, bd.R.columns)


def _frame(rows, cols) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(cols), dtype=float)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(cols)).fillna(0.0).sort_index()


def _lookback_ret(px, d, months, skip=0):
    end = px.loc[: d - pd.DateOffset(months=skip)].iloc[-1] if skip else px.loc[d]
    start = px.loc[: d - pd.DateOffset(months=months)]
    if not len(start):
        return pd.Series(np.nan, index=px.columns)
    return end / start.iloc[-1] - 1.0


# ---------------------------------------------------------------- strategies
def mom_12_1(bd):
    px = bd.px

    def rule(d, u):
        r = _lookback_ret(px, d, 12, skip=1)[u].dropna()
        return _equal(list(r.sort_values(ascending=False).index[:10])) if len(r) >= 20 else None
    return _monthly(bd, rule)


def mom_ls30(bd):
    """Long top 5 / short bottom 5 of the 30 most liquid by 12-1 momentum, 100/100."""
    px = bd.px

    def rule(d, u):
        r = _lookback_ret(px, d, 12, skip=1)[u[:30]].dropna().sort_values(ascending=False)
        if len(r) < 20:
            return None
        w = _equal(list(r.index[:5]), 1.0)
        w.update({s: -0.2 for s in r.index[-5:]})
        return w
    return _monthly(bd, rule)


def low_vol(bd):
    vol = bd.R.rolling(126, min_periods=100).std()

    def rule(d, u):
        v = vol.loc[d, u].dropna()
        return _equal(list(v.sort_values().index[:10])) if len(v) >= 20 else None
    return _monthly(bd, rule)


def weekly_reversal(bd):
    px = bd.px
    rows = {}
    we = _week_ends(bd.R.index)
    for i in range(1, len(we)):
        d, prev = we[i], we[i - 1]
        u = bd.universe(d)
        if len(u) < 20:
            continue
        r = (px.loc[d, u] / px.loc[prev, u] - 1.0).dropna().sort_values()
        rows[d] = _equal(list(r.index[:10]))
    return _frame(rows, bd.R.columns)


def high_52w(bd):
    px = bd.px
    hi = px.rolling(252, min_periods=200).max()

    def rule(d, u):
        ratio = (px.loc[d, u] / hi.loc[d, u]).dropna()
        return _equal(list(ratio.sort_values(ascending=False).index[:10])) if len(ratio) >= 20 else None
    return _monthly(bd, rule)


def max_avoid(bd):
    mx = bd.R.rolling(21, min_periods=15).max()

    def rule(d, u):
        m = mx.loc[d, u].dropna()
        if len(m) < 20:
            return None
        keep = m[m <= m.quantile(0.8)].index
        return _equal(list(keep))
    return _monthly(bd, rule)


def illiquid(bd):
    med = bd.turnover.rolling(63, min_periods=30).median()

    def rule(d, u):
        t = med.loc[d, u].dropna()
        return _equal(list(t.sort_values().index[:10])) if len(t) >= 20 else None
    return _monthly(bd, rule)


def trend_xu100(bd):
    lvl = (1.0 + bd.index_R.fillna(0.0)).cumprod()
    sma = lvl.rolling(200, min_periods=200).mean()

    def rule(d, u):
        if pd.isna(sma.get(d)):
            return None
        return _equal(u[:30]) if lvl[d] > sma[d] else {u[0]: 0.0}
    return _monthly(bd, rule)


def corr_regime(bd):
    R = bd.R

    def avg_corr(d, names):
        win = R.loc[:d, names].tail(60).dropna(axis=1, thresh=50)
        if win.shape[1] < 10:
            return np.nan
        c = win.corr().to_numpy()
        return float(c[np.triu_indices_from(c, 1)].mean())

    hist: dict = {}

    def rule(d, u):
        hist[d] = avg_corr(d, u[:30])
        past = pd.Series(hist).loc[:d].dropna()
        past = past[past.index > d - pd.DateOffset(months=12)]
        if len(past) < 12 or pd.isna(hist[d]):
            return None
        return {u[0]: 0.0} if hist[d] > past.quantile(0.8) else _equal(u[:30])
    return _monthly(bd, rule)


def _pairs(bd, short_leader: bool, max_pairs: int = 5, corr_min: float = 0.8, z_in: float = 2.0, z_out: float = 0.5, max_days: int = 20):
    """Correlation-breakdown pairs: highly correlated liquid stocks that diverge are expected to reconverge.

    Pairs are re-chosen monthly from the 30 most liquid (120-day return correlation > 0.8).
    Daily: spread = log price A - log price B; z-score over 60 days. Enter when |z| > 2:
    long the laggard (and, if `short_leader`, short the leader). Exit at |z| < 0.5 or after 20 days.
    """
    logp = np.log(bd.px)
    me = month_end_dates(bd.R.index)
    pairs_by_month = {}
    for d in me:
        u = [s for s in bd.universe(d)[:30]]
        if len(u) < 10:
            continue
        win = bd.R.loc[:d, u].tail(120).dropna(axis=1, thresh=100)
        c = win.corr()
        cand = [(c.iat[i, j], c.index[i], c.columns[j]) for i in range(len(c)) for j in range(i + 1, len(c)) if c.iat[i, j] > corr_min]
        pairs_by_month[d] = [(a, b) for _, a, b in sorted(cand, reverse=True)]

    open_pos: dict = {}   # (a, b) -> (long, short, entry_index)
    rows = {}
    keys = sorted(pairs_by_month)
    idx = bd.R.index
    k = 0
    current: list = []
    for t, d in enumerate(idx):
        while k < len(keys) and keys[k] <= d:
            current = pairs_by_month[keys[k]]
            k += 1
        for pair, (lng, sht, t0) in list(open_pos.items()):
            a, b = pair
            spread = (logp[a] - logp[b]).loc[:d].tail(60)
            z = (spread.iloc[-1] - spread.mean()) / spread.std() if spread.std() > 0 else 0.0
            if abs(z) < z_out or t - t0 >= max_days or pair not in current:
                del open_pos[pair]
        for a, b in current:
            if len(open_pos) >= max_pairs:
                break
            if (a, b) in open_pos:
                continue
            spread = (logp[a] - logp[b]).loc[:d].tail(60)
            if spread.notna().sum() < 50 or not spread.std() > 0:
                continue
            z = (spread.iloc[-1] - spread.mean()) / spread.std()
            if abs(z) > z_in:
                lng, sht = (b, a) if z > 0 else (a, b)   # z > 0: A ran ahead of B -> B is the laggard
                open_pos[(a, b)] = (lng, sht, t)
        w = {}
        for lng, sht, _ in open_pos.values():
            w[lng] = w.get(lng, 0.0) + 1.0 / max_pairs
            if short_leader:
                w[sht] = w.get(sht, 0.0) - 1.0 / max_pairs
        rows[d] = w or {bd.R.columns[0]: 0.0}
    return _changes_only(_frame(rows, bd.R.columns))


def pairs_long(bd):
    return _pairs(bd, short_leader=False)


def pairs_ls(bd):
    return _pairs(bd, short_leader=True)


BIST_LIBRARY: list[Spec] = [
    Spec("bist_mom_12_1", "momentum", "bist", mom_12_1, "Jegadeesh & Titman (1993); emerging-market evidence: Rouwenhorst (1999)", "winners over the last year (skipping the last month) keep outperforming"),
    Spec("bist_mom_ls30", "momentum", "bist", mom_ls30, "Jegadeesh & Titman (1993), long-short form", "momentum spread is the purer test: long winners, short losers among the most liquid"),
    Spec("bist_low_vol", "risk", "bist", low_vol, "Baker, Bradley & Wurgler (2011), Benchmarks as limits to arbitrage", "low-volatility stocks earn similar or higher returns with less risk"),
    Spec("bist_weekly_reversal", "mean_reversion", "bist", weekly_reversal, "Lehmann (1990); Jegadeesh (1990)", "last week's biggest losers rebound as liquidity-driven selling reverses"),
    Spec("bist_52w_high", "momentum", "bist", high_52w, "George & Hwang (2004), The 52-week high and momentum investing", "investors anchor on the 52-week high and under-react to news that pushes prices near it"),
    Spec("bist_max_avoid", "risk", "bist", max_avoid, "Bali, Cakici & Whitelaw (2011), Maxing out", "lottery-like stocks with extreme recent up-days are overpriced; avoiding them helps"),
    Spec("bist_illiquid", "liquidity", "bist", illiquid, "Amihud (2002), Illiquidity and stock returns", "less liquid stocks pay a premium for being harder to trade"),
    Spec("bist_trend_xu100", "trend", "bist", trend_xu100, "Faber (2007) applied to the BIST100", "exit Turkish equities when the index is below its 200-day average"),
    Spec("bist_corr_regime", "regime", "bist", corr_regime, "Pollet & Wilson (2010) find the OPPOSITE sign; tested as a risk-off rule", "when liquid stocks all move together, diversification has failed; step aside (expected sign contested)"),
    Spec("bist_pairs_long", "correlation_arbitrage", "bist", pairs_long, "Gatev, Goetzmann & Rouwenhorst (2006), Pairs trading", "correlated stocks that diverge tend to reconverge; long-only version buys the laggard"),
    Spec("bist_pairs_ls", "correlation_arbitrage", "bist", pairs_ls, "Gatev, Goetzmann & Rouwenhorst (2006), Pairs trading", "full pairs trade: long the laggard, short the leader, within the shortable top 30"),
]
