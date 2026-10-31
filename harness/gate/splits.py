"""Out-of-sample splitting: walk-forward and purged/embargoed K-fold.

Both work on a family of variant return series (columns) that were computed
causally over the full sample, and both *select* a variant on training data
before scoring it on unseen data. Selection is the step that overfits, so it
must happen inside every split, never once up front.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.metrics import sharpe


def _best(train: pd.DataFrame) -> str:
    scores = train.apply(lambda s: sharpe(s.to_numpy()))
    if scores.isna().all():
        return train.columns[0]
    return scores.idxmax()


def walk_forward(variants: pd.DataFrame, train_years: float = 3.0, test_months: int = 6) -> tuple[pd.Series, list[dict]]:
    """Rolling train/test: pick the best variant on each train window, hold it for the test window.

    Returns the stitched out-of-sample return series and the selection log.
    """
    idx = variants.index
    start = idx[0]
    oos, log = [], []
    while True:
        train_end = start + pd.DateOffset(months=int(round(train_years * 12)))
        test_end = train_end + pd.DateOffset(months=test_months)
        if train_end >= idx[-1]:
            break
        train = variants.loc[(idx >= start) & (idx < train_end)]
        test = variants.loc[(idx >= train_end) & (idx < test_end)]
        if len(test) == 0:
            break
        pick = _best(train)
        oos.append(test[pick])
        log.append({"train_start": str(start.date()), "test_start": str(train_end.date()), "pick": str(pick)})
        start = start + pd.DateOffset(months=test_months)
    if not oos:
        raise ValueError("sample too short for one walk-forward step")
    return pd.concat(oos), log


def purged_kfold(n: int, k: int = 5, purge: int = 21, embargo: int = 21):
    """Yield (train_idx, test_idx) for contiguous folds.

    Drops `purge` observations before each test fold (their holding periods would
    overlap the fold) and `embargo` observations after it (serial correlation leak).
    """
    if k < 2 or n < k:
        raise ValueError("need k >= 2 and n >= k")
    edges = np.linspace(0, n, k + 1).astype(int)
    all_idx = np.arange(n)
    for a, b in zip(edges[:-1], edges[1:]):
        keep = (all_idx < a - purge) | (all_idx >= b + embargo)
        yield all_idx[keep], all_idx[a:b]


def cv_oos(variants: pd.DataFrame, k: int = 5, purge: int = 21, embargo: int = 21) -> pd.Series:
    """Concatenated out-of-fold returns, selecting the best variant on each purged training set."""
    out = []
    for tr, te in purged_kfold(len(variants), k, purge, embargo):
        pick = _best(variants.iloc[tr])
        out.append(variants.iloc[te][pick])
    return pd.concat(out)
