"""Calibration strategy with no skill: random monthly in/out per asset.

Used only to prove the gate rejects luck. Logged under venue "calibration" so
it never inflates the trial count of a real venue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.returns import month_end_dates


def targets(returns: pd.DataFrame, seed: int, risky: list[str], p_in: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    me = month_end_dates(returns.index)
    rows = {}
    for d in me:
        live = [s for s in risky if pd.notna(returns.at[d, s])]
        if not live:
            continue
        on = [s for s in live if rng.random() < p_in]
        rows[d] = {s: (1.0 / len(live) if s in on else 0.0) for s in returns.columns}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=returns.columns).fillna(0.0)
