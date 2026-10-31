"""Append-only, hash-chained trial ledger.

Every backtest the harness runs -- pass or fail -- is one ledger row. The row
count per venue is the trial count N used by the Deflated Sharpe Ratio, so the
ledger must be impossible to quietly shrink. Each row stores the SHA-256 of the
previous row; deleting or editing any row breaks the chain and `verify` reports it.

There is deliberately no delete or update function.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64
REQUIRED = ("spec_id", "venue", "snapshot_id", "params", "metrics", "verdict")
VERDICTS = ("pass", "fail", "error")


class LedgerError(Exception):
    pass


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _row_hash(row: dict) -> str:
    body = {k: v for k, v in row.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


class TrialLedger:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, record: dict, now: datetime | None = None) -> dict:
        missing = [k for k in REQUIRED if k not in record]
        if missing:
            raise LedgerError(f"trial record is missing {missing}")
        if record["verdict"] not in VERDICTS:
            raise LedgerError(f"verdict must be one of {VERDICTS}")
        problems = self.verify()
        if problems:
            raise LedgerError("ledger chain is broken, refusing to append: " + "; ".join(problems))
        rows = self.rows()
        row = dict(record)
        row["trial_no"] = len(rows) + 1
        row["recorded_at"] = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        row["prev_hash"] = rows[-1]["hash"] if rows else GENESIS
        try:
            row["hash"] = _row_hash(row)
        except ValueError as exc:  # NaN/inf in metrics
            raise LedgerError(f"record is not valid JSON (NaN or inf?): {exc}") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(_canonical(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    def verify(self) -> list[str]:
        problems, prev = [], GENESIS
        for i, row in enumerate(self.rows(), start=1):
            if row.get("trial_no") != i:
                problems.append(f"row {i}: trial_no is {row.get('trial_no')} (rows removed or reordered)")
            if row.get("prev_hash") != prev:
                problems.append(f"row {i}: prev_hash does not match row {i - 1}")
            if row.get("hash") != _row_hash(row):
                problems.append(f"row {i}: content was edited after it was written")
            prev = row.get("hash")
        return problems

    def count(self, venue: str | None = None) -> int:
        """Trial count N, optionally for one venue. Failed trials count too.

        A row that tested a family of variants counts as `n_variants` trials,
        because each variant was a chance to get lucky.
        """
        return sum(int(r.get("n_variants", 1)) for r in self.rows() if venue is None or r["venue"] == venue)

    def sr_batches(self, venue: str) -> list[tuple[int, float, float]]:
        """(n, mean, variance) of per-period Sharpes for each logged batch, for pooled DSR variance."""
        out = []
        for r in self.rows():
            m = r.get("metrics", {})
            if r["venue"] == venue and "variant_sr_mean" in m:
                out.append((int(r.get("n_variants", 1)), float(m["variant_sr_mean"]), float(m.get("variant_sr_var", 0.0))))
        return out
