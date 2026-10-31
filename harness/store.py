"""Append-only snapshot store for daily price data.

Design
------
Every pull of a source is stored as a *delta*: only the rows that are new, or
whose values changed versus everything stored before. Reconstructing the data
"as known on day D" replays all deltas with snapshot id <= D, last value wins
per (symbol, date). This gives three properties the research harness needs:

* Reproducibility: a backtest names a snapshot id and always sees the same data.
* Revision tracking: when a vendor silently rewrites history (splits, fixes),
  the rewritten rows appear in a delta and are counted in the manifest.
* Small git history: a daily pull usually adds a handful of rows, not a full copy.

Files are never overwritten. Each delta's SHA-256 is recorded in manifest.json,
and `verify` recomputes them so tampering or corruption is detected.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["symbol", "date"]
SNAPSHOT_ID = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class StoreError(Exception):
    """Raised when a write or read would break the store's guarantees."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical frame: key columns first, date as python date, sorted."""
    missing = [c for c in KEY if c not in df.columns]
    if missing:
        raise StoreError(f"frame is missing key columns {missing}")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str)
    dt = pd.to_datetime(out["date"])
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    # Daily sources keep plain dates; intraday sources (any non-midnight bar) keep timestamps (UTC).
    out["date"] = dt.dt.date if (dt == dt.dt.normalize()).all() else dt
    if out.duplicated(KEY).any():
        dups = out[out.duplicated(KEY, keep=False)].head(5)
        raise StoreError(f"duplicate (symbol, date) rows, e.g.\n{dups}")
    value_cols = sorted(c for c in out.columns if c not in KEY)
    for c in value_cols:
        out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")
    return out[KEY + value_cols].sort_values(KEY).reset_index(drop=True)


def _changed_mask(new: pd.DataFrame, old: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Row-wise: does `new` differ from `old` in any column (NaN == NaN)?"""
    changed = np.zeros(len(new), dtype=bool)
    for c in cols:
        a = new[c].to_numpy(dtype="float64")
        b = old[c].to_numpy(dtype="float64")
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-10, atol=0.0)
        changed |= ~(both_nan | close)
    return changed


def _files(entry: dict) -> list[tuple[str, str]]:
    if "parts" in entry:
        return [(p["file"], p["sha256"]) for p in entry["parts"]]
    return [(entry["file"], entry["sha256"])]


class SnapshotStore:
    def __init__(self, root: str | os.PathLike, rows_per_part: int = 400_000):
        self.root = Path(root)
        self.rows_per_part = rows_per_part

    # ---- manifest -------------------------------------------------------
    def _dir(self, source: str) -> Path:
        if not re.fullmatch(r"[a-z0-9_]+", source):
            raise StoreError(f"invalid source name {source!r}")
        return self.root / source

    def manifest(self, source: str) -> dict:
        path = self._dir(source) / "manifest.json"
        if not path.exists():
            return {"source": source, "snapshots": []}
        return json.loads(path.read_text())

    def _write_manifest(self, source: str, manifest: dict) -> None:
        path = self._dir(source) / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)  # atomic on POSIX

    def snapshot_ids(self, source: str) -> list[str]:
        return [s["id"] for s in self.manifest(source)["snapshots"]]

    def sources(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "manifest.json").exists())

    # ---- read -----------------------------------------------------------
    def load(self, source: str, as_of: str | None = None) -> pd.DataFrame:
        """Data as known on snapshot `as_of` (default: latest). Verifies hashes."""
        entries = self.manifest(source)["snapshots"]
        if as_of is not None:
            if as_of not in [e["id"] for e in entries]:
                raise StoreError(f"{source}: no snapshot {as_of!r}")
            entries = [e for e in entries if e["id"] <= as_of]
        if not entries:
            raise StoreError(f"{source}: no snapshots stored")
        frames = []
        for e in entries:
            for f, h in _files(e):
                path = self._dir(source) / f
                if _sha256(path) != h:
                    raise StoreError(f"{source}: hash mismatch in {f} (file was altered)")
                frames.append(pd.read_parquet(path))
        frames = [f for f in frames if len(f)]
        if not frames:
            raise StoreError(f"{source}: snapshots contain no rows")
        state = pd.concat(frames, ignore_index=True)
        state = state.drop_duplicates(KEY, keep="last")  # later delta wins
        return normalize(state)

    def _latest_state(self, source: str) -> pd.DataFrame | None:
        if not self.snapshot_ids(source):
            return None
        try:
            return self.load(source)
        except StoreError as exc:
            if "no rows" in str(exc):
                return None
            raise

    # ---- write ----------------------------------------------------------
    def write(self, source: str, df: pd.DataFrame, snapshot_id: str, notes: list[str] | None = None) -> dict:
        """Store the delta between `df` and the current state as snapshot `snapshot_id`."""
        if not SNAPSHOT_ID.match(snapshot_id):
            raise StoreError(f"snapshot id must be YYYY-MM-DD, got {snapshot_id!r}")
        ids = self.snapshot_ids(source)
        if snapshot_id in ids:
            raise StoreError(f"{source}: snapshot {snapshot_id} already exists; snapshots are never overwritten")
        if ids and snapshot_id < ids[-1]:
            raise StoreError(f"{source}: snapshot {snapshot_id} is older than latest {ids[-1]}")

        new = normalize(df)
        state = self._latest_state(source)
        if state is None:
            delta, n_new, n_revised = new, len(new), 0
        else:
            if set(state.columns) != set(new.columns):
                raise StoreError(
                    f"{source}: column set changed {sorted(state.columns)} -> {sorted(new.columns)}"
                )
            merged = new.merge(state, on=KEY, how="left", suffixes=("", "__old"), indicator=True)
            is_new = (merged["_merge"] == "left_only").to_numpy()
            value_cols = [c for c in new.columns if c not in KEY]
            old_vals = merged[[f"{c}__old" for c in value_cols]].set_axis(value_cols, axis=1)
            is_revised = ~is_new & _changed_mask(merged[value_cols], old_vals, value_cols)
            delta = new[is_new | is_revised].reset_index(drop=True)
            n_new, n_revised = int(is_new.sum()), int(is_revised.sum())

        d = self._dir(source)
        d.mkdir(parents=True, exist_ok=True)
        # Large deltas (a first pull of hundreds of stocks) are split by symbol so no
        # single file approaches GitHub's 100 MB per-file limit.
        symbols = sorted(delta["symbol"].unique()) if len(delta) else []
        n_parts = max(1, -(-len(delta) // self.rows_per_part))
        groups = [symbols[i::n_parts] for i in range(n_parts)] if n_parts > 1 else [None]
        parts = []
        for i, g in enumerate(groups):
            fname = f"{snapshot_id}.parquet" if n_parts == 1 else f"{snapshot_id}-part{i:02d}.parquet"
            path = d / fname
            if path.exists():
                raise StoreError(f"{path} exists but is not in the manifest; refusing to overwrite")
            chunk = delta if g is None else delta[delta["symbol"].isin(g)]
            chunk.to_parquet(path, index=False, compression="zstd")
            parts.append({"file": fname, "sha256": _sha256(path)})

        entry = {
            "id": snapshot_id,
            **({"file": parts[0]["file"], "sha256": parts[0]["sha256"]} if n_parts == 1 else {"parts": parts}),
            "rows_pulled": int(len(new)),
            "rows_new": n_new,
            "rows_revised": n_revised,
            "min_date": str(new["date"].min()) if len(new) else None,
            "max_date": str(new["date"].max()) if len(new) else None,
            "symbols": sorted(new["symbol"].unique().tolist()),
            "notes": notes or [],
        }
        manifest = self.manifest(source)
        manifest["snapshots"].append(entry)
        self._write_manifest(source, manifest)
        return entry

    # ---- integrity ------------------------------------------------------
    def verify(self, source: str) -> list[str]:
        """Return a list of integrity problems (empty list = healthy)."""
        problems = []
        d = self._dir(source)
        entries = self.manifest(source)["snapshots"]
        listed = {f for e in entries for f, _ in _files(e)}
        ids = [e["id"] for e in entries]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            problems.append(f"{source}: snapshot ids are not strictly increasing")
        for e in entries:
            for f, h in _files(e):
                path = d / f
                if not path.exists():
                    problems.append(f"{source}: missing file {f}")
                elif _sha256(path) != h:
                    problems.append(f"{source}: hash mismatch in {f}")
        for p in d.glob("*.parquet"):
            if p.name not in listed:
                problems.append(f"{source}: unlisted file {p.name}")
        return problems
