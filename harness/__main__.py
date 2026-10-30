"""Command line: python -m harness {pull,verify,status,ledger}"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone

from .config import SOURCES
from .ledger import TrialLedger
from .pull import pull_source
from .store import SnapshotStore


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def cmd_pull(args) -> int:
    store = SnapshotStore(args.data_root)
    snap = date.fromisoformat(args.snapshot_date) if args.snapshot_date else _today_utc()
    names = args.source or list(SOURCES)
    failed = False
    for name in names:
        res = pull_source(store, name, SOURCES[name], snap)
        if res.written:
            e = res.entry
            print(f"[ok]   {name}: {e['rows_pulled']} pulled, {e['rows_new']} new, {e['rows_revised']} revised, up to {e['max_date']}")
        else:
            failed = True
            print(f"[FAIL] {name}: " + "; ".join(res.errors))
        for w in res.warnings:
            print(f"       warning: {w}")
    return 1 if failed else 0


def cmd_verify(args) -> int:
    store = SnapshotStore(args.data_root)
    problems = [p for s in store.sources() for p in store.verify(s)]
    problems += [f"ledger: {p}" for p in TrialLedger(args.ledger).verify()]
    for p in problems:
        print(f"[FAIL] {p}")
    if not problems:
        print(f"[ok]   {len(store.sources())} sources and the trial ledger are intact")
    return 1 if problems else 0


def cmd_status(args) -> int:
    store = SnapshotStore(args.data_root)
    today = _today_utc()
    print(f"{'source':<10} {'snapshots':>9} {'rows':>8} {'last bar':>11} {'age':>5} {'revisions':>9}")
    for name in store.sources():
        entries = store.manifest(name)["snapshots"]
        df = store.load(name)
        last = df["date"].max()
        revisions = sum(e["rows_revised"] for e in entries)
        print(f"{name:<10} {len(entries):>9} {len(df):>8} {str(last):>11} {(today - last).days:>4}d {revisions:>9}")
    return 0


def cmd_ledger(args) -> int:
    led = TrialLedger(args.ledger)
    rows = led.rows()
    venues = sorted({r["venue"] for r in rows})
    print(f"trials: {len(rows)}")
    for v in venues:
        print(f"  {v}: {led.count(v)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--ledger", default="ledger/trials.jsonl")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pull", help="fetch sources and store today's snapshot")
    pp.add_argument("--source", action="append", choices=list(SOURCES))
    pp.add_argument("--snapshot-date", help="YYYY-MM-DD, default today (UTC)")
    pp.set_defaults(func=cmd_pull)
    sub.add_parser("verify", help="check hashes and the ledger chain").set_defaults(func=cmd_verify)
    sub.add_parser("status", help="summary per source").set_defaults(func=cmd_status)
    sub.add_parser("ledger", help="trial counts per venue").set_defaults(func=cmd_ledger)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
