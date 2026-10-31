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
        if res.skipped:
            print(f"[skip] {name}: snapshot {snap.isoformat()} already exists")
            continue
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


def cmd_research(args) -> int:
    from . import research

    store, led = SnapshotStore(args.data_root), TrialLedger(args.ledger)
    if args.strategy == "library":
        from . import phase3
        import os
        res = phase3.run(store, led, args.as_of, gate_library=not args.selectors_only, selector_version=args.selector_version)
        os.makedirs("reports", exist_ok=True)
        path = f"reports/phase3-{res.snapshot}-{args.selector_version}.md"
        open(path, "w").write(phase3.to_markdown(res))
        print(f"report written to {path}")
        return 0
    if args.strategy == "bist":
        from . import phase5
        import os
        out = phase5.run(store, led, args.as_of)
        os.makedirs("reports", exist_ok=True)
        path = f"reports/phase5-bist-{out['notes']['snapshot']}.md"
        open(path, "w").write(phase5.to_markdown(out))
        print(f"report written to {path}")
        return 0
    if args.strategy == "tsmom":
        rep = research.run_tsmom(store, led, args.as_of, vault=args.vault)
    else:
        if args.vault:
            print("calibration runs never use the vault")
            return 2
        rep = research.run_calibration(store, led, args.as_of)
    print(rep.table())
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
    rp = sub.add_parser("research", help="run a strategy family through the validation gate (logs to the ledger)")
    rp.add_argument("strategy", choices=["tsmom", "calibration", "library", "bist"])
    rp.add_argument("--selectors-only", action="store_true", help="library: skip re-gating the strategies (avoids double-counting trials)")
    rp.add_argument("--selector-version", default="v2")
    rp.add_argument("--as-of", help="snapshot id to use, default latest")
    rp.add_argument("--vault", action="store_true", help="one-time test on the locked last 18 months")
    rp.set_defaults(func=cmd_research)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
