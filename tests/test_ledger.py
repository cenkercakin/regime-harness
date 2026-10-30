import json

import pytest

from harness.ledger import LedgerError, TrialLedger


def rec(venue="crypto", verdict="fail", **kw):
    r = {"spec_id": "tsmom-12m", "venue": venue, "snapshot_id": "2026-10-31", "params": {"lookback": 252}, "metrics": {"sharpe": 0.4}, "verdict": verdict}
    r.update(kw)
    return r


def test_append_and_count(tmp_path):
    led = TrialLedger(tmp_path / "t.jsonl")
    led.append(rec())
    led.append(rec(verdict="pass"))
    led.append(rec(venue="us_etfs"))
    assert led.count() == 3 and led.count("crypto") == 2 and led.count("us_etfs") == 1
    assert led.verify() == []
    assert [r["trial_no"] for r in led.rows()] == [1, 2, 3]


def test_edit_is_detected(tmp_path):
    p = tmp_path / "t.jsonl"
    led = TrialLedger(p)
    led.append(rec())
    led.append(rec())
    lines = p.read_text().splitlines()
    row = json.loads(lines[0])
    row["metrics"]["sharpe"] = 2.5  # someone "improves" a result
    lines[0] = json.dumps(row)
    p.write_text("\n".join(lines) + "\n")
    assert any("edited" in x for x in led.verify())
    with pytest.raises(LedgerError):
        led.append(rec())


def test_deleting_a_failed_trial_is_detected(tmp_path):
    p = tmp_path / "t.jsonl"
    led = TrialLedger(p)
    for _ in range(3):
        led.append(rec())
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")
    problems = led.verify()
    assert problems and any("prev_hash" in x or "trial_no" in x for x in problems)


def test_bad_records_refused(tmp_path):
    led = TrialLedger(tmp_path / "t.jsonl")
    with pytest.raises(LedgerError):
        led.append({"spec_id": "x"})
    with pytest.raises(LedgerError):
        led.append(rec(verdict="maybe"))
    with pytest.raises(LedgerError):
        led.append(rec(metrics={"sharpe": float("nan")}))
    assert led.count() == 0
