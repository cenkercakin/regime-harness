import numpy as np
import pandas as pd
import pytest

from harness.store import SnapshotStore, StoreError


def test_first_write_stores_everything(tmp_path, bars):
    st = SnapshotStore(tmp_path)
    df = bars(["A", "B"], "2026-01-01", 30)
    e = st.write("src", df, "2026-02-01")
    assert e["rows_new"] == 60 and e["rows_revised"] == 0
    assert len(st.load("src")) == 60


def test_identical_repull_writes_empty_delta(tmp_path, bars):
    st = SnapshotStore(tmp_path)
    df = bars(["A"], "2026-01-01", 30)
    st.write("src", df, "2026-02-01")
    e = st.write("src", df, "2026-02-02")
    assert e["rows_new"] == 0 and e["rows_revised"] == 0
    assert len(pd.read_parquet(tmp_path / "src" / "2026-02-02.parquet")) == 0
    assert len(st.load("src")) == 30


def test_new_and_revised_rows_and_point_in_time_load(tmp_path, bars):
    st = SnapshotStore(tmp_path)
    day1 = bars(["A"], "2026-01-01", 30)
    st.write("src", day1, "2026-02-01")

    day2 = bars(["A"], "2026-01-01", 31)          # one new bar
    day2.loc[5, "close"] = day2.loc[5, "close"] * 1.10  # vendor revises one old bar
    e = st.write("src", day2, "2026-02-02")
    assert e["rows_new"] == 1 and e["rows_revised"] == 1

    as_of_1 = st.load("src", as_of="2026-02-01")
    pd.testing.assert_frame_equal(as_of_1, st.load("src", as_of="2026-02-01"))
    assert len(as_of_1) == 30
    assert as_of_1.loc[5, "close"] == pytest.approx(day1.loc[5, "close"])
    latest = st.load("src")
    assert len(latest) == 31
    assert latest.loc[5, "close"] == pytest.approx(day2.loc[5, "close"])


def test_rows_missing_from_later_pull_are_kept(tmp_path, bars):
    """A delisted symbol vanishing from the vendor must not vanish from our history."""
    st = SnapshotStore(tmp_path)
    st.write("src", bars(["A", "DEAD"], "2026-01-01", 10), "2026-02-01")
    st.write("src", bars(["A"], "2026-01-01", 11), "2026-02-02")
    assert "DEAD" in set(st.load("src")["symbol"])


def test_nan_values_are_not_false_revisions(tmp_path, bars):
    st = SnapshotStore(tmp_path)
    df = bars(["A"], "2026-01-01", 5)
    df.loc[2, "volume"] = np.nan
    st.write("src", df, "2026-02-01")
    assert st.write("src", df, "2026-02-02")["rows_revised"] == 0


@pytest.mark.parametrize("bad", ["2026-02-01", "2026-01-15", "20260202", "../x"])
def test_write_guards(tmp_path, bars, bad):
    st = SnapshotStore(tmp_path)
    st.write("src", bars(["A"], "2026-01-01", 5), "2026-02-01")
    with pytest.raises(StoreError):
        st.write("src", bars(["A"], "2026-01-01", 5), bad)


def test_bad_source_name_and_duplicates_refused(tmp_path, bars):
    st = SnapshotStore(tmp_path)
    with pytest.raises(StoreError):
        st.write("../etc", bars(["A"], "2026-01-01", 5), "2026-02-01")
    df = bars(["A"], "2026-01-01", 5)
    with pytest.raises(StoreError):
        st.write("src", pd.concat([df, df.head(1)]), "2026-02-01")


def test_column_change_refused(tmp_path, bars):
    st = SnapshotStore(tmp_path)
    st.write("src", bars(["A"], "2026-01-01", 5), "2026-02-01")
    with pytest.raises(StoreError):
        st.write("src", bars(["A"], "2026-01-01", 5).drop(columns="volume"), "2026-02-02")


def test_tampering_is_detected(tmp_path, bars):
    st = SnapshotStore(tmp_path)
    st.write("src", bars(["A"], "2026-01-01", 5), "2026-02-01")
    path = tmp_path / "src" / "2026-02-01.parquet"
    df = pd.read_parquet(path)
    df.loc[0, "close"] = 1.0
    df.to_parquet(path, index=False)
    assert any("hash mismatch" in p for p in st.verify("src"))
    with pytest.raises(StoreError, match="hash mismatch"):
        st.load("src")


def test_unlisted_file_is_reported(tmp_path, bars):
    st = SnapshotStore(tmp_path)
    st.write("src", bars(["A"], "2026-01-01", 5), "2026-02-01")
    bars(["A"], "2026-01-01", 5).to_parquet(tmp_path / "src" / "2026-03-01.parquet")
    assert any("unlisted" in p for p in st.verify("src"))
    assert st.verify("src") and SnapshotStore(tmp_path).sources() == ["src"]


def test_large_delta_is_split_into_parts_and_reloads(tmp_path, bars):
    st = SnapshotStore(tmp_path, rows_per_part=50)
    df = bars([f"S{i}" for i in range(8)], "2026-01-01", 30)   # 240 rows -> 5 parts
    e = st.write("src", df, "2026-02-01")
    assert len(e["parts"]) == 5
    pd.testing.assert_frame_equal(st.load("src").reset_index(drop=True), st.load("src", as_of="2026-02-01"))
    assert len(st.load("src")) == 240 and st.verify("src") == []
    # a later small delta goes back to a single file, and both load together
    df2 = bars([f"S{i}" for i in range(8)], "2026-01-01", 31)
    e2 = st.write("src", df2, "2026-02-02")
    assert "file" in e2 and len(st.load("src")) == 248


def test_intraday_timestamps_round_trip(tmp_path):
    idx = pd.date_range("2026-10-01", periods=600, freq="5min")
    df = pd.DataFrame({"symbol": "BTC/USD", "date": idx, "open": 1.0, "high": 2.0, "low": 0.5, "close": np.linspace(1, 2, 600), "volume": 1.0})
    st = SnapshotStore(tmp_path)
    st.write("m5", df, "2026-10-04")
    out = st.load("m5")
    assert len(out) == 600 and pd.api.types.is_datetime64_any_dtype(out["date"])
    more = pd.DataFrame({"symbol": "BTC/USD", "date": pd.date_range("2026-10-01", periods=700, freq="5min"), "open": 1.0, "high": 2.0, "low": 0.5,
                         "close": np.r_[np.linspace(1, 2, 600), np.ones(100)], "volume": 1.0})
    e = st.write("m5", more, "2026-10-05")
    assert e["rows_new"] == 100 and e["rows_revised"] == 0
    assert e["max_date"].startswith("2026-10-03")
