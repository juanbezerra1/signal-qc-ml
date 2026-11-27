from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from signal_qc_ml import columns as cols
from signal_qc_ml import extract
from signal_qc_ml.data import read_table

from .helpers import make_table, snapshot_path


def test_every_column_of_the_table_is_made_by_a_query() -> None:
    recordings = extract.query_text("recordings.sql")
    for name in (*cols.required_columns(),):
        assert re.search(rf"\b{name}\b", recordings), f"{name} is not in recordings.sql"


def test_the_query_names_the_columns_in_the_order_of_the_groups() -> None:
    select = extract.query_text("recordings.sql")
    order = [select.index(re.search(rf"\b(AS )?{n}\b", select).group(0)) for n in ("recording_id", "qc")]  # type: ignore[union-attr]
    assert order == sorted(order)


def test_the_signals_are_summarised_with_exact_arithmetic() -> None:
    text = extract.query_text("signals.sql")
    assert "::numeric" in text and "round(" in text  # float sums depend on the order of the rows


def test_neither_query_reads_the_review_into_a_known_column() -> None:
    text = extract.query_text("recordings.sql")
    assert "known before the review" in text and "produced by the review" in text


def test_the_database_comes_from_the_argument_then_the_environment_then_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LABTRACK_DSN", raising=False)
    assert extract.get_dsn() == extract.DEFAULT_DSN
    monkeypatch.setenv("LABTRACK_DSN", "postgresql://env")
    assert extract.get_dsn() == "postgresql://env"
    assert extract.get_dsn("postgresql://arg") == "postgresql://arg"


def test_a_missing_driver_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(extract.ExtractError, match="needs psycopg"):
        extract.fetch("postgresql://x")


def test_a_database_that_is_not_there_is_an_error_that_says_so() -> None:
    pytest.importorskip("psycopg")
    with pytest.raises(extract.ExtractError, match="could not connect to the database"):
        extract.fetch("postgresql://nobody:none@127.0.0.1:1/none?connect_timeout=2")


def recordings_and_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    recordings = pd.DataFrame(
        {
            "recording_id": [3, 1, 2],
            "recorded_on": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]).date,
            "duration_s": [10.123456789, 20.0, 30.5],
            "qc": ["passed", "flagged", "pending"],
        }
    )
    signals = pd.DataFrame(
        {
            "recording_id": [1, 1, 2, 3],
            "feature": ["rms", "mnf", "rms", "mnf"],
            "mean": [1.0, 2.0, 3.0, 4.0],
            "sd": [0.1, 0.2, 0.3, 0.4],
        }
    )
    return recordings, signals


def test_the_two_queries_become_one_wide_table() -> None:
    table = extract.assemble(*recordings_and_signals())
    assert table["recording_id"].tolist() == [1, 2, 3]  # sorted: the same order on every machine
    assert [c for c in table.columns if c.startswith("sig_")] == [
        "sig_mnf_mean",
        "sig_mnf_sd",
        "sig_rms_mean",
        "sig_rms_sd",
    ]
    assert table.loc[table["recording_id"] == 2, "sig_mnf_mean"].isna().all()  # a recording without that feature
    assert table.loc[table["recording_id"] == 1, "sig_rms_sd"].item() == 0.1
    assert table["recorded_on"].tolist() == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert table["duration_s"].tolist() == [20.0, 30.5, 10.123457]  # six decimals, as in the files


def test_the_hash_is_the_one_of_the_text_whether_the_file_is_gzipped_or_not(tmp_path: Path) -> None:
    table = extract.assemble(*recordings_and_signals())
    plain = extract.write_table(table, tmp_path / "t.csv")
    packed = extract.write_table(table, tmp_path / "t.csv.gz")
    assert plain == packed == hashlib.sha256((tmp_path / "t.csv").read_bytes()).hexdigest()
    assert extract.sha256_of_text(tmp_path / "t.csv.gz") == plain
    assert gzip.decompress((tmp_path / "t.csv.gz").read_bytes()) == (tmp_path / "t.csv").read_bytes()


def test_the_text_has_unix_line_endings_and_no_index(tmp_path: Path) -> None:
    extract.write_table(extract.assemble(*recordings_and_signals()), tmp_path / "t.csv")
    raw = (tmp_path / "t.csv").read_bytes()
    assert b"\r" not in raw and raw.startswith(b"recording_id,recorded_on,")


def test_the_provenance_file_has_the_name_of_the_table_without_its_endings() -> None:
    assert extract.meta_path("data/recordings-small.csv") == Path("data/recordings-small.meta.json")
    assert extract.meta_path("data/recordings-large.csv.gz") == Path("data/recordings-large.meta.json")
    assert extract.meta_path("x.csv").name == "x.meta.json"


def test_a_table_is_checked_against_its_provenance(tmp_path: Path) -> None:
    table = make_table(participants=5)
    path = tmp_path / "t.csv"
    assert extract.verify(path) is None  # no file at all: nothing to compare with
    sha = extract.write_table(table, path)
    assert extract.verify(path) is None  # a table with no provenance file
    meta = extract.provenance(table, path, sha, {"latest_migration": 13})
    extract.meta_path(path).write_text(json.dumps(meta), encoding="utf-8")
    assert extract.verify(path) is True
    path.write_text(path.read_text(encoding="utf-8").replace("S-01", "S-99", 1), encoding="utf-8", newline="\n")
    assert extract.verify(path) is False


def test_the_provenance_counts_what_is_in_the_table() -> None:
    table = make_table(participants=10, per_participant=8, pending=4)
    meta = extract.provenance(table, Path("t.csv"), "abc", {"latest_migration": 13})
    assert meta["table"] == {
        "file": "t.csv", "sha256": "abc", "rows": 80, "reviewed": 76, "pending": 4, "participants": 10,
        "columns": len(table.columns),
    }  # fmt: skip
    assert meta["source"] == {"labtrack_db_latest_migration": 13}
    assert set(meta["queries"]) == {"recordings.sql", "signals.sql"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in meta["queries"].values())


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_the_tables_that_ship_are_the_ones_that_their_provenance_describes(size: str) -> None:
    path = snapshot_path(size)
    assert extract.verify(path) is True
    meta = json.loads(extract.meta_path(path).read_text(encoding="utf-8"))
    frame = read_table(path)
    assert meta["table"]["rows"] == len(frame)
    assert meta["table"]["participants"] == frame["participant_id"].nunique()
    assert meta["table"]["reviewed"] == int((frame["qc"] != "pending").sum())
    assert meta["table"]["file"] == path.name
    queries = {name: hashlib.sha256(extract.query_text(name).encode("utf-8")).hexdigest() for name in meta["queries"]}
    assert queries == meta["queries"], "the queries changed after the tables were extracted: extract them again"


@pytest.mark.skipif(not os.environ.get("SIGNAL_QC_ML_TEST_DSN"), reason="needs a labtrack-db database (small seed)")
def test_extracting_again_from_the_database_gives_the_same_table(tmp_path: Path) -> None:
    meta = extract.snapshot(os.environ["SIGNAL_QC_ML_TEST_DSN"], tmp_path / "again.csv")
    shipped = json.loads(extract.meta_path(snapshot_path("small")).read_text(encoding="utf-8"))
    assert meta["table"]["sha256"] == shipped["table"]["sha256"]
    assert extract.sha256_of_text(tmp_path / "again.csv") == extract.sha256_of_text(snapshot_path("small"))
