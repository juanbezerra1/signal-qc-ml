from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from signal_qc_ml import __version__, cli

from .helpers import make_table, snapshot_path, write_csv

QUICK = ["--folds", "3", "--repeats", "1", "--bootstrap", "30", "--permutations", "10"]


def run(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, str, str]:
    code = cli.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.strip() == f"signal-qc-ml {__version__}"


def test_a_command_is_needed() -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main([])
    assert stop.value.code == 2


def test_check_counts_the_rows_and_compares_with_the_provenance(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "check", snapshot_path("small"))
    assert code == 0
    assert "460 recordings of 59 participants" in out and "SHA-256 matches recordings-small.meta.json" in out


def test_check_says_when_a_table_is_not_the_one_that_was_extracted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = tmp_path / "recordings-small.csv"
    copy.write_bytes(snapshot_path("small").read_bytes().replace(b"FFB-01", b"FFB-02", 1))
    (tmp_path / "recordings-small.meta.json").write_bytes(
        (snapshot_path("small").parent / "recordings-small.meta.json").read_bytes()
    )
    code, _, err = run(capsys, "check", copy)
    assert code == 1 and "SHA-256 differs" in err


def test_check_on_a_table_with_no_provenance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(make_table(participants=10), tmp_path / "mine.csv")
    code, out, _ = run(capsys, "check", path)
    assert code == 0 and "no mine.meta.json next to it" in out


def test_a_file_that_is_not_there_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, "check", tmp_path / "none.csv")
    assert code == 2 and out == "" and "error: there is no file" in err


def test_evaluate_prints_the_tables_and_writes_the_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(make_table(participants=40, seed=1), tmp_path / "t.csv")
    code, out, err = run(capsys, "evaluate", path, "--out", tmp_path / "out" / "r.json", "--skip-audit", *QUICK)
    assert code == 0, err
    assert "| nothing (the share seen in training) | prior | 0 | 0.500 |" in out
    assert "Mean when results are moved" in out and "skipped (--skip-audit)" in out
    results = json.loads((tmp_path / "out" / "r.json").read_text(encoding="utf-8"))
    assert results["audit"] == {"skipped": True}
    assert results["design"] == {"folds": 3, "repeats": 1, "seed": 0, "bootstrap": 30, "permutations": 10}
    assert results["table"]["recordings"] == 320
    assert results["table"]["has_provenance"] is False
    assert {"python", "numpy", "pandas", "scikit-learn"} <= set(results["environment"])


def test_evaluate_audits_the_splits_unless_told_not_to(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("table_audit")
    path = write_csv(make_table(participants=40, seed=1), tmp_path / "t.csv")
    out_file = tmp_path / "r.json"
    code, _, _ = run(capsys, "evaluate", path, "--out", out_file, *QUICK)
    assert code == 0
    audit = json.loads(out_file.read_text(encoding="utf-8"))["audit"]
    assert audit["splits"] == 3 and audit["splits_with_errors"] == 0 and audit["tool"].startswith("table-audit")


def test_the_controls_need_the_primary_test(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(make_table(participants=30), tmp_path / "t.csv")
    code, _, err = run(capsys, "evaluate", path, "--controls", "--skip-audit", "--permutations", "0", "--repeats", "1")
    assert code == 2 and "need permutations" in err


def test_the_controls_are_in_the_results_when_asked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(make_table(participants=40, seed=1), tmp_path / "t.csv")
    out_file = tmp_path / "r.json"
    code, out, err = run(
        capsys,
        "evaluate",
        path,
        "--out",
        out_file,
        "--controls",
        "--skip-audit",
        "--replicates",
        "2",
        "--simulations",
        "2",
        *QUICK,
    )
    assert code == 0, err
    controls = json.loads(out_file.read_text(encoding="utf-8"))["controls"]
    assert len(controls["power"]["points"]) == 8 and controls["identity_leak"]["simulations"] >= 1
    assert "Planted shift" in out and "At random (ignores participants)" in out


def test_a_table_that_changed_after_it_was_extracted_is_not_evaluated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = tmp_path / "recordings-small.csv"
    copy.write_bytes(snapshot_path("small").read_bytes().replace(b"FFB-01", b"FFB-02", 1))
    (tmp_path / "recordings-small.meta.json").write_bytes(
        (snapshot_path("small").parent / "recordings-small.meta.json").read_bytes()
    )
    code, _, err = run(capsys, "evaluate", copy, "--skip-audit", *QUICK)
    assert code == 2 and "was changed after it was extracted" in err


def test_rank_refuses_when_nothing_is_there(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(make_table(participants=60, pending=6, seed=1), tmp_path / "t.csv")
    code, out, err = run(
        capsys, "rank", path, "--out", tmp_path / "ranking.csv", "--folds", "4", "--repeats", "2", "--bootstrap", "100"
    )
    assert code == 1 and out == ""
    assert "refused: the logistic model is not better than chance" in err and "nothing was ranked" in err
    assert not (tmp_path / "ranking.csv").exists()


def test_rank_does_not_offer_an_option_that_it_would_ignore(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(["rank", str(snapshot_path("small")), "--out", str(tmp_path / "r.csv"), "--permutations", "10"])
    assert stop.value.code == 2


def test_rank_writes_the_order_when_there_is_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(make_table(participants=60, pending=6, effect=2.0, seed=1), tmp_path / "t.csv")
    code, out, err = run(
        capsys,
        "rank",
        path,
        "--out",
        tmp_path / "r" / "ranking.csv",
        "--folds",
        "4",
        "--repeats",
        "2",
        "--bootstrap",
        "100",
    )
    assert code == 0, err
    ranking = pd.read_csv(tmp_path / "r" / "ranking.csv")
    assert list(ranking.columns) == ["rank", "recording_id", "participant_code", "score"] and len(ranking) == 6
    assert "6 recordings ranked by the logistic model" in out and "participants never seen" in out
    assert b"\r" not in (tmp_path / "r" / "ranking.csv").read_bytes()


def test_report_prints_sections_of_a_results_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(make_table(participants=40, seed=1), tmp_path / "t.csv")
    run(capsys, "evaluate", path, "--out", tmp_path / "r.json", "--skip-audit", *QUICK)
    code, out, _ = run(capsys, "report", tmp_path / "r.json", "--section", "primary")
    assert code == 0 and out.startswith("| AUC of the first pass |") and "prior" not in out
    code, out, _ = run(capsys, "report", tmp_path / "r.json", "--lang", "pt", "--section", "primary")
    assert code == 0 and "AUC da primeira passada" in out


def test_report_of_something_that_is_not_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "x.json").write_text("{not json", encoding="utf-8")
    code, _, err = run(capsys, "report", tmp_path / "x.json")
    assert code == 2 and "is not a results file" in err
    code, _, err = run(capsys, "report", tmp_path / "missing.json")
    assert code == 2 and "there is no file" in err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--folds", "1"], "--folds must be at least 2"),
        (["--repeats", "0"], "--repeats must be at least 1"),
        (["--bootstrap", "0"], "--bootstrap must be at least 1"),
        (["--permutations", "-1"], "cannot be negative"),
        (["--jobs", "0"], "--jobs must be at least 1"),
    ],
)
def test_the_design_is_checked_before_anything_runs(tmp_path: Path, argv: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        cli.main(["evaluate", str(snapshot_path("small")), "--skip-audit", *argv])


def test_what_the_command_prints_is_plain_ascii(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_csv(make_table(participants=40, seed=1), tmp_path / "t.csv")
    code, out, err = run(capsys, "evaluate", path, "--skip-audit", *QUICK)
    assert code == 0
    for text in (out, err):
        text.encode("ascii")  # a Windows console in a legacy code page must be able to show all of it
    _, out, _ = run(capsys, "report", snapshot_path("small").parent.parent / "results" / "small.json", "--lang", "pt")
    out.encode("ascii")
