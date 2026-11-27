"""The results that ship with the project are results of the tables that ship with it, and say what the README says."""

from __future__ import annotations

import json
from typing import Any

import pytest

from signal_qc_ml import controls, report
from signal_qc_ml.data import counts, read_table
from signal_qc_ml.evaluate import BOTH_NAME, REVIEW_ONLY_NAME, Design
from signal_qc_ml.extract import meta_path, sha256_of_text
from signal_qc_ml.stats import theory_auc

from .helpers import RESULTS, SIZES, snapshot_path

# How far the measured AUC of a planted column may be from the theory, by table size: the smaller the table, the more
# a mean of 20 tables moves by chance (and a column with a weak effect gets the sign of its coefficient wrong now and then).
TOLERANCE = {"small": 0.06, "medium": 0.03, "large": 0.02}


def results_of(size: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"{size}.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(params=SIZES)
def size(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_there_is_a_results_file_for_every_table_and_no_other() -> None:
    assert sorted(p.stem for p in RESULTS.glob("*.json")) == sorted(SIZES)


def test_the_results_were_made_from_the_table_that_ships(size: str) -> None:
    results = results_of(size)
    path = snapshot_path(size)
    recorded = json.loads(meta_path(path).read_text(encoding="utf-8"))["table"]["sha256"]
    assert results["table"]["file"] == path.name
    assert results["table"]["has_provenance"] is True
    assert results["table"]["sha256"] == recorded == sha256_of_text(path)


def test_the_counts_are_the_counts_of_the_table(size: str) -> None:
    results = results_of(size)["table"]
    numbers = counts(read_table(snapshot_path(size)))
    for name in ("recordings", "reviewed", "pending", "participants", "positives"):
        assert results[name] == numbers[name], name
    assert results["prevalence"] == pytest.approx(results["positives"] / results["reviewed"])


def test_the_results_used_the_default_design(size: str) -> None:
    assert results_of(size)["design"] == Design().as_dict()


def test_the_prior_is_exactly_a_coin(size: str) -> None:
    prior = next(r for r in results_of(size)["rows"] if r["model"] == "prior")
    assert prior["auc"] == pytest.approx(0.5, abs=1e-12)
    assert prior["auc_ci95"] == pytest.approx([0.5, 0.5], abs=1e-12)


def test_what_the_review_produces_gives_the_answer_away(size: str) -> None:
    rows = results_of(size)["rows"]
    leaking = [r for r in rows if r["features"] in (REVIEW_ONLY_NAME, BOTH_NAME)]
    assert len(leaking) == 3
    assert all(r["auc"] > 0.95 and not r["known_before_review"] for r in leaking)
    assert all(r["known_before_review"] for r in rows if r not in leaking)


def known_before(size: str) -> list[dict[str, Any]]:
    """The rows that read only what is known before the review, without the prior, in the order of the table."""
    return [r for r in results_of(size)["rows"] if r["known_before_review"] and r["model"] != "prior"]


def test_the_primary_test_sees_nothing_on_any_of_the_tables(size: str) -> None:
    """The answer that the seed builds in is "nothing predicts it", and the one test that was declared says so."""
    assert results_of(size)["primary"]["p_value"] > 0.05


def test_what_the_readme_says_of_the_rows_known_before_the_review_holds_on_the_large_table() -> None:
    rows = known_before("large")
    assert [r["features"] for r in rows[:3]] == ["acquisition", "+ session", "+ device"]
    assert len(rows) == 6
    aucs = [r["auc"] for r in rows]
    assert max(aucs) - 0.5 < 0.02 and 0.5 - min(aucs) < 0.02
    assert max(aucs[:5]) - aucs[0] < 0.002  # "they stay there as groups of columns are added"
    # "the lower ends of the first three are above 0.5 by less than 0.002, and the three after them reach 0.5"
    for row in rows[:3]:
        assert 0.5 < row["auc_ci95"][0] < 0.502, row["name"]
    for row in rows[3:]:
        assert row["auc_ci95"][0] <= 0.5 <= row["auc_ci95"][1], row["name"]


def test_the_primary_test_has_a_reference_at_chance(size: str) -> None:
    primary = results_of(size)["primary"]
    assert primary["permutations"] == Design().permutations
    assert primary["null_mean"] == pytest.approx(0.5, abs=0.01)
    assert 1 / (primary["permutations"] + 1) <= primary["p_value"] <= 1
    assert primary["critical_95"] > primary["null_mean"]
    everything_known = [r for r in known_before(size) if r["model"] == "logistic"][-1]  # the last group that is added
    assert everything_known["features"] == "+ signal"
    # the first pass is one of the ten that the row averages, so the two are close
    assert abs(primary["auc_first_pass"] - everything_known["auc"]) < 0.06


def test_the_planted_effect_is_found_in_proportion_to_its_size(size: str) -> None:
    section = results_of(size)["controls"]["power"]
    points = section["points"]
    assert [p["shift"] for p in points] == list(controls.SHIFTS)
    assert all(p["replicates"] == 20 for p in points)
    for point in points:
        assert point["auc_theory"] == pytest.approx(theory_auc(point["shift"]))
        assert abs(point["auc_alone"] - point["auc_theory"]) < TOLERANCE[size], point["shift"]
    assert points[0]["detected"] <= 0.15  # with no effect, about one table in twenty
    assert points[-1]["detected"] >= points[0]["detected"]
    assert section["critical_95"] == pytest.approx(results_of(size)["primary"]["critical_95"])
    assert section["required_power"] == controls.POWER


def test_the_smallest_effect_that_is_seen_is_where_the_curve_crosses_the_line(size: str) -> None:
    section = results_of(size)["controls"]["power"]
    seen = [p["shift"] for p in section["points"] if p["shift"] > 0 and p["detected"] >= controls.POWER]
    assert section["smallest_detected_shift"] == (seen[0] if seen else None)


def test_a_split_at_random_reports_more_than_a_split_by_participant(size: str) -> None:
    leak = results_of(size)["controls"]["identity_leak"]
    assert leak["concentration"] == 3.3
    assert leak["auc_random"] > leak["auc_grouped"] + 0.02
    assert leak["random_higher"] >= leak["simulations"] - 1  # one simulation in ten may go the other way by chance


def test_the_audit_found_nothing_in_the_splits_used_and_everything_in_the_random_ones(size: str) -> None:
    results = results_of(size)
    grouped, random = results["audit"], results["controls"]["random_split_audit"]
    assert grouped["kind"] == "grouped" and grouped["splits"] == Design().folds * Design().repeats
    assert grouped["splits_with_errors"] == 0 and grouped["checks"] == {}
    assert random["kind"] == "random" and random["splits"] == random["splits_with_errors"] == 5
    assert random["checks"] == {"group-overlap": 5}


def test_every_section_of_the_report_can_be_made_from_every_results_file(size: str) -> None:
    results = results_of(size)
    for lang in sorted(report.TEXT):
        for section in report.SECTIONS:
            assert report.render(results, (section,), lang).strip(), (section, lang)


def test_the_tables_are_the_synthetic_ones_that_the_seed_makes() -> None:
    """The README says that a recording of the seed is `passed` 88%, `flagged` 7% and `rejected` 5% of the time."""
    frame = read_table(snapshot_path("large"))
    reviewed = frame[frame["qc"] != "pending"]["qc"].value_counts(normalize=True)
    assert reviewed["passed"] == pytest.approx(0.88, abs=0.02)
    assert reviewed["flagged"] == pytest.approx(0.07, abs=0.02)
    assert reviewed["rejected"] == pytest.approx(0.05, abs=0.02)
