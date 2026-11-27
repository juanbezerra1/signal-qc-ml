from __future__ import annotations

import pandas as pd
import pytest

from signal_qc_ml import evaluate, rank

from .helpers import make_table

PLAN = evaluate.Design(folds=4, repeats=2, seed=0, bootstrap=100, permutations=0)


def test_a_model_that_is_not_better_than_chance_ranks_nothing() -> None:
    frame = make_table(participants=60, pending=6, seed=1)
    with pytest.raises(
        rank.RankRefused, match=r"not better than chance.*AUC 0\.\d{3}, 95% interval .* and a coin gives 0\.5"
    ):
        rank.rank_pending(frame, PLAN)


def test_a_model_that_finds_something_puts_the_recordings_in_order() -> None:
    frame = make_table(participants=60, pending=6, effect=2.0, seed=1)
    ranking, row = rank.rank_pending(frame, PLAN)
    assert not row.interval_has_chance and row.ci_low > 0.5
    pending = frame[frame["qc"] == "pending"]
    assert sorted(ranking["recording_id"]) == sorted(pending["recording_id"])  # only what is waiting for a review
    assert ranking["rank"].tolist() == list(range(1, len(pending) + 1))
    assert ranking["score"].is_monotonic_decreasing
    assert ranking["score"].between(0, 1).all()
    assert list(ranking.columns) == ["rank", "recording_id", "participant_code", "score"]


def test_the_order_follows_the_planted_column() -> None:
    frame = make_table(participants=80, pending=0, effect=2.0, seed=2)
    frame = pd.concat([frame, make_table(participants=1, pending=0, seed=9).iloc[:0]])
    waiting = frame.tail(6).copy()
    waiting["qc"] = "pending"
    waiting["sig_alpha_mean"] = [-3.0, -2.0, -1.0, 2.0, 3.0, 4.0]  # a recording that needs attention sits high on it
    frame = pd.concat([frame.iloc[:-6], waiting])
    ranking, _ = rank.rank_pending(frame, PLAN)
    top = ranking.head(3)["recording_id"].tolist()
    assert set(top) == set(waiting["recording_id"].iloc[3:])


def test_nothing_is_waiting_for_a_review() -> None:
    frame = make_table(participants=60, pending=0, effect=2.0, seed=1)
    with pytest.raises(rank.RankRefused, match="nothing to put in order"):
        rank.rank_pending(frame, PLAN)


def test_a_model_that_does_worse_than_chance_ranks_nothing_either(monkeypatch: pytest.MonkeyPatch) -> None:
    worse = evaluate.Row("n", "logistic", "f", 3, 0.4, 0.35, 0.45, 0.1, 0.1, True)
    monkeypatch.setattr(rank, "evaluate_one", lambda *a, **k: (worse, None))
    with pytest.raises(rank.RankRefused, match="worse than chance"):
        rank.rank_pending(make_table(participants=30, pending=3, seed=1), PLAN)


def test_the_boosting_model_can_rank_too() -> None:
    frame = make_table(participants=60, pending=6, effect=2.0, seed=1)
    ranking, row = rank.rank_pending(frame, PLAN, model="boosting")
    assert row.model == "boosting" and len(ranking) == 6
