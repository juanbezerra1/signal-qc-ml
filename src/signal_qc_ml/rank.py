"""Putting the recordings that have not been reviewed in order of how likely the review is to want a second look.

This is the one place where the evaluation has a consequence, and it is a refusal when the evaluation found nothing:
a ranking made by a model that is not better than chance on participants it has not seen is an invented order that
looks like a result.
"""

from __future__ import annotations

import pandas as pd

from . import columns as cols
from . import cv
from .data import reviewed_and_pending
from .evaluate import Design, Row, before_review, evaluate_one
from .models import build_pipeline, design


class RankRefused(RuntimeError):
    """The model is not better than chance on participants it has not seen, so it does not give an order."""


def rank_pending(frame: pd.DataFrame, plan: Design, model: str = "logistic", jobs: int = 1) -> tuple[pd.DataFrame, Row]:
    """The recordings waiting for a review, most likely to need attention first, and the evidence for the order.

    The model is evaluated first, on the recordings that have been reviewed, with the same cross-validation by
    participant as everywhere else. If the 95% interval of its AUC includes 0.5, nothing is ranked.
    """
    reviewed, pending = reviewed_and_pending(frame)
    if pending.empty:
        raise RankRefused("no recording is waiting for a review: there is nothing to put in order")
    spec = before_review(list(frame.columns))
    splits = cv.grouped_splits(reviewed.y, reviewed.groups, plan.folds, plan.repeats, plan.seed)
    row, _ = evaluate_one(model, spec, reviewed, splits, plan, jobs)
    if row.interval_has_chance:
        raise RankRefused(
            f"the {model} model is not better than chance on participants it has not seen "
            f"(AUC {row.auc:.3f}, 95% interval {row.ci_low:.3f} to {row.ci_high:.3f}, and a coin gives 0.5), "
            "so it would put the recordings in an order that means nothing, and nothing was ranked"
        )
    if row.ci_high < 0.5:
        raise RankRefused(
            f"the {model} model does worse than chance on participants it has not seen "
            f"(AUC {row.auc:.3f}, 95% interval {row.ci_low:.3f} to {row.ci_high:.3f}); nothing was ranked"
        )
    pipeline = build_pipeline(model, spec, plan.seed)
    pipeline.fit(design(reviewed.frame, spec), reviewed.y)
    score = pipeline.predict_proba(design(pending, spec))[:, 1]
    ranking = pd.DataFrame(
        {"recording_id": pending["recording_id"], cols.GROUP_LABEL: pending[cols.GROUP_LABEL], "score": score}
    )
    ranking = ranking.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["score"] = ranking["score"].round(6)
    return ranking, row
