"""The few statistics the evaluation needs, written out so that they can be checked against scikit-learn."""

from __future__ import annotations

import math

import numpy as np


class RankedScores:
    """The ties of a vector of scores, worked out once, so that the AUC of many weightings of the rows is cheap.

    The area under the ROC curve is the chance that a recording that needs attention gets a higher score than one
    that does not, counting a tie as half. With a weight on every row (how many times its participant was drawn in a
    bootstrap sample, say) it is a sum over the distinct scores, two ``bincount`` calls and a cumulative sum.
    """

    def __init__(self, score: np.ndarray) -> None:
        distinct, inverse = np.unique(score, return_inverse=True)
        self.inverse = inverse.reshape(-1)
        self.size = int(distinct.size)

    def auc(self, y: np.ndarray, weights: np.ndarray | None = None) -> float:
        """The AUC of these scores for the 0/1 labels ``y``; NaN if (after weighting) one class is empty."""
        w = np.ones(y.shape[0]) if weights is None else weights
        negative = np.bincount(self.inverse, weights=w * (1 - y), minlength=self.size)
        positive = np.bincount(self.inverse, weights=w * y, minlength=self.size)
        pairs = positive.sum() * negative.sum()
        if pairs == 0:
            return math.nan
        below = np.cumsum(negative) - negative
        return float((positive * (below + 0.5 * negative)).sum() / pairs)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """The area under the ROC curve (see ``RankedScores``)."""
    return RankedScores(score).auc(y)


def _mean_ignoring_nan(values: list[float]) -> float:
    kept = [v for v in values if not math.isnan(v)]
    return float(np.mean(kept)) if kept else math.nan


class FoldedAUC:
    """The AUC of cross-validation: one AUC for each fold of each pass, averaged.

    The models of two folds are different models, so their scores are not on one scale; pooling the scores of all the
    folds before ranking them mixes the scales (a model fitted on a fold with fewer recordings that need attention
    gives lower probabilities to a fold that has more of them) and pulls the AUC under 0.5 even when nothing is
    learnt. Ranking inside each fold and averaging is what scikit-learn's ``cross_val_score`` does with
    ``scoring="roc_auc"``; it also makes the AUC of a model that says the same thing for every recording exactly 0.5.

    ``score[p, i]`` is the score that recording ``i`` got in pass ``p``, and ``fold[p, i]`` the fold that tested it.
    """

    def __init__(self, score: np.ndarray, fold: np.ndarray) -> None:
        score, fold = np.atleast_2d(score), np.atleast_2d(fold)
        if score.shape != fold.shape:
            raise ValueError("score and fold must have the same shape")
        self.parts: list[tuple[np.ndarray, RankedScores]] = []
        for scores_of_pass, folds_of_pass in zip(score, fold, strict=True):
            for k in np.unique(folds_of_pass):
                rows = np.flatnonzero(folds_of_pass == k)
                self.parts.append((rows, RankedScores(scores_of_pass[rows])))

    def auc(self, y: np.ndarray, weights: np.ndarray | None = None) -> float:
        """The mean AUC over the folds of all the passes; a fold where a class is missing does not count."""
        return _mean_ignoring_nan(
            [ranked.auc(y[rows], None if weights is None else weights[rows]) for rows, ranked in self.parts]
        )


def participant_bootstrap(
    y: np.ndarray, folded: FoldedAUC, groups: np.ndarray, draws: int, seed: int
) -> tuple[float, float]:
    """A 95% interval for the AUC of ``folded``, from resampling participants.

    The recordings of a participant resemble each other, so the unit of the sample is the participant: each draw
    takes as many participants as there are, with replacement, and every recording of a participant counts as many
    times as the participant was drawn. The scores are not recomputed, so the interval says how much the AUC moves with
    the participants in the test sets, not how much it moves with the data the models were fitted on.
    """
    _, codes = np.unique(groups, return_inverse=True)
    codes = codes.reshape(-1)
    n_groups = int(np.max(codes)) + 1
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for draw in range(draws):
        counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
        values[draw] = folded.auc(y, counts[codes].astype(float))
    if np.isnan(values).all():
        return math.nan, math.nan
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def theory_auc(shift: float) -> float:
    """The AUC of a score that is standard normal in one class and standard normal moved by ``shift`` in the other."""
    return 0.5 * (1.0 + math.erf(shift / 2.0))  # Phi(shift / sqrt(2))


def p_value(observed: float, null: np.ndarray) -> float:
    """The chance, under the null distribution, of a value at least as large as ``observed`` (counting itself)."""
    return float((1 + np.sum(null >= observed)) / (1 + null.size))
