"""Splitting by participant, and the predictions that come out of the splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .models import FeatureSpec, build_pipeline, design


class LeakageError(RuntimeError):
    """A participant is on both sides of a split: the scores that come out of it cannot be trusted."""


@dataclass(frozen=True)
class Split:
    """One train/test pair: row numbers, the pass (repeat) it belongs to and its fold."""

    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Predictions:
    """What cross-validation gives back: ``score[p, i]`` is the score of recording ``i`` in pass ``p`` (from a model
    that was fitted without it) and ``fold[p, i]`` the fold of pass ``p`` in which it was tested."""

    score: np.ndarray
    fold: np.ndarray


def grouped_splits(y: np.ndarray, groups: np.ndarray, folds: int = 5, repeats: int = 10, seed: int = 0) -> list[Split]:
    """``repeats`` passes of ``folds``-fold cross-validation in which a participant is in a single fold.

    Within a pass every recording is tested once, by a model that has never seen any recording of its participant.
    The share of recordings that need attention is kept about the same in every fold, and each pass shuffles the
    participants in another way.
    """
    out: list[Split] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            out.append(Split(repeat, fold, train, test))
    assert_no_shared_groups(out, groups)
    return out


def random_splits(y: np.ndarray, folds: int = 5, repeats: int = 10, seed: int = 0) -> list[Split]:
    """The same passes, cutting the recordings at random and ignoring who they belong to.

    This is the split that leaks when recordings of a participant resemble each other. It exists to show that, and
    to show what the audit says about it; nothing else in the project evaluates with it.
    """
    out: list[Split] = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
            out.append(Split(repeat, fold, train, test))
    return out


def shared_groups(split: Split, groups: np.ndarray) -> np.ndarray:
    """The participants that have recordings in the train side and in the test side of ``split``."""
    return np.intersect1d(groups[split.train], groups[split.test])


def assert_no_shared_groups(splits: list[Split], groups: np.ndarray) -> None:
    """Raise ``LeakageError`` if any split has a participant on both sides."""
    for split in splits:
        shared = shared_groups(split, groups)
        if shared.size:
            raise LeakageError(
                f"{shared.size} participants are on both sides of fold {split.fold} of pass {split.repeat} "
                f"(for example {', '.join(str(g) for g in shared[:3])})"
            )


def _fit_predict(model: str, spec: FeatureSpec, X: pd.DataFrame, y: np.ndarray, split: Split, seed: int) -> np.ndarray:
    pipeline = build_pipeline(model, spec, seed)
    pipeline.fit(X.iloc[split.train], y[split.train])
    return np.asarray(pipeline.predict_proba(X.iloc[split.test]))[:, 1]


def out_of_fold(
    model: str,
    spec: FeatureSpec,
    frame: pd.DataFrame,
    y: np.ndarray,
    splits: list[Split],
    seed: int = 0,
    jobs: int = 1,
) -> Predictions:
    """Fit ``model`` on the train side of every split and score the test side."""
    X = design(frame, spec)
    passes = max(s.repeat for s in splits) + 1
    score = np.full((passes, len(y)), np.nan)
    fold = np.full((passes, len(y)), -1, dtype=int)
    probabilities = Parallel(n_jobs=jobs)(delayed(_fit_predict)(model, spec, X, y, s, seed) for s in splits)
    for split, proba in zip(splits, probabilities, strict=True):
        score[split.repeat, split.test] = proba
        fold[split.repeat, split.test] = split.fold
    if np.isnan(score).any():
        raise LeakageError("some recording was never tested: the splits do not cover the table")
    return Predictions(score, fold)


def permute_between(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``y`` with the results of whole participants moved to other participants who have as many recordings, and
    then mixed among the recordings of the participant that got them.

    How many recordings of a participant need attention stays what it was in the table (some participants have
    many, most have none), which is what makes the recordings of one participant alike. Which recording they are
    is drawn again, and so is who has them, so every link between the results and the columns is broken: those of
    the participant, those of the session and those of the recording. (Moving a participant's results as they are
    would carry the order of its recordings along, and that order says which modality, block or channel count a
    recording has: the table would keep a link that happened to be there by chance.) A participant that nobody else
    matches in size keeps its own count.
    """
    _, codes = np.unique(groups, return_inverse=True)
    codes = codes.reshape(-1)
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    moved = y.copy()
    for size in np.unique(sizes):
        members = np.flatnonzero(sizes == size)
        rows = order[starts[members][:, None] + np.arange(size)[None, :]]  # one line per participant of this size
        labels = y[rows[rng.permutation(members.size)]] if members.size > 1 else y[rows]
        moved[rows] = rng.permuted(labels, axis=1)
    return moved
