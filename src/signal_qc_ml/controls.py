"""Controls: what the evaluation says when there is something to find, and when it is allowed to cheat.

A result of "nothing predicts it" is worth something only if the same procedure finds a signal that is really there.
So the table of recordings is kept as it is, and a signal is put in it on purpose.

* ``power_curve`` adds a column that carries the result of the review with a known strength, and asks how often the
  primary test sees it. The results are moved between participants first, so that the columns of the table know
  nothing about them and the planted column is the only thing there is to find. The answer says how large an effect
  the "no" of the evaluation rules out.
* ``identity_leak`` gives each participant a tendency to have recordings that need attention, and shows what a random
  split reports, and what a split by participant reports, when the columns say who the participant is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from . import cv, stats
from .data import Reviewed
from .evaluate import Design, before_review, single_pass_auc
from .models import FeatureSpec

#: Sizes of the planted effect: how many standard deviations the column moves for a recording that needs attention.
SHIFTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
PLANTED = "planted"  # the name of the column that carries the effect
POWER = 0.8  # the share of detections that counts as "the test sees it"


@dataclass(frozen=True)
class PowerPoint:
    shift: float
    auc_theory: float  # of the planted column alone, in the limit of many recordings
    auc_alone: float  # of the planted column alone, as the evaluation measures it
    auc_with_known: float  # of everything known before the review plus the planted column
    detected: float  # the share of replicates whose AUC is above the critical value of the primary test
    replicates: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def planted_column(y: np.ndarray, shift: float, rng: np.random.Generator) -> np.ndarray:
    """A number for each recording: noise, moved up by ``shift`` for the recordings that need attention."""
    return rng.standard_normal(y.size) + shift * y


def _replicate(data: Reviewed, spec: FeatureSpec, shift: float, folds: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = cv.permute_between(data.y, data.groups, rng)  # results that the columns of the table know nothing about
    frame = data.frame.assign(**{PLANTED: planted_column(y, shift, rng)})
    alone = single_pass_auc("logistic", FeatureSpec(PLANTED, (PLANTED,)), frame, y, data.groups, folds, seed)
    with_known = single_pass_auc("logistic", spec.plus(PLANTED), frame, y, data.groups, folds, seed)
    return alone, with_known


def power_curve(
    data: Reviewed,
    critical: float,
    design: Design,
    shifts: tuple[float, ...] = SHIFTS,
    replicates: int = 20,
    jobs: int = 1,
) -> list[PowerPoint]:
    """How often the primary test sees a planted column of each strength.

    Every replicate moves the results between participants anew, plants the column and measures one pass of
    cross-validation. ``critical`` is the AUC that 95% of the tables with the results moved stay under (the
    ``critical_95`` of the primary test): a replicate counts as detected when the AUC of everything known before the
    review plus the planted column is above it, so with no effect (shift 0) about one replicate in twenty is.
    """
    if PLANTED in data.frame.columns:
        raise ValueError(f"the table already has a column called {PLANTED}")
    spec = before_review(list(data.frame.columns))
    seeds = np.random.SeedSequence([design.seed, 7]).spawn(len(shifts) * replicates)
    tasks = [
        (shift, int(seeds[i * replicates + r].generate_state(1)[0]))
        for i, shift in enumerate(shifts)
        for r in range(replicates)
    ]
    results = Parallel(n_jobs=jobs)(delayed(_replicate)(data, spec, s, design.folds, seed) for s, seed in tasks)
    points = []
    for i, shift in enumerate(shifts):
        chunk = results[i * replicates : (i + 1) * replicates]
        alone = [a for a, _ in chunk]
        known = [k for _, k in chunk]
        points.append(
            PowerPoint(
                shift=shift,
                auc_theory=stats.theory_auc(shift),
                auc_alone=float(np.mean(alone)),
                auc_with_known=float(np.mean(known)),
                detected=float(np.mean([k > critical for k in known])),
                replicates=replicates,
            )
        )
    return points


def smallest_detected(points: list[PowerPoint], power: float = POWER) -> PowerPoint | None:
    """The weakest planted effect of the grid that is seen at least ``power`` of the time (not the zero effect)."""
    for point in sorted(points, key=lambda p: p.shift):
        if point.shift > 0 and point.detected >= power:
            return point
    return None


@dataclass(frozen=True)
class IdentityLeak:
    concentration: float
    icc: float  # how alike the recordings of one participant are, 0 (not at all) to 1 (identical)
    prevalence: float
    simulations: int
    auc_random: float
    auc_grouped: float
    random_higher: int  # in how many simulations the random split reports the higher AUC

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def simulate_tendency(
    groups: np.ndarray, prevalence: float, concentration: float, rng: np.random.Generator
) -> np.ndarray:
    """Results where each participant has its own chance of a recording that needs attention.

    The chance of a participant is drawn from a Beta distribution with mean ``prevalence``; a small ``concentration``
    spreads the participants (some have many such recordings, most have none), a large one makes them all alike.
    """
    _, codes = np.unique(groups, return_inverse=True)
    codes = codes.reshape(-1)
    chance = np.asarray(
        rng.beta(prevalence * concentration, (1 - prevalence) * concentration, size=int(np.max(codes)) + 1)
    )
    return (rng.random(codes.size) < chance[codes]).astype(np.int8)


def _simulation(data: Reviewed, spec: FeatureSpec, concentration: float, folds: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = simulate_tendency(data.groups, data.prevalence, concentration, rng)
    if y.min() == y.max():
        return math.nan, math.nan
    out = []
    for splits in (cv.random_splits(y, folds, 1, seed), cv.grouped_splits(y, data.groups, folds, 1, seed)):
        predictions = cv.out_of_fold("boosting", spec, data.frame, y, splits, seed)
        out.append(stats.FoldedAUC(predictions.score, predictions.fold).auc(y))
    return out[0], out[1]


def identity_leak(
    data: Reviewed, design: Design, concentration: float = 3.3, simulations: int = 10, jobs: int = 1
) -> IdentityLeak:
    """AUC of boosting on the columns known before the review, with results that follow the participant.

    The columns do not know anything about the results, but several of them (the session, the device, the age) are
    the same for the recordings of one participant, so a model that sees the participant in training can learn the
    participant's tendency. A random split lets it; a split by participant does not.
    """
    spec = before_review(list(data.frame.columns))
    seeds = np.random.SeedSequence([design.seed, 11]).spawn(simulations)
    seed_values = [int(s.generate_state(1)[0]) for s in seeds]
    pairs = Parallel(n_jobs=jobs)(
        delayed(_simulation)(data, spec, concentration, design.folds, seed) for seed in seed_values
    )
    kept = [(r, g) for r, g in pairs if not (math.isnan(r) or math.isnan(g))]
    if not kept:
        raise ValueError("no simulation gave results with both classes")
    random_auc = np.array([r for r, _ in kept])
    grouped_auc = np.array([g for _, g in kept])
    return IdentityLeak(
        concentration=concentration,
        icc=1.0 / (concentration + 1.0),
        prevalence=data.prevalence,
        simulations=len(kept),
        auc_random=float(random_auc.mean()),
        auc_grouped=float(grouped_auc.mean()),
        random_higher=int((random_auc > grouped_auc).sum()),
    )
