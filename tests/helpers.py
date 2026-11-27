"""Small tables for the tests: the same columns as the real one, made by a random number generator.

``effect`` puts a link between one signal column and the result of the review (how many standard deviations the
column moves for a recording that needs attention); ``tendency`` gives every participant its own chance of having
recordings that need attention, so that recordings of one participant resemble each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signal_qc_ml import columns as cols
from signal_qc_ml.data import Reviewed, reviewed_and_pending

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SIZES = ("small", "medium", "large")


def snapshot_path(size: str) -> Path:
    for candidate in (DATA / f"recordings-{size}.csv", DATA / f"recordings-{size}.csv.gz"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(size)


def make_table(
    participants: int = 30,
    per_participant: int = 8,
    effect: float = 0.0,
    tendency: float = 0.0,
    pending: int = 4,
    prevalence: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """A table that passes ``data.validate``, with ``participants * per_participant`` recordings."""
    rng = np.random.default_rng(seed)
    n = participants * per_participant
    participant = np.repeat(np.arange(1, participants + 1), per_participant)
    session = (participant - 1) * 2 + np.tile(np.repeat([1, 2], per_participant // 2), participants)
    chance = np.full(participants, prevalence)
    if tendency:
        chance = rng.beta(prevalence * tendency, (1 - prevalence) * tendency, size=participants)
    needs_attention = rng.random(n) < chance[participant - 1]
    qc = np.where(needs_attention, rng.choice(["flagged", "rejected"], size=n, p=[0.6, 0.4]), "passed").astype(object)
    reviewed_last = n - pending
    qc[reviewed_last:] = cols.PENDING
    y = needs_attention.astype(float)

    per_session = rng.normal(size=int(session.max()) + 1)
    n_trials = rng.integers(20, 40, size=n)
    rejected = np.where(needs_attention, rng.integers(8, 20, size=n), rng.integers(0, 3, size=n))
    rejected = np.minimum(rejected, n_trials)
    frame = pd.DataFrame(
        {
            "recording_id": np.arange(1, n + 1),
            "session_id": session,
            "participant_id": participant,
            "participant_code": [f"P{p:04d}" for p in participant],
            "recorded_on": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, size=n), unit="D"),
            "study_code": rng.choice(["S-01", "S-02"], size=n),
            "modality": rng.choice(["emg", "force", "eeg"], size=n),
            "block_no": rng.integers(1, 4, size=n),
            "sampling_rate_hz": rng.choice([1000, 2000], size=n),
            "n_channels": rng.choice([1, 4], size=n),
            "duration_s": rng.normal(350, 40, size=n).round(1),
            "n_trials": n_trials,
            "bytes_per_sample": 4 + rng.normal(0, 0.01, size=n),
            "recorded_hour": rng.integers(8, 18, size=n),
            "recorded_weekday": rng.integers(1, 6, size=n),
            "visit_no": rng.integers(1, 4, size=n),
            "room": rng.choice(["R01", "R02", "R03"], size=n),
            "protocol_version": rng.choice(["v1", "v2"], size=n),
            "session_minutes": (90 + 10 * per_session[session]).round(1),
            "recordings_in_session": per_participant // 2,
            "operator": rng.integers(1, 6, size=n),
            "operator_role": rng.choice(["postdoc", "student"], size=n),
            "device_id": rng.integers(1, 6, size=n),
            "device_age_days": rng.integers(100, 2000, size=n),
            "days_since_calibration": (30 + 10 * per_session[session]).round(1),
            "calibration_covers": rng.integers(0, 2, size=n),
            "age_years": (20 + (participant % 40)),
            "sex": np.where(participant % 2, "F", "M"),
            "dominant_hand": np.where(participant % 5, "right", "left"),
            "group_label": np.where(participant % 3, "control", "intervention"),
            "n_trials_rejected": rejected,
            "valid_fraction": 1 - rejected / n_trials,
            "qc": qc,
        }
    )
    frame["recorded_on"] = frame["recorded_on"].dt.strftime("%Y-%m-%d")
    for feature in ("alpha", "beta", "gamma"):
        for stat in ("mean", "sd"):
            frame[f"sig_{feature}_{stat}"] = rng.normal(size=n)
    frame["sig_alpha_mean"] = frame["sig_alpha_mean"] + effect * y
    frame.loc[frame["modality"] != "eeg", ["sig_gamma_mean", "sig_gamma_sd"]] = np.nan  # a feature of one modality
    return frame


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def reviewed_of(**kwargs: Any) -> Reviewed:
    """The reviewed recordings of a table made by ``make_table`` with none of them pending."""
    return reviewed_and_pending(make_table(pending=0, **kwargs))[0]
