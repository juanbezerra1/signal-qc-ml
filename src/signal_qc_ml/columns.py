"""Which column of the table is what.

The whole project rests on this file. A column is one of four things:

* an **identifier**: who the row is about (the recording, its session, its participant) and when. Never a feature;
* the **target**: ``qc``, the quality-control state of the recording;
* a **feature known before the review**: how the recording was made, the session, the device and the participant,
  and what the signal looked like. This is what a model may use to say, before anyone looks, which recordings need
  attention;
* a column **produced by the review** itself: the trials thrown out are decided in the same review as the recording,
  so a model that reads them is reading the answer. They are kept in the table on purpose, to show what that does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

IDENTIFIERS = ("recording_id", "session_id", "participant_id", "participant_code", "recorded_on")
TARGET = "qc"
GROUP = "participant_id"  # whose recordings must never be on both sides of a split
GROUP_LABEL = "participant_code"  # the same, as the audit's CSV files name it

PENDING = "pending"  # not reviewed yet: no answer to learn from, only rows to score
ATTENTION = ("flagged", "rejected")  # the classes of the question "does this recording need attention?"
STATES = ("passed", "flagged", "rejected", PENDING)

SIGNAL_PREFIX = "sig_"  # sig_<feature>_mean and sig_<feature>_sd


@dataclass(frozen=True)
class Group:
    """A named set of columns. ``categorical`` ones are text or codes, ``numeric`` ones are quantities."""

    name: str
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


ACQUISITION = Group(
    "acquisition",
    numeric=(
        "block_no",
        "sampling_rate_hz",
        "n_channels",
        "duration_s",
        "n_trials",
        "bytes_per_sample",
        "recorded_hour",
        "recorded_weekday",
    ),
    categorical=("study_code", "modality"),
)
SESSION = Group(
    "session",
    numeric=("visit_no", "session_minutes", "recordings_in_session"),
    categorical=("room", "protocol_version", "operator", "operator_role"),
)
DEVICE = Group(
    "device",
    numeric=("device_age_days", "days_since_calibration", "calibration_covers"),
    categorical=("device_id",),
)
PARTICIPANT = Group(
    "participant",
    numeric=("age_years",),
    categorical=("sex", "dominant_hand", "group_label"),
)
REVIEW = Group("review", numeric=("n_trials_rejected", "valid_fraction"))

#: The groups of features known before the review, in the order in which the evaluation adds them.
BEFORE_REVIEW = (ACQUISITION, SESSION, DEVICE, PARTICIPANT)
#: Names of the steps of the evaluation: the four groups above, then the signal.
STEPS = ("acquisition", "session", "device", "participant", "signal")


def signal_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """The ``sig_*`` columns of a table, in the order they have in it."""
    return tuple(c for c in columns if c.startswith(SIGNAL_PREFIX))


def group_named(name: str, columns: Sequence[str]) -> Group:
    """One of the groups of ``STEPS`` (or ``review``), with the signal columns found in ``columns``."""
    if name == "signal":
        return Group("signal", numeric=signal_columns(columns))
    for group in (*BEFORE_REVIEW, REVIEW):
        if group.name == name:
            return group
    raise KeyError(f"no group of columns is called {name!r}; they are {', '.join((*STEPS, REVIEW.name))}")


def required_columns() -> tuple[str, ...]:
    """Columns that every table must have, besides the ``sig_*`` ones."""
    names = [*IDENTIFIERS, TARGET]
    for group in (*BEFORE_REVIEW, REVIEW):
        names.extend(group.columns)
    return tuple(names)
