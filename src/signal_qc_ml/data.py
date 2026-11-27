"""Reading the table of recordings and cutting it into what the models need."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import columns as cols


class TableError(ValueError):
    """The file is not a table that this tool can use; the message says what is wrong with it."""


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a table written by ``extract``: one row per recording, UTF-8, a point for decimals."""
    try:
        frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except FileNotFoundError as error:
        raise TableError(f"there is no file {path}") from error
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as error:
        raise TableError(f"{path} is not a table of recordings: {error}") from error
    validate(frame, str(path))
    return frame


def validate(frame: pd.DataFrame, name: str = "the table") -> None:
    """Raise ``TableError`` unless ``frame`` has the columns and the values that the rest of the code relies on."""
    missing = [c for c in cols.required_columns() if c not in frame.columns]
    if missing:
        raise TableError(f"{name} has no column {', '.join(missing)}; was it written by `signal-qc-ml extract`?")
    unknown = sorted(set(frame[cols.TARGET].astype(str)) - set(cols.STATES))
    if unknown:
        raise TableError(f"{name}: qc has values that are not one of {', '.join(cols.STATES)}: {', '.join(unknown)}")
    if frame["recording_id"].duplicated().any():
        raise TableError(f"{name}: a recording_id appears more than once")
    for column in cols.signal_columns(list(frame.columns)):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TableError(f"{name}: the signal column {column} is not made of numbers")


@dataclass(frozen=True)
class Reviewed:
    """The recordings that someone has reviewed: the only ones that have an answer to learn from."""

    frame: pd.DataFrame
    y: np.ndarray  # 1 when the recording was flagged or rejected, 0 when it passed
    groups: np.ndarray  # the participant of each row

    @property
    def n_participants(self) -> int:
        return int(np.unique(self.groups).size)

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())


def target_of(qc: pd.Series) -> np.ndarray:
    """1 for a recording that needs attention (flagged or rejected), 0 for one that passed."""
    return qc.isin(cols.ATTENTION).to_numpy(dtype=np.int8)


def reviewed_and_pending(frame: pd.DataFrame) -> tuple[Reviewed, pd.DataFrame]:
    """Split the table in the recordings that have a review (to learn from) and those that do not (to score).

    ``pending`` is not a quality-control result, it is the absence of one: it is never a class of the target.
    """
    is_pending = frame[cols.TARGET] == cols.PENDING
    done = frame[~is_pending].reset_index(drop=True)
    pending = frame[is_pending].reset_index(drop=True)
    if done.empty:
        raise TableError("no recording in the table has been reviewed, so there is nothing to learn from")
    y = target_of(done[cols.TARGET])
    if y.min() == y.max():
        raise TableError("every reviewed recording has the same result, so there is nothing to tell apart")
    return Reviewed(done, y, done[cols.GROUP].to_numpy()), pending


def counts(frame: pd.DataFrame) -> dict[str, int]:
    """The numbers that every report starts with."""
    reviewed, pending = reviewed_and_pending(frame)
    return {
        "recordings": len(frame),
        "reviewed": len(reviewed.frame),
        "pending": len(pending),
        "participants": reviewed.n_participants,
        "positives": reviewed.n_positive,
    }
