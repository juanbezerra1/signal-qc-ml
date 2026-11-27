"""The whole analysis of one table, written down as one JSON document."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from . import __version__, audit, controls, cv
from . import columns as cols
from .data import TableError, counts, read_table, reviewed_and_pending
from .evaluate import DEFAULT_DESIGN, Design, evaluate
from .extract import meta_path, sha256_of_text, verify

Progress = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


def environment() -> dict[str, str]:
    """The versions that the numbers depend on a little (the last digit of an AUC can change with a BLAS)."""
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def run(
    path: str | Path,
    plan: Design = DEFAULT_DESIGN,
    *,
    with_controls: bool = False,
    skip_audit: bool = False,
    jobs: int = 1,
    replicates: int = 20,
    simulations: int = 10,
    progress: Progress = _quiet,
) -> dict[str, Any]:
    """Evaluate the table in ``path``: the results table, the primary test, the audit and, if asked, the controls."""
    started = time.monotonic()
    frame = read_table(path)
    if verify(path) is False:
        raise TableError(
            f"{path} is not the table that {meta_path(path).name} describes (its SHA-256 differs): "
            "it was changed after it was extracted"
        )
    data, _ = reviewed_and_pending(frame)
    if with_controls and not plan.permutations:
        raise ValueError("the controls compare with the primary test, so they need permutations (use --permutations)")
    result: dict[str, Any] = {
        "tool": f"signal-qc-ml {__version__}",
        "table": {
            "file": Path(path).name,
            "sha256": sha256_of_text(path),
            "has_provenance": meta_path(path).is_file(),
            **counts(frame),
            "prevalence": data.prevalence,
        },
        "design": plan.as_dict(),
    }

    progress("evaluating the columns, group by group")
    evaluation = evaluate(data, plan, jobs)
    result["rows"] = [row.as_dict() for row in evaluation.rows]
    if evaluation.primary is not None:
        result["primary"] = evaluation.primary.as_dict()

    splits = cv.grouped_splits(data.y, data.groups, plan.folds, plan.repeats, plan.seed)
    if skip_audit:
        result["audit"] = {"skipped": True}
    else:
        progress(f"auditing {len(splits)} splits with table-audit")
        summary = audit.audit_splits(data.frame, splits, "grouped")
        audit.require_clean(summary)
        result["audit"] = summary.as_dict()

    if with_controls:
        assert evaluation.primary is not None
        progress("planting an effect of known size")
        points = controls.power_curve(data, evaluation.primary.critical_95, plan, replicates=replicates, jobs=jobs)
        smallest = controls.smallest_detected(points)
        progress("simulating a tendency of each participant")
        leak = controls.identity_leak(data, plan, simulations=simulations, jobs=jobs)
        section: dict[str, Any] = {
            "power": {
                "critical_95": evaluation.primary.critical_95,
                "required_power": controls.POWER,
                "points": [p.as_dict() for p in points],
                "smallest_detected_shift": smallest.shift if smallest else None,
            },
            "identity_leak": leak.as_dict(),
        }
        if not skip_audit:
            progress("auditing a split that ignores the participants")
            random = audit.audit_splits(data.frame, cv.random_splits(data.y, plan.folds, 1, plan.seed), "random")
            section["random_split_audit"] = random.as_dict()
        result["controls"] = section

    result["environment"] = environment()
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def columns_by_group(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Which columns of ``frame`` belong to which group: what the evaluation was allowed to read."""
    out = {group.name: list(group.columns) for group in (*cols.BEFORE_REVIEW, cols.REVIEW)}
    out["signal"] = list(cols.signal_columns(list(frame.columns)))
    return out
