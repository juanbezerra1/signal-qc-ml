"""Break the project in one small way at a time and check that the tests notice.

    py scripts/mutation_check.py                the mutations, one by one, with the verdict of each (a few minutes)
    py scripts/mutation_check.py fold ranking   only the mutations whose name contains one of these words
    py scripts/mutation_check.py --list         the names, and nothing else

It needs what the tests need. The project is copied to a temporary folder and the copy is the one that gets broken,
so your files are never touched, even if you press Ctrl+C.

A test-suite that passes proves little until you have seen it fail. Each mutation below is a plausible mistake (a
tie counted as a loss, a split that forgets the participants, a refusal that is never made). The tests named next to
it must fail. If one does not, the script says SURVIVED and exits with 1: either the tests miss something, or the
mutation is harmless and should go.

tests/test_docs.py fails when the line a mutation targets is no longer in the code exactly once, so a refactor cannot
quietly turn this check into a no-op.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path("src") / "signal_qc_ml"


@dataclass(frozen=True)
class Mutation:
    name: str
    file: str  # inside src/signal_qc_ml
    old: str  # must appear exactly once in the file
    new: str
    tests: tuple[str, ...]  # arguments for pytest: the files, and perhaps -k "..."


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "a tie between scores counts as a loss, not a half",
        "stats.py",
        "(below + 0.5 * negative)",
        "(below)",
        ("tests/test_stats.py",),
    ),
    Mutation(
        "the AUC of folds ranks all the folds together",
        "stats.py",
        "rows = np.flatnonzero(folds_of_pass == k)",
        "rows = np.arange(folds_of_pass.size)",
        ("tests/test_stats.py",),
    ),
    Mutation(
        "the interval is 90 percent, not 95",
        "stats.py",
        "[2.5, 97.5]",
        "[5, 95]",
        ("tests/test_stats.py",),
    ),
    Mutation(
        "the bootstrap draws the participants without replacement",
        "stats.py",
        "counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))",
        "counts = np.ones(n_groups, dtype=int)",
        ("tests/test_stats.py",),
    ),
    Mutation(
        "the p-value does not count the observed value",
        "stats.py",
        "return float((1 + np.sum(null >= observed)) / (1 + null.size))",
        "return float(np.sum(null >= observed) / null.size)",
        ("tests/test_stats.py",),
    ),
    Mutation(
        "the AUC of a shift is Phi(shift), not Phi(shift / sqrt 2)",
        "stats.py",
        "math.erf(shift / 2.0)",
        "math.erf(shift)",
        ("tests/test_stats.py",),
    ),
    Mutation(
        "the splits cut the recordings and forget the participants",
        "cv.py",
        "splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)\n"
        "        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):",
        "splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)\n"
        "        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):",
        ("tests/test_cv.py",),
    ),
    Mutation(
        "every pass cuts the participants in the same way",
        "cv.py",
        "random_state=seed + repeat)\n"
        "        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):",
        "random_state=seed)\n"
        "        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):",
        ("tests/test_cv.py",),
    ),
    Mutation(
        "the test of a split lets a participant on both sides pass",
        "cv.py",
        "if shared.size:",
        "if shared.size > 10**9:",
        ("tests/test_cv.py",),
    ),
    Mutation(
        "moving results does not move them",
        "cv.py",
        "labels = y[rows[rng.permutation(members.size)]] if members.size > 1 else y[rows]",
        "labels = y[rows]",
        ("tests/test_cv.py",),
    ),
    Mutation(
        "moved results keep the order of the recordings",
        "cv.py",
        "moved[rows] = rng.permuted(labels, axis=1)",
        "moved[rows] = labels",
        ("tests/test_cv.py",),
    ),
    Mutation(
        "moved results are shuffled among all the recordings",
        "cv.py",
        "labels = y[rows[rng.permutation(members.size)]] if members.size > 1 else y[rows]",
        "labels = rng.permutation(y)[rows]",
        ("tests/test_cv.py",),
    ),
    Mutation(
        "flagged is not a recording that needs attention",
        "data.py",
        "return qc.isin(cols.ATTENTION).to_numpy(dtype=np.int8)",
        'return (qc == "rejected").to_numpy(dtype=np.int8)',
        ("tests/test_data.py",),
    ),
    Mutation(
        "a recording that waits for a review is a recording that passed",
        "data.py",
        "is_pending = frame[cols.TARGET] == cols.PENDING",
        "is_pending = frame[cols.TARGET] == 'no-such-state'",
        ("tests/test_data.py",),
    ),
    Mutation(
        "the columns that the review produces are known before the review",
        "evaluate.py",
        "return spec_for(cols.STEPS, columns, name=BEFORE_REVIEW_NAME)",
        'return spec_for([*cols.STEPS, "review"], columns, name=BEFORE_REVIEW_NAME)',
        ("tests/test_evaluate.py", "-k", "no_set_of_columns"),
    ),
    Mutation(
        "the prior learns something",
        "models.py",
        'return Pipeline([("model", DummyClassifier(strategy="prior"))])',
        'return Pipeline([("model", DummyClassifier(strategy="stratified", random_state=seed))])',
        ("tests/test_models.py", "tests/test_cv.py", "-k", "prior"),
    ),
    Mutation(
        "the logistic regression reads the numbers as they are",
        "models.py",
        '("scale", StandardScaler()),',
        '("scale", "passthrough"),',
        ("tests/test_models.py",),
    ),
    Mutation(
        "codes are read as quantities",
        "models.py",
        "X[column] = X[column].astype(str)",
        "pass",
        ("tests/test_models.py",),
    ),
    Mutation(
        "the interval of a row that touches 0.5 on one side no longer has chance in it",
        "evaluate.py",
        "return not (self.ci_low > 0.5 or self.ci_high < 0.5)",
        "return not (self.ci_low >= 0.5 or self.ci_high <= 0.5)",
        ("tests/test_evaluate.py", "-k", "interval_has_chance"),
    ),
    Mutation(
        "the primary test is on boosting",
        "evaluate.py",
        'PRIMARY_MODEL = "logistic"',
        'PRIMARY_MODEL = "boosting"',
        ("tests/test_evaluate.py", "-k", "primary_test_is_of"),
    ),
    Mutation(
        "the planted column does not depend on the result",
        "controls.py",
        "return rng.standard_normal(y.size) + shift * y",
        "return rng.standard_normal(y.size) + shift",
        ("tests/test_controls.py", "-k", "planted_column"),
    ),
    Mutation(
        "the zero effect counts as the smallest one that is seen",
        "controls.py",
        "if point.shift > 0 and point.detected >= power:",
        "if point.detected >= power:",
        ("tests/test_controls.py", "-k", "smallest"),
    ),
    Mutation(
        "the planted effect is planted in the real results, not in moved ones",
        "controls.py",
        "y = cv.permute_between(data.y, data.groups, rng)  # results that the columns of the table know nothing about",
        "y = data.y",
        ("tests/test_controls.py", "-k", "replicate or a_planted_column_alone or planted_curve"),
    ),
    Mutation(
        "every participant has the same chance",
        "controls.py",
        "rng.beta(prevalence * concentration, (1 - prevalence) * concentration, size=int(np.max(codes)) + 1)",
        "np.full(int(np.max(codes)) + 1, prevalence)",
        ("tests/test_controls.py", "-k", "tendency"),
    ),
    Mutation(
        "the ranking is made even when the interval has chance in it",
        "rank.py",
        "if row.interval_has_chance:",
        "if False:",
        ("tests/test_rank.py",),
    ),
    Mutation(
        "the ranking puts the least likely first",
        "rank.py",
        'ranking.sort_values("score", ascending=False, kind="stable")',
        'ranking.sort_values("score", ascending=True, kind="stable")',
        ("tests/test_rank.py",),
    ),
    Mutation(
        "the audit is not told who the participants are",
        "audit.py",
        '        "--group",\n        cols.GROUP_LABEL,\n',
        "",
        ("tests/test_audit.py", "-k", "ignore_participants"),
    ),
    Mutation(
        "the audit is given the ids of the recordings",
        "audit.py",
        'DROPPED = ("recording_id", "session_id", "participant_id")',
        "DROPPED = ()",
        ("tests/test_audit.py", "-k", "no_ids"),
    ),
    Mutation(
        "the hash of a gzipped table is the hash of the file",
        "extract.py",
        'data = gzip.decompress(target.read_bytes()) if target.suffix == ".gz" else target.read_bytes()',
        "data = target.read_bytes()",
        ("tests/test_extract.py",),
    ),
    Mutation(
        "a refusal exits like a mistake in the command",
        "cli.py",
        '        _say(f"refused: {error}")\n        return EXIT_REFUSED',
        '        _say(f"refused: {error}")\n        return EXIT_USAGE',
        ("tests/test_cli.py", "-k", "rank_refuses"),
    ),
)


def problems_with_the_list(root: Path = ROOT) -> list[str]:
    """What is wrong with the list itself: a target that is missing or ambiguous."""
    found: list[str] = []
    names = [m.name for m in MUTATIONS]
    found.extend(f"the name {n!r} is used twice" for n in sorted(set(names)) if names.count(n) > 1)
    for m in MUTATIONS:
        path = root / PACKAGE / m.file
        count = path.read_text(encoding="utf-8").count(m.old) if path.exists() else 0
        if count != 1:
            found.append(f"{m.name}: the text to break appears {count} times in {m.file} (it must be exactly once)")
        if m.old == m.new:
            found.append(f"{m.name}: the mutation changes nothing")
        for argument in m.tests:
            if argument.startswith("tests/") and not (root / argument).exists():
                found.append(f"{m.name}: {argument} does not exist")
    return found


def run(copy: Path, mutation: Mutation, env: dict[str, str]) -> tuple[str, str]:
    """Break one thing in the copy, run its tests, put it back. Returns (verdict, what failed first)."""
    target = copy / PACKAGE / mutation.file
    original = target.read_text(encoding="utf-8")
    target.write_text(original.replace(mutation.old, mutation.new, 1), encoding="utf-8")
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-rf", "-p", "no:cacheprovider", *mutation.tests],
            cwd=copy,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        target.write_text(original, encoding="utf-8")
    first = re.search(r"^FAILED (\S+)", done.stdout, re.MULTILINE)
    if done.returncode == 1:
        return "caught", first.group(1).split("::", 1)[-1] if first else "?"
    if done.returncode == 0:
        return "SURVIVED", "every test passed"
    return "ERROR", f"pytest exited with {done.returncode}: {(done.stdout + done.stderr).strip()[-300:]}"


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    words = [a for a in arguments if not a.startswith("--")]
    flags = [a for a in arguments if a.startswith("--")]
    chosen = [m for m in MUTATIONS if not words or any(w.lower() in m.name.lower() for w in words)]
    if "--list" in flags:
        print("\n".join(m.name for m in chosen))
        return 0
    if not chosen:
        print("no mutation has that word in its name; try --list", file=sys.stderr)
        return 2
    broken = problems_with_the_list()
    if broken:
        print("the list of mutations no longer fits the code:\n  " + "\n  ".join(broken), file=sys.stderr)
        return 2

    survivors = 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="signal-qc-ml-mutation-") as tmp:
        copy = Path(tmp) / "project"
        shutil.copytree(
            ROOT,
            copy,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".*_cache", "*.egg-info", "build", "dist", ".venv", "venv"
            ),
        )
        # The copy's own source comes first on the path, whatever way the package was installed.
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(copy / "src"), os.environ.get("PYTHONPATH", "")])}
        seen = subprocess.run(
            [sys.executable, "-c", "import signal_qc_ml; print(signal_qc_ml.__file__)"],
            cwd=copy,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not seen.startswith(str(copy)):
            print(f"the tests would not run on the copy ({seen}); is signal_qc_ml importable?", file=sys.stderr)
            return 2

        for number, mutation in enumerate(chosen, start=1):
            verdict, detail = run(copy, mutation, env)
            survivors += verdict != "caught"
            print(f"{number:2}/{len(chosen)}  {verdict:<8} {mutation.name}  ->  {detail}", flush=True)
    seconds = time.monotonic() - started
    print(f"\n{len(chosen) - survivors} of {len(chosen)} mutations were caught in {seconds:.0f} s.")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
