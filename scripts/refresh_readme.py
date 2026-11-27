"""Write README.md and README.pt-BR.md from docs/templates/ and results/*.json, so that each number in them is a result.

    py scripts/refresh_readme.py            rewrite the two READMEs
    py scripts/refresh_readme.py --check    fail if the READMEs are not what the templates and the results make

A template is the README with {{...}} where a number or a table goes:

    {{table:evaluation:large}}      the table of one section of one results file (see report.py for the sections)
    {{auc:large:+ signal:logistic}} the AUC of one row (the columns it reads, then the model)
    {{ci:large:+ signal:logistic}}  its 95% interval, as "0.489 to 0.523" ("0,489 a 0,523" in Portuguese)
    {{n:large:+ signal:logistic}}   how many columns that row reads
    {{lowest:large}}                the lowest AUC among the rows that read only what is known before the review
    {{highest:large}}               the highest
    {{primary:large:p_value}}       a field of the primary test (a number with 3 decimals)
    {{count:large:participants}}    a count from the table of that results file, with thousands separators
    {{share:large:prevalence}}      a share from that table, as a percentage with one decimal
    {{seen:medium:0.4}}             the share of planted tables that the primary test saw, at a shift (a percentage)
    {{theory:0.4}}                  the AUC that a planted column of that shift would have on its own
    {{leak:large:auc_random}}       a field of the simulation of a tendency of each participant (a count as it is)
    {{smallest:large}}              the AUC (alone) of the weakest planted effect that was seen at least 80% of the time
    {{shift:large}}                 the shift (in standard deviations) of that effect
    {{seconds:large}}               how long that run took, in minutes
    {{out:check data/recordings-small.csv}}
                                    what the program prints for those arguments (run here, from the root of the project)

Numbers in the prose follow the language (0,503 and 1.779 in Portuguese); the tables are what ``report`` prints, with a
decimal point in both, like the rest of the output of the program.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from signal_qc_ml import report  # noqa: E402
from signal_qc_ml.cli import main as run_cli  # noqa: E402
from signal_qc_ml.stats import theory_auc  # noqa: E402

TARGETS = {
    "en": ("docs/templates/README.md.in", "README.md"),
    "pt": ("docs/templates/README.pt-BR.md.in", "README.pt-BR.md"),
}
PLACEHOLDER = re.compile(r"\{\{(.+?)\}\}")


def load_results(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "results").glob("*.json"))}


def _row(results: dict[str, Any], features: str, model: str) -> dict[str, Any]:
    for row in results["rows"]:
        if row["features"] == features and row["model"] == model:
            return row  # type: ignore[no-any-return]
    raise KeyError(f"no row for {features!r} / {model!r}")


def _decimal(value: float, digits: int, lang: str) -> str:
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "pt" else text


def _whole(value: int, lang: str) -> str:
    text = f"{value:,}"
    return text.replace(",", ".") if lang == "pt" else text


def with_slashes(text: str, separator: str = os.sep) -> str:
    """``text`` with the separator that the system writes between folders turned into a slash.

    The program prints a path the way the system writes it (``data\\recordings-small.csv`` on Windows). The README shows
    the same text wherever it is made, and that is the one with slashes.
    """
    return text.replace(separator, "/")


def printed_by_the_program(arguments: str) -> str:
    """What ``py -m signal_qc_ml <arguments>`` prints (both streams, in order), run from the root of the project."""
    buffer = io.StringIO()
    here = Path.cwd()
    os.chdir(ROOT)
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            run_cli(shlex.split(arguments))
    finally:
        os.chdir(here)
    return with_slashes(buffer.getvalue().rstrip("\n"))


def resolve(key: str, all_results: dict[str, dict[str, Any]], lang: str) -> str:
    kind, _, rest = key.strip().partition(":")
    if kind == "out":
        return printed_by_the_program(rest)
    args = rest.split(":")
    to = report.TEXT[lang]["to"]
    if kind == "table":
        return report.render(all_results[args[1]], (args[0],), lang).strip()
    if kind == "theory":
        return _decimal(theory_auc(float(args[0])), 3, lang)
    results = all_results[args[0]]
    if kind == "auc":
        return _decimal(_row(results, args[1], args[2])["auc"], 3, lang)
    if kind in ("lowest", "highest"):
        known = [r["auc"] for r in results["rows"] if r["known_before_review"] and r["model"] != "prior"]
        return _decimal(min(known) if kind == "lowest" else max(known), 3, lang)
    if kind == "n":
        return str(_row(results, args[1], args[2])["n_features"])
    if kind == "ci":
        low, high = _row(results, args[1], args[2])["auc_ci95"]
        return f"{_decimal(low, 3, lang)} {to} {_decimal(high, 3, lang)}"
    if kind == "primary":
        value = results["primary"][args[1]]
        return str(value) if args[1] == "permutations" else _decimal(value, 3, lang)
    if kind == "count":
        return _whole(results["table"][args[1]], lang)
    if kind == "share":
        return _decimal(100 * results["table"][args[1]], 1, lang) + "%"
    if kind == "seen":
        point = next(p for p in results["controls"]["power"]["points"] if abs(p["shift"] - float(args[1])) < 1e-9)
        return f"{round(100 * point['detected'])}%"
    if kind == "leak":
        value = results["controls"]["identity_leak"][args[1]]
        return str(value) if isinstance(value, int) else _decimal(value, 3, lang)
    if kind == "smallest":
        shift = results["controls"]["power"]["smallest_detected_shift"]
        return "none" if shift is None else _decimal(theory_auc(shift), 3, lang)
    if kind == "shift":
        shift = results["controls"]["power"]["smallest_detected_shift"]
        return "none" if shift is None else _decimal(shift, 1, lang)
    if kind == "seconds":
        return _decimal(results["seconds"] / 60, 0, lang)
    raise KeyError(f"unknown placeholder {key!r}")


def render(template: str, all_results: dict[str, dict[str, Any]], lang: str) -> str:
    return PLACEHOLDER.sub(lambda m: resolve(m.group(1), all_results, lang), template)


def expected(root: Path = ROOT) -> dict[str, str]:
    all_results = load_results(root)
    out = {}
    for lang, (template, target) in TARGETS.items():
        text = (root / template).read_text(encoding="utf-8")
        out[target] = render(text, all_results, lang)
    return out


def main(argv: list[str]) -> int:
    made = expected()
    if "--check" in argv:
        stale = [name for name, text in made.items() if (ROOT / name).read_text(encoding="utf-8") != text]
        for name in stale:
            print(
                f"{name} is not what its template and results/*.json make: run py scripts/refresh_readme.py",
                file=sys.stderr,
            )
        return 1 if stale else 0
    for name, text in made.items():
        (ROOT / name).write_text(text, encoding="utf-8", newline="\n")
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
