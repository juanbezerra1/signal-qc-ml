"""The command line: ``signal-qc-ml extract | check | evaluate | rank | report``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import report as report_module
from .analysis import run
from .audit import AuditUnavailable
from .cv import LeakageError
from .data import TableError, counts, read_table
from .evaluate import Design
from .extract import ExtractError, meta_path, snapshot, verify
from .rank import RankRefused, rank_pending

EXIT_OK = 0
EXIT_REFUSED = 1  # the tool looked and said no: a leak, or a model that does not beat chance
EXIT_USAGE = 2  # the command or an input could not be understood


def _design_options(parser: argparse.ArgumentParser, permutations: bool = False) -> None:
    group = parser.add_argument_group("how the evaluation is done")
    group.add_argument("--folds", type=int, default=5, metavar="N", help="folds of the cross-validation (default: 5)")
    group.add_argument(
        "--repeats", type=int, default=10, metavar="N", help="passes, each with other folds (default: 10)"
    )
    group.add_argument("--bootstrap", type=int, default=1000, metavar="N", help="draws of the interval (default: 1000)")
    if permutations:  # only the primary test of `evaluate` moves the results; `rank` does not run it
        group.add_argument(
            "--permutations", type=int, default=500, metavar="N", help="moves of the primary test (default: 500)"
        )
    group.add_argument("--seed", type=int, default=0, help="seed of everything that is random (default: 0)")
    group.add_argument("--jobs", type=int, default=1, metavar="N", help="processes to use (default: 1)")


def _plan(args: argparse.Namespace) -> Design:
    if args.folds < 2:
        raise SystemExit("signal-qc-ml: error: --folds must be at least 2")
    for name in ("repeats", "bootstrap"):
        if getattr(args, name) < 1:
            raise SystemExit(f"signal-qc-ml: error: --{name} must be at least 1")
    permutations = getattr(args, "permutations", 0)
    if permutations < 0 or args.jobs < 1:
        raise SystemExit("signal-qc-ml: error: --permutations cannot be negative and --jobs must be at least 1")
    return Design(args.folds, args.repeats, args.seed, args.bootstrap, permutations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signal-qc-ml",
        description="Does anything in the database predict the quality-control result of a recording?",
    )
    parser.add_argument("--version", action="version", version=f"signal-qc-ml {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="{extract,check,evaluate,rank,report}")

    extract = commands.add_parser("extract", help="read the table from a labtrack-db database")
    extract.add_argument("--dsn", help="the database (default: $LABTRACK_DSN, then the docker-compose default)")
    extract.add_argument("--out", required=True, type=Path, help="where to write the table (.csv or .csv.gz)")

    check = commands.add_parser("check", help="read a table, check it against the provenance file and count its rows")
    check.add_argument("file", type=Path)

    evaluate = commands.add_parser("evaluate", help="evaluate what predicts the review, on participants never seen")
    evaluate.add_argument("file", type=Path)
    evaluate.add_argument("--out", type=Path, help="write the results as JSON")
    evaluate.add_argument("--controls", action="store_true", help="also plant an effect and simulate a leak (slower)")
    evaluate.add_argument(
        "--replicates", type=int, default=20, metavar="N", help="tables per planted effect (default: 20)"
    )
    evaluate.add_argument("--simulations", type=int, default=10, metavar="N", help="simulated leaks (default: 10)")
    evaluate.add_argument(
        "--skip-audit", action="store_true", help="do not audit the splits with table-audit (the results say so)"
    )
    _design_options(evaluate, permutations=True)

    rank = commands.add_parser("rank", help="put the recordings waiting for a review in order, if the evidence allows")
    rank.add_argument("file", type=Path)
    rank.add_argument("--out", required=True, type=Path, help="where to write the ranking (CSV)")
    rank.add_argument("--model", choices=["logistic", "boosting"], default="logistic")
    _design_options(rank)

    show = commands.add_parser("report", help="print the tables of a results file")
    show.add_argument("results", type=Path)
    show.add_argument("--section", choices=report_module.SECTIONS, action="append", help="a section (default: all)")
    show.add_argument("--lang", choices=sorted(report_module.TEXT), default="en")
    return parser


def _say(text: str) -> None:
    print(f"signal-qc-ml: {text}", file=sys.stderr)


def _do_check(args: argparse.Namespace) -> int:
    frame = read_table(args.file)
    numbers = counts(frame)
    verdict = verify(args.file)
    print(f"{args.file}: {numbers['recordings']} recordings of {numbers['participants']} participants")
    print(f"reviewed {numbers['reviewed']} (needing attention: {numbers['positives']}), pending {numbers['pending']}")
    if verdict is None:
        print(f"no {meta_path(args.file).name} next to it: where the table came from is not recorded")
        return EXIT_OK
    if not verdict:
        _say(f"{args.file} is not the table that {meta_path(args.file).name} describes (its SHA-256 differs)")
        return EXIT_REFUSED
    print(f"SHA-256 matches {meta_path(args.file).name}")
    return EXIT_OK


def _do_evaluate(args: argparse.Namespace) -> int:
    results = run(
        args.file,
        _plan(args),
        with_controls=args.controls,
        skip_audit=args.skip_audit,
        jobs=args.jobs,
        replicates=args.replicates,
        simulations=args.simulations,
        progress=_say,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(report_module.render(results))
    if args.out:
        print(f"results written to {args.out}")
    return EXIT_OK


def _do_rank(args: argparse.Namespace) -> int:
    frame = read_table(args.file)
    ranking, row = rank_pending(frame, _plan(args), args.model, args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out, index=False, lineterminator="\n", encoding="utf-8")
    print(
        f"{len(ranking)} recordings ranked by the {args.model} model "
        f"(AUC on participants never seen {row.auc:.3f}, 95% interval {row.ci_low:.3f} to {row.ci_high:.3f})"
    )
    print(f"ranking written to {args.out}")
    return EXIT_OK


def _do_report(args: argparse.Namespace) -> int:
    try:
        results = json.loads(args.results.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise TableError(f"there is no file {args.results}") from None
    except json.JSONDecodeError as error:
        raise TableError(f"{args.results} is not a results file: {error}") from error
    sections = tuple(args.section) if args.section else report_module.SECTIONS
    print(report_module.render(results, sections, args.lang))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # a Windows console in a legacy code page must not crash on an accent
            reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract":
            meta = snapshot(args.dsn, args.out)
            print(
                f"{meta['table']['rows']} recordings written to {args.out} (SHA-256 {meta['table']['sha256'][:12]}...)"
            )
            return EXIT_OK
        if args.command == "check":
            return _do_check(args)
        if args.command == "evaluate":
            return _do_evaluate(args)
        if args.command == "rank":
            return _do_rank(args)
        return _do_report(args)
    except (RankRefused, LeakageError) as error:
        _say(f"refused: {error}")
        return EXIT_REFUSED
    except (TableError, ExtractError, AuditUnavailable, ValueError) as error:
        _say(f"error: {error}")
        return EXIT_USAGE
