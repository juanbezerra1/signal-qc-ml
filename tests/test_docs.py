"""The documentation is part of the project, so it is tested like the rest: the numbers in the README are the numbers of
the code and of ``results/``, every command in it parses, every link points at something, and the two READMEs say the
same thing. A README that has drifted from the code is a bug that nobody reports.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import re
import shlex
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from signal_qc_ml import cli, controls
from signal_qc_ml import columns as cols
from signal_qc_ml.evaluate import Design

from .helpers import DATA, RESULTS, ROOT

README = {"en": "README.md", "pt": "README.pt-BR.md"}
TEMPLATE = {"en": "docs/templates/README.md.in", "pt": "docs/templates/README.pt-BR.md.in"}
DOCS = (*README.values(), "docs/design-notes.md", "docs/columns.md")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")


def script(name: str) -> ModuleType:
    """One of the scripts of the project, loaded as a module (they are not part of the installed package)."""
    spec = importlib.util.spec_from_file_location(f"project_script_{name}", ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---- reading markdown, just enough


def without_fences(markdown: str) -> str:
    return re.sub(r"^```[^\n]*\n.*?^```$", "", markdown, flags=re.MULTILINE | re.DOTALL)


def fences(markdown: str) -> list[str]:
    return re.findall(r"^```[^\n]*\n(.*?)^```$", markdown, flags=re.MULTILINE | re.DOTALL)


def headings(markdown: str) -> list[str]:
    return [line for line in without_fences(markdown).splitlines() if re.match(r"#{1,6} ", line)]


def slug(heading: str) -> str:
    """The anchor that GitHub makes of a heading."""
    words = heading.lstrip("#").strip().lower()
    return re.sub(r"[^\w\- ]", "", words).replace(" ", "-")


def link_targets(markdown: str) -> list[str]:
    return re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", without_fences(markdown))


def code_spans(markdown: str) -> list[str]:
    return re.findall(r"`([^`\n]+)`", without_fences(markdown))


def table_under(markdown: str, heading_start: str) -> list[list[str]]:
    """The rows of the first table under a heading, as lists of cells, without its header and delimiter rows."""
    lines = markdown.splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith(heading_start))
    rows: list[list[str]] = []
    for line in lines[at + 1 :]:
        if line.startswith("## ") and rows:
            break
        if not line.startswith("|"):
            if rows:
                break
            continue
        if re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", line):
            continue
        rows.append([cell.strip() for cell in line[1:-1].split("|")])
    return rows[1:]


def commands_in(markdown: str) -> Iterator[list[str]]:
    """The arguments of every ``$ py -m signal_qc_ml ...`` line of the console blocks."""
    for block in fences(markdown):
        for line in block.splitlines():
            if line.startswith("$ py -m signal_qc_ml "):
                yield shlex.split(line.removeprefix("$ py -m signal_qc_ml "))


def local_paths(markdown: str) -> set[str]:
    """The files and folders that a document links to, without the anchors."""
    return {t.split("#")[0] for t in link_targets(markdown) if not t.startswith(("http", "#"))}


def digits(cell: str) -> list[str]:
    """The numbers in a cell without their separators (1,779 in English is 1.779 in Portuguese)."""
    return [re.sub(r"[.,]", "", n) for n in re.findall(r"\d[\d.,]*\d|\d", cell)]


# ---- the READMEs are what the templates and the results make


def test_the_readmes_are_what_the_templates_and_the_results_make() -> None:
    made = script("refresh_readme").expected()
    assert set(made) == set(README.values())
    for name, expected in made.items():
        assert text(name) == expected, f"{name} is stale: run py scripts/refresh_readme.py"
    assert not (ROOT / "ranking.csv").exists()  # `rank` refused, so it wrote nothing


def test_a_placeholder_that_is_not_known_is_an_error() -> None:
    refresh = script("refresh_readme")
    with pytest.raises(KeyError):
        refresh.resolve("nonsense:large", refresh.load_results(), "en")


def test_the_numbers_of_the_prose_follow_the_language() -> None:
    refresh = script("refresh_readme")
    all_results = refresh.load_results()
    assert refresh.resolve("count:large:participants", all_results, "en").replace(",", "") == str(
        all_results["large"]["table"]["participants"]
    )
    assert "," in refresh.resolve("auc:large:acquisition:logistic", all_results, "pt")
    assert "." in refresh.resolve("auc:large:acquisition:logistic", all_results, "en")


def test_the_figures_are_what_the_results_draw() -> None:
    drawn = script("make_figures").figures()
    assert sorted(drawn) == ["auc.svg", "leak.svg", "power.svg"]
    for name, svg in drawn.items():
        assert (ROOT / "docs" / "img" / name).read_text(encoding="utf-8") == svg, name
        assert svg.startswith("<svg") and "<title" in svg and "<desc" in svg  # a figure that a screen reader can say


def test_the_two_readmes_say_the_same_thing() -> None:
    placeholders = {lang: re.findall(r"\{\{(.+?)\}\}", text(TEMPLATE[lang])) for lang in README}
    assert placeholders["en"] == placeholders["pt"]
    en, pt = text(README["en"]), text(README["pt"])
    assert len(headings(en)) == len(headings(pt))
    assert [h.count("#") for h in headings(en)] == [h.count("#") for h in headings(pt)]
    assert len(fences(en)) == len(fences(pt))
    # the commands are the same, except for the language that `report` is asked for
    assert [c[:2] for c in commands_in(en)] == [c[:2] for c in commands_in(pt)]
    figures_en = [t for t in link_targets(en) if t.startswith("docs/img/")]
    figures_pt = [t for t in link_targets(pt) if t.startswith("docs/img/")]
    assert figures_en == figures_pt and len(figures_en) == 3
    assert local_paths(pt) - {"README.md"} == local_paths(en) - {"README.pt-BR.md"}
    assert [digits(row[1]) for row in table_under(en, "## At a glance")] == [
        digits(row[1]) for row in table_under(pt, "## Em resumo")
    ]


def test_each_readme_links_to_the_other() -> None:
    assert "README.pt-BR.md" in link_targets(text(README["en"]))
    assert "README.md" in link_targets(text(README["pt"]))


# ---- every command, link and name points at something


def test_every_command_of_the_readmes_is_one_that_the_program_has() -> None:
    parser = cli.build_parser()
    seen = 0
    for lang in README:
        for arguments in commands_in(text(README[lang])):
            seen += 1
            try:
                parser.parse_args(arguments)
            except SystemExit as stop:  # argparse exits when the line is not one the program understands
                raise AssertionError(f"{README[lang]}: {arguments} is not a valid command") from stop
    assert seen >= 10


def test_every_link_points_at_a_file_and_every_anchor_at_a_heading() -> None:
    for name in DOCS:
        markdown = text(name)
        base = (ROOT / name).parent
        for target in link_targets(markdown):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, anchor = target.partition("#")
            document = (base / path).resolve() if path else ROOT / name
            assert document.exists(), f"{name}: {target} does not exist"
            if anchor and document.suffix == ".md":
                anchors = {slug(h) for h in headings(document.read_text(encoding="utf-8"))}
                assert anchor in anchors, f"{name}: no heading for #{anchor}"


def test_every_script_that_the_docs_name_exists() -> None:
    for name in DOCS:
        for script_name in re.findall(r"scripts/(\w+\.py)", text(name)):
            assert (ROOT / "scripts" / script_name).is_file(), f"{name}: scripts/{script_name}"


def test_every_test_that_the_docs_name_exists() -> None:
    defined = {
        node.name
        for path in (ROOT / "tests").glob("test_*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    named = {span for name in DOCS for span in code_spans(text(name)) if re.fullmatch(r"test_\w+", span)}
    assert named, "the design notes name the tests that pin what went wrong"
    assert named <= defined, sorted(named - defined)


FILE_EXTENSIONS = {"py", "sql", "md", "json", "csv", "gz", "yml", "toml", "svg", "in"}


def test_every_function_that_the_docs_name_exists() -> None:
    modules = {"cv", "stats", "evaluate", "controls", "audit", "rank", "data", "extract", "models", "columns", "report"}
    checked = 0
    for name in DOCS:
        for span in code_spans(text(name)):
            match = re.fullmatch(r"(\w+)\.([A-Za-z_]\w*)", span)
            if match and match.group(1) in modules and match.group(2) not in FILE_EXTENSIONS:
                module = importlib.import_module(f"signal_qc_ml.{match.group(1)}")
                assert hasattr(module, match.group(2)), f"{name}: {span}"
                checked += 1
    assert checked >= 3


def test_every_option_that_the_readmes_show_exists() -> None:
    """The options of this program: in its command lines, and where the prose names one on its own, as `--flag`."""
    known = {option for action in _all_actions(cli.build_parser()) for option in action.option_strings}
    shown = 0
    for lang in README:
        markdown = text(README[lang])
        flags = {
            f
            for block in fences(markdown)
            for f in re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", block)
            if "signal_qc_ml" in block
        }
        flags |= {span for span in code_spans(markdown) if re.fullmatch(r"--[a-z][a-z-]+", span)}
        for flag in flags:
            assert flag in known, f"{README[lang]}: {flag}"
        shown += len(flags)
    assert shown >= 10


def _all_actions(parser: argparse.ArgumentParser) -> Iterator[argparse.Action]:
    for action in parser._actions:  # noqa: SLF001 (argparse has no public way to list the options)
        yield action
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for sub in action.choices.values():
                yield from _all_actions(sub)


# ---- the numbers that the READMEs count are the numbers of the code


def test_the_commands_are_counted_as_the_program_has_them() -> None:
    subparsers = next(a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction))  # noqa: SLF001
    for lang, heading in (("en", "## At a glance"), ("pt", "## Em resumo")):
        rows = {row[0]: row[1] for row in table_under(text(README[lang]), heading)}
        assert int(rows["Commands" if lang == "en" else "Comandos"]) == len(subparsers.choices)


def test_the_mutations_are_counted_as_the_check_has_them() -> None:
    mutations = len(script("mutation_check").MUTATIONS)
    en, pt = text(README["en"]), text(README["pt"])
    assert f"| {mutations} |" in en and f"| {mutations} |" in pt
    assert f"{mutations} small ways" in en and f"{mutations} jeitos pequenos" in pt
    assert f"{mutations} small ways" in text(".github/workflows/ci.yml")


def test_the_dependencies_are_counted_as_pyproject_lists_them() -> None:
    n = len(PYPROJECT["project"]["dependencies"])
    for lang, heading in (("en", "## At a glance"), ("pt", "## Em resumo")):
        row = next(r for r in table_under(text(README[lang]), heading) if r[0] in ("Dependencies", "Dependências"))
        assert digits(row[1])[0] == str(n)


def test_the_python_versions_of_the_docs_are_those_of_the_project() -> None:
    assert PYPROJECT["project"]["requires-python"] == ">=3.11"
    classifiers = [
        c for c in PYPROJECT["project"]["classifiers"] if c.startswith("Programming Language :: Python :: 3.")
    ]
    versions = sorted(c.rsplit(" ", 1)[1] for c in classifiers)
    assert versions == ["3.11", "3.12", "3.13", "3.14"]
    workflow = text(".github/workflows/ci.yml")
    for version in versions:
        assert f'"{version}"' in workflow, version
    assert "3.11 or newer" in text(README["en"]) and "3.11 ou mais novo" in text(README["pt"])


def test_the_design_that_the_readmes_describe_is_the_default_design() -> None:
    design = Design()
    assert (design.folds, design.repeats) == (5, 10)
    assert design.folds * design.repeats == 50  # "50 per table" in the audit section
    assert "five folds" in text(README["en"]) and "cinco dobras" in text(README["pt"])
    assert "repeated ten times" in text(README["en"]) and "repetido dez vezes" in text(README["pt"])
    replicates = inspect.signature(controls.power_curve).parameters["replicates"].default
    assert replicates == 20
    assert "20 times for every shift" in text(README["en"]) and "20 vezes para cada" in text(README["pt"])
    assert "(50 per table)" in text(README["en"]) and "(50 por tabela)" in text(README["pt"])


def test_the_tendency_that_the_readmes_describe_is_the_one_that_is_simulated() -> None:
    concentration = inspect.signature(controls.identity_leak).parameters["concentration"].default
    correlation = f"{1 / (concentration + 1):.2f}"
    assert correlation == "0.23"
    assert correlation in text(README["en"]) and correlation.replace(".", ",") in text(README["pt"])


def test_the_margin_that_the_readmes_give_for_twenty_tables_is_the_one_of_the_arithmetic() -> None:
    margin = 1.96 * (controls.POWER * (1 - controls.POWER) / 20) ** 0.5
    assert round(100 * margin) == 18
    assert "plus or minus 18 points" in text(README["en"]) and "mais ou menos 18 pontos" in text(README["pt"])


def test_the_columns_page_lists_the_columns_of_the_code_and_no_others() -> None:
    page = text("docs/columns.md")
    named = {span for span in code_spans(page) if re.fullmatch(r"[a-z_]+", span)}
    in_code = set(cols.required_columns())
    assert in_code <= named, sorted(in_code - named)
    # what else the page puts in code: the states of the review, the modalities, and a command
    assert named - in_code == {"passed", "flagged", "rejected", "pending", "emg", "force", "eeg", "rank"}
    assert "`sig_<feature>_mean`" in page and "17 features, 34 columns" in page


def test_the_tables_that_ship_have_the_columns_of_the_page_and_the_signals_the_page_counts() -> None:
    header = (DATA / "recordings-small.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert len(cols.signal_columns(header)) == 34
    assert set(cols.required_columns()) <= set(header)
    assert set(header) - set(cols.required_columns()) == set(cols.signal_columns(header))


# ---- nobody's private data is in the project


EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")


def project_text_files() -> Iterator[Path]:
    skip = {".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build", ".venv", "venv"}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and not skip & set(path.relative_to(ROOT).parts) and path.suffix not in {".gz", ".pyc"}:
            yield path


def test_no_e_mail_address_is_in_the_project_but_the_noreply_one() -> None:
    found = {
        (path.relative_to(ROOT).as_posix(), address)
        for path in project_text_files()
        for address in EMAIL.findall(path.read_text(encoding="utf-8", errors="replace"))
        if not address.endswith("@users.noreply.github.com")
    }
    assert found == set()


def test_no_path_of_the_machine_that_made_the_files_is_in_them() -> None:
    for path in [*RESULTS.glob("*.json"), *DATA.glob("*.meta.json"), *(ROOT / name for name in DOCS)]:
        content = path.read_text(encoding="utf-8")
        for fragment in ("/tmp/", "/home/", "/Users/", "C:\\Users", "C:\\\\Users", "C:/Users"):
            assert fragment not in content, f"{path.name}: {fragment}"


def test_the_owner_in_the_links_is_one_name() -> None:
    owners = set()
    for name in README.values():
        owners |= set(re.findall(r"github\.com/([\w-]+)/", text(name)))
    assert len(owners) == 1, owners
