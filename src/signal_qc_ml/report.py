"""Markdown tables from the JSON that ``analysis.run`` writes: the same code makes the README and the console output.

The text is plain ASCII on purpose: a Windows console that is not set to UTF-8 would turn anything else into
question marks. A test compares the tables in the README with what this module makes from ``results/*.json``.
"""

from __future__ import annotations

from typing import Any

SECTIONS = ("evaluation", "primary", "power", "leak", "audit")

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "columns": "Columns the model reads",
        "model": "Model",
        "n": "Columns",
        "auc": "AUC",
        "interval": "95% interval",
        "ap": "Avg. precision",
        "brier": "Brier",
        "known": "Known before the review",
        "yes": "yes",
        "no": "no",
        "nothing": "nothing (the share seen in training)",
        "to": "to",
        "p_auc": "AUC of the first pass",
        "p_mean": "Mean when results are moved",
        "p_sd": "Spread (sd)",
        "p_crit": "95th percentile",
        "p_p": "p-value",
        "p_n": "Moves",
        "w_shift": "Planted shift (sd)",
        "w_theory": "AUC of the column (theory)",
        "w_alone": "AUC of the column (measured)",
        "w_known": "AUC with everything known",
        "w_seen": "Seen by the primary test",
        "l_split": "Split",
        "l_random": "At random (ignores participants)",
        "l_grouped": "By participant",
        "l_auc": "AUC of boosting",
        "a_what": "Splits audited",
        "a_errors": "Splits with errors",
        "a_checks": "Checks that fired (splits)",
        "a_none": "none",
        "a_skipped": "skipped (--skip-audit)",
        "a_by": "By participant",
        "a_random": "At random",
    },
    "pt": {
        "columns": "Colunas que o modelo le",
        "model": "Modelo",
        "n": "Colunas",
        "auc": "AUC",
        "interval": "Intervalo de 95%",
        "ap": "Precisao media",
        "brier": "Brier",
        "known": "Conhecidas antes da revisao",
        "yes": "sim",
        "no": "nao",
        "nothing": "nada (a proporcao vista no treino)",
        "to": "a",
        "p_auc": "AUC da primeira passada",
        "p_mean": "Media com resultados trocados",
        "p_sd": "Dispersao (dp)",
        "p_crit": "Percentil 95",
        "p_p": "Valor p",
        "p_n": "Trocas",
        "w_shift": "Deslocamento plantado (dp)",
        "w_theory": "AUC da coluna (teoria)",
        "w_alone": "AUC da coluna (medido)",
        "w_known": "AUC com tudo que se sabe",
        "w_seen": "Vista pelo teste principal",
        "l_split": "Divisao",
        "l_random": "Ao acaso (ignora participantes)",
        "l_grouped": "Por participante",
        "l_auc": "AUC do boosting",
        "a_what": "Divisoes auditadas",
        "a_errors": "Divisoes com erros",
        "a_checks": "Verificacoes que dispararam (divisoes)",
        "a_none": "nenhuma",
        "a_skipped": "nao feita (--skip-audit)",
        "a_by": "Por participante",
        "a_random": "Ao acaso",
    },
}

FEATURE_NAMES_PT = {
    "acquisition": "aquisicao",
    "+ session": "+ sessao",
    "+ device": "+ equipamento",
    "+ participant": "+ participante",
    "+ signal": "+ sinal",
    "everything known before the review": "tudo que se sabe antes da revisao",
    "only what the review produces": "so o que a revisao produz",
    "everything known before + what the review produces": "tudo antes da revisao + o que a revisao produz",
}


def _table(header: list[str], rows: list[list[str]], right: set[int]) -> str:
    align = ["---:" if i in right else "---" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _interval(low: float, high: float, words: dict[str, str]) -> str:
    return f"{low:.3f} {words['to']} {high:.3f}"


def evaluation(results: dict[str, Any], lang: str = "en") -> str:
    t = TEXT[lang]
    header = [t["columns"], t["model"], t["n"], t["auc"], t["interval"], t["ap"], t["brier"], t["known"]]
    rows = []
    for row in results["rows"]:
        name = row["features"]
        if name == "nothing":
            name = t["nothing"]
        elif lang == "pt":
            name = FEATURE_NAMES_PT.get(name, name)
        low, high = row["auc_ci95"]
        rows.append(
            [
                name,
                row["model"],
                str(row["n_features"]),
                f"{row['auc']:.3f}",
                _interval(low, high, t),
                f"{row['average_precision']:.3f}",
                f"{row['brier']:.3f}",
                t["yes"] if row["known_before_review"] else t["no"],
            ]
        )
    return _table(header, rows, {2, 3, 5, 6})


def primary(results: dict[str, Any], lang: str = "en") -> str:
    t = TEXT[lang]
    p = results.get("primary")
    if p is None:
        return ""
    header = [t["p_auc"], t["p_mean"], t["p_sd"], t["p_crit"], t["p_p"], t["p_n"]]
    row = [
        f"{p['auc_first_pass']:.3f}",
        f"{p['null_mean']:.3f}",
        f"{p['null_sd']:.3f}",
        f"{p['critical_95']:.3f}",
        f"{p['p_value']:.3f}",
        str(p["permutations"]),
    ]
    return _table(header, [row], set(range(6)))


def power(results: dict[str, Any], lang: str = "en") -> str:
    t = TEXT[lang]
    section = results.get("controls", {}).get("power")
    if section is None:
        return ""
    header = [t["w_shift"], t["w_theory"], t["w_alone"], t["w_known"], t["w_seen"]]
    rows = [
        [
            f"{p['shift']:.1f}",
            f"{p['auc_theory']:.3f}",
            f"{p['auc_alone']:.3f}",
            f"{p['auc_with_known']:.3f}",
            f"{round(100 * p['detected'])}% ({p['replicates']})",
        ]
        for p in section["points"]
    ]
    return _table(header, rows, set(range(5)))


def leak(results: dict[str, Any], lang: str = "en") -> str:
    t = TEXT[lang]
    section = results.get("controls", {}).get("identity_leak")
    if section is None:
        return ""
    header = [t["l_split"], t["l_auc"]]
    rows = [
        [t["l_random"], f"{section['auc_random']:.3f}"],
        [t["l_grouped"], f"{section['auc_grouped']:.3f}"],
    ]
    return _table(header, rows, {1})


def audit(results: dict[str, Any], lang: str = "en") -> str:
    t = TEXT[lang]
    main = results.get("audit", {})
    random = results.get("controls", {}).get("random_split_audit")
    header = ["", t["a_what"], t["a_errors"], t["a_checks"]]

    def line(label: str, section: dict[str, Any]) -> list[str]:
        checks = ", ".join(f"{name} ({n})" for name, n in section["checks"].items()) or t["a_none"]
        return [label, str(section["splits"]), str(section["splits_with_errors"]), checks]

    if main.get("skipped"):
        return _table(header, [[t["a_by"], t["a_skipped"], "", ""]], {1, 2})
    rows = [line(t["a_by"], main)]
    if random is not None:
        rows.append(line(t["a_random"], random))
    return _table(header, rows, {1, 2})


RENDERERS = {"evaluation": evaluation, "primary": primary, "power": power, "leak": leak, "audit": audit}


def render(results: dict[str, Any], sections: tuple[str, ...] = SECTIONS, lang: str = "en") -> str:
    """The tables of ``sections`` (those that the results have), one after another."""
    parts = []
    for name in sections:
        text = RENDERERS[name](results, lang)
        if text:
            parts.append(text)
    return "\n".join(parts)
