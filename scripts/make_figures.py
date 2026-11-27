"""Draw the three figures of the README from results/*.json, as SVG, with nothing but the standard library.

    py scripts/make_figures.py            writes docs/img/auc.svg, docs/img/power.svg and docs/img/leak.svg
    py scripts/make_figures.py --check    draws them again and fails if the files in docs/img are not what comes out

The colours are the first three slots of a categorical palette that passes the colour-blind and contrast checks in
both modes (blue, orange, aqua); the SVG switches to the dark values when the reader's system is dark. Every number
that is drawn is also in a table of the README, so nothing is only in the picture.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "img"
SIZES = ("small", "medium", "large")

STYLE = """
  :root {
    --surface:#fcfcfb; --text:#0b0b0b; --text2:#52514e; --grid:#e3e2dd; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface:#1a1a19; --text:#f0efec; --text2:#c3c2b7; --grid:#33332f; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
    }
  }
  text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--text2); font-size: 12px; }
  .title { fill: var(--text); font-size: 15px; font-weight: 600; }
  .sub { font-size: 12px; }
  .row { fill: var(--text); }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .ref { stroke: var(--text2); stroke-width: 1; }
  .surface { fill: var(--surface); }
  .k1 { fill: var(--s1); } .k2 { fill: var(--s2); } .k3 { fill: var(--s3); }
  .l1 { stroke: var(--s1); } .l2 { stroke: var(--s2); } .l3 { stroke: var(--s3); }
  .ln { stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; }
  .dot { stroke: var(--surface); stroke-width: 2; }
"""


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Canvas:
    def __init__(self, width: int, height: int, title: str, desc: str) -> None:
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-labelledby="t d">',
            f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>',
            f"<style>{STYLE}</style>",
            f'<rect class="surface" width="{width}" height="{height}"/>',
        ]

    def text(self, x: float, y: float, s: str, cls: str = "", anchor: str = "start") -> None:
        extra = f' class="{cls}"' if cls else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>{esc(s)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str) -> None:
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>')

    def path(self, points: list[tuple[float, float]], cls: str) -> None:
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
        self.parts.append(f'<path d="{d}" class="ln {cls}"/>')

    def dot(self, x: float, y: float, cls: str, r: float = 5) -> None:
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="dot {cls}"/>')

    def svg(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def scale(value: float, low: float, high: float, start: float, end: float) -> float:
    return start + (value - low) / (high - low) * (end - start)


Results = dict[str, Any]


def axis(low: float, high: float, start: float, end: float) -> Callable[[float], float]:
    """Where, in the drawing (``start`` to ``end``), a value of the data (``low`` to ``high``) goes."""

    def position(value: float) -> float:
        return scale(value, low, high, start, end)

    return position


def load(size: str) -> Results:
    results: Results = json.loads((RESULTS / f"{size}.json").read_text(encoding="utf-8"))
    return results


ROW_LABELS = {
    "nothing": "prior: the share seen in training",
    "acquisition": "acquisition",
    "+ session": "+ session",
    "+ device": "+ device",
    "+ participant": "+ participant",
    "+ signal": "+ signal (everything known before)",
    "everything known before the review": "boosting, everything known before",
    "only what the review produces": "only what the review produces",
    "everything known before + what the review produces": "known before + review columns",
}


def nice_range(
    values: list[float], pad: float, step: float, include: float | None = None, ceiling: float = 1.0
) -> tuple[float, float]:
    """A range that holds ``values`` (and ``include``) with some room, whose ends are multiples of ``step``.

    An AUC cannot be above 1, so the range stops there.
    """
    everything = [*values, include] if include is not None else values
    low = math.floor(round((min(everything) - pad) / step, 6)) * step
    high = math.ceil(round((max(everything) + pad) / step, 6)) * step
    return round(low, 6), round(min(high, ceiling), 6)


def ticks(low: float, high: float, step: float) -> list[float]:
    count = round((high - low) / step)
    return [round(low + i * step, 6) for i in range(count + 1)]


AUC_PANELS = (
    # name, class of the marks, rows that belong to it, room around the values, step of the ticks, reference line
    ("known before the review", "1", True, 0.01, 0.02, 0.5),
    ("produced by the review itself", "2", False, 0.005, 0.01, None),
)


def auc_figure(results: Results, size: str) -> str:
    """Two panels with their own axes: what is known before the review is all within a hair of 0.5, and what the review
    produces is near 1, so on one axis the first would be a smudge around the reference line."""
    table = results["table"]
    left, right, step, width = 300, 40, 28, 760
    # first work out where everything goes, then draw: the canvas needs its height before the first mark
    layout = []
    top = 62
    for name, cls, known, pad, tick_step, reference in AUC_PANELS:
        rows = [r for r in results["rows"] if r["known_before_review"] == known]
        first_row = top + 36
        bottom = first_row + step * (len(rows) - 1) + 14
        layout.append((name, cls, rows, pad, tick_step, reference, first_row, bottom))
        top = bottom + 28
    height = layout[-1][-1] + 18 + 20
    canvas = Canvas(
        width,
        height,
        "AUC of each set of columns, with its 95% interval",
        f"For {table['participants']} participants: every set of columns known before the review is near 0.5, the AUC "
        "of a coin; the columns that the review itself produces are near 1.",
    )
    canvas.text(24, 28, "What predicts the result of the review, for participants never seen", "title")
    canvas.text(
        24,
        48,
        f"AUC and 95% interval, {table['recordings']:,} recordings of {table['participants']:,} participants",
        "sub",
    )
    for name, cls, rows, pad, tick_step, reference, first_row, bottom in layout:
        low, high = nice_range([v for r in rows for v in (r["auc"], *r["auc_ci95"])], pad, tick_step, reference)
        x = axis(low, high, left, width - right)
        header_y = first_row - 20
        canvas.dot(30, header_y - 4, f"k{cls}", r=5)
        canvas.text(44, header_y, name, "row")
        for tick in ticks(low, high, tick_step):
            if reference is None or abs(tick - reference) > 1e-9:
                canvas.line(x(tick), first_row - 14, x(tick), bottom, "grid")
            canvas.text(x(tick), bottom + 18, f"{tick:.2f}", anchor="middle")
        if reference is not None:
            canvas.line(x(reference), first_row - 14, x(reference), bottom, "ref")
            canvas.text(x(reference) + 6, header_y, f"{reference}: a coin")
        for i, row in enumerate(rows):
            y = first_row + step * i
            label = ROW_LABELS.get(row["features"], row["features"])
            if row["model"] == "boosting" and row["features"] == "everything known before + what the review produces":
                label += ", boosting"
            canvas.text(left - 14, y + 4, label, "row", anchor="end")
            low_ci, high_ci = row["auc_ci95"]
            canvas.line(x(low_ci), y, x(high_ci), y, f"ln l{cls}")
            canvas.dot(x(row["auc"]), y, f"k{cls}")
    return canvas.svg()


def power_figure(all_results: dict[str, Results]) -> str:
    width, height = 760, 430
    left, right, top, bottom = 64, 150, 82, 64
    low, high = 0.5, 0.72
    canvas = Canvas(
        width,
        height,
        "How often the primary test sees an effect that was planted",
        "The share of tables in which the primary test sees a planted column, against the AUC that column would have "
        "alone, for the three table sizes.",
    )
    canvas.text(24, 28, "What the test could have seen", "title")
    canvas.text(24, 48, "Share of tables in which the primary test sees a column planted with a known strength", "sub")
    x = axis(low, high, left, width - right)
    y = axis(0, 100, height - bottom, top)
    for percent in (0, 25, 50, 75, 100):
        canvas.line(left, y(percent), width - right, y(percent), "grid")
        canvas.text(left - 10, y(percent) + 4, f"{percent}%", anchor="end")
    for tick in (0.5, 0.55, 0.6, 0.65, 0.7):
        canvas.text(x(tick), height - bottom + 20, f"{tick:.2f}", anchor="middle")
    canvas.text(
        (left + width - right) / 2,
        height - bottom + 42,
        "AUC that the planted column would have on its own",
        anchor="middle",
    )
    canvas.line(left, y(80), width - right, y(80), "ref")
    canvas.text(width - right + 8, y(80) + 4, "80%: counts as seen")
    for i, size in enumerate(SIZES):
        points = all_results[size]["controls"]["power"]["points"]
        coords = [(x(p["auc_theory"]), y(100 * p["detected"])) for p in points]
        cls = str(i + 1)
        canvas.path(coords, f"l{cls}")
        for cx, cy in coords:
            canvas.dot(cx, cy, f"k{cls}", r=4)
    legend_x = left
    for i, size in enumerate(SIZES):
        participants = all_results[size]["table"]["participants"]
        cls = str(i + 1)
        canvas.dot(legend_x + 6, 66, f"k{cls}", r=4)
        canvas.text(legend_x + 18, 70, f"{participants:,} participants")
        legend_x += 170
    return canvas.svg()


def leak_figure(all_results: dict[str, Results]) -> str:
    width, height = 760, 300
    left, right, top, step = 170, 40, 92, 52
    low, high = 0.45, 0.75
    canvas = Canvas(
        width,
        height,
        "AUC of a model when the recordings are split at random and when they are split by participant",
        "With results that follow the participant, a split at random reports an AUC well above 0.5 and a split by "
        "participant does not, for the three table sizes.",
    )
    canvas.text(24, 28, "What a split that ignores participants reports", "title")
    canvas.text(24, 48, "Boosting on the columns known before the review; results that follow the participant", "sub")
    x = axis(low, high, left, width - right)
    plot_bottom = top + step * len(SIZES) - 18
    for tick in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        if abs(tick - 0.5) > 1e-9:
            canvas.line(x(tick), top - 18, x(tick), plot_bottom, "grid")
        canvas.text(x(tick), plot_bottom + 18, f"{tick:.2f}", anchor="middle")
    canvas.line(x(0.5), top - 18, x(0.5), plot_bottom, "ref")
    canvas.text(x(0.5), top - 24, "0.5: a coin", anchor="middle")
    for i, size in enumerate(SIZES):
        leak = all_results[size]["controls"]["identity_leak"]
        participants = all_results[size]["table"]["participants"]
        row_y = top + step * i
        canvas.text(left - 14, row_y + 4, f"{participants:,} participants", "row", anchor="end")
        canvas.line(x(leak["auc_grouped"]), row_y, x(leak["auc_random"]), row_y, "ln l2")
        canvas.dot(x(leak["auc_grouped"]), row_y, "k1")
        canvas.dot(x(leak["auc_random"]), row_y, "k2")
    legend_y = height - 14
    canvas.dot(left + 6, legend_y - 4, "k1")
    canvas.text(left + 18, legend_y, "split by participant")
    canvas.dot(left + 170, legend_y - 4, "k2")
    canvas.text(left + 182, legend_y, "split at random")
    return canvas.svg()


def figures() -> dict[str, str]:
    all_results = {size: load(size) for size in SIZES}
    return {
        "auc.svg": auc_figure(all_results["large"], "large"),
        "power.svg": power_figure(all_results),
        "leak.svg": leak_figure(all_results),
    }


def main(argv: list[str]) -> int:
    drawn = figures()
    if "--check" in argv:
        stale = [
            name
            for name, text in drawn.items()
            if not (OUT / name).is_file() or (OUT / name).read_text(encoding="utf-8") != text
        ]
        for name in stale:
            print(f"docs/img/{name} is not what results/*.json draws: run py scripts/make_figures.py", file=sys.stderr)
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in drawn.items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"docs/img/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
