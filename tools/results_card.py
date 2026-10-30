"""Draw the evaluation card in docs/results/ from the latest runs in evals/results/.

    uv run python tools/results_card.py

Reads the newest map-first and vector runs and writes results-light.svg and results-dark.svg, so the
card in the README is always the numbers of a real run, never typed by hand.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
OUT = ROOT / "docs" / "results"

SANS = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace"

THEMES = {
    "light": {
        "bg": "#F4F7F5", "stroke": "#D3DFD8", "text": "#0D262B", "muted": "#5B6F6A",
        "track": "#E1E9E4", "vector": "#A3B3AD", "a": "#2F7A5A", "b": "#5FB38A",
    },
    "dark": {
        "bg": "#0D262B", "stroke": "#1F4047", "text": "#E4EBE7", "muted": "#8FA5A0",
        "track": "#163339", "vector": "#5E7772", "a": "#2F7A5A", "b": "#8FD1AD",
    },
}  # fmt: skip

# (summary key, label, unit, higher is better)
METRICS = [
    ("correct", "Correct, by the calibrated judge", "%", True),
    ("retrieval_recall", "Read the notes the answer needs", "%", True),
    ("citation_recall", "Cited the notes the answer needs", "%", True),
    ("found_matches_expected", "Answered, or declined, when it should", "%", True),
    ("tokens", "Tokens per question", "", False),
    ("seconds", "Seconds per question", "s", False),
]


def latest(prefix: str) -> dict:
    runs = sorted(RESULTS.glob(f"{prefix}-*.json"))
    if not runs:
        raise SystemExit(f"no {prefix} run in {RESULTS}")
    return json.loads(runs[-1].read_text(encoding="utf-8"))


def fmt(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value * 100:.1f}%"
    if unit == "s":
        return f"{value:.1f} s"
    return f"{value:,.0f}"


def card(theme: dict, runs: dict[str, dict]) -> str:
    t = theme
    width, top, row_h = 960, 168, 58
    height = top + row_h * len(METRICS) + 92
    stamp = runs["map"]["run_name"].split("-")[-2]  # map-<model>-YYYYMMDD-HHMM
    when = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
    items = runs["map"]["items"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Map-first vs vector retrieval">',
        "<title>Map-first vs vector retrieval</title>",
        '<defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="0">'
        f'<stop offset="0" stop-color="{t["a"]}"/><stop offset="1" stop-color="{t["b"]}"/>'
        "</linearGradient></defs>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="20" '
        f'fill="{t["bg"]}" stroke="{t["stroke"]}"/>',
        f'<text x="40" y="52" font-family="{SANS}" font-size="11" letter-spacing="2.2" '
        f'fill="{t["muted"]}">SECOND-BRAIN-AGENT · EVALUATION</text>',
        f'<text x="40" y="88" font-family="{SANS}" font-size="30" font-weight="700" '
        f'letter-spacing="-0.5" fill="{t["text"]}">Map-first vs vector retrieval</text>',
        f'<text x="40" y="112" font-family="{SANS}" font-size="13" fill="{t["muted"]}">'
        f"{items} questions · same prompts · agent {escape(runs['map']['model'])} · "
        f"{when:%d %b %Y} · Langfuse experiments</text>",
    ]
    # legend
    lx = 700
    out += [
        f'<rect x="{lx}" y="62" width="14" height="14" rx="4" fill="url(#g)"/>',
        f'<text x="{lx + 22}" y="74" font-family="{SANS}" font-size="13" font-weight="600" '
        f'fill="{t["text"]}">Map-first</text>',
        f'<rect x="{lx + 110}" y="62" width="14" height="14" rx="4" fill="{t["vector"]}"/>',
        f'<text x="{lx + 132}" y="74" font-family="{SANS}" font-size="13" font-weight="600" '
        f'fill="{t["text"]}">Vector</text>',
    ]
    bar_x, bar_w = 40, 640
    for i, (key, label, unit, higher) in enumerate(METRICS):
        y = top + i * row_h
        values = {name: float(run["summary"][key]) for name, run in runs.items()}
        scale = 1.0 if unit == "%" else max(values.values()) * 1.08
        better = max(values, key=values.get) if higher else min(values, key=values.get)
        hint = "" if higher else "  ·  lower is better"
        out.append(
            f'<text x="{bar_x}" y="{y}" font-family="{SANS}" font-size="14" font-weight="600" '
            f'fill="{t["text"]}">{escape(label)}<tspan font-weight="400" fill="{t["muted"]}" '
            f'font-size="12">{hint}</tspan></text>'
        )
        for j, name in enumerate(("map", "vector")):
            by = y + 10 + j * 14
            w = max(4.0, bar_w * values[name] / scale)
            fill = "url(#g)" if name == "map" else t["vector"]
            out += [
                f'<rect x="{bar_x}" y="{by}" width="{bar_w}" height="9" rx="4.5" '
                f'fill="{t["track"]}"/>',
                f'<rect x="{bar_x}" y="{by}" width="{w:.1f}" height="9" rx="4.5" fill="{fill}"/>',
            ]
            weight = "700" if name == better else "400"
            out.append(
                f'<text x="{width - 40}" y="{by + 9}" text-anchor="end" font-family="{MONO}" '
                f'font-size="13" font-weight="{weight}" fill="{t["text"]}">'
                f"{'map' if name == 'map' else 'vec'}  {fmt(values[name], unit)}</text>"
            )
    fy = top + row_h * len(METRICS) + 26
    out += [
        f'<line x1="40" x2="{width - 40}" y1="{fy - 18}" y2="{fy - 18}" stroke="{t["stroke"]}"/>',
        f'<text x="40" y="{fy + 6}" font-family="{SANS}" font-size="13" fill="{t["muted"]}">'
        f'<tspan font-weight="600" fill="{t["text"]}">Judge calibrated first.</tspan> '
        "Agreement with hand labels went from 65.1% (prompt v1) to 97.7% (v2) "
        "before any of these scores were trusted.</text>",
        f'<text x="40" y="{fy + 28}" font-family="{SANS}" font-size="13" fill="{t["muted"]}">'
        "Vector misses came from chunk boundaries: a two-note question and a client name "
        "split from its plan.</text>",
        "</svg>",
    ]
    return "\n".join(out)


def main() -> None:
    runs = {"map": latest("map"), "vector": latest("vector")}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (OUT / f"results-{name}.svg").write_text(card(theme, runs), encoding="utf-8")
    print(f"wrote {OUT / 'results-light.svg'} and results-dark.svg")


if __name__ == "__main__":
    main()
