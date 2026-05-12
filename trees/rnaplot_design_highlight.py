#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


TEXT_RE = re.compile(
    r'(?P<indent>\s*)<text class="nucleotide" x="(?P<x>-?\d+(?:\.\d+)?)" y="(?P<y>-?\d+(?:\.\d+)?)">(?P<base>[^<]+)</text>'
)

ROLE_STYLES = {
    "null": ("#e5e7eb", "#9ca3af", 0.78),
    "unpaired": ("#fef3c7", "#d97706", 0.84),
    "loop": ("#fef3c7", "#d97706", 0.84),
    "left": ("#dbeafe", "#2563eb", 0.84),
    "right": ("#dcfce7", "#16a34a", 0.84),
    "paired": ("#e0f2fe", "#0284c7", 0.84),
}
BASE_CONSTRAINT_FILL = "#c026d3"
BASE_CONSTRAINT_STROKE = "#111827"


def restyle_rnaplot_svg(text: str) -> str:
    """Make ViennaRNA RNAplot diagrams easier to read under design-target bubbles."""
    text = re.sub(
        r"\.nucleotide\s*\{[^}]*\}",
        """.nucleotide {
        font-family: Arial, Helvetica, DejaVu Sans, Liberation Sans, sans-serif;
        font-size: 8.0px;
        font-weight: 650;
        text-anchor: middle;
        dominant-baseline: central;
      }""",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\.backbone\s*\{[^}]*\}",
        """.backbone {
        stroke: #9ca3af;
        fill: none;
        stroke-width: 1.05;
        stroke-linecap: round;
        stroke-linejoin: round;
        opacity: 0.88;
      }""",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\.basepairs\s*\{[^}]*\}",
        """.basepairs {
        stroke: #dc2626;
        fill: none;
        stroke-width: 1.65;
        stroke-linecap: round;
        opacity: 0.92;
      }""",
        text,
        flags=re.S,
    )
    return text


def highlight_svg_bases(
    svg_path: Path,
    highlights: dict[int, str],
    *,
    draw_null_bubbles: bool = True,
    base_constraint_indices: set[int] | None = None,
    label_overrides: dict[int, str] | None = None,
) -> None:
    """Add design-target markers to a ViennaRNA RNAplot SVG.

    Indices in ``highlights`` are zero-based sequence positions. Positions not
    present in ``highlights`` are drawn as neutral gray "null target" bubbles
    when ``draw_null_bubbles`` is true.
    """
    text = restyle_rnaplot_svg(svg_path.read_text(encoding="utf-8"))
    matches = list(TEXT_RE.finditer(text))
    if not matches:
        raise RuntimeError(f"could not find nucleotide labels in {svg_path}")

    base_constraint_indices = base_constraint_indices or set()
    label_overrides = label_overrides or {}
    for index in set(highlights) | base_constraint_indices | set(label_overrides):
        if index < 0 or index >= len(matches):
            raise ValueError(f"highlight index {index} is outside sequence length {len(matches)} for {svg_path}")

    bubble_indices = range(len(matches)) if draw_null_bubbles else sorted(highlights)
    # RNAplot stores nucleotide labels in a translated group so ordinary text
    # appears visually near each backbone vertex. We instead center the labels
    # ourselves, so both bubbles and labels should use the raw vertex coordinates.
    circles: list[str] = ['    <g id="design-target-bubbles">']
    for index in bubble_indices:
        role = highlights.get(index, "null")
        match = matches[index]
        fill, stroke, opacity = ROLE_STYLES.get(role, ROLE_STYLES["paired"])
        x = match.group("x")
        y = match.group("y")
        circles.extend(
            [
                f'      <circle cx="{x}" cy="{y}" r="5.1" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="0.85" opacity="{opacity:.2f}">',
                f"        <title>design target: position {index + 1}, {role}</title>",
                "      </circle>",
            ]
        )
    circles.append("    </g>")

    old_marker = '    <g transform="translate(-4.6, 4)" id="seq">'
    new_marker = '    <g id="seq">'
    if old_marker not in text and new_marker not in text:
        raise RuntimeError(f"could not find sequence group in {svg_path}")
    if old_marker in text:
        text = text.replace(old_marker, f"{chr(10).join(circles)}\n{new_marker}", 1)
    else:
        text = text.replace(new_marker, f"{chr(10).join(circles)}\n{new_marker}", 1)

    selected = set(highlights)
    position_counter = -1

    def style_text(match: re.Match[str]) -> str:
        nonlocal position_counter
        position_counter += 1
        has_base_constraint = position_counter in base_constraint_indices
        weight = "800" if has_base_constraint else "650"
        fill = BASE_CONSTRAINT_FILL if has_base_constraint else "#374151"
        label = label_overrides.get(position_counter, match.group("base"))
        return (
            f'{match.group("indent")}<text class="nucleotide" x="{match.group("x")}" y="{match.group("y")}" '
            f'text-anchor="middle" dominant-baseline="central" style="font-weight:{weight}; fill:{fill}; '
            f'paint-order:stroke; stroke:{BASE_CONSTRAINT_STROKE if has_base_constraint else "none"}; '
            f'stroke-width:{0.45 if has_base_constraint else 0}; stroke-linejoin:round">'
            f'{label}</text>'
        )

    text = TEXT_RE.sub(style_text, text)
    svg_path.write_text(text, encoding="utf-8")
