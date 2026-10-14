"""Decorative SVG artwork for the UI: paddy-field silhouettes and the paddy loader."""

import random
from functools import lru_cache
from urllib.parse import quote


@lru_cache
def paddy_field(width: int, height: int, count: int, color: str, grain: str, alpha: float = 1.0, seed: int = 7) -> str:
    """A row of paddy stalks (some with drooping grain heads) as a CSS-ready SVG data URI."""
    rnd = random.Random(seed)  # fixed seed -> identical artwork on every rerun
    parts = []
    for _ in range(count):
        x = rnd.uniform(0, width)
        stalk_h = rnd.uniform(0.4, 1.0) * height
        lean = rnd.uniform(-0.2, 0.2) * stalk_h
        tip_x, tip_y = x + lean, height - stalk_h
        parts.append(
            f"<path d='M{x:.1f} {height} Q{x + lean * 0.15:.1f} {height - stalk_h * 0.55:.1f} {tip_x:.1f} {tip_y:.1f}' "
            f"stroke='{color}' stroke-width='{rnd.uniform(1.2, 2.6):.1f}' fill='none' stroke-linecap='round' "
            f"opacity='{rnd.uniform(0.35, 0.95):.2f}'/>"
        )
        if rnd.random() < 0.35:  # drooping panicle of grains, bending the way the stalk leans
            side = 1 if lean >= 0 else -1
            for k in range(6):
                gx, gy = tip_x + side * k * 2.6, tip_y + k * k * 0.75
                parts.append(f"<ellipse cx='{gx:.1f}' cy='{gy:.1f}' rx='1.5' ry='3.1' fill='{grain}' "
                             f"transform='rotate({side * (20 + k * 9)} {gx:.1f} {gy:.1f})'/>")
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
           f"viewBox='0 0 {width} {height}'><g opacity='{alpha}'>{''.join(parts)}</g></svg>")
    return "data:image/svg+xml," + quote(svg)


def _loader_stalk(x: float) -> str:
    grains = "".join(
        f"<ellipse class='grain' cx='{x + 2 + k * 2.4:.1f}' cy='{7 + k * k * 0.8:.1f}' rx='1.6' ry='3' "
        f"transform='rotate({25 + k * 10} {x + 2 + k * 2.4:.1f} {7 + k * k * 0.8:.1f})'/>"
        for k in range(5)
    )
    return f"<g class='stalk'><path d='M{x} 34 Q{x - 1} 20 {x + 1} 6'/>{grains}</g>"


def loader_html(stalks: int = 5) -> str:
    """Centered card where paddy stalks rise one after another, left to right (like loading dots).
    Shown via CSS only while Streamlit is running."""
    width = 12 * stalks + 4
    body = "".join(_loader_stalk(6 + 12 * i) for i in range(stalks))
    return (f"<div class='paddy-loader' role='status' aria-label='Loading'>"
            f"<svg viewBox='0 0 {width} 36' width='{width * 1.6:.0f}' height='{36 * 1.6:.0f}'>"
            f"{body}<path class='soil' d='M2 34.5 H{width - 2}'/></svg></div>")
