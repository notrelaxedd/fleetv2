"""Server-side SVG for the Models screen: the "Growth of $100" chart and the small trend lines.

Pure functions (numbers in, an SVG string out), no libraries. The SVGs use a viewBox, so
they scale to the width of their box; the style sheet sets their text size per screen width.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from html import escape
from typing import Any, Sequence

ACCENT = "#5EE0A8"
MUTED = "#93A0AD"
GRID = "#252D36"
WIDTH, HEIGHT = 640, 300
LEFT, RIGHT, TOP, BOTTOM = 58, 14, 12, 28
SPARK_W, SPARK_H = 96, 28
X_TICKS = 4
Y_TICKS = 5


def _num(value: float) -> str:
    """A short number for path data."""
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _points(values: Sequence[float | None]) -> list[tuple[int, float]]:
    """(index, value) for the values that exist: None and NaN are skipped."""
    out = []
    for i, v in enumerate(values):
        if v is None:
            continue
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            continue
        out.append((i, v))
    return out


def sparkline(values: Sequence[float | None], width: int = SPARK_W, height: int = SPARK_H) -> str:
    """A small trend line: one accent polyline, no axes. Fewer than two points gives ''."""
    pts = _points(values)
    if len(pts) < 2:
        return ""
    ys = [v for _, v in pts]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    pad = 2.0
    last = len(pts) - 1
    coords = []
    for k, (_, v) in enumerate(pts):
        x = pad + (width - 2 * pad) * k / last
        y = height - pad - (height - 2 * pad) * ((v - lo) / span if hi != lo else 0.5)
        coords.append(f"{_num(x)},{_num(y)}")
    return (f'<svg class="spark" data-spark viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'aria-hidden="true" focusable="false">'
            f'<polyline points="{" ".join(coords)}" fill="none" stroke="{ACCENT}" stroke-width="1.6" '
            f'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')


def nice_ticks(lo: float, hi: float, count: int = Y_TICKS) -> tuple[list[float], float]:
    """Round tick values inside [lo, hi] (steps of 1, 2, 2.5 or 5 times a power of ten) and the step."""
    span = hi - lo
    if span <= 0:
        return [lo], 1.0
    raw = span / max(1, count - 1)
    power = 10 ** math.floor(math.log10(raw))
    step = power
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * power
        if step >= raw:
            break
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    return [round(k * step, 6) for k in range(first, last + 1)], step


def money(value: float, step: float = 1.0) -> str:
    """$100, $1,250, $102.5"""
    if float(step).is_integer():
        return f"${value:,.0f}"
    return f"${value:,.1f}" if step >= 0.1 else f"${value:,.2f}"


def date_label(epoch: float, long_span: bool = True) -> str:
    """"Jan 2025" (or "Jan 5, 2025" when the whole chart covers only a few months)."""
    d = datetime.fromtimestamp(float(epoch), timezone.utc)
    return d.strftime("%b %Y") if long_span else f"{d.strftime('%b')} {d.day}, {d.year}"


def empty_chart(message: str = "No chart yet: there is not enough history to draw one.",
                width: int = WIDTH, height: int = HEIGHT) -> str:
    return (f'<svg class="chart" data-chart="growth" data-empty="true" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{escape(message)}">'
            f'<text x="{width / 2:g}" y="{height / 2:g}" text-anchor="middle" fill="{MUTED}" font-size="14">'
            f'{escape(message)}</text></svg>')


def _path(pts: list[tuple[float, float]]) -> str:
    return " ".join(("M" if k == 0 else "L") + f"{_num(x)} {_num(y)}" for k, (x, y) in enumerate(pts))


def growth_chart(t: Sequence[float], model: Sequence[float | None], benchmark: Sequence[float | None] | None = None,
                 model_label: str = "Model", benchmark_label: str = "Buy and hold",
                 title: str = "Growth of $100", width: int = WIDTH, height: int = HEIGHT) -> str:
    """The Growth of $100 line chart: the model as a solid accent line, buy and hold as a dashed
    muted line, a thin reference line at $100, a few $ values down the side and dates along the
    bottom. Missing (None) points are skipped; with nothing to draw it returns a friendly empty SVG."""
    benchmark = list(benchmark or [])
    model = list(model or [])
    t = list(t or [])
    n = len(t)
    model_pts = [(i, v) for i, v in _points(model) if i < n]
    bench_pts = [(i, v) for i, v in _points(benchmark) if i < n]
    if len(model_pts) < 2:
        return empty_chart(width=width, height=height)

    used = [i for i, _ in model_pts] + [i for i, _ in bench_pts]
    t0, t1 = float(t[min(used)]), float(t[max(used)])
    if t1 <= t0:
        return empty_chart(width=width, height=height)
    values = [v for _, v in model_pts] + [v for _, v in bench_pts] + [100.0]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.06 or 5.0
    lo, hi = lo - pad, hi + pad
    ticks, step = nice_ticks(lo, hi)
    if 100.0 not in ticks:
        ticks = sorted(t_ for t_ in ticks if abs(t_ - 100.0) > step * 0.45) + [100.0]
        ticks.sort()

    plot_w, plot_h = width - LEFT - RIGHT, height - TOP - BOTTOM

    def x_of(epoch: float) -> float:
        return LEFT + plot_w * (float(epoch) - t0) / (t1 - t0)

    def y_of(value: float) -> float:
        return TOP + plot_h * (1 - (value - lo) / (hi - lo))

    parts = [f'<svg class="chart" data-chart="growth" viewBox="0 0 {width} {height}" role="img" '
             f'aria-labelledby="chart-title chart-desc">',
             f'<title id="chart-title">{escape(title)}</title>',
             f'<desc id="chart-desc">{escape(model_label)} (solid line)'
             + (f' against {escape(benchmark_label)} (dashed line)' if bench_pts else "")
             + f', starting from $100, {date_label(t0, False)} to {date_label(t1, False)}.</desc>']

    parts.append('<g class="grid" data-axis="y">')
    for v in ticks:
        y = y_of(v)
        if v != 100.0:
            parts.append(f'<line class="gridline" x1="{LEFT}" x2="{width - RIGHT}" y1="{_num(y)}" y2="{_num(y)}" '
                         f'stroke="{GRID}" stroke-width="1" vector-effect="non-scaling-stroke"/>')
        parts.append(f'<text class="ylabel" data-tick="{v:g}" x="{LEFT - 8}" y="{_num(y)}" text-anchor="end" '
                     f'dominant-baseline="central" fill="{MUTED}" font-size="12">{money(v, step)}</text>')
    parts.append("</g>")

    long_span = (t1 - t0) >= 120 * 86400
    parts.append('<g class="axis" data-axis="x">')
    for k in range(X_TICKS):
        frac = k / (X_TICKS - 1)
        anchor = "start" if k == 0 else "end" if k == X_TICKS - 1 else "middle"
        x = LEFT + plot_w * frac
        parts.append(f'<text class="xlabel" x="{_num(x)}" y="{height - 8}" text-anchor="{anchor}" fill="{MUTED}" '
                     f'font-size="12">{date_label(t0 + (t1 - t0) * frac, long_span)}</text>')
    parts.append("</g>")

    y100 = y_of(100.0)
    parts.append(f'<line class="ref" data-ref="100" x1="{LEFT}" x2="{width - RIGHT}" y1="{_num(y100)}" y2="{_num(y100)}" '
                 f'stroke="{MUTED}" stroke-opacity=".55" stroke-width="1" vector-effect="non-scaling-stroke"/>')

    if bench_pts:
        d = _path([(x_of(t[i]), y_of(v)) for i, v in bench_pts])
        parts.append(f'<path class="series benchmark" data-series="benchmark" data-label="{escape(benchmark_label)}" '
                     f'd="{d}" fill="none" stroke="{MUTED}" stroke-width="1.6" stroke-dasharray="6 5" '
                     f'stroke-linejoin="round" vector-effect="non-scaling-stroke"><title>{escape(benchmark_label)}</title></path>')
    d = _path([(x_of(t[i]), y_of(v)) for i, v in model_pts])
    parts.append(f'<path class="series model" data-series="model" data-label="{escape(model_label)}" d="{d}" '
                 f'fill="none" stroke="{ACCENT}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" '
                 f'vector-effect="non-scaling-stroke"><title>{escape(model_label)}</title></path>')
    parts.append("</svg>")
    return "".join(parts)


def chart_for(chart: dict[str, Any] | None) -> str:
    """The SVG for models_view's `selected.chart` dict."""
    chart = chart or {}
    return growth_chart(chart.get("t") or [], chart.get("model") or [], chart.get("benchmark") or [],
                        chart.get("model_label") or "Model", chart.get("benchmark_label") or "Buy and hold",
                        chart.get("title") or "Growth of $100")
