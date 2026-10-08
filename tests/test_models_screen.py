"""The Models screen: ranked list, selected model, chart, metric cards, buttons, fragment, charts.py.

Tests look for data-* hooks (not CSS classes) in the rendered HTML."""
from __future__ import annotations

import copy
import re
from html import unescape

import pytest

from coordinator import charts, models, models_view, trading_view
from fleet2.models import REGISTRY
from fleet2.models.base import params_with_defaults
from fleet2.sim.backtest import Limits
from fleet2.worker.backtest_job import backtest_periods
from tests.test_dashboard import attr_tags, element, text
from tests.test_lookahead import synthetic

_REAL: dict | None = None


def real_metrics() -> dict:
    """A real {"train", "held_out", "split_t"} dict from a backtest on synthetic prices."""
    global _REAL
    if _REAL is None:
        module = REGISTRY["momentum"]
        data = synthetic("stocks", 700)
        _REAL = backtest_periods(data, module, params_with_defaults(module, None), Limits(), 0.3)
    return copy.deepcopy(_REAL)


def metrics_with(roi: float, trades: int) -> dict:
    m = real_metrics()
    m["held_out"].update(roi=roi, trades=trades, enough_trades=trades >= 100)
    return m


@pytest.fixture
def stored(conn):
    """momentum +20% (150 trades), dip_buy +5% (150), pairs +50% (12 trades: not enough), crypto_trend untested."""
    models.store_backtest(conn, "momentum", metrics_with(0.20, 150), None)
    models.store_backtest(conn, "dip_buy", metrics_with(0.05, 150), None)
    models.store_backtest(conn, "pairs", metrics_with(0.50, 12), None)
    return conn


def page_ids(html: str) -> list[str]:
    return re.findall(r'data-model="([^"]+)"', html)


def test_list_is_ranked_with_rank_and_signed_roi(client, stored):
    html = client.get("/models").text
    # enough trades first by ROI, then not enough trades, then not tested
    assert page_ids(html) == ["momentum", "dip_buy", "pairs", "crypto_trend"]
    assert re.findall(r'data-rank="(\d+)"', html) == ["1", "2", "3", "4"]
    rows = html.split("data-models-list", 1)[1]
    rois = [re.sub(r"\s+", " ", unescape(m)).strip() for m in re.findall(r"<[^>]*data-roi[^>]*>(.*?)</", rows, re.S)]
    assert rois == ["+20.0%", "+5.0%", "+50.0%", "-"]
    assert 'data-roi data-tone="gain"' in html


def test_negative_roi_has_a_minus_sign_and_loss_tone(client, conn):
    models.store_backtest(conn, "momentum", metrics_with(-0.123, 150), None)
    html = client.get("/models").text
    row = html[html.index('data-model="momentum"'):]
    assert re.search(r'data-roi data-tone="loss">\s*−12\.3%', row)


def test_row_shows_description_tags_and_a_trend_line(client, stored):
    html = client.get("/models").text
    row = html[html.index('data-model="momentum"'):html.index('data-model="dip_buy"')]
    assert REGISTRY["momentum"].DESCRIPTION in unescape(row)
    assert element(row, 'data-tag="market"') == "Stocks"
    assert element(row, 'data-tag="status"') == "Backtested"
    assert "data-tag=\"not-enough\"" not in row
    assert "<polyline" in row and "data-spark" in row
    untested = html[html.index('data-model="crypto_trend"'):]
    assert element(untested, 'data-tag="status"') == "Not tested yet"
    assert "<polyline" not in untested.split("</li>")[0]


def test_not_enough_trades_tag(client, stored):
    html = client.get("/models").text
    tags = attr_tags(html, 'data-tag="not-enough"')
    assert len(tags) == 1
    pairs = html[html.index('data-model="pairs"'):html.index('data-model="crypto_trend"')]
    assert element(pairs, 'data-tag="not-enough"') == "Not enough trades"
    # the selected model's own tags carry it too
    html = client.get("/models?id=pairs").text
    detail = html[html.index("data-model-tags"):html.index("data-how-it-works")]
    assert "Not enough trades" in text(detail)


def test_first_ranked_is_selected_by_default_and_id_selects(client, stored):
    html = client.get("/models").text
    assert re.findall(r'data-model="([^"]+)"[^>]*data-selected="true"', html) == ["momentum"]
    assert element(html, "data-model-name") == "Momentum"
    html = client.get("/models?id=dip_buy").text
    assert re.findall(r'data-model="([^"]+)"[^>]*data-selected="true"', html) == ["dip_buy"]
    assert element(html, "data-model-name") == "Dip buyer"
    assert 'data-selected-id="dip_buy"' in html
    assert 'href="/models?id=pairs"' in html
    # an unknown id falls back to the first
    assert element(client.get("/models?id=nope").text, "data-model-name") == "Momentum"


def test_header_is_shared_and_models_nav_is_current(client, stored):
    html = client.get("/models").text
    assert 'data-nav="models" class="current" aria-current="page"' in html
    assert 'data-nav="fleet" class="current"' not in html
    assert element(html, 'data-pill="mode"') == "Paper"
    assert element(html, 'data-action="pause"') == "Pause all trading"


def test_detail_has_name_tags_explanation_and_origin(client, stored):
    html = client.get("/models?id=momentum").text
    assert [text(t).strip() for t in re.findall(r'<span class="tag[^"]*" data-tag>(.*?)</span>', html)] == ["Stocks", "Backtested"]
    assert element(html, "data-how-it-works") == REGISTRY["momentum"].HOW_IT_WORKS.strip()
    assert element(html, "data-origin") == "Starter model"


def test_eight_metric_cards_with_labels_and_descriptions_word_for_word(client, stored):
    html = client.get("/models?id=momentum").text
    cards = attr_tags(html, "data-metric")
    assert [re.search(r'data-metric="(\w+)"', c).group(1) for c in cards] == [k for k, _, _ in models_view.METRICS]
    body = text(html)
    assert "What the numbers mean" in body
    for _, label, description in models_view.METRICS:
        assert label in body
        assert description in body
    cards_html = html[html.index("data-metrics"):]
    assert element(cards_html, 'data-metric="roi"') != ""
    roi_card = cards_html[cards_html.index('data-metric="roi"'):cards_html.index('data-metric="vs_buy_and_hold"')]
    assert element(roi_card, "data-value") == "+20.0%" and 'data-tone="gain"' in roi_card
    trades_card = cards_html[cards_html.index('data-metric="trades"'):]
    assert element(trades_card, "data-value") == "150"


def test_a_card_note_shows_only_when_there_is_one(client, stored):
    html = client.get("/models?id=pairs").text
    trades = html[html.index('data-metric="trades"'):html.index('data-metric="avg_hold_s"')]
    assert 'data-tone="warn"' in trades
    assert "Not enough trades: these results could be luck" in text(trades)
    roi = html[html.index('data-metric="roi"'):html.index('data-metric="vs_buy_and_hold"')]
    assert "data-note" not in roi


def test_chart_has_two_labelled_series_solid_and_dashed(client, stored):
    html = client.get("/models?id=momentum").text
    assert element(html, "data-chart-card").startswith("Growth of $100")
    model = attr_tags(html, 'data-series="model"')[0]
    bench = attr_tags(html, 'data-series="benchmark"')[0]
    assert "stroke-dasharray" not in model and "stroke-dasharray" in bench
    assert 'data-label="Momentum"' in model and 'data-label="Buy and hold SPY"' in bench
    assert element(html, 'data-legend="model"') == "Momentum"
    assert element(html, 'data-legend="benchmark"') == "Buy and hold SPY"
    assert "$100" in element(html, "data-chart-card")
    assert element(html, "data-period").startswith("Held-out period ")
    assert element(html, "data-training").startswith("Training period ROI ")
    assert 'viewBox="0 0 640 300"' in html


def test_crypto_model_chart_names_btc(client, conn):
    models.store_backtest(conn, "crypto_trend", metrics_with(0.1, 150), None)
    html = client.get("/models?id=crypto_trend").text
    assert element(html, 'data-legend="benchmark"') == "Buy and hold BTC"


def test_untested_model_shows_the_untested_message_and_no_chart_or_cards(client, stored):
    html = client.get("/models?id=crypto_trend").text
    assert element(html, "data-untested") == "Not tested yet. Press Run backtest to see how it would have done."
    assert "data-chart" not in html and "data-metric" not in html and "What the numbers mean" not in html


def test_buttons_and_their_hooks(client, stored):
    html = client.get("/models?id=momentum").text
    run = attr_tags(html, 'data-action="run-backtest"')
    assert len(run) == 1 and 'data-model-id="momentum"' in run[0]
    assert element(html, 'data-action="run-backtest"') == "Run backtest"
    paper = attr_tags(html, 'data-action="paper-start"')[0]
    assert "disabled" not in paper and 'data-model-id="momentum"' in paper
    assert element(html, 'data-action="paper-start"') == "Start paper trading"
    search = attr_tags(html, 'data-action="search-start"')
    assert len(search) == 1
    assert element(html, 'data-action="search-start"') == "Start model search"
    assert "data-paper-reason" not in html and "data-search-detail" not in html


def test_paper_button_is_disabled_with_a_reason_when_untested_or_retired(client, stored, conn):
    html = client.get("/models?id=crypto_trend").text
    assert "disabled" in attr_tags(html, 'data-action="paper-start"')[0]
    assert "Run a backtest first" in element(html, "data-paper-reason")
    conn.execute("UPDATE models SET status = 'retired' WHERE id = 'dip_buy'")
    html = client.get("/models?id=dip_buy").text
    assert "disabled" in attr_tags(html, 'data-action="paper-start"')[0]
    assert "retired" in element(html, "data-paper-reason")


def test_paper_trading_model_offers_stop_and_shows_the_paper_line(client, stored, conn, monkeypatch):
    conn.execute("UPDATE models SET status = 'paper_trading' WHERE id = 'momentum'")
    monkeypatch.setattr(trading_view, "paper_summaries",
                        lambda c: {"momentum": {"line": "Paper trading since Oct 1: +$120.00 (+1.2%)", "tone": "gain"}})
    html = client.get("/models?id=momentum").text
    assert element(html, 'data-action="paper-stop"') == "Stop paper trading"
    assert "disabled" not in attr_tags(html, 'data-action="paper-stop"')[0]
    assert "data-action=\"paper-start\"" not in html
    assert element(html, "data-paper-line") == "Paper trading since Oct 1: +$120.00 (+1.2%)"
    assert 'data-paper-line data-tone="gain"' in html
    # another model has no line
    assert "data-paper-line" not in client.get("/models?id=dip_buy").text


def test_search_button_text_and_detail_come_from_the_search_status(client, stored, monkeypatch):
    from coordinator import search
    monkeypatch.setattr(search, "search_status",
                        lambda c: {"running": True, "button": "Stop model search", "detail": "Tried 12 of 40 ideas"})
    html = client.get("/models").text
    assert element(html, 'data-action="search-stop"') == "Stop model search"
    assert "search-start" not in html
    assert element(html, "data-search-detail") == "Tried 12 of 40 ideas"


def test_no_models_shows_one_empty_state_line(client, conn):
    conn.execute("DELETE FROM models")
    html = client.get("/models").text
    assert element(html, 'data-empty="models"') == "No models yet."
    assert "data-model-detail" not in html and "data-models-list" not in html


def test_fragment_has_the_header_and_models_regions(client, stored):
    html = client.get("/fragments/models?id=dip_buy").text
    regions = re.findall(r'data-region="([\w-]+)"', html)
    assert regions == ["status", "banner", "models-search", "models-list", "model-head", "model-actions", "model-results"]
    assert 'data-pill="mode"' in html and 'data-action="pause"' in html
    assert re.findall(r'data-model="([^"]+)"[^>]*data-selected="true"', html) == ["dip_buy"]
    assert element(html, "data-model-name") == "Dip buyer"
    assert "<html" not in html


def test_fragment_shows_the_pause_banner(client, stored):
    client.post("/api/trading/pause")
    html = client.get("/fragments/models").text
    assert "data-banner" in html and "All trading is paused" in text(html)


def test_fragment_regions_match_the_page(client, stored):
    page = client.get("/models?id=momentum").text
    frag = client.get("/fragments/models?id=momentum").text
    for name in ("models-list", "model-head", "model-actions", "model-results"):
        live = re.search(rf'<div data-region="{name}">(.*?)</div>\s*(?=<(?:div|p|section|/section|/div))', page, re.S)
        assert live and name in frag


def test_page_loads_app_js_and_marks_the_screen(client, stored):
    html = client.get("/models").text
    assert "/static/app.js" in html and "data-models-screen" in html
    js = client.get("/static/app.js").text
    assert "/fragments/models" in js and "/api/search/" in js and "run-backtest" in js


# ---- charts.py -----------------------------------------------------------------------

T = [1_704_067_200 + i * 86_400 * 30 for i in range(12)]  # Jan 1 2024, then every 30 days
UP = [100 + 5 * i for i in range(12)]


def d_of(svg: str, series: str) -> str:
    m = re.search(rf'data-series="{series}"[^>]*\sd="([^"]+)"', svg)
    assert m, f"no {series} path"
    return m.group(1)


def test_growth_chart_draws_both_series_and_skips_nulls():
    bench = [100, None, 102, 104, None, 108, 110, None, 114, 116, 118, 120]
    svg = charts.growth_chart(T, UP, bench, "Momentum", "Buy and hold SPY")
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert d_of(svg, "model").count("L") == 11
    assert d_of(svg, "benchmark").count("L") == 8 and d_of(svg, "benchmark").count("M") == 1
    assert not re.search(r"\bnan\b", svg.lower()) and "None" not in svg
    assert 'stroke-dasharray' in re.search(r'<path[^>]*data-series="benchmark"[^>]*>', svg).group(0)
    assert "Momentum" in svg and "Buy and hold SPY" in svg


def test_growth_chart_without_a_benchmark_has_only_the_model_line():
    svg = charts.growth_chart(T, UP, [None] * 12, "Momentum", "Buy and hold SPY")
    assert 'data-series="model"' in svg and 'data-series="benchmark"' not in svg


def test_y_axis_includes_100_and_has_money_labels():
    svg = charts.growth_chart(T, [100 + 40 * i for i in range(12)], None)
    labels = re.findall(r'class="ylabel" data-tick="([\d.]+)"[^>]*>([^<]+)<', svg)
    assert 3 <= len(labels) <= 7
    values = [float(v) for v, _ in labels]
    assert 100.0 in values and min(values) <= 100 <= max(values)
    assert all(t.startswith("$") for _, t in labels)
    assert ('data-tick="100"' in svg) and ">$100<" in svg
    assert 'data-ref="100"' in svg


def test_y_axis_still_includes_100_when_everything_is_below_it():
    svg = charts.growth_chart(T, [100 - 3 * i for i in range(12)], [100 - i for i in range(12)])
    values = [float(v) for v in re.findall(r'data-tick="([\d.]+)"', svg)]
    assert 100.0 in values and min(values) < 100


def test_reference_line_sits_at_the_100_label():
    svg = charts.growth_chart(T, [90 + 3 * i for i in range(12)], None)
    ref_y = float(re.search(r'data-ref="100"[^>]*y1="([\d.]+)"', svg).group(1))
    label_y = float(re.search(r'data-tick="100"[^>]*y="([\d.]+)"', svg).group(1))
    assert ref_y == pytest.approx(label_y, abs=0.1)


def test_x_axis_dates_look_like_jan_2025():
    t0 = 1_735_689_600  # Jan 1 2025
    svg = charts.growth_chart([t0 + i * 86_400 * 30 for i in range(12)], UP, None)
    labels = re.findall(r'class="xlabel"[^>]*>([^<]+)<', svg)
    assert len(labels) == 4 and labels[0] == "Jan 2025"
    assert all(re.fullmatch(r"[A-Z][a-z]{2} \d{4}", lab) for lab in labels)


def test_a_short_period_uses_full_dates_so_labels_do_not_repeat():
    t0 = 1_735_689_600
    svg = charts.growth_chart([t0 + i * 86_400 * 3 for i in range(12)], UP, None)
    labels = re.findall(r'class="xlabel"[^>]*>([^<]+)<', svg)
    assert len(set(labels)) == len(labels) and labels[0] == "Jan 1, 2025"


@pytest.mark.parametrize("args", [([], [], []), ([T[0]], [100], [100]), (T, [None] * 12, [None] * 12)])
def test_empty_input_gives_a_friendly_empty_svg(args):
    svg = charts.growth_chart(*args)
    assert svg.startswith("<svg") and 'data-empty="true"' in svg
    assert "No chart yet" in svg and "<path" not in svg


def test_chart_labels_are_escaped():
    svg = charts.growth_chart(T, UP, UP, "<b>x</b>", "A & B")
    assert "<b>x</b>" not in svg and "&lt;b&gt;" in svg and "A &amp; B" in svg


def test_all_points_stay_inside_the_view_box():
    svg = charts.growth_chart(T, [100, 300, 20, 500, 100, 90, 100, 100, 100, 100, 100, 1000], UP)
    for d in (d_of(svg, "model"), d_of(svg, "benchmark")):
        for x, y in re.findall(r"[ML]([\d.]+) ([\d.]+)", d):
            assert 0 <= float(x) <= 640 and 0 <= float(y) <= 300


def test_nice_ticks_are_round_and_inside_the_range():
    ticks, step = charts.nice_ticks(93.0, 148.0)
    assert ticks == [100, 120, 140] or ticks[1] - ticks[0] == step
    assert all(93 <= t <= 148 for t in ticks)
    assert charts.money(1250) == "$1,250" and charts.money(102.5, 2.5) == "$102.5"


def test_sparkline_is_one_polyline_with_no_axes():
    svg = charts.sparkline([100, 102, 101, 105, 110])
    assert svg.count("<polyline") == 1 and "<line" not in svg and "<text" not in svg
    assert "#5EE0A8" in svg
    pts = re.search(r'points="([^"]+)"', svg).group(1).split()
    assert len(pts) == 5
    ys = [float(p.split(",")[1]) for p in pts]
    assert ys[-1] == min(ys) and ys[0] == max(ys)  # the highest value is drawn highest (smallest y)


def test_sparkline_skips_nulls_and_handles_flat_and_short_input():
    assert len(re.search(r'points="([^"]+)"', charts.sparkline([1, None, 3, float("nan"), 5])).group(1).split()) == 3
    flat = charts.sparkline([100, 100, 100])
    assert not re.search(r"\bnan\b", flat.lower())
    assert charts.sparkline([]) == "" and charts.sparkline([5]) == ""
