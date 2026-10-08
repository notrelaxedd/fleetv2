"""The Trading mode panel: markup on both screens, the pill that opens it, and the live API it uses."""
from __future__ import annotations

import re

import pytest


@pytest.mark.parametrize("path", ["/fleet", "/models"])
def test_panel_markup_is_on_each_screen(client, path):
    html = client.get(path).text
    assert 'id="live-panel"' in html
    assert "Trading mode" in html
    for line in ("ALPACA_LIVE=true in .env on box1", "Live Alpaca keys in .env", "Your confirmation here",
                 "The mode changes only when the coordinator restarts. It never switches by itself."):
        assert line in html
    assert 'placeholder="Type TRADE REAL MONEY"' in html
    assert "Confirm live trading" in html and "Withdraw confirmation" in html
    assert 'data-action="live-close"' in html


@pytest.mark.parametrize("path", ["/fleet", "/models", "/fragments/fleet", "/fragments/models"])
def test_mode_pill_is_the_panel_trigger(client, path):
    html = client.get(path).text
    assert re.search(r'<button[^>]*data-pill="mode"[^>]*data-action="open-live"', html)


def test_live_state_defaults_to_paper_and_env_off(client):
    data = client.get("/api/live").json()
    assert data["mode"] == "paper"
    assert data["env_allows_live"] is False
    assert data["confirmed"] is False
    assert data["phrase"] == "TRADE REAL MONEY"


def test_confirm_without_env_switch_is_refused(client):
    r = client.post("/api/live/confirm", json={"confirm": "TRADE REAL MONEY"})
    assert r.status_code == 409
    assert "ALPACA_LIVE=true" in r.json()["detail"]
    assert client.get("/api/live").json()["confirmed"] is False


def test_confirm_with_wrong_phrase_is_a_bad_request(client):
    r = client.post("/api/live/confirm", json={"confirm": "yes"})
    assert r.status_code == 400
    assert "TRADE REAL MONEY" in r.json()["detail"]


def test_withdraw_works_and_leaves_paper_mode(client):
    r = client.post("/api/live/withdraw")
    assert r.status_code == 200
    assert "withdrawn" in r.json()["message"]
    state = client.get("/api/live").json()
    assert state["confirmed"] is False and state["mode"] == "paper"
