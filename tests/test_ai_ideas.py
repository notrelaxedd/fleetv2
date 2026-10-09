"""Claude Haiku writing recipes (coordinator/ai_ideas.py, docs/AI_PLAN.md stage B), with a
stand-in for the Anthropic client: nothing here calls Anthropic or spends anything.

What these prove: the key and the monthly cap gate every call; every call's cost is
recorded; Haiku's answer only becomes recipes that pass the same checks as a random one;
Haiku is shown training numbers only, never held-out or lockbox ones; searches take its
recipes beside the random ones and report their training results back."""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from coordinator import ai_ideas, futures_models, futures_view, queue
from coordinator.ai_ideas import AiSettings
from fleet2.models.futures import recipe as R
from fleet2.models.futures.base import params_with_defaults
from tests.conftest import enroll

NOW = datetime(2026, 10, 9, 15, 0, tzinfo=timezone.utc)
SETTINGS = AiSettings()
GOOD = [
    {"signal": "range_break", "direction": "follow", "filters": ["quiet_day"], "exit": "vwap_cross",
     "entries": "first", "side": "both", "idea": "Calm mornings that break their range tend to keep going."},
    {"signal": "gap", "direction": "fade", "filters": [], "exit": "vwap_return", "entries": "first",
     "side": "both", "idea": "Big opening gaps often drift back toward the day's average price."},
]


class FakeClient:
    """Stands in for anthropic.Anthropic(): records each request, answers with `answer`."""

    def __init__(self, answer=None, stop_reason="end_turn", usage=(3000, 900), error=None):
        self.answer = {"recipes": GOOD} if answer is None else answer
        self.stop_reason, self.usage, self.error = stop_reason, usage, error
        self.requests: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        text = self.answer if isinstance(self.answer, str) else json.dumps(self.answer)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=self.usage[0], output_tokens=self.usage[1]))


def futures_search(conn) -> str:
    result = queue.create_job(conn, "model_search", {"markets": ["futures"]}, queue.AUTO)
    return str(result.jobs[0]["id"])


def write(conn, client, settings=SETTINGS, key="sk-test", now=NOW):
    return ai_ideas.write_recipes(conn, settings, key, client=client, now=now)


# ------------------------------------------------------------------ settings and gates


def test_the_settings_file_has_the_cap_the_model_and_the_prices():
    s = ai_ideas.load_settings(Path(__file__).resolve().parents[1] / "config" / "ai.toml")
    assert s.model == "claude-haiku-5-5" and s.monthly_cap_usd == 5.0 and s.effort == "low"
    assert (s.input_per_million, s.output_per_million) == (0.10, 0.50)
    assert s.cost(1_000_000, 0) == pytest.approx(0.10) and s.cost(0, 1_000_000) == pytest.approx(0.50)
    assert ai_ideas.load_settings(Path("/nowhere.toml")) == AiSettings()


def test_bad_settings_are_refused(tmp_path):
    bad = tmp_path / "ai.toml"
    bad.write_text('[recipes]\neffort = "maximum"\n')
    with pytest.raises(ValueError, match="effort"):
        ai_ideas.load_settings(bad)


def test_no_key_or_no_futures_search_means_no_call(conn):
    client = FakeClient()
    assert write(conn, client, key="")["skipped"] == "no ANTHROPIC_API_KEY in .env on box1"
    assert write(conn, client)["skipped"] == "no futures model search is running"
    off = AiSettings(enabled=False)
    futures_search(conn)
    assert write(conn, client, settings=off)["skipped"] == "turned off in config/ai.toml"
    assert client.requests == []


def test_the_monthly_cap_is_checked_before_every_call(conn):
    futures_search(conn)
    client = FakeClient()
    ai_ideas.record_call(conn, "claude-haiku-5-5", 0, 0, 4.999, "ok")  # this month, so far
    reply = write(conn, client)
    assert "cap of $5.00" in reply["skipped"] and client.requests == []
    conn.execute("UPDATE ai_calls SET at = %s", (NOW - timedelta(days=40),))  # last month's spend does not count
    assert write(conn, client)["written"] == 2


def test_at_most_calls_per_hour(conn):
    futures_search(conn)
    settings = AiSettings(calls_per_hour=1, queue=100)
    assert write(conn, FakeClient(), settings=settings)["written"] == 2
    again = write(conn, FakeClient(answer={"recipes": []}), settings=settings)
    assert again.get("skipped") == "1 calls this hour already", again


def test_no_call_while_enough_recipes_wait(conn):
    futures_search(conn)
    assert write(conn, FakeClient(), settings=AiSettings(queue=2))["written"] == 2
    assert write(conn, FakeClient(), settings=AiSettings(queue=2))["skipped"] == "enough recipes are waiting"


# ------------------------------------------------------------------ the call and the answer


def test_a_call_asks_haiku_for_recipes_in_a_fixed_shape_and_records_its_cost(conn):
    futures_search(conn)
    client = FakeClient(usage=(3000, 900))
    reply = write(conn, client)
    assert reply["written"] == 2 and reply["cost"] == pytest.approx(3000 * 0.1e-6 + 900 * 0.5e-6)
    req = client.requests[0]
    assert req["model"] == "claude-haiku-5-5" and req["max_tokens"] == 8000
    assert req["output_config"]["effort"] == "low"
    fmt = req["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["properties"]["recipes"]["items"]["additionalProperties"] is False
    assert not {"temperature", "top_p", "top_k"} & set(req)  # Haiku 5.5 refuses sampling settings
    text = req["system"] + req["messages"][0]["content"]
    for name in [*R.SIGNALS, *R.FILTERS, *R.EXITS]:
        assert name in text  # every building block is described
    rows = conn.execute("SELECT * FROM recipe_ideas ORDER BY id").fetchall()
    assert [r["family"] for r in rows] == [R.name_of({k: v for k, v in g.items() if k != "idea"}) for g in GOOD]
    assert rows[0]["by"] == "haiku" and rows[0]["status"] == "queued" and rows[0]["note"].startswith("Calm mornings")
    call = conn.execute("SELECT * FROM ai_calls").fetchone()
    assert call["outcome"] == "ok" and call["input_tokens"] == 3000 and call["output_tokens"] == 900
    assert ai_ideas.month_spend(conn, NOW) == pytest.approx(reply["cost"])


def test_only_valid_new_recipes_are_kept(conn):
    futures_search(conn)
    answer = {"recipes": [
        GOOD[0],
        {**GOOD[1], "filters": ["quiet_day", "busy_day"]},  # contradicts itself
        {**GOOD[1], "signal": "astrology"},                 # not a building block
        GOOD[0],                                            # the same recipe twice
    ]}
    reply = write(conn, FakeClient(answer=answer), settings=AiSettings(per_call=4))
    assert reply["written"] == 1 and len(reply["dropped"]) == 3
    # Written again later: already queued, so dropped.
    conn.execute("DELETE FROM ai_calls")
    assert write(conn, FakeClient(answer={"recipes": [GOOD[0]]}))["written"] == 0


@pytest.mark.parametrize("client, outcome", [
    (FakeClient(stop_reason="refusal"), "refused"),
    (FakeClient(stop_reason="max_tokens"), "bad: the answer was cut off (max_tokens)"),
    (FakeClient(answer="not json"), "bad: unreadable answer"),
    (FakeClient(error=type("AuthenticationError", (Exception,), {"status_code": 401})("no")),
     "error: Anthropic refused ANTHROPIC_API_KEY in .env on box1"),
    (FakeClient(error=ConnectionError("down")), "error: ConnectionError: down"),
])
def test_problems_are_recorded_and_shown_never_raised(conn, client, outcome):
    futures_search(conn)
    reply = write(conn, client)
    assert reply["written"] == 0
    row = conn.execute("SELECT outcome FROM ai_calls").fetchone()
    assert row["outcome"].startswith(outcome)
    assert conn.execute("SELECT count(*) AS n FROM recipe_ideas").fetchone()["n"] == 0
    line = ai_ideas.status_line(conn, SETTINGS, "sk-test", NOW)
    assert line["tone"] == "warn" and "last call: " + outcome in line["text"]


# ------------------------------------------------------------------ training numbers only


def test_haiku_is_shown_training_numbers_only(conn):
    job = futures_search(conn)
    r = R.random_recipe(random.Random("shown"))
    sneaky = {"name": R.name_of(r), "recipe": r, "tried": 16, "best_score": 0.12, "passes": True, "days_traded": 140,
              "pnl_double": 812.5, "pnl_normal": 1404.0,
              "held_out": {"pass_rate": 0.9}, "lockbox": {"pass_rate": 1.0}, "sizes": [1, 2]}
    assert ai_ideas.record_results(conn, job, [sneaky]) == 1
    stored = conn.execute("SELECT result FROM recipe_ideas").fetchone()["result"]
    assert set(stored) == set(ai_ideas.TRAIN_KEYS)
    client = FakeClient()
    write(conn, client)
    text = client.requests[0]["messages"][0]["content"]
    assert "[random]" in text and "best score 0.120" in text and "140 days traded" in text
    assert "held" not in text.lower() and "lockbox" not in text.lower() and "pass rate" not in text.lower()


def test_results_mark_haiku_recipes_tried_and_keep_random_ones_up_to_date(conn):
    job = futures_search(conn)
    write(conn, FakeClient())
    idea = ai_ideas.take(conn, job, 1)[0]
    entry = {"name": idea["family"], "recipe": idea["recipe"], "idea_id": idea["id"], "tried": 16,
             "best_score": None, "passes": False, "days_traded": 12, "pnl_double": -90.0, "pnl_normal": None}
    ai_ideas.record_results(conn, job, [entry])
    row = conn.execute("SELECT * FROM recipe_ideas WHERE id = %s", (idea["id"],)).fetchone()
    assert row["status"] == "tried" and row["tries"] == 16 and row["result"]["days_traded"] == 12
    r = R.random_recipe(random.Random("random one"))
    base = {"name": R.name_of(r), "recipe": r, "tried": 16, "passes": False, "days_traded": 50,
            "pnl_double": 10.0, "pnl_normal": None}
    ai_ideas.record_results(conn, job, [{**base, "best_score": 0.05}])
    ai_ideas.record_results(conn, job, [{**base, "best_score": 0.02, "tried": 4}])  # worse: the best is kept
    rows = conn.execute("SELECT * FROM recipe_ideas WHERE by = 'random'").fetchall()
    assert len(rows) == 1 and rows[0]["tries"] == 20 and rows[0]["result"]["best_score"] == 0.05
    assert rows[0]["result"]["tried"] == 20
    wrong = {**base, "name": "recipe_0000000000"}  # a name that is not its recipe's
    assert ai_ideas.record_results(conn, job, [wrong]) == 0


def test_searches_take_the_oldest_waiting_recipes_and_return_unfinished_ones(conn):
    job = futures_search(conn)
    write(conn, FakeClient())
    first = ai_ideas.take(conn, job, 1)
    assert len(first) == 1 and first[0]["note"].startswith("Calm mornings")
    assert ai_ideas.take(conn, job, 5)[0]["id"] == first[0]["id"] + 1
    assert ai_ideas.take(conn, job, 5) == []
    conn.execute("UPDATE jobs SET status = 'cancelled'")  # that search ended before trying them
    again = futures_search(conn)
    assert len(ai_ideas.take(conn, again, 5)) == 2


# ------------------------------------------------------------------ through the coordinator's routes


def test_the_routes_give_recipes_only_while_haiku_is_on(client, conn):
    w = enroll(client, conn, "w1")
    hdr = {"Authorization": "Bearer " + w["worker_token"]}
    job = futures_search(conn)
    write(conn, FakeClient())
    client.app.state.ai_key = ""
    assert client.post("/api/v1/search/ideas", json={"job_id": job, "take": 3}, headers=hdr).json() == []
    client.app.state.ai_key = "sk-test"
    got = client.post("/api/v1/search/ideas", json={"job_id": job, "take": 3}, headers=hdr).json()
    assert [g["family"] for g in got] == [r["family"] for r in conn.execute("SELECT family FROM recipe_ideas ORDER BY id")]
    entry = {"name": got[0]["family"], "recipe": got[0]["recipe"], "idea_id": got[0]["id"], "tried": 16,
             "best_score": 0.2, "passes": True, "days_traded": 120, "pnl_double": 50.0, "pnl_normal": 90.0}
    r = client.post("/api/v1/search/tries", json={"job_id": job, "tries": {"recipe": {"n": 16, "sum": 1.0, "sq": 0.2}},
                                                   "recipes": [entry]}, headers=hdr)
    assert r.status_code == 200
    assert conn.execute("SELECT status FROM recipe_ideas WHERE id = %s", (got[0]["id"],)).fetchone()["status"] == "tried"


def test_a_kept_haiku_recipe_says_so_with_its_idea(conn):
    job = futures_search(conn)
    write(conn, FakeClient())
    idea = ai_ideas.take(conn, job, 1)[0]
    module = R.family(idea["recipe"])
    from tests.test_futures_screen import FEES, noise, real_metrics

    m = real_metrics()
    m["train"]["daily"] = {"days": list(range(20240101, 20240161)), "pnl": noise(3)}
    body = {"module": module.__name__, "params": params_with_defaults(module, None), "train_score": 1.0,
            "seed": "1:1:x:0", "parent": None, "metrics": m, "idea_id": idea["id"]}
    reply = futures_models.store_found(conn, None, body)
    row = conn.execute("SELECT * FROM models WHERE id = %s", (reply["id"],)).fetchone()
    assert row["metrics"]["recipe_by"] == "haiku" and row["metrics"]["idea"].startswith("Calm mornings")
    origin = futures_view.detail(row, FEES, {}, None, False, "synthetic")["origin"]
    assert "recipe put together by Claude Haiku" in origin and "Calm mornings" in origin
    line = ai_ideas.status_line(conn, SETTINGS, "sk-test", NOW)
    assert "recipe models kept now: 1 by Haiku, 0 random" in line["text"]
    # An idea id that is not this recipe's does not make it Haiku's.
    other = {**body, "idea_id": idea["id"] + 1}
    other["metrics"] = {**m, "train": {**m["train"], "daily": {"days": list(range(20240101, 20240161)),
                                                               "pnl": noise(99)}}}
    reply = futures_models.store_found(conn, None, other)
    if reply["kept"]:
        row = conn.execute("SELECT metrics FROM models WHERE id = %s", (reply["id"],)).fetchone()
        assert row["metrics"]["recipe_by"] == "random"


def test_the_futures_screen_says_whether_haiku_is_on(client, conn):
    client.app.state.ai_key = ""
    html = client.get("/models?market=futures").text
    assert "Claude Haiku is off. To let it write recipes for model search, add ANTHROPIC_API_KEY" in html
    client.app.state.ai_key = "sk-test"
    ai_ideas.record_call(conn, "claude-haiku-5-5", 3000, 900, 0.00075, "ok")
    html = client.get("/models?market=futures").text
    assert "Claude Haiku: 0 recipes written this month" in html and "$0.00 of $5.00 spent this month" in html
