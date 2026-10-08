"""Worker side: the runner child reports progress and detail, the bundle ships fleet2/,
and /install.sh carries the coordinator URL and the v1/v2 switch command."""
from __future__ import annotations

import io
import tarfile
import time

from coordinator.bundle import build_bundle
from fleet2.worker.runner import Runner


def test_runner_reports_progress_detail_and_result():
    runner = Runner({"id": "j1", "kind": "sleep", "params": {"seconds": 2}})
    runner.start()
    deadline = time.monotonic() + 15
    while runner.outcome is None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert runner.outcome == "done"
    assert runner.result == {"slept": 2, "summary": "Slept 2 s"}
    assert runner.detail == "Test job: 2 of 2 seconds"
    assert runner.snapshot()[1] == 1.0


def test_runner_stops_on_sigterm_with_checkpoint():
    runner = Runner({"id": "j2", "kind": "sleep", "params": {"seconds": 30}})
    runner.start()
    time.sleep(2.5)
    runner.stop(grace=3.0)
    assert runner.outcome == "stopped"
    assert runner.snapshot()[0]["elapsed"] >= 1


def test_bundle_has_one_fleet2_tree_with_version():
    bundle = build_bundle()
    with tarfile.open(fileobj=io.BytesIO(bundle.data), mode="r:gz") as tar:
        names = tar.getnames()
    assert {n.split("/")[0] for n in names} == {"fleet2"}
    assert "fleet2/__init__.py" in names and "fleet2/worker/agent.py" in names
    assert "fleet2/VERSION" in names
    assert not any(n.endswith(".pyc") for n in names)


def test_install_script_is_served_with_url_and_switch(client):
    text = client.get("/install.sh").text
    assert 'DEFAULT_HOST_URL="http://127.0.0.1:8090"' in text
    assert "__FLEET2_SWITCH__" not in text
    assert "fleet2 use v1|v2" in text and "sudo" not in text
    assert "Conflicts=fleet-worker.service" in text
    assert "/var/lib/fleet2" in text
