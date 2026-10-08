"""Start bookkeeping for self-update rollback.

Files under app/ (next to the version directories and the current symlink):
  previous      the version app/current pointed at before the last self-update
  pending.json  {"version": v, "starts": n}: v was installed by a self-update and has
                not registered successfully yet; n counts its start attempts
  bad_versions  one version per line; the agent never updates to a listed version

python3 -m fleet2.worker run calls note_start() before importing the agent. When the
pending version failed MAX_FAILED_STARTS times, rollback() repoints app/current at
previous, lists the version in bad_versions and the process exits 75 so systemd
restarts the old code. Agent._apply_register clears pending.json on the first
successful register.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger("fleet2.launch")

PENDING_NAME = "pending.json"
PREVIOUS_NAME = "previous"
BAD_VERSIONS_NAME = "bad_versions"
MAX_FAILED_STARTS = 3
EXIT_ROLLED_BACK = 75


def current_target(app: str) -> str | None:
    """The version name app/current points at, or None."""
    try:
        return os.path.basename(os.readlink(os.path.join(app, "current")).rstrip("/"))
    except OSError:
        return None


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def read_previous(app: str) -> str | None:
    text = _read_text(os.path.join(app, PREVIOUS_NAME))
    return text.strip() or None if text else None


def record_previous(app: str, version: str) -> None:
    _write_text(os.path.join(app, PREVIOUS_NAME), version + "\n")


def bad_versions(app: str) -> set[str]:
    text = _read_text(os.path.join(app, BAD_VERSIONS_NAME)) or ""
    return {line.strip() for line in text.splitlines() if line.strip()}


def mark_bad(app: str, version: str) -> None:
    listed = bad_versions(app)
    listed.add(version)
    _write_text(os.path.join(app, BAD_VERSIONS_NAME), "".join(v + "\n" for v in sorted(listed)))


def read_pending(app: str) -> dict[str, Any] | None:
    text = _read_text(os.path.join(app, PENDING_NAME))
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or not data.get("version"):
        return None
    return data


def write_pending(app: str, version: str, starts: int) -> None:
    _write_text(os.path.join(app, PENDING_NAME), json.dumps({"version": version, "starts": int(starts)}) + "\n")


def clear_pending(app: str) -> None:
    try:
        os.unlink(os.path.join(app, PENDING_NAME))
    except OSError:
        pass


def note_start(app: str) -> int | None:
    """Count one start of the pending version. Returns the start count, or None
    when no pending version applies (none recorded, or current points elsewhere)."""
    pending = read_pending(app)
    if pending is None:
        return None
    if current_target(app) != pending["version"]:
        clear_pending(app)
        return None
    starts = int(pending.get("starts") or 0) + 1
    write_pending(app, str(pending["version"]), starts)
    return starts


def rollback(app: str) -> str | None:
    """Repoint app/current at the previous version and blacklist the pending one.
    Returns the version rolled back to, or None when there is nothing to roll back to."""
    from fleet2.worker.update import swap_current

    pending = read_pending(app)
    previous = read_previous(app)
    bad = pending["version"] if pending else current_target(app)
    if not previous or previous == bad or not os.path.isdir(os.path.join(app, previous)):
        log.error("cannot roll back: no usable previous version (previous=%r)", previous)
        return None
    if bad:
        mark_bad(app, bad)
    swap_current(app, previous)
    clear_pending(app)
    log.error("rolled back app/current from %s to %s", bad, previous)
    return previous


def failed_start(app: str, starts: int | None) -> int:
    """Exit code after a failed start: 75 after a rollback, else 1."""
    if starts is None or starts < MAX_FAILED_STARTS:
        return 1
    log.error("pending version failed to start %d times; rolling back", starts)
    if rollback(app) is None:
        return 1
    return EXIT_ROLLED_BACK
