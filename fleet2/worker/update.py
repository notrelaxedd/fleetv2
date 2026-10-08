"""Self-update: download the worker tarball, verify, extract, repoint app/current.

Layout under the state dir: app/<version>/fleet2/... and the symlink app/current.
Before the swap the old target is written to app/previous and app/pending.json is
created, so fleet2.worker.launch can roll back when the new code fails to start.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import tarfile
from typing import Any

from fleet2.common import http
from fleet2.worker import launch

log = logging.getLogger("fleet2.update")

VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class UpdateError(Exception):
    """The update could not be applied; the running code is untouched."""


def _check_member(member: tarfile.TarInfo) -> None:
    """Reject anything outside fleet2/ or anything that is not a plain file or dir."""
    name = member.name
    if name.startswith("/") or name.startswith("\\"):
        raise UpdateError(f"absolute path in tarball: {name}")
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise UpdateError(f"unsafe path in tarball: {name}")
    if parts[0] != "fleet2":
        raise UpdateError(f"unexpected top-level entry in tarball: {name}")
    if not (member.isfile() or member.isdir()):
        raise UpdateError(f"unsupported member type in tarball: {name}")
    if "__pycache__" in parts or name.endswith(".pyc"):
        raise UpdateError(f"compiled file in tarball: {name}")


def extract_tarball(data: bytes, dest: str) -> None:
    """Extract a verified worker tarball into dest (must not exist yet)."""
    try:
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise UpdateError(f"bad tarball: {exc}") from None
    with tar:
        members = tar.getmembers()
        for member in members:
            _check_member(member)
        os.makedirs(dest, exist_ok=False)
        for member in members:
            target = os.path.join(dest, member.name)
            if member.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            os.chmod(target, 0o644)
    if not os.path.isfile(os.path.join(dest, "fleet2", "__init__.py")):
        raise UpdateError("tarball has no fleet2/__init__.py")


def swap_current(app: str, version: str) -> None:
    """Atomically point app/current at app/<version> via a temp symlink and os.replace."""
    current = os.path.join(app, "current")
    tmp = os.path.join(app, f".current.{os.getpid()}.tmp")
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    os.symlink(version, tmp)
    os.replace(tmp, current)


def current_version_dir(app: str) -> str | None:
    """Realpath of app/current, or None when the symlink is missing."""
    try:
        return os.path.realpath(os.path.join(app, "current"))
    except OSError:
        return None


def install_version(app: str, version: str, data: bytes) -> str:
    """Extract data into app/<version> (replacing a stale copy) and return the path."""
    if not VERSION_RE.match(version):
        raise UpdateError(f"refusing to install version with odd name: {version!r}")
    target = os.path.join(app, version)
    if os.path.exists(target):
        if current_version_dir(app) == os.path.realpath(target):
            raise UpdateError(f"{target} is the running version")
        shutil.rmtree(target)
    staging = os.path.join(app, f".{version}.{os.getpid()}.staging")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    try:
        extract_tarball(data, staging)
        os.replace(staging, target)
    finally:
        if os.path.exists(staging):
            shutil.rmtree(staging, ignore_errors=True)
    return target


def self_update(host_url: str, app: str, running_version: str, timeout: float = 60.0) -> str | None:
    """Download, verify and install the host's current code. Returns the new version
    when app/current was swapped, None when the host serves the running version."""
    try:
        info: Any = http.get_json(host_url + "/dl/version", timeout=10.0)
    except (http.HttpError, http.HttpConnectionError) as exc:
        raise UpdateError(f"/dl/version: {exc}") from None
    if not isinstance(info, dict) or not info.get("code_version") or not info.get("sha256"):
        raise UpdateError(f"/dl/version answered {info!r}")
    version = str(info["code_version"])
    expected = str(info["sha256"]).lower()
    if version == running_version:
        return None
    if version in launch.bad_versions(app):
        raise UpdateError(f"version {version} is listed in app/bad_versions (rolled back earlier); not updating")
    try:
        data = http.get_bytes(host_url + "/dl/worker.tar.gz", timeout=timeout)
    except (http.HttpError, http.HttpConnectionError) as exc:
        raise UpdateError(f"/dl/worker.tar.gz: {exc}") from None
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise UpdateError(f"sha256 mismatch: expected {expected}, got {actual}")
    os.makedirs(app, exist_ok=True)
    install_version(app, version, data)
    previous = launch.current_target(app)
    if previous and previous != version:
        launch.record_previous(app, previous)
    launch.write_pending(app, version, 0)
    swap_current(app, version)
    log.info("installed version %s (%d bytes) and repointed app/current (previous %s)", version, len(data), previous)
    return version
