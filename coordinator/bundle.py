"""Build the worker tarball served from /dl, once at host start."""
from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

VERSION_FILE = "VERSION"
TOP = "fleet2"


@dataclass(frozen=True)
class Bundle:
    """The in-memory worker tarball and its identity."""

    code_version: str
    sha256: str
    data: bytes


def fleet_package_dir() -> Path:
    """Directory of the installed `fleet2` package (agent, models, backtester)."""
    import fleet2

    return Path(fleet2.__file__).resolve().parent


def source_files(root: Path) -> list[tuple[str, bytes]]:
    """(relative posix path, bytes) for every *.py under root, sorted by path."""
    files: list[tuple[str, bytes]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        files.append((path.relative_to(root).as_posix(), path.read_bytes()))
    return files


def compute_code_version(files: list[tuple[str, bytes]]) -> str:
    """First 12 hex of sha256 over sorted relative paths and file bytes."""
    digest = hashlib.sha256()
    for rel, data in files:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\n")
        digest.update(data)
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def _tarinfo(name: str, size: int, is_dir: bool = False) -> tarfile.TarInfo:
    """A tar entry with all metadata fixed, so the archive is reproducible."""
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.DIRTYPE if is_dir else tarfile.REGTYPE
    info.mode = 0o755 if is_dir else 0o644
    return info


def build_tarball(files: list[tuple[str, bytes]], code_version: str) -> bytes:
    """A gzip tarball with one top-level `fleet2/` dir, *.py files and VERSION."""
    dirs: set[str] = set()
    for rel, _ in files:
        parent = Path(rel).parent.as_posix()
        while parent not in ("", "."):
            dirs.add(parent)
            parent = Path(parent).parent.as_posix()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        tar.addfile(_tarinfo(TOP, 0, is_dir=True))
        for d in sorted(dirs):
            tar.addfile(_tarinfo(f"{TOP}/{d}", 0, is_dir=True))
        version_bytes = (code_version + "\n").encode("utf-8")
        tar.addfile(_tarinfo(f"{TOP}/{VERSION_FILE}", len(version_bytes)), io.BytesIO(version_bytes))
        for rel, data in files:
            tar.addfile(_tarinfo(f"{TOP}/{rel}", len(data)), io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def build_bundle(root: Path | None = None) -> Bundle:
    """Build the worker bundle from the installed fleet2 package."""
    files = source_files(root or fleet_package_dir())
    code_version = compute_code_version(files)
    data = build_tarball(files, code_version)
    return Bundle(code_version=code_version, sha256=hashlib.sha256(data).hexdigest(), data=data)
