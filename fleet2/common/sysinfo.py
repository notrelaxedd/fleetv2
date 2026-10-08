"""CPU, RAM and boot id from /proc. Every reader tolerates a missing file."""

from __future__ import annotations

import os

PROC_STAT = "/proc/stat"
PROC_MEMINFO = "/proc/meminfo"
PROC_BOOT_ID = "/proc/sys/kernel/random/boot_id"


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def _read_cpu_counters(path: str = PROC_STAT) -> tuple[int, int] | None:
    """Return (total_jiffies, idle_jiffies) from the aggregate cpu line."""
    text = _read_text(path)
    if text is None:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "cpu":
            try:
                values = [int(p) for p in parts[1:]]
            except ValueError:
                return None
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return sum(values), idle
    return None


class CpuMeter:
    """CPU percentage from the delta between two /proc/stat samples."""

    def __init__(self, path: str = PROC_STAT) -> None:
        self._path = path
        self._last: tuple[int, int] | None = None

    def sample(self) -> float:
        """Return busy percent since the previous call (0.0 on the first call)."""
        current = _read_cpu_counters(self._path)
        if current is None:
            return 0.0
        previous = self._last
        self._last = current
        if previous is None:
            return 0.0
        total = current[0] - previous[0]
        idle = current[1] - previous[1]
        if total <= 0:
            return 0.0
        busy = 100.0 * (total - idle) / total
        return round(max(0.0, min(100.0, busy)), 1)


_default_meter = CpuMeter()


def cpu_pct() -> float:
    """Module-level CPU meter (first call returns 0.0)."""
    return _default_meter.sample()


def _meminfo_kb(path: str = PROC_MEMINFO) -> dict[str, int]:
    text = _read_text(path)
    if text is None:
        return {}
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                out[parts[0][:-1]] = int(parts[1])
            except ValueError:
                continue
    return out


def ram_total_mb(path: str = PROC_MEMINFO) -> int | None:
    """Total RAM in MiB, or None when /proc/meminfo is unreadable."""
    info = _meminfo_kb(path)
    if "MemTotal" not in info:
        return None
    return info["MemTotal"] // 1024


def ram_used_mb(path: str = PROC_MEMINFO) -> int | None:
    """Used RAM in MiB (MemTotal - MemAvailable), or None when unreadable."""
    info = _meminfo_kb(path)
    if "MemTotal" not in info:
        return None
    available = info.get("MemAvailable", info.get("MemFree", 0))
    return max(0, info["MemTotal"] - available) // 1024


def boot_id(path: str = PROC_BOOT_ID) -> str | None:
    """Kernel boot id, or None when unreadable."""
    text = _read_text(path)
    if text is None:
        return None
    return text.strip() or None


def hostname() -> str:
    """Short host name."""
    try:
        return os.uname().nodename
    except (AttributeError, OSError):
        return "unknown"


# ---------------------------------------------------------------- per-session RSS

PROC_ROOT = "/proc"


def _stat_session(path: str) -> int | None:
    """Session id (field 6 of /proc/<pid>/stat), parsed after the last ')' so a
    command name containing spaces or parentheses cannot shift the fields.
    None for a zombie or dead process: it holds no memory and cannot be signalled."""
    text = _read_text(path)
    if text is None:
        return None
    head, sep, tail = text.rpartition(")")
    if not sep:
        return None
    fields = tail.split()
    # tail fields: state ppid pgrp session ...
    if len(fields) < 4 or fields[0] in ("Z", "X"):
        return None
    try:
        return int(fields[3])
    except ValueError:
        return None


def _kb_fields(path: str, names: tuple[str, ...]) -> dict[str, int] | None:
    """The named "Key:   123 kB" fields of a /proc file; None when the file is unreadable."""
    text = _read_text(path)
    if text is None:
        return None
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":") and parts[0][:-1] in names:
            try:
                out[parts[0][:-1]] = int(parts[1])
            except ValueError:
                continue
    return out


def process_mem_kb(pid_dir: str) -> int:
    """Memory one process really holds, in kB: proportional (shared pages are split
    between the processes mapping them, so a forked child that shares its parent's
    buffer is not counted again) and anonymous or shmem only (file pages are cache the
    kernel can drop). Pss_Anon + Pss_Shmem from smaps_rollup, else Pss, else
    RssAnon + RssShmem from status, else VmRSS; 0 when the process vanished."""
    rollup = _kb_fields(os.path.join(pid_dir, "smaps_rollup"), ("Pss_Anon", "Pss_Shmem", "Pss"))
    if rollup:
        if "Pss_Anon" in rollup:
            return rollup["Pss_Anon"] + rollup.get("Pss_Shmem", 0)
        if "Pss" in rollup:
            return rollup["Pss"]
    status = _kb_fields(os.path.join(pid_dir, "status"), ("RssAnon", "RssShmem", "VmRSS"))
    if not status:
        return 0
    if "RssAnon" in status:
        return status["RssAnon"] + status.get("RssShmem", 0)
    return status.get("VmRSS", 0)


def session_pids(sid: int, proc_root: str = PROC_ROOT) -> list[int]:
    """Pids of every live process whose session id is sid. Processes that vanish
    while the listing is read and zombies waiting for their reaper are skipped."""
    try:
        names = os.listdir(proc_root)
    except OSError:
        return []
    out: list[int] = []
    for name in names:
        if not name.isdigit():
            continue
        if _stat_session(os.path.join(proc_root, name, "stat")) == sid:
            out.append(int(name))
    return sorted(out)


def session_rss_kb(sid: int, proc_root: str = PROC_ROOT) -> int:
    """Sum of process_mem_kb over every process in session sid.

    The runner child is started with start_new_session=True, so its pid is the
    session id of itself and of every grandchild it spawns; this is what the memory
    watchdog measures. Vanished processes count as 0.
    """
    total = 0
    for pid in session_pids(sid, proc_root):
        total += process_mem_kb(os.path.join(proc_root, str(pid)))
    return total
