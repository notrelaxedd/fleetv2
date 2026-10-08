"""Temperature, boot media, disk writes and wear from /proc and /sys (no root needed).

Every reader takes the filesystem roots as arguments so tests can point them at a fake
tree, and every reader tolerates missing files: an old board without sensors reports
None, never an error. Nothing here writes to disk.

Wear comes from two places only: eMMC exposes `device/life_time` to any user, and for
SATA or NVMe drives the root-owned `fleet-wear` timer (installed by install_worker.sh)
writes SMART's figure to /run/fleet-wear/wear.json once an hour. USB flash sticks
report nothing, so their wear is None and the dashboard shows GB written since boot.
"""

from __future__ import annotations

import json
import os

SYS_ROOT = "/sys"
PROC_ROOT = "/proc"
WEAR_FILE = "/run/fleet-wear/wear.json"

# hwmon drivers that measure the CPU package; other sensors (disks, chipset, wifi)
# are only used when none of these exist.
CPU_SENSORS = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "k8temp", "via_cputemp", "acpitz")
VIRTUAL_DISKS = ("loop", "ram", "zram", "nbd")
MIN_PLAUSIBLE_C = 1.0
MAX_PLAUSIBLE_C = 125.0


def _read(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _listdir(path: str) -> list[str]:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def _millideg(path: str) -> float | None:
    text = _read(path)
    try:
        value = int(text) / 1000.0 if text is not None else None
    except ValueError:
        return None
    if value is None or not MIN_PLAUSIBLE_C <= value <= MAX_PLAUSIBLE_C:
        return None
    return value


def temp_c(sys_root: str = SYS_ROOT) -> float | None:
    """Hottest CPU sensor in degrees C (any hwmon sensor, then thermal zones, as fallbacks)."""
    cpu: list[float] = []
    other: list[float] = []
    hwmon = os.path.join(sys_root, "class", "hwmon")
    for entry in _listdir(hwmon):
        base = os.path.join(hwmon, entry)
        name = _read(os.path.join(base, "name")) or ""
        for fname in _listdir(base):
            if fname.startswith("temp") and fname.endswith("_input"):
                value = _millideg(os.path.join(base, fname))
                if value is not None:
                    (cpu if name in CPU_SENSORS else other).append(value)
    if not cpu and not other:
        thermal = os.path.join(sys_root, "class", "thermal")
        for entry in _listdir(thermal):
            if entry.startswith("thermal_zone"):
                value = _millideg(os.path.join(thermal, entry, "temp"))
                if value is not None:
                    other.append(value)
    found = cpu or other
    return round(max(found), 1) if found else None


def _root_devnum(proc_root: str) -> tuple[str | None, str | None]:
    """(major:minor, mount source) of the filesystem mounted at / (the last one wins)."""
    text = _read(os.path.join(proc_root, "self", "mountinfo"))
    devnum = source = None
    for line in (text or "").splitlines():
        head, sep, tail = line.partition(" - ")
        fields = head.split()
        if not sep or len(fields) < 5 or fields[4] != "/":
            continue
        devnum = fields[2]
        rest = tail.split()
        source = rest[1] if len(rest) > 1 else None
    return devnum, source


def _disk_of(block_dir: str, sys_root: str, depth: int = 0) -> str | None:
    """The whole-disk sysfs directory behind a block device directory (partition ->
    parent disk; device-mapper or md -> the disk behind its first slave)."""
    if depth > 4:
        return None
    real = os.path.realpath(block_dir)
    if os.path.exists(os.path.join(real, "partition")):
        real = os.path.dirname(real)
    slaves = _listdir(os.path.join(real, "slaves"))
    if slaves:
        return _disk_of(os.path.join(sys_root, "class", "block", slaves[0]), sys_root, depth + 1)
    return real if os.path.isdir(real) else None


def boot_disk(sys_root: str = SYS_ROOT, proc_root: str = PROC_ROOT) -> str | None:
    """Sysfs directory of the disk the root filesystem lives on, or None."""
    devnum, source = _root_devnum(proc_root)
    if devnum and not devnum.startswith("0:"):
        found = _disk_of(os.path.join(sys_root, "dev", "block", devnum), sys_root)
        if found:
            return found
    if source and source.startswith("/dev/"):
        # /dev/sda2 directly; /dev/mapper/vg-root or /dev/disk/by-uuid/... through the
        # symlink to its dm-N or sdXN node.
        for name in (os.path.basename(source), os.path.basename(os.path.realpath(source))):
            found = _disk_of(os.path.join(sys_root, "class", "block", name), sys_root)
            if found:
                return found
    return None


def boot_media(disk: str | None) -> str:
    """flash (USB, removable, SD or eMMC), ssd, hdd or unknown (also for loop, ram and
    network block devices, whose real medium is elsewhere)."""
    if not disk:
        return "unknown"
    name = os.path.basename(disk)
    if name.startswith(VIRTUAL_DISKS):
        return "unknown"
    if "/usb" in disk or name.startswith("mmcblk") or _read(os.path.join(disk, "removable")) == "1":
        return "flash"
    rotational = _read(os.path.join(disk, "queue", "rotational"))
    if rotational == "0":
        return "ssd"
    if rotational == "1":
        return "hdd"
    return "unknown"


def gb_written(disk: str | None) -> float | None:
    """GB written to the disk since boot (sectors written, field 7 of its stat file,
    are always 512-byte units)."""
    text = _read(os.path.join(disk, "stat")) if disk else None
    fields = (text or "").split()
    try:
        return round(int(fields[6]) * 512 / 1e9, 2) if len(fields) > 6 else None
    except ValueError:
        return None


def _emmc_wear(disk: str) -> float | None:
    """eMMC life_time: two hex bands 0x01 (0-10% used) .. 0x0B (past its rated life);
    the upper bound of the worse band, so the figure never flatters the chip."""
    text = _read(os.path.join(disk, "device", "life_time"))
    try:
        bands = [int(part, 16) for part in (text or "").split()]
    except ValueError:
        return None
    bands = [b for b in bands if 1 <= b <= 11]
    return float(min(100, max(bands) * 10)) if bands else None


def wear_pct(disk: str | None, wear_file: str = WEAR_FILE) -> float | None:
    """Percent of rated life used: SMART (from the fleet-wear timer) or eMMC; else None."""
    if not disk:
        return None
    text = _read(wear_file)
    if text:
        try:
            entry = (json.loads(text).get("devices") or {}).get(os.path.basename(disk)) or {}
            value = entry.get("wear_pct")
        except (ValueError, AttributeError):
            value = None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1000:
            return float(min(value, 100.0))
    return _emmc_wear(disk)
