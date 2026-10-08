"""Worker stats for the heartbeat: CPU %, RAM and CPU temperature.

Uses psutil when it is installed and falls back to the stdlib readers in sysinfo and
hwinfo otherwise. Every field is None when it cannot be read (never a made-up number),
sample() never raises, never sleeps and never writes to disk.
"""

from __future__ import annotations

import os
from typing import Any

from fleet2.common import hwinfo, sysinfo

try:  # python3-psutil is optional on the workers
    import psutil
except ImportError:
    psutil = None

_MIB = 1024 * 1024


def _round_pct(value: Any) -> float | None:
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(100.0, pct)), 1) if pct == pct else None


class StatsReader:
    def __init__(self, sys_root: str = "/sys", proc_root: str = "/proc", use_psutil: bool = True) -> None:
        self.sys_root = sys_root
        self.proc_root = proc_root
        self._use_psutil = use_psutil
        self._stat_path = os.path.join(proc_root, "stat")
        self._meminfo_path = os.path.join(proc_root, "meminfo")
        self._meter = sysinfo.CpuMeter(self._stat_path)

    def _ps(self) -> Any:
        return psutil if self._use_psutil else None

    def sample(self) -> dict[str, Any]:
        """One heartbeat sample; keys cpu_pct, ram_pct, ram_used_mb, ram_total_mb, temp_c."""
        ram = self._safe(self._ram) or (None, None, None)
        return {
            "cpu_pct": self._safe(self._cpu),
            "ram_pct": ram[0],
            "ram_used_mb": ram[1],
            "ram_total_mb": ram[2],
            "temp_c": self._safe(self._temp),
        }

    @staticmethod
    def _safe(reader: Any) -> Any:
        try:
            return reader()
        except Exception:  # a broken sensor or psutil bug must not kill the heartbeat
            return None

    def _cpu(self) -> float | None:
        ps = self._ps()
        if ps is not None:
            return _round_pct(ps.cpu_percent(interval=None))
        if not os.path.isfile(self._stat_path):
            return None
        return _round_pct(self._meter.sample())

    def _ram(self) -> tuple[float | None, int | None, int | None]:
        ps = self._ps()
        if ps is not None:
            vm = ps.virtual_memory()
            return (_round_pct(vm.percent), max(0, int(vm.total - vm.available)) // _MIB,
                    int(vm.total) // _MIB)
        total = sysinfo.ram_total_mb(self._meminfo_path)
        used = sysinfo.ram_used_mb(self._meminfo_path)
        if not total or used is None:
            return None, None, None
        return _round_pct(100.0 * used / total), used, total

    def _psutil_temp(self) -> float | None:
        ps = self._ps()
        if ps is None or not hasattr(ps, "sensors_temperatures"):
            return None
        cpu: list[float] = []
        other: list[float] = []
        for name, readings in (ps.sensors_temperatures() or {}).items():
            for reading in readings:
                value = float(reading.current)
                if hwinfo.MIN_PLAUSIBLE_C <= value <= hwinfo.MAX_PLAUSIBLE_C:
                    (cpu if name in hwinfo.CPU_SENSORS else other).append(value)
        found = cpu or other
        return round(max(found), 1) if found else None

    def _temp(self) -> float | None:
        value = self._safe(self._psutil_temp)
        return value if value is not None else hwinfo.temp_c(self.sys_root)
