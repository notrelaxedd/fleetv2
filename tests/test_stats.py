"""StatsReader: psutil path, /sys and /proc fallbacks, and never raising."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from fleet2.common import stats

MIB = 1024 * 1024


def fake_psutil(temps: Any = None, cpu: float = 12.34, with_sensors: bool = True) -> SimpleNamespace:
    ns = SimpleNamespace(
        cpu_percent=lambda interval=None: cpu,
        virtual_memory=lambda: SimpleNamespace(percent=41.26, total=8192 * MIB, available=4096 * MIB + 1),
    )
    if with_sensors:
        ns.sensors_temperatures = (temps if callable(temps) else (lambda: temps or {}))
    return ns


def t(*values: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(label="", current=v, high=None, critical=None) for v in values]


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


@pytest.fixture
def roots(tmp_path: Any) -> tuple[str, str]:
    return str(tmp_path / "sys"), str(tmp_path / "proc")


def reader(roots: tuple[str, str], **kw: Any) -> stats.StatsReader:
    return stats.StatsReader(sys_root=roots[0], proc_root=roots[1], **kw)


def test_psutil_values_and_rounding(monkeypatch: Any, roots: Any) -> None:
    monkeypatch.setattr(stats, "psutil", fake_psutil({"coretemp": t(51.04, 55.55)}))
    s = reader(roots).sample()
    assert s == {"cpu_pct": 12.3, "ram_pct": 41.3, "ram_used_mb": 4095, "ram_total_mb": 8192, "temp_c": 55.5}


def test_cpu_is_clamped(monkeypatch: Any, roots: Any) -> None:
    monkeypatch.setattr(stats, "psutil", fake_psutil(cpu=250.0))
    assert reader(roots).sample()["cpu_pct"] == 100.0


def test_prefers_cpu_sensor_over_hotter_other(monkeypatch: Any, roots: Any) -> None:
    monkeypatch.setattr(stats, "psutil", fake_psutil({"nvme": t(70.0), "coretemp": t(48.0, 50.0)}))
    assert reader(roots).sample()["temp_c"] == 50.0


def test_other_sensor_used_when_no_cpu_sensor(monkeypatch: Any, roots: Any) -> None:
    monkeypatch.setattr(stats, "psutil", fake_psutil({"nvme": t(40.0)}))
    assert reader(roots).sample()["temp_c"] == 40.0


def test_implausible_readings_ignored(monkeypatch: Any, roots: Any) -> None:
    monkeypatch.setattr(stats, "psutil", fake_psutil({"coretemp": t(0.0, -20.0, 200.0, 45.0)}))
    assert reader(roots).sample()["temp_c"] == 45.0
    monkeypatch.setattr(stats, "psutil", fake_psutil({"coretemp": t(0.0, 255.0)}))
    assert reader(roots).sample()["temp_c"] is None


def _no_sensors() -> list[Any]:
    def boom() -> Any:
        raise RuntimeError("sensors broke")
    return [fake_psutil({}), fake_psutil(None, with_sensors=False), fake_psutil(boom), fake_psutil(lambda: None)]


@pytest.mark.parametrize("ps", _no_sensors())
def test_psutil_without_sensors_falls_back_to_sys(monkeypatch: Any, roots: Any, ps: Any) -> None:
    monkeypatch.setattr(stats, "psutil", ps)
    write(os.path.join(roots[0], "class", "thermal", "thermal_zone0", "temp"), "47000\n")
    assert reader(roots).sample()["temp_c"] == 47.0


def test_hwmon_fallback(monkeypatch: Any, roots: Any) -> None:
    monkeypatch.setattr(stats, "psutil", fake_psutil({}))
    base = os.path.join(roots[0], "class", "hwmon", "hwmon0")
    write(os.path.join(base, "name"), "coretemp\n")
    write(os.path.join(base, "temp1_input"), "52500\n")
    assert reader(roots).sample()["temp_c"] == 52.5


@pytest.mark.parametrize("use_psutil", [True, False])
def test_no_sensor_anywhere_is_none(monkeypatch: Any, roots: Any, use_psutil: bool) -> None:
    monkeypatch.setattr(stats, "psutil", fake_psutil({}))
    assert reader(roots, use_psutil=use_psutil).sample()["temp_c"] is None


def test_proc_fallback(monkeypatch: Any, roots: Any) -> None:
    monkeypatch.setattr(stats, "psutil", None)
    proc = roots[1]
    write(os.path.join(proc, "meminfo"), "MemTotal: 4194304 kB\nMemFree: 100 kB\nMemAvailable: 3145728 kB\n")
    write(os.path.join(proc, "stat"), "cpu  100 0 100 800 0 0 0 0 0 0\n")
    r = reader(roots)
    first = r.sample()
    assert first["cpu_pct"] == 0.0
    assert (first["ram_total_mb"], first["ram_used_mb"], first["ram_pct"]) == (4096, 1024, 25.0)
    write(os.path.join(proc, "stat"), "cpu  200 0 200 900 0 0 0 0 0 0\n")
    assert r.sample()["cpu_pct"] == 66.7


def test_use_psutil_false_ignores_installed_psutil(monkeypatch: Any, roots: Any) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("psutil must not be used")
    monkeypatch.setattr(stats, "psutil", SimpleNamespace(cpu_percent=boom, virtual_memory=boom, sensors_temperatures=boom))
    s = reader(roots, use_psutil=False).sample()
    assert s == {"cpu_pct": None, "ram_pct": None, "ram_used_mb": None, "ram_total_mb": None, "temp_c": None}


def test_missing_proc_gives_none_not_zero(monkeypatch: Any, roots: Any) -> None:
    monkeypatch.setattr(stats, "psutil", None)
    s = reader(roots).sample()
    assert s["cpu_pct"] is None and s["ram_pct"] is None


def test_sample_never_raises_when_psutil_raises(monkeypatch: Any, roots: Any) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise OSError("nope")
    monkeypatch.setattr(stats, "psutil", SimpleNamespace(cpu_percent=boom, virtual_memory=boom, sensors_temperatures=boom))
    s = reader(roots).sample()
    assert set(s) == {"cpu_pct", "ram_pct", "ram_used_mb", "ram_total_mb", "temp_c"}
    assert all(v is None for v in s.values())
