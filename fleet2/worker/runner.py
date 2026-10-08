"""Job runner: the child process entry point and the parent-side Runner handle.

Child (python3 -m fleet2.worker.runner): reads one job JSON object from stdin, runs it
and prints one JSON object per line on stdout:
  {"checkpoint": {...}, "progress": 0.42, "detail": "..."}   after every unit of work
                                            (progress null = "Live", for paper trading;
                                            detail is the one line the Fleet card shows)
  {"done": true, "result": {...}}            on success
  {"stopped": true, "checkpoint": {...}, "progress": 0.42}   after SIGTERM
  {"error": "..."}                           on failure
Every line is flushed. The process exits 0 in all cases. The job's "context" (the
coordinator address and this worker's token, so the job can fetch price bars into
memory) is merged into the params the job function receives as params["_context"];
params are otherwise untouched.

Test hook: env FLEET2_TEST_JOBS="module:attr" names a dict of extra job functions that
the child merges over the registry (tests inject kinds that echo their inputs instead
of running the real backtest).

Parent: Runner starts the child, keeps the last checkpoint/progress/result/error it
printed and can stop (SIGTERM, grace, SIGKILL) or kill it.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import traceback
from typing import Any

log = logging.getLogger("fleet2.runner")

MODULE = "fleet2.worker.runner"


# ---------------------------------------------------------------- child side


def _print_line(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _read_job() -> dict[str, Any]:
    raw = sys.stdin.read()
    job = json.loads(raw)
    if not isinstance(job, dict):
        raise ValueError("job must be a JSON object")
    return job


def _registry() -> dict[str, Any]:
    """The job registry, with the FLEET2_TEST_JOBS extras merged over it."""
    from fleet2.worker.jobs import JOBS

    jobs: dict[str, Any] = dict(JOBS)
    spec = os.environ.get("FLEET2_TEST_JOBS", "").strip()
    if spec:
        module_name, _, attr = spec.partition(":")
        extra = getattr(importlib.import_module(module_name), attr or "TEST_JOBS")
        jobs.update(extra)
    return jobs


def _params_with_context(job: dict[str, Any]) -> dict[str, Any]:
    """A copy of the job's params with job["context"] merged in as params["_context"]."""
    params = dict(job.get("params") or {})
    if isinstance(job.get("context"), dict):
        params["_context"] = job["context"]
    params["_job_id"] = job.get("id")
    return params


def run_child() -> int:
    """Entry point of the child process. Returns the exit code (always 0)."""
    from fleet2.worker.jobs import JobStopped

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        job = _read_job()
    except Exception as exc:
        _print_line({"error": f"malformed job: {exc}"})
        return 0

    kind = job.get("kind")
    try:
        func = _registry().get(kind) if isinstance(kind, str) else None
    except Exception as exc:
        _print_line({"error": f"job registry unavailable: {exc}"})
        return 0
    if func is None:
        _print_line({"error": f"unknown job kind: {kind!r}"})
        return 0

    params = _params_with_context(job)
    checkpoint = job.get("checkpoint")
    if not isinstance(checkpoint, dict):
        checkpoint = None
    last: dict[str, Any] = {"checkpoint": checkpoint, "progress": 0.0}

    def emit(cp: dict[str, Any], progress: float | None, detail: str | None = None) -> None:
        last["checkpoint"] = cp
        last["progress"] = None if progress is None else float(progress)
        line: dict[str, Any] = {"checkpoint": cp, "progress": last["progress"]}
        if detail is not None:
            line["detail"] = str(detail)[:200]
        _print_line(line)

    try:
        result = func(params, checkpoint, emit, stop.is_set)
    except JobStopped:
        _print_line({"stopped": True, "checkpoint": last["checkpoint"], "progress": last["progress"]})
        return 0
    except Exception:
        _print_line({"error": traceback.format_exc(limit=8)})
        return 0
    _print_line({"done": True, "result": result})
    return 0


# --------------------------------------------------------------- parent side


def _package_root() -> str:
    """Directory that contains the fleet2 package (what PYTHONPATH must include)."""
    import fleet2

    return os.path.dirname(os.path.dirname(os.path.abspath(fleet2.__file__)))


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the child: parent env, OMP_NUM_THREADS=1, fleet2 on PYTHONPATH."""
    env = dict(os.environ if base is None else base)
    env["OMP_NUM_THREADS"] = "1"
    root = _package_root()
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root not in parts:
        parts.insert(0, root)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _seed_progress(value: Any) -> float:
    """The job's stored progress (a resumed job reports it until its first unit finishes)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


class Runner:
    """Parent-side handle on one runner child process.

    The child starts in its own session (and process group). terminate() signals the
    group (the child's search workers sit in groups of their own, so only the child
    sees the stop and ends them itself, fleet2/sim/parallel.py); kill() and
    reap_group() SIGKILL the group and then every other process of the session, so
    no grandchild outlives a kill.
    """

    def __init__(
        self,
        job: dict[str, Any],
        python: str | None = None,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.job = job
        self.job_id = str(job.get("id", ""))
        self._python = python or sys.executable
        self._command = command
        self._env = env
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self.last_checkpoint: dict[str, Any] | None = job.get("checkpoint") if isinstance(job.get("checkpoint"), dict) else None
        self.last_progress: float | None = _seed_progress(job.get("progress"))
        self.detail: str | None = None
        self.checkpoint_seq: int = 0
        self.result: Any = None
        self.done: bool = False
        self.error: str | None = None
        self.stopped: bool = False
        self.term_sent: bool = False
        self.kill_sent: bool = False
        self.group_killed: bool = False

    # lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Spawn the child and feed it the job."""
        command = self._command or [self._python, "-m", MODULE]
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=child_env(self._env),
            close_fds=True,
            start_new_session=True,
        )
        payload = {
            "id": self.job.get("id"),
            "kind": self.job.get("kind"),
            "params": self.job.get("params") or {},
            "checkpoint": self.job.get("checkpoint"),
            "context": self.job.get("context"),
        }
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(payload).encode("utf-8"))
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        self._reader = threading.Thread(target=self._read_loop, name=f"runner-{self.job_id[:8]}", daemon=True)
        self._reader.start()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def poll(self) -> int | None:
        """Exit code of the child, or None while it runs."""
        if self._proc is None:
            return None
        return self._proc.poll()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to timeout seconds; True when the child has exited."""
        if self._proc is None:
            return True
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _signal_group(self, sig: int) -> None:
        """Signal the child's whole process group; fall back to the child alone."""
        assert self._proc is not None
        try:
            os.killpg(self._proc.pid, sig)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                self._proc.send_signal(sig)
            except OSError:
                pass

    def _kill_session(self) -> int:
        """SIGKILL every process left in the child's session (its pid is the session
        id); the number signalled."""
        from fleet2.common.sysinfo import session_pids

        assert self._proc is not None
        count = 0
        for pid in session_pids(self._proc.pid):
            try:
                os.kill(pid, signal.SIGKILL)
                count += 1
            except OSError:
                pass
        return count

    def terminate(self) -> None:
        """Send SIGTERM to the process group (the child sets its stop flag)."""
        if self._proc is None or self._proc.poll() is not None:
            return
        self.term_sent = True
        self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        """Send SIGKILL to the process group, then to the rest of the session."""
        if self._proc is None:
            return
        self.kill_sent = True
        self._signal_group(signal.SIGKILL)
        self._kill_session()

    def stop(self, grace: float = 3.0) -> None:
        """SIGTERM, wait up to grace seconds, then SIGKILL survivors."""
        self.terminate()
        if not self.wait(grace):
            log.warning("runner %s ignored SIGTERM for %.1fs, killing", self.job_id, grace)
            self.kill()
            self.wait(5.0)
        self.reap_group()

    def join_reader(self, timeout: float = 2.0) -> None:
        if self._reader is not None:
            self._reader.join(timeout)

    def reap_group(self, timeout: float = 2.0) -> None:
        """Call once the child exited: a grandchild still holding its stdout keeps the
        reader alive, so SIGKILL the whole group and then wait for the reader; then
        SIGKILL whatever is left in the session (processes in other groups, such as
        search workers, do not hold the pipe and would otherwise run on unseen)."""
        self.join_reader(0.2)
        if self._reader is not None and self._reader.is_alive():
            log.warning("runner %s left children behind; killing its process group", self.job_id)
            self.group_killed = True
            if self._proc is not None:
                self._signal_group(signal.SIGKILL)
            self.join_reader(timeout)
        if self._proc is not None and self._proc.poll() is not None and self._kill_session():
            log.warning("runner %s left processes in its session; killed them", self.job_id)
            self.group_killed = True

    # state -----------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.poll() is not None

    @property
    def outcome(self) -> str | None:
        """None while running, else one of done, error, stopped, crashed."""
        if not self.finished:
            return None
        self.join_reader(1.0)
        with self._lock:
            if self.done:
                return "done"
            if self.error is not None:
                return "error"
            if self.stopped:
                return "stopped"
            return "crashed"

    def snapshot(self) -> tuple[dict[str, Any] | None, float | None, int]:
        """(last_checkpoint, last_progress, checkpoint_seq) under the lock."""
        with self._lock:
            return self.last_checkpoint, self.last_progress, self.checkpoint_seq

    # reader thread ---------------------------------------------------------

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for raw in self._proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                log.warning("runner %s: unparseable line: %s", self.job_id, line[:200])
                continue
            if isinstance(msg, dict):
                self._apply(msg)

    def _apply(self, msg: dict[str, Any]) -> None:
        with self._lock:
            if "checkpoint" in msg and isinstance(msg["checkpoint"], dict):
                self.last_checkpoint = msg["checkpoint"]
                self.checkpoint_seq += 1
            if "progress" in msg:
                if msg["progress"] is None:
                    self.last_progress = None
                else:
                    try:
                        self.last_progress = float(msg["progress"])
                    except (TypeError, ValueError):
                        pass
            if isinstance(msg.get("detail"), str):
                self.detail = msg["detail"][:200]
            if msg.get("done"):
                self.done = True
                self.result = msg.get("result")
                self.last_progress = 1.0
            if msg.get("stopped"):
                self.stopped = True
            if "error" in msg:
                self.error = str(msg["error"])


if __name__ == "__main__":
    sys.exit(run_child())
