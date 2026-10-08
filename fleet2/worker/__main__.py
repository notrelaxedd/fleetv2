"""Command line: python3 -m fleet2.worker {run|enroll|status}."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import sys

import fleet2 as fleet
from fleet2.common import http, sysinfo
from fleet2.worker import config, launch

ENROLL_TOKEN_ENV = "FLEET_ENROLL_TOKEN"


def _setup_logging(verbose: bool) -> None:
    """journald (and any non-tty stderr) adds its own timestamp; add ours only on a tty.

    Warnings only by default: some workers boot from 16 GB flash drives, so the agent
    does not write a line per heartbeat or per job. -v turns on info and debug lines."""
    fmt = "%(levelname)s %(name)s: %(message)s"
    if sys.stderr.isatty() and not os.environ.get("JOURNAL_STREAM"):
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if verbose else logging.WARNING, format=fmt)


def cmd_run(args: argparse.Namespace) -> int:
    """Run the agent until SIGTERM/SIGINT; propagate its exit code.

    The agent import and start are guarded: when a self-updated version fails to
    start MAX_FAILED_STARTS times, launch.failed_start rolls app/current back to the
    previous version and exits 75 so systemd restarts the old code.
    """
    log = logging.getLogger("fleet2.agent")
    state_dir = config.state_dir()
    app = config.app_dir(state_dir)
    starts = launch.note_start(app)
    if starts is not None:
        log.info("start %d of pending version %s", starts, launch.current_target(app))
    try:
        from fleet2.worker.agent import Agent

        agent = Agent(state_dir=state_dir)
    except Exception:
        log.exception("agent failed to start")
        return launch.failed_start(app, starts)

    def _request_stop(signum: int, frame: object) -> None:
        log.info("signal %d received, stopping", signum)
        agent.stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        return agent.run_forever()
    except Exception:
        log.exception("agent crashed")
        if starts is not None and launch.read_pending(app) is not None:
            return launch.failed_start(app, starts)
        return 1


def enroll(host_url: str, token: str, name: str | None, state_dir: str) -> dict:
    """Register with an enroll token and write worker.conf (0600)."""
    host_url = host_url.rstrip("/")
    payload = {
        "enroll_token": token,
        "hostname": sysinfo.hostname(),
        "name": name or sysinfo.hostname(),
        "python_version": platform.python_version(),
        "code_version": fleet.__version__,
        "boot_id": sysinfo.boot_id(),
    }
    resp = http.post_json(host_url + "/api/v1/workers/register", payload, timeout=10.0)
    if not isinstance(resp, dict) or not resp.get("worker_id") or not resp.get("worker_token"):
        raise SystemExit(f"unexpected register response: {resp!r}")
    conf = {"host_url": host_url, "worker_id": str(resp["worker_id"]), "worker_token": str(resp["worker_token"])}
    config.save_conf(state_dir, conf)
    return conf


def cmd_enroll(args: argparse.Namespace) -> int:
    state_dir = config.state_dir()
    token = args.token or os.environ.get(ENROLL_TOKEN_ENV, "")
    if not token:
        print(f"enroll failed: give --token or set {ENROLL_TOKEN_ENV}", file=sys.stderr)
        return 2
    try:
        conf = enroll(args.host, token, args.name, state_dir)
    except http.HttpError as exc:
        print(f"enroll failed: {exc}", file=sys.stderr)
        return 1
    except http.HttpConnectionError as exc:
        print(f"enroll failed: host unreachable: {exc}", file=sys.stderr)
        return 1
    print(f"enrolled as {conf['worker_id']}; wrote {config.conf_path(state_dir)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print the conf (token redacted) and the last heartbeat result."""
    state_dir = config.state_dir()
    try:
        conf = config.load_conf(state_dir)
    except config.ConfMissing as exc:
        print(f"not enrolled: {exc}")
        return 1
    out = {"conf": config.redacted(conf), "code_version": fleet.__version__, "status": config.load_status(state_dir)}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m fleet2.worker", description="fleet-v2 worker agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the agent (systemd)")
    run.set_defaults(func=cmd_run)
    enr = sub.add_parser("enroll", help="register with an enroll token and write worker.conf")
    enr.add_argument("--host", required=True, help="host URL, e.g. http://100.x.y.z:8080")
    enr.add_argument(
        "--token",
        default=None,
        help=f"enroll token from the dashboard (or set env {ENROLL_TOKEN_ENV}, which keeps it off the command line)",
    )
    enr.add_argument("--name", default=None, help="worker name (default: hostname)")
    enr.set_defaults(func=cmd_enroll)
    status = sub.add_parser("status", help="print the conf and the last heartbeat result")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
