"""Coordinator command line: python -m coordinator.cli <command>.

  migrate                apply database migrations
  enroll-token           print a single-use token and the worker install command
  workers                list workers
  send-test-job [--seconds N] [--target auto|all_idle|<worker name>]
                         a sleep job, to watch progress on the Fleet screen
"""
from __future__ import annotations

import argparse
import json
import sys

from coordinator import auth, db, queue
from coordinator.config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m coordinator.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate")
    sub.add_parser("enroll-token")
    sub.add_parser("workers")
    test = sub.add_parser("send-test-job")
    test.add_argument("--seconds", type=int, default=60)
    test.add_argument("--target", default="auto")
    args = parser.parse_args(argv)
    config = Config.from_env()
    if args.command == "migrate":
        print("applied:", db.migrate(config.database_url) or "none")
        return 0
    with db.connect(config.database_url) as conn:
        if args.command == "enroll-token":
            token, expires = auth.create_enroll_token(conn)
            url = config.public_url
            print(f"token (single use, expires {expires:%Y-%m-%d %H:%M} UTC): {token}")
            print(f"on the worker, as root (su -): curl -fsSL {url}/install.sh | bash -s -- {url} {token} --name w<N>")
        elif args.command == "workers":
            for row in conn.execute("SELECT id, name, enabled, last_heartbeat_at, cpu_pct, ram_pct, temp_c FROM workers ORDER BY name"):
                print(json.dumps({k: (str(v) if v is not None else None) for k, v in row.items()}))
        elif args.command == "send-test-job":
            target = args.target
            if target not in (queue.AUTO, queue.ALL_IDLE):
                row = conn.execute("SELECT id FROM workers WHERE name = %s OR id = %s", (target, target)).fetchone()
                if row is None:
                    print(f"no worker named {target!r}", file=sys.stderr)
                    return 1
                target = row["id"]
            result = queue.create_job(conn, "sleep", {"seconds": args.seconds}, target)
            for job in result.jobs:
                print(f"job {job['id']} -> {job['target_worker_id'] or 'waiting for a free worker'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
