"""Coordinator command line: python -m coordinator.cli <command>.

  migrate                apply database migrations
  enroll-token           print a single-use token and the worker install command
  workers                list workers
  send-test-job [--seconds N] [--target auto|all_idle|<worker name>]
                         a sleep job, to watch progress on the Fleet screen
  futures-prices         a Futures prices job: download MES and MNQ 1-minute prices
                         (the Models screen's Futures view has the same button)
  reset-search-log [--market stocks|crypto|futures] [--family NAME]
                         forget what model search has tried (all of it, one market or one
                         model file/recipe), so families that gave up try again
  news-replay --since YYYY-MM-DD [--until YYYY-MM-DD] --max-usd N [--yes]
                         replay news published after Claude's training cutoff as if it
                         were live and score the predictions (docs/NEWS_PLAN.md, section
                         1). Without --yes it only says what it would do and spends
                         nothing; --until is the start of that day (UTC), default now
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone

from coordinator import auth, db, haiku, limits as limits_mod, news_feed, news_replay, queue, research, search
from coordinator.config import Config


def _day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date like 2026-08-01") from None


def _news_replay(config: Config, args: argparse.Namespace) -> int:
    """Without --yes: say what a replay would do. With it: run the replay (autocommit, so
    every article's payment is saved as it is made and a stop loses nothing)."""
    now = datetime.now(timezone.utc)
    since = news_replay.day_start(args.since)
    until = news_replay.day_start(args.until) if args.until else now
    limits = limits_mod.load_limits(config.limits_path)
    settings = haiku.load_settings(config.ai_path)
    analyst = research.make_analyst(config, settings)
    try:
        news_replay.check_window(limits, since, until, now)
        if not args.max_usd > 0:
            raise ValueError("--max-usd must be more than zero.")
        with db.connect(config.database_url, autocommit=True) as conn:
            if not args.yes:
                counts = news_replay.window_counts(conn, since, until)
                print(f"News replay of {since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M} UTC, spending at most "
                      f"${args.max_usd:,.2f}.")
                print(f"Stored for this window: {counts['stored']:,} articles; {counts['replayed']:,} already "
                      f"replayed, {counts['waiting']:,} waiting.")
                if counts["waiting"]:
                    print(f"Rough cost of the waiting ones: {counts['waiting']:,} x about "
                          f"${news_replay.ESTIMATE_PER_ARTICLE_USD:.2f} = "
                          f"${counts['waiting'] * news_replay.ESTIMATE_PER_ARTICLE_USD:,.2f}.")
                else:
                    print("Nothing is waiting yet: a real run first downloads the window's articles from "
                          "Alpaca's news feed (free), then replays them.")
                if not analyst.connected:
                    print(f"Claude is not connected ({analyst.problem}); a real run would refuse.")
                print("Nothing was spent. Add --yes to run it; it stops at the amount above and can be run again "
                      "to carry on.")
                return 0
            news_replay.replay(conn, news_feed.make_news_source(config), analyst, settings, limits, since, until,
                               args.max_usd, now)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m coordinator.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate")
    sub.add_parser("enroll-token")
    sub.add_parser("workers")
    test = sub.add_parser("send-test-job")
    test.add_argument("--seconds", type=int, default=60)
    test.add_argument("--target", default="auto")
    sub.add_parser("futures-prices")
    reset = sub.add_parser("reset-search-log")
    reset.add_argument("--market", choices=search.LOG_MARKETS)
    reset.add_argument("--family")
    replay = sub.add_parser("news-replay")
    replay.add_argument("--since", required=True, type=_day, metavar="YYYY-MM-DD")
    replay.add_argument("--until", type=_day, metavar="YYYY-MM-DD")
    replay.add_argument("--max-usd", required=True, type=float)
    replay.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    config = Config.from_env()
    if args.command == "migrate":
        print("applied:", db.migrate(config.database_url) or "none")
        return 0
    if args.command == "news-replay":
        return _news_replay(config, args)
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
        elif args.command == "futures-prices":
            result = queue.create_job(conn, "data_refresh", {"markets": ["futures"]}, queue.AUTO)
            job = result.jobs[0]
            print(f"futures prices job {job['id']} -> {job['target_worker_id'] or 'waiting for a free worker'}")
        elif args.command == "reset-search-log":
            deleted = search.reset(conn, args.market, args.family)
            what = " ".join(x for x in (args.market, args.family) if x) or "everything"
            print(f"search log reset ({what}): {deleted:,} row{'s' if deleted != 1 else ''} deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
