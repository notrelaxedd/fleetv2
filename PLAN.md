# fleet-v2: v1 summary and build plan

Status: draft for the owner's OK. No code has been written yet.
v1 source read: `notrelaxedd/polymarket-fleet` @ 9c5485a, cloned read-only to
`/home/user/notrelaxedd/polymarket-fleet` (never edited).

---

## Part 1: how v1 works

### 1. Coordinator and workers talking
- Plain HTTPS + JSON over the tailnet. The coordinator ("host") is a FastAPI app
  (`host/api/app.py`) backed by Postgres, run by Docker Compose (`db`, `host`, `exchange`)
  on the Windows 11 PC, published with `tailscale serve`.
- Workers only ever call the coordinator, never the other way round
  (`docs/PROTOCOL.md`):
  - `POST /api/v1/workers/register`: a one-time enroll token gives a worker id and a
    bearer token. The token rotates on every agent start (`host/auth.py`, `host/api/workers.py`).
  - `POST /api/v1/workers/{id}/heartbeat` every 3 to 5 s. It sends CPU, RAM, temperature
    (`fleet/common/sysinfo.py`, `fleet/common/hwinfo.py`), its role, and the progress and
    checkpoint of each running job. The reply carries newly claimed jobs, jobs to stop
    (`preempt`, `cancel`, `lost`), the kill flag and the current code version.
  - `/api/v1/jobs/{id}/checkpoint | complete | fail`, fenced by a per-lease token.
  - Self-update: the worker downloads `/dl/worker.tar.gz` when the code version
    changes, checks the sha256 and rolls back after 3 failed starts (`fleet/worker/update.py`).
- Owner pages: the `Tailscale-User-Login` header must match `FLEET_OWNER_LOGIN`.

### 2. Queueing, assigning and reporting jobs
- One `jobs` table in Postgres: `queued -> leased -> succeeded | failed | cancelled`
  (`host/migrations/0001_init.sql`, `host/leases.py`, `host/heartbeat.py`).
- Claiming happens inside the heartbeat with `FOR UPDATE SKIP LOCKED`, so no job is
  ever taken twice. A lease lasts 30 s and every heartbeat renews it.
- Workers have a role (idle, backtest, model_search, train, trade). A job sent to
  "any idle" or to a chosen worker flips that worker's role, and the worker goes back to
  idle on its own afterwards. This uses an epoch handshake (`host/scheduling.py`).
- A background loop runs every 5 s (`host/loop.py`, `host/recovery.py`):
  - the reaper requeues a job whose lease expired (dead worker), keeping its
    checkpoint, and fails it after 3 expiries
  - the dispatcher hands waiting jobs to idle workers
- Each job runs in a child process (`fleet/worker/runner.py`). It prints one JSON line
  per unit of work with its checkpoint and progress (0 to 1), and each unit takes under
  3 s so a stop is always quick. The agent forwards progress on every heartbeat.

### 3. The model-search loop
- There is no LLM and no API key. It is a seeded random parameter search
  (`fleet/sim/search.py`, `fleet/models/search_space.py`):
  - Each model family has `search_space(rng)`, which draws one random set of parameters.
  - Candidate `i` uses `random.Random(f"{seed}:{i}")`, so a resumed job repeats exactly.
- Each candidate is backtested on the search era and ranked by "shrunk ROI"
  (`roi * n/(n+100)`). The best 5 are kept.
- The 5 kept candidates are then re-run once on a held-out validation era that the
  search never saw (`fleet/sim/validate.py`). They also get stress tests (worse costs,
  nudged parameters) and the flags overfit, fragile and regime_dependent
  (`fleet/sim/stress.py`, `fleet/sim/stats.py`).
- The coordinator stores the survivors as `models` rows with a templated three-sentence
  summary (`fleet/models/summary_text.py`). Searches can use several cores
  (`fleet/sim/parallel.py`).

### 4. How the dashboard is served
- The same FastAPI process serves Jinja2 server-rendered pages (`host/templates/`) with
  one stylesheet (`host/static/style.css`), self-hosted woff2 fonts and a small vanilla
  `app.js`.
- `app.js` re-fetches HTML fragments every few seconds (`/fragments/fleet`,
  `/fragments/topbar`).
- The one exception is a React + three.js "3D city" page built with Vite (`fleet-ui/`).
- There is no other build step.

### What v1 already says about Alpaca
`docs/ALPACA.md` plans stocks and crypto as a future "step 9". Only a read-only probe was
built, and it is not on the trading path. v2 replaces that plan.

---

## Part 2: build plan

### What gets copied from v1 and adapted (no Polymarket or Kalshi code)
| v2 | From v1 | Change |
|---|---|---|
| `coordinator/db.py`, `migrations/` | `host/db.py`, `0001_init.sql` | fresh schema: workers, jobs, job_events, models, bars, orders, positions, settings, audit |
| `coordinator/leases.py`, `heartbeat.py`, `recovery.py`, `loop.py`, `queue.py` | same names in `host/` | roles and the epoch handshake are dropped; a worker claims jobs aimed at it, or untargeted ones when idle. "Auto" picks the online idle worker with the lowest CPU. A dead worker's job is **failed** (see Q3). |
| `coordinator/auth.py`, `bundle.py` (worker tarball and self-update) | `host/auth.py`, `host/bundle.py` | as is |
| `worker/agent.py`, `runner.py`, `posts.py`, `update.py`, `watchdog.py`, `config.py`, `launch.py` | `fleet/worker/*` | trade and NFL bits removed; heartbeat interval 5 s |
| `worker/stats.py` | `fleet/common/sysinfo.py`, `hwinfo.temp_c` | `psutil` first, then `/sys/class/thermal` and hwmon, else "no sensor" |
| `sim/search.py`, `parallel.py`, `stats.py` | `fleet/sim/*` | same seeded search; returns-based instead of bets-based |
| `deploy/install_worker.sh`, `fleet2-worker.service` | `deploy/*` | new paths and names so v1 and v2 never collide |
| `dashboard/templates`, `static/` | `host/templates`, `app.js` refresh pattern | two screens only; the 3D page is not copied |

### New in v2
- `coordinator/broker.py` is the only module that imports `alpaca-py`.
  - It uses `TradingClient` for the account, clock, orders and positions, and the
    historical data clients for bars.
  - It is paper by default. Live needs `ALPACA_LIVE=true` in `.env` **and** a typed
    confirmation in dashboard settings.
- `coordinator/safety.py` holds the pause switch, the daily loss check and the
  per-model and per-position caps (from `config/limits.toml`). Every order request goes
  through one function, `approve_and_place()`, which logs model, worker, time and reason.
- `coordinator/data.py` handles data refresh:
  - free IEX bars for stocks and Alpaca crypto bars, cached in Postgres
  - served to workers as gzipped JSON with an ETag, which workers keep in memory only
- `models/` has one file per model:
  - the interface is `target_positions(history, params) -> {symbol: weight}`
  - each file carries `NAME`, `MARKET`, `DESCRIPTION`, `HOW_IT_WORKS` and `SEARCH_SPACE`
  - starter files: `momentum.py`, `dip_buy.py`, `crypto_trend.py` and `pairs.py`
- `sim/backtest.py` is the honest backtester:
  - **Lookahead guard:** a `History` view that can only see bars before T and raises on
    anything later.
  - **Costs:** slippage on every fill, plus a crypto taker fee of 0.25% (Alpaca's lowest tier).
  - **Data split:** a train period and a held-out period; ranking uses held-out results only.
  - **Trade minimum:** fewer than 100 trades means "not enough trades", and such a model is never ranked first.
  - **Benchmark:** SPY or BTC buy-and-hold over the same period.
  - **Metrics:** all 8 metrics from the spec.
- Paper trade job:
  - The worker fetches fresh bars and computes targets, then posts a signal to the
    coordinator.
  - The coordinator checks it, sizes it within the model's $10,000 and $1,000 per
    position, and places the order.
  - Each model has its own virtual book; account value comes from Alpaca.
  - The job reports "Live" instead of a percent.

### Alpaca facts (looked up; Alpaca's own site is blocked from this sandbox, so these come from search results quoting the docs)
- Trading API: 200 calls/min per account. Paper and live are counted separately.
- Market data on the free plan: 200 calls/min, IEX feed for stocks. The coordinator is
  the only caller and throttles itself to stay under the limit.
- Crypto:
  - Symbols are written like `BTC/USD`.
  - Trading runs 24/7.
  - Crypto cannot be shorted and cannot be bought on margin.
  - Allowed time-in-force values are `gtc` and `ioc`.
  - Fractional amounts are allowed.
- Crypto fees: 0.15% maker and 0.25% taker at the lowest volume tier.
- Pattern day trader rule: Alpaca says FINRA retired it on **June 4, 2026**, with an
  intraday margin framework replacing it, and the paper account no longer simulates
  PDT checks. So v2 adds no PDT guard. Re-check this on box1 with a real paper key.
- I cannot reach Alpaca's API from here. All Alpaca calls are tested against a fake
  broker, and the first real call happens on box1.

### Switching a worker between v1 and v2
- v2 installs as `fleet2-worker.service`, with user `fleet2`, state in
  `/var/lib/fleet2` and `Conflicts=fleet-worker.service`. systemd will never run both at
  once, and v2 never touches `/var/lib/fleet`.
- To move a worker to v2:
  1. In the v1 dashboard, set the box idle and disable it. Never move a box in the
     trade role.
  2. Run `sudo fleet2 use v2`. This runs `systemctl disable --now fleet-worker` and
     then `enable --now fleet2-worker`.
- To move it back: run `sudo fleet2 use v1`, then re-enable the box in v1.
- I will not run either command on any machine. You run them, one box at a time.

### Disk on flash-drive workers
- No data on disk; bars are held in memory and `/tmp` (a tmpfs via `PrivateTmp`).
- Logs: warnings only, plus a rate limit in the unit.
- State is only `worker.conf` and the code, and it changes only on update.

### Stages (each ends with tests green, a local commit and a demo before the next one)
1. Worker stats agent and the Fleet screen with real data (header, tiles, worker cards,
   Assign job, Previous jobs, using a sleep test job).
2. Data refresh and backtest jobs with the four starter models.
3. The Models screen with real metrics, the Growth of $100 chart and metric cards.
4. Paper trading through the coordinator, with pause, daily loss and limits. A fresh
   Opus reviewer checks the backtester and safety code before this stage is called done.
5. Model search.

### Agents
- Main session (Opus): reading v1, architecture, backtester, lookahead guard, broker
  and safety, and checking every subagent by running its output.
- Sonnet subagents, each on files it alone owns:
  - dashboard
  - worker stats and data refresh
  - tests and README
- One fresh Opus reviewer before stage 4 is done.
- No Haiku anywhere.

### Tests
- backtester (costs, metrics, benchmark)
- the lookahead guard: a model that peeks at future bars must raise
- the daily loss auto-pause
- the pause switch, including an order arriving mid-pause
- the queue: a dead worker's job is failed and can be re-queued

### Open questions (see chat)
