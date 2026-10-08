# fleet-v2

Alpaca stock and crypto research and paper trading on a rack of old computers. One
coordinator (box1) holds the Alpaca keys, the price data and the dashboard, and is the
only thing that places orders. The workers (w1, w2, ...) run backtests, model searches
and paper-trading models and send their decisions to the coordinator. A worker never
holds a key.

Paper trading only by default. Real money needs several deliberate steps (see
"Safety controls"). The build plan and its reasons are in `PLAN.md`.

All commands below are run as root (`su -` first). Debian 13 often has no `sudo`, so none
of them use it.

## Start the coordinator (box1, Debian with Docker)

```bash
git clone https://github.com/notrelaxedd/fleetv2.git fleet-v2 && cd fleet-v2
cp .env.example .env        # then edit it (see below)
chmod 600 .env
docker compose up -d --build
curl -s http://127.0.0.1:8090/healthz      # {"ok": true, "db": true}
tailscale serve --bg --https=443 http://127.0.0.1:8090
tailscale funnel status                     # must show nothing public
```

In `.env`, set `FLEET_PUBLIC_URL` (your `https://box1.<your-tailnet>.ts.net`),
`FLEET_OWNER_LOGIN` (your Tailscale login) and `POSTGRES_PASSWORD`. Open the
`FLEET_PUBLIC_URL` address from any device signed in to Tailscale as that login.

## Where the Alpaca paper keys go

Only in `.env` on box1, never anywhere else (workers never see them):

```
ALPACA_PAPER_KEY_ID=...
ALPACA_PAPER_SECRET_KEY=...
```

Make them in the Alpaca dashboard under Paper account, API Keys. Then
`docker compose up -d coordinator` to load them. Until keys are present the Account tile
says "Add your Alpaca paper keys to .env on box1".

Stock prices come from `ALPACA_DATA_FEED=iex`, Alpaca's free feed. `sip` is Alpaca's paid
feed: set it only with the owner's OK.

## Add a worker

On box1, make a one-time token (valid for one hour):

```bash
docker compose exec coordinator python -m coordinator.cli enroll-token
```

On the worker (Debian with python3 3.11 or newer, run as root), with the token and the
worker's name:

```bash
curl -fsSL https://box1.<tailnet>.ts.net/install.sh | bash -s -- https://box1.<tailnet>.ts.net <token> --name w5
```

It installs `python3-psutil` and `python3-numpy` from apt, puts fleet-v2 in
`/var/lib/fleet2` as user `fleet2`, adds the `fleet2` switch command and starts
`fleet2-worker`. The worker updates itself from box1 when the code changes.

### Workers that also have v1

v1 (polymarket-fleet) keeps running on some workers. The installer never stops, disables
or removes v1. If the v1 agent is running, it enrolls the box but does **not** start
fleet-v2. Switch when you are ready, one box at a time:

1. In the v1 dashboard, set the box idle and disable it. Never switch a box that is in
   v1's trade role.
2. `fleet2 use v2`

To go back: wait for its v2 job to finish (or cancel it), then `fleet2 use v1` and
re-enable the box in v1. `fleet2 status` shows which agent is running. The two can never
run at once (`Conflicts=` in the service file).

## Using it

The dashboard has two screens, **Fleet** and **Models**. The header shows the trading
mode, whether stocks and crypto are open, and the Pause all trading button.

![Models screen](docs/screenshots/stage5-models-desktop.png)

(The screenshots use demo data, see Development.)

**Fleet** shows every worker's CPU, RAM and temperature ("no sensor" when the machine
has none), refreshed every 5 seconds, the Alpaca account tiles, the running jobs and
the previous jobs. **Assign a job** sends one of four jobs to Auto (the idle worker
with the lowest CPU), to one worker, or to every idle worker:

- **Data refresh**: download the latest prices to the coordinator.
- **Backtest**: test one model on past prices and save its results.
- **Paper trade**: run one model against the Alpaca paper account.
- **Model search**: invent new models, test them, keep the good ones, repeat until stopped.

First run, in order:

1. Assign a **Data refresh** and wait for it to finish. Backtests need the prices.
2. **Run backtest** for each of the four starter models (Momentum, Dip buyer, Crypto
   trend, Pairs), from the Models screen or Assign a job.
3. Open **Models** to compare them.

![Fleet screen](docs/screenshots/stage5-fleet-desktop.png)

**Models** lists every model ranked by ROI on the held-out period (the last part of the
prices, see "Honest backtesting"). A model with under 100 trades says "Not enough
trades" and is never ranked first. Pick a model to see its Growth of $100 chart against
buying and holding SPY (or BTC for crypto), and eight numbers: ROI, vs. buy and hold,
max drawdown, Sharpe ratio, win rate, profit factor, trades and average hold. Each has a
plain-words explanation under it.

**Start paper trading** (on a tested model) gives the model its own virtual book of
$10,000 and a worker to run it. The worker sends the model's decisions to the
coordinator, which places the orders in the Alpaca paper account. **Stop paper trading**
sells the model's positions and closes its book. While models paper trade, the
coordinator keeps their prices fresh by itself.

**Start model search** tries new settings for the model files. It draws seeded random
settings, tests each on the training period only, keeps the best, and then measures it
once on the held-out period it never saw. It repeats until you press Stop model search,
and keeps at most five found models per model file (the one with the lowest training
score is dropped when a better one is found).

## Safety controls

- **Pause all trading**: one button in the header. While paused no model places an
  order. Backtests and model searches keep running. Resume trading is the same button.
- **Daily loss limit**: if the Alpaca account is down more than 2% on the day, all
  trading pauses by itself and says why. Only you resume it.
- **Per-model limits** in `config/limits.toml`: starting balance 10,000, max per position
  1,000, max per model 10,000 (dollars). A worker cannot raise them. Edit the file, then
  `docker compose restart coordinator`. The daily loss percent and the temperature at
  which a worker card turns amber are in the same file.
- **Every order is logged** with the model, the worker, the time and the reason,
  including orders that were blocked and why. See a model's last 100 orders at
  `GET /api/models/<id>/orders` (open `https://box1.<tailnet>.ts.net/api/models/<id>/orders`
  in the browser).
- **Market hours**: stock orders wait for the market to open (the model shows "Waiting
  for the stock market to open"). Crypto trades 24/7.
- **Real money** needs all of these: `ALPACA_LIVE=true` and the live keys
  (`ALPACA_LIVE_KEY_ID`, `ALPACA_LIVE_SECRET_KEY`) in `.env` on box1, you typing
  "TRADE REAL MONEY" in the dashboard's Trading mode panel (click the Paper pill in the
  header), and a coordinator restart. The mode never switches by itself, and the pill
  says Live when it is live.

## Honest backtesting

- **No lookahead.** A model only sees bars that closed before its decision, and asking
  for a later bar raises an error. `tests/test_lookahead.py` proves it.
- **Costs on every fill.** Stocks: 5 bp slippage, no commission. Crypto: 10 bp slippage
  plus Alpaca's 25 bp taker fee.
- **Training vs held-out.** The last 25% of the price history is held out. Model search
  never sees it, and models are ranked on it only.
- **Benchmark.** Buy and hold SPY for stock models, BTC for crypto models, over the same
  period.
- **Same limits as paper trading.** Backtests use the same balance and the same
  per-position and per-model caps.

## When a worker goes offline

Within about 30 seconds its card says "Offline" and its job fails with "Worker w3 went
offline". **Run again** sends the job to another worker. A paper-trading model keeps its
book and positions, and Run again resumes it. Stopping a model on purpose is different:
start it again from the Models screen.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[coordinator,dev]'
# a local Postgres 16 superuser at postgresql://postgres:postgres@127.0.0.1:5432/postgres
.venv/bin/python -m pytest tests -q
```

Demo mode: `FLEET_FAKE_BROKER=1` uses made-up prices and a fake Alpaca account, so the
whole thing can be tried with no keys. The dashboard labels it "demo data" on screen.
Never use it with real keys. For a local run also set `FLEET_DEV=1` (no Tailscale login
needed) and `DATABASE_URL` to a Postgres database, then `python -m coordinator.main`.

## Future options (not built)

Neither is switched on without the owner's OK. See `PLAN.md`.

- **Paid data**: Alpaca's SIP feed (`ALPACA_DATA_FEED=sip`), with each bar set recording
  which feed it came from.
- **Topstep funded futures**: a second broker beside Alpaca, with its own safety checks.
  Topstep's terms and API cost need checking first.
