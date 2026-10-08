# fleet-v2

Alpaca stock and crypto research and paper trading on a rack of old computers. One
coordinator (box1) holds the Alpaca keys, the price data and the dashboard; the
workers (w1, w2, ...) run backtests, model searches and paper-trading models and send
their signals to the coordinator, which is the only thing that places orders.

Paper trading only by default. Status: **stage 1 of 5** (worker stats and the Fleet
screen). The plan is in `PLAN.md`.

## Start the coordinator (box1, Debian with Docker)

```bash
git clone https://github.com/notrelaxedd/fleetv2.git fleet-v2 && cd fleet-v2
cp .env.example .env        # then edit it (see below)
chmod 600 .env
docker compose up -d --build
curl -s http://127.0.0.1:8090/healthz      # {"ok": true, "db": true}
sudo tailscale serve --bg --https=443 http://127.0.0.1:8090
tailscale funnel status                     # must show nothing public
```

Open `https://box1.<your-tailnet>.ts.net` from any device signed in to Tailscale as
the login in `FLEET_OWNER_LOGIN`.

## Where the Alpaca paper keys go

Only in `.env` on box1, never anywhere else (workers never see them):

```
ALPACA_PAPER_KEY_ID=...
ALPACA_PAPER_SECRET_KEY=...
```

Make them in the Alpaca dashboard under Paper account, API Keys. Then
`docker compose up -d coordinator`. Until keys are present the Account tiles say
"Add your Alpaca paper keys to .env on box1". Live trading stays impossible unless
`ALPACA_LIVE=true` is set **and** you confirm it in the dashboard (stage 4).

Money limits are in `config/limits.toml` (edit, then `docker compose restart coordinator`).

## Add a worker

On box1, make a one-time token:

```bash
docker compose exec coordinator python -m coordinator.cli enroll-token
```

On the worker (Debian, as root), with the token and the worker's name:

```bash
curl -fsSL https://box1.<tailnet>.ts.net/install.sh | sudo bash -s -- https://box1.<tailnet>.ts.net <token> --name w5
```

It installs `python3-psutil` and `python3-numpy` from apt, puts fleet-v2 in
`/var/lib/fleet2` as user `fleet2` and starts `fleet2-worker`. It never touches v1.

### Workers that also have v1

If the v1 agent is running, the installer enrolls the box but does **not** start
fleet-v2. Switch when you are ready, one box at a time:

1. In the v1 dashboard, set the box idle and disable it (never switch a box in v1's trade role).
2. `sudo fleet2 use v2`

To go back: wait for its v2 job to finish (or cancel it), then `sudo fleet2 use v1`
and re-enable the box in v1. `fleet2 status` shows which agent runs. The two can never
run at once (`Conflicts=` in the service file).

## Try stage 1

- The Fleet screen shows every worker's CPU, RAM and temperature ("no sensor" when the
  machine has none), refreshed every 5 seconds.
- Send a test job and watch its progress bar:
  `docker compose exec coordinator python -m coordinator.cli send-test-job --seconds 120`
  (`--target w3` or `--target all_idle` also work).
- Unplug or power off a worker mid-job: within about 30 seconds the card says
  "Offline", and the job shows as Failed "Worker w3 went offline" with a Run again button.
- Pause all trading / Resume trading flips the banner (no orders exist yet).
- The four job types in "Assign a job" switch on in stages 2, 4 and 5.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[coordinator,dev]'
# a local Postgres 16 superuser at postgresql://postgres:postgres@127.0.0.1:5432/postgres
.venv/bin/python -m pytest tests -q
```
