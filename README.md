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

## Futures prices (for Topstep)

The Futures market backtests and searches for day-trading models on the CME micro
futures MES (micro S&P 500) and MNQ (micro Nasdaq-100), paper trades them on Alpaca, and
trades the ones ready for a Combine on Topstep (see "Futures trading: Alpaca paper, then
Topstep"). Research uses 1-minute bars of the regular session only, 8:30 to 15:00
Chicago time (to 12:00 on half days), from May 2019.

Where the prices come from:

- **Databento** (paid, pay-as-you-go), when `.env` on box1 has a key:

  ```
  DATABENTO_API_KEY=...
  ```

  Then `docker compose up -d coordinator`. Only the coordinator reads the key, like the
  Alpaca keys. Before every download the coordinator asks Databento what bringing MES and
  MNQ up to date will cost, and refuses when that is more than `max_download_usd` in
  `config/topstep.toml` (10 dollars to start; raise it yourself if a first full download
  costs more). A refused download says the price and changes nothing.
  Databento's pay-as-you-go history stops about a day before now (the latest day needs
  a live-data license), so the futures prices end there; research only needs whole past days.
- **Proxy** (free), when there is no Databento key: SPY and QQQ 1-minute bars from Alpaca
  (your paper keys) standing in for MES and MNQ, scaled to about index points. Good for
  building and testing; the dashboard labels it "proxy" everywhere, and it can never
  earn a "ready for a Combine" verdict.
- **Synthetic** in demo mode (`FLEET_FAKE_BROKER=1`): made-up prices, labelled so.

To load them, press **Load futures prices** on the Models screen's Futures view (or
run `docker compose exec coordinator python -m coordinator.cli futures-prices` on box1).
It downloads a month of one symbol at a
time (about 90 steps for the full history, so give it a while the first time), and only
adds what is new after that. If you add a Databento key later, the next Futures prices
job replaces the proxy bars with Databento's, never mixing the two.

The futures history is then split by date into three periods, fixed once and never
moved: **training** (the first 60% of the days, the only part model search sees),
**held-out** (the next 25%, for ranking) and the **lockbox** (the last 15%, opened once
per model by Final check). Prices after the lockbox are not used yet. The split happens
the first time something needs it, so load the full history first.

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

The dashboard has two screens, **Fleet** and **Models** (with a Futures view, see
"Futures on the Models screen"). The header shows the trading
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

## Futures models

Five model files under `fleet2/models/futures/`, each a well-known day-trading idea.
That is deliberate: they are honest starting points, and whether any still works after
costs is exactly what the fleet finds out.

- **Opening range**: trades the first break above or below the range of the first 15
  to 60 minutes.
- **VWAP revert**: on quiet days, fades a price stretched far from the day's
  volume-weighted average price.
- **Trend day**: in the afternoon, trades in the direction of the first hour's move.
- **Gap fade**: bets that part of a large overnight gap closes in the first hours
  (skipped on days the contract rolled).
- **Pullback**: in a strong intraday trend, buys short dips (or sells short rallies).

Each can go long or short, decides on 1-, 3-, 5- or 15-minute bars, and has an optional
stop and target in ticks; model search tries all of these. Every one passes the
cut-off test (`tests/test_futures_cutoff.py`).

### Recipes: new ideas from building blocks

Model search can only tune the settings of an idea, so besides the five files it also
tries **recipes** (`fleet2/models/futures/recipe.py`): models put together from building
blocks, never from new code. A recipe picks:

- one **signal**: opening-range break, stretch from VWAP, a short and a longer average
  of today's prices crossing, a gap from yesterday, a move from today's open, a new high
  or low of the day, or short-term momentum. It can **follow** the signal or **fade** it;
- up to two **filters**: a quiet day, a busy day, on the trade's side of VWAP or of the
  open, after a gap, or without one;
- one **exit**: hold to the stop, target or close, out at a cross of VWAP, profit at
  VWAP, out on the opposite signal, or out after a number of bars;
- the day's first signal only or every signal, and long, short or both.

Every round of a futures search adds three new random recipes, each tried with a full
set of settings, and keeps tuning the recipes of models already kept. At most 15 recipe
models are kept at a time (a new find replaces the weakest when it scores higher), and
all recipe tries count together in the "chance this is luck" figure. A recipe model's
page says so ("recipe put together by a random mix of building blocks"), and its
description is written from its blocks.

Each building block uses only the bar itself and earlier bars, and
`tests/test_futures_recipes.py` runs the cut-off test on every block and on many random
recipes.

### Claude Haiku writing recipes

With a Claude API key, Claude Haiku 5.5 also writes recipes while a futures model search
runs (`coordinator/ai_ideas.py`, plan in `docs/AI_PLAN.md`). To turn it on, put the key in
`.env` on box1 (only the coordinator reads it) and run `docker compose up -d coordinator`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

- Haiku is shown the building blocks and every recipe tried so far with its **training**
  results only (best score, days traded, profit at double costs). Held-out and lockbox
  results never reach it, so it cannot tune its ideas to the tests that judge them.
- Its answer has a fixed shape (structured output), and each recipe must pass the same
  checks as a random one. Nothing it writes is ever run as code.
- Each search round takes up to three of its recipes **beside** the round's random ones,
  so the Futures view can show how many recipe models each has kept.
- Every call's cost is recorded. `config/ai.toml` sets a monthly cap ($5 to start), the
  model, how many recipes per call, at most six calls an hour, and the prices it counts
  with. A call is only made when the month's spend plus the most it could cost stays
  under the cap. At about $0.001 a call, $5 covers thousands of calls.
- The Futures view says whether Haiku is on, what it has written and kept this month,
  what it has cost against the cap, and the last problem if a call failed. A model whose
  recipe Haiku wrote says so on its page, with Haiku's one sentence on the idea.

### Claude Haiku's reviews and daily market note

With the same key, and within the same monthly cap:

- **Reviews** (`coordinator/ai_reviews.py`). Each futures model's page has a "Claude
  Haiku's review" card: a verdict word (promising, unclear or weak), what could be luck,
  what looks fragile and what to watch on paper, written from the numbers on that page.
  Reviews are **automatic** for models that beat their coin flip, have a Final check or
  trade, whenever their results change (a new backtest, a Final check, five more paper
  days). **Ask Claude Haiku for a review** asks for one on any tested model. At most 20
  a day (`[reviews]` in `config/ai.toml`, where automatic reviews can be turned off).
  Advice only: a review changes nothing, and the recipe writer never sees reviews.
- **Market note** (`coordinator/market_note.py`), an experiment. On each trading day at
  07:45 Chicago time, Haiku reads the headlines since the last close from Alpaca's news
  feed (the Alpaca paper keys) and writes a short note on the Futures view: the day's
  scheduled events, whether the news is quiet, normal or heavy, and a flag for an
  unusual day. Haiku may remember past markets, so the note can only be tested going
  forward: the Futures view compares futures paper results on flagged days with the
  other days, counting only notes written before the open, and says when that rests on
  too few days (under 40). Nothing trades on it.

## Futures on the Models screen

![Futures view](docs/screenshots/futures-models-desktop.png)

(Demo data: synthetic prices and made-up test fees, so it is labelled "Synthetic prices"
and can never be ready.)

**Models → Futures** (the tab above the list) is the futures research screen. At the
top it says where the futures prices come from ("proxy" when they are the SPY/QQQ
stand-in, "synthetic" in demo mode), with two buttons:

- **Load futures prices** starts a Futures prices job (see "Futures prices").
- **Start model search** searches the five futures model files and new recipes. It stays greyed out,
  with "Set the fee in config/topstep.toml" under it, until the commissions are filled
  in there. One search runs at a time, stocks and crypto or futures.

How futures model search works:

- **Score**: training is cut into four back-to-back parts. In each part the model's
  daily profit and loss at one contract and **double slippage** gives a Sharpe ratio
  (return against how bumpy it was). The score is the worst of the four, so a setting
  that only worked in one stretch scores low.
- **Gates**: it must trade on at least 100 different training days and make money at
  normal and at double slippage.
- **Proposals**: half of each round's settings are random, half are small changes to
  the models already kept. Every setting can be made again from its seed text, which
  names the parent model for a change.
- **Robustness**: the best setting of a round is tried again with each number moved 10%
  down and 10% up. If the middle one of those scores under half as well, the winner sat
  on a lucky peak and is dropped.
- **Keeping**: at most five found models per model file. A new find replaces the kept
  model with the most similar settings when it scores higher. A find whose daily
  results move more than 90% like a kept model's is the same idea twice: it only
  replaces that model if it scores higher.
- **Chance this is luck**: every setting tried is counted. The more tries, the more one
  of them looks good by luck alone; this figure (the deflated Sharpe ratio) says how
  likely that is for each model. Lower is better.
- Settings are tried on every core of the worker at once, and the prices are downloaded
  once and kept until they change. Stop model search stops it within a second.

The list is ranked by **expected net dollars per Combine attempt on the held-out
period**, at the model's best contract size. Only a model that beats its **coin-flip
twin** (a higher pass rate and more money per attempt than the same trades with random
directions) gets a rank, and nothing is ranked until the fees and the payout cap are set.
Pick a model to see its pass rate, median days to pass, expected payout and net, contract
size, worst day, best-day share, trades per day, average hold, its result at double
slippage and the chance it is luck, each with the coin-flip twin's number beside it,
plus a chart of its held-out profit and loss against the twin's.

**Run backtest** tests a futures model on training and held-out prices. **Final check**
opens the lockbox, the last 15% of the prices, for that one model: it runs once, at the
contract size the held-out test chose, and its result is kept forever (the database
refuses to change or delete it). It needs real Databento prices, so the lockbox is never
spent on stand-in prices.

## Futures trading: Alpaca paper, then Topstep

A futures model goes to Topstep only after it has traded on live prices in the Alpaca
paper account. Both are started from its page on the Futures view.

**Start paper trading on Alpaca** (any tested futures model, once the commissions are set
in `config/topstep.toml`). Alpaca has no futures, so each micro contract is traded as its
share equivalent in the paper account, long or short:

- 1 MES = 50 SPY shares (the S&P 500 is about 10 x SPY and MES pays $5 a point), about
  $33,000;
- 1 MNQ = 82 QQQ shares (the Nasdaq-100 is about 41 x QQQ and MNQ pays $2 a point).

A one-dollar move in SPY on 50 shares is then the same $50 as the matching move on one
MES contract, so the results read in futures dollars. The model decides on SPY and QQQ
1-minute prices from Alpaca, scaled to index points, exactly as in its proxy backtests.
It is never traded with real money: in live mode the coordinator refuses.

How it runs: a worker replays today through the backtester every few seconds, on the
newest closed minute, and sends the coordinator the contracts the model wants from the
next minute on (after its stop, target and the cut-off, the same logic the backtest
used). The coordinator places the orders and books the fills. Each day's result counts
the commission per contract, though Alpaca paper charges none, so it compares with the
backtest. The model's page shows what it holds, today's result, every day so far and
its live signal. `GET /api/models/<id>/futures/orders` lists its orders.

Its own limits, in `config/limits.toml` `[futures]`: at most 2 micro contracts per model
(`futures_paper_max_contracts`) and $150,000 of SPY and QQQ for all futures paper models
together (`futures_paper_max_dollars`). The Alpaca account's 2% daily loss limit and the
pause switch apply as for stocks.

**Start trading on Topstep** is only offered for a model whose whole "ready for a Combine"
checklist is ticked, including 20 Alpaca paper days. To connect TopstepX (a separate
paid API subscription at Topstep; make the key under Settings > API in TopstepX), put
these in `.env` on box1:

```
TOPSTEPX_USERNAME=...
TOPSTEPX_API_KEY=...
TOPSTEPX_ACCOUNT=...      # the account name (or id) to trade, e.g. your Combine
```

Then type **TRADE ON TOPSTEP** in the Topstep box on the Futures view and run
`docker compose restart coordinator`. Topstep stays off until all of that is done, and
never switches on by itself. On Topstep the model decides on the real contract's
1-minute prices from TopstepX and trades real micro contracts at its best contract size.

**Trading by hand instead:** the Futures view lists every trading model's live signal
(for example "long 1 MES"). You can follow it on TopstepX yourself rather than connecting
the API.

Safety rules for futures, checked on every order:

- **Flat every day**: no new trade in the 10 minutes before the flat time, and
  everything closed 2 minutes before 15:00 Chicago time (`flat_margin_minutes`, ahead of
  Topstep's 15:10 deadline). Closing for the day still happens while trading is paused,
  so an account is never left holding overnight.
- **A worker that stops sending decisions** (crashed, offline) gets its model closed out
  after 180 seconds (`stale_decision_seconds`).
- **Topstep's loss limit**: the coordinator counts the loss floor from the account's
  balance at the end of each day (`[live] account_start_balance`, the same rule as the
  simulator) and stops trading on Topstep, closing every position, once 80% of the loss
  limit is used (`stop_at_loss_share`). Only you resume it ("Resume Topstep trading").
  Open losses count toward it; open gains do not.
- **Topstep's contract limit**: no model above its own size, and never more than
  `max_micro_contracts` (50) on the account in total.
- **Every order is logged**, blocked ones too, with the model, the worker, the time and
  the reason.

### What "ready for a Combine" requires

A model is shown as ready for a Combine only when every line of its checklist is ticked:

1. **Real futures prices** (Databento). Proxy or synthetic prices never count.
2. **Held-out**: expected net dollars per attempt above zero (so the fees must be set).
3. **Held-out**: it passes at least 10 percentage points more often than its coin-flip
   twin (`min_pass_rate_edge` in `config/topstep.toml`).
4. **Held-out**: it makes money at double slippage.
5. **Lockbox Final check**: it makes money, and its pass rate is at most 15 points below
   the held-out one (`max_lockbox_drop`).
6. **Alpaca paper trading**: at least 20 finished trading days on live prices
   (`min_shadow_days`), with at most 10% of them (`max_days_outside`) outside the range
   of daily results the held-out backtest saw (its 1st to 99th percentile, scaled to the
   contracts traded), and a profit overall.

Only then can it trade on Topstep. Expect most searches to end with "nothing good enough
yet"; that answer is worth having before paying for a Combine.

## Topstep's rules (config/topstep.toml)

Futures models are judged on what Topstep pays for, not on ROI. A simulator
(`fleet2/sim/topstep.py`) replays a model's days through Topstep's Combine (the paid
test) and then the Express Funded account (where payouts happen), starting from every
day of a period. Every rule it uses is in `config/topstep.toml`. The starting values
come from third-party summaries in October 2026; you confirmed them on Oct 8, 2026.
Topstep changes its rules from time to time: check them at help.topstep.com now and then
and update the "Last checked" lines. The fees and the payout cap are still to fill in.
Edit the file, then `docker compose restart coordinator`.

What each number means:

- **size** (50,000): the account balance you start with. All money below is in dollars.
- **profit_target** (3,000): the Combine passes once the balance is this much above the
  start.
- **max_loss_limit** (2,000): the loss floor starts this far below the starting balance
  (48,000). At the end of every day it moves up to the day's closing balance minus
  2,000 if that is higher, never down, and it stops at the starting balance (50,000).
  The floor is watched all day, open trades included: touch it once and the account
  fails, even if the day ends green. Example: the balance closes at 50,500, so the floor
  moves to 48,500; a loss the next day leaves it at 48,500.
- **daily_loss_limit** (0, none): an optional Topstep add-on. With a number, a day that
  falls that far ends at that loss.
- **max_micro_contracts** (50): the most micro contracts at once. The simulator never
  tests more.
- **flat_by** (15:10 Chicago time): Topstep's deadline to be out of every trade. The
  backtester is out by 15:00.
- **consistency** (kind "target", best_day_share 0.5): no single day may be more than
  half of the profit target ($1,500). Sources disagree: another says 55% of the total
  profit (kind "total_profit", 0.55), where you keep trading until it holds. Set the one
  Topstep uses now.
- **payout_winning_days** (5) and **payout_min_day_profit** (150): in the Express Funded
  account you may ask for a payout after 5 days of at least $150 profit each, counted
  since the start or the last payout.
- **payout_share_of_balance** (0.5): a payout is at most half of the profit in the
  account.
- **max_payout** (unset): the most one payout can be. Topstep caps it by account size and
  "path"; the plan has no number, so fill it in.
- **profit_split** (0.9): your share of each payout.
- **floor_to_start_after_payout** (true): after a payout the loss floor moves up to the
  starting balance.
- **[fees]** (all unset): commission plus exchange fees per contract per side for MES and
  MNQ, the Combine's monthly fee and the activation fee for the Express Funded account.
  Until you fill them in, futures backtests and model search do not start, and the
  dashboard says "Set the fee in config/topstep.toml" instead of a number.
- **[live]**: the starting balance the loss floor is counted from, stopping at 80% of
  the loss limit, the 2-minute margin before 15:00, and the 180 seconds after which a
  silent worker's model is closed out (see "Futures trading").
- **[costs] slippage_ticks** (1), **[simulator]** and **[ready]**: our own assumptions
  (slippage, how long to play the Express Funded account, how many coin-flip twins,
  and what "ready for a Combine" needs), not Topstep's rules.

What the simulator reports for a model, at each contract size it tries:

- **Pass rate**: one Combine attempt starts on every day of the period that has 60
  trading days left after it; the pass rate is the share that pass. An attempt that has
  neither passed nor failed after 60 trading days counts as given up, after paying about
  three months of fees (`combine_max_days`, our own assumption).
- **Median days to pass**: trading days from start to pass, for the attempts that
  passed.
- **Expected payout**: the average paid out per attempt in the Express Funded account
  (played for at most 120 trading days).
- **Expected net per attempt**: your share of that, minus the monthly Combine fees and
  the activation fee. The best contract size is the one with the most.
- **Coin-flip twin**: the same model with the same trade times but each trade's
  direction decided by a coin. With a $3,000 target and a $2,000 limit, even a coin
  passes now and then, so a model only counts when it clearly beats its twin.
- Sizes stop where the model's worst stretch in training, times the size, would use
  more than half the loss limit.

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
- **Futures models** have their own rules on top of these (flat every day, Topstep's loss
  limit and contract limit): see "Futures trading: Alpaca paper, then Topstep".
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

Futures models have their own backtester (`fleet2/sim/futures_backtest.py`) with the
same promise, adapted to day trading:

- **No lookahead.** A model decides on 1-, 3-, 5- or 15-minute bars built from 1-minute
  bars, and each decision fills at the open of the next 1-minute bar. Every futures model
  must pass the cut-off test: its answers up to any bar stay exactly the same when the
  prices after that bar are cut off.
- **Whole contracts, long or short**, with slippage on every fill (1 tick each side by
  default) and the commission and exchange fees per contract per side from
  `config/topstep.toml`.
- **Stops and targets assume the worst.** A stop and a target in the same minute: the
  stop fills. A minute that opens past the stop fills at that open. A target only fills
  once the price trades through it.
- **Flat every day.** No new trades from 14:50 Chicago time, everything closed at 15:00
  (Topstep's own deadline is 15:10). Half days close at 12:00.
- **Results by day:** profit or loss, the worst moment of the day including open trades
  (that is what Topstep's loss limit watches), trades and minutes held.

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
- **Topstep's overnight session**: futures models trade the regular session only
  (8:30 to 15:00 Chicago time).
