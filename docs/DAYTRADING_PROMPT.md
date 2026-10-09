# Prompt: build the futures day-trading research pipeline

Paste everything below the line into a new Claude Code session on `notrelaxedd/fleetv2`.

---

You are working on fleet-v2 (`notrelaxedd/fleetv2`). It is a coordinator plus a rack of
worker PCs that backtest, search for and paper-trade trading models; see `README.md` and
`PLAN.md`. Your job is to add a third market, **Futures**, for day trading CME micro
index futures (MES, MNQ). The end goal is models that earn payouts on Topstep's
funded-account program.

The full design is in `docs/DAYTRADING_PLAN.md`. If it isn't on your branch, it is on
`claude/clever-brown-s0eulq`. Read it first, then read the code it touches:
- `fleet2/universe.py`
- `fleet2/sim/backtest.py`, `metrics.py`, `marketdata.py`
- `fleet2/models/`
- `fleet2/worker/search_job.py`, `paper_job.py`
- `coordinator/data.py`, `search.py`, `models_view.py`, `migrations/`
- `config/limits.toml`
- `tests/`

## What to build

Stages 1 to 5 of the plan, in order:

1. **Futures prices.**
   - A Databento source for `GLBX.MDP3` `ohlcv-1m` bars of MES and MNQ, regular session
     only. Store the instrument id on every bar.
   - Check the cost before every download and refuse above a configured cap.
   - A free stand-in source that uses Alpaca SPY/QQQ 1-minute bars, labelled "proxy".
   - A migration that adds `'1Min'` and `'futures'`.
   - An `.npz` bar payload with an ETag for the futures market.
   - A CME session module in America/Chicago time: trading day of each bar, holidays,
     half days.
2. **A futures backtester** beside the current one. Do not change the existing one.
   - Whole contracts, long and short.
   - Each model returns targets from -1 to +1 for whole arrays of bars, computed with
     numpy.
   - Fills at the next 1-minute open, plus slippage ticks and per-side fees.
   - Optional stops and targets, with pessimistic intrabar fills.
   - No entries after the cut-off. Flat by 15:00 CT.
   - Decisions on 1/3/5/15-minute bars resampled from 1-minute bars.
   - Results: per-day P&L, worst intraday dip, trades.
3. **The Topstep rules simulator** (`fleet2/sim/topstep.py`), with every rule in a new
   `config/topstep.toml`.
   - It covers the Combine, then the Express Funded account:
     - trailing end-of-day loss limit, enforced on intraday dips
     - consistency rule, in both kinds
     - contract limit
     - payout rules, caps and split
     - loss-limit reset after a payout
     - fees
   - It reports: pass rate, median days to pass, expected net dollars per attempt, the
     best contract size, and the same numbers for a coin-flip twin (same trade times,
     random direction).
4. **Five futures model files** under `fleet2/models/futures/`: opening range, VWAP
   revert, trend day, gap fade, pullback.
   - Each has plain-English `DESCRIPTION` and `HOW_IT_WORKS` text, like the current
     models.
   - Each has a `SEARCH_SPACE` that includes bar size.
5. **Futures model search and Models screen.**
   - Score: the worst of four training periods' daily Sharpe, at double slippage.
   - Gates: trades on at least 100 different days; positive P&L at normal and double
     slippage.
   - Proposals: half random, half small changes to models already kept, with the parent
     recorded in the seed text.
   - Robustness: drop a winner when its median ±10% neighbour scores below half of it.
   - Keeping: replace the closest kept model, drop finds whose daily P&L is more than
     90% correlated with a kept one.
   - Count tries and show a deflated-Sharpe "chance this is luck" figure.
   - Run candidates in a process pool across all cores, and cache prices by ETag.
   - Rank futures models by held-out expected net payout. Only models that beat their
     coin-flip twin get a rank.
   - Add a lockbox "Final check" that runs once per model and is stored forever.

Do **not** build stage 6. That means no shadow trading, no TopstepX/ProjectX
connection, and no orders of any kind. Nothing in this work may place an order.

## Rules

- **Leave stocks and crypto alone.** Their code paths, data format and tests must keep
  working unchanged. Add futures beside them, not inside them.
- **Honest backtesting stays the core promise.**
  - Every futures model must pass a cut-off test. For random cut-off bars t, signals
    computed on prices ending at t must equal the full-data signals up to t.
  - Training, held-out and lockbox periods are separate dates. Nothing later in that
    list is ever read by anything earlier.
  - Write tests that prove both.
- **Never guess Topstep's rules or fees.**
  - Put the values from the plan's table into `config/topstep.toml`. Give each a
    comment with its source and "check at help.topstep.com".
  - Leave fees unset.
  - When a needed value is unset, the simulator and the dashboard say "set the fee in
    config/topstep.toml" instead of computing a number.
- **Test the simulator against hand-worked examples.** For instance: a $50,000 account
  whose end-of-day balance rises to $50,500 moves the loss floor from $48,000 to
  $48,500. A loss the next day does not lower it. The floor stops at $50,000.
  - Also cover: an intraday dip through the floor fails the account even when the day
    closes green; the consistency rule in both kinds; a payout and the floor reset.
- **Keys.** `DATABENTO_API_KEY` goes in `.env` on box1 and is read only by the
  coordinator, like the Alpaca keys. Add it to `.env.example` with no value.
  - Without a key, everything must still run on the proxy source or on synthetic data,
    and be labelled as such on screen.
  - A "ready for a Combine" verdict is never shown for proxy data.
- **Match the repo's style.** Comments, docs and on-screen text in plain English with
  no jargon, the way `README.md` and the current model files are written.
- **Each stage is its own commit**, with tests:
  - Run `python -m pytest tests -q` before each commit. It needs a local Postgres 16; see
    `README.md` → Development.
  - Update `README.md` for anything the owner sees or sets.
- **If you get stuck** (no Postgres, Databento unreachable, a rule you can't pin down),
  stop and say exactly what is blocking rather than working around it silently.

## Done means

- All five stages are committed and tests pass, including the cut-off tests, the
  separate-periods tests and the simulator's worked examples.
- A model search can be started on the futures market from the dashboard, on proxy or
  Databento data.
- The Models screen shows futures models ranked by held-out expected payout, with the
  coin-flip twin's numbers beside them.
- The README explains, in plain words:
  - how to load futures prices
  - what each Topstep number means
  - what "ready for a Combine" requires
- A short report back covers:
  - what was built
  - what was not, and why
  - the speed of one search candidate on 7 years of 1-minute MES data, or synthetic
    data of that size
  - every Topstep value the owner still needs to check
