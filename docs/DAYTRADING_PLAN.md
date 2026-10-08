# Day trading for Topstep: plan

Status: stages 1 to 6 built, 2026-10-08 (see README.md). Written 2026-10-08.

What changed from this plan when it was built, by the owner's decisions:
- Stage 6's shadow trading is real paper trading on Alpaca: each micro contract is traded
  as its SPY or QQQ share equivalent in the Alpaca paper account (1 MES = 50 SPY shares,
  1 MNQ = 82 QQQ shares), and the checklist's last line counts those days.
- Topstep trading is through the TopstepX API, behind keys in .env, a typed confirmation
  and a restart, and only for models whose whole checklist is ticked. The dashboard also
  lists live signals for trading by hand.
- The simulator gives up a Combine attempt after 60 trading days (combine_max_days), so
  no attempt is cut short by the end of the prices.

Goal: find day-trading models that are likely to earn Topstep payouts, and be honest
about the ones that are not. The fleet keeps doing what it does now (search, backtest,
rank); this plan adds a third market, "Futures", built around Topstep's rules.

No code can promise a profitable model. What this plan does promise: a model only
reaches a paid Topstep Combine after it has made money on four sets of prices it was
never tuned on, at worse-than-expected costs, and has beaten a coin-flip version of
itself. Many searches will end with "nothing good enough yet", and that answer is
worth having before paying for a Combine.

---

## 1. Why the current setup cannot do this

- **Topstep only trades CME futures.** It does not trade stocks or crypto, so none of
  today's models or data can be traded there.
- **The data is the wrong size.** Stocks are daily bars and crypto hourly, so no model
  can buy and sell within a day.
- **The models hold for days to months.** That is momentum over months, dip buys held
  up to 20 days, and so on. Topstep requires every position to be closed every day.
- **Costs are too high where the models trade.** Stocks cost about 10 bp per round
  trip and crypto about 70 bp. Day-trading edges are only a few bp.
- **The score is wrong for the goal.** It measures ROI on a long-only book. Topstep pays
  for staying inside a trailing loss limit while hitting a profit target, then earning
  payouts.

## 2. How it can work

**Trade micro index futures: MES (micro S&P 500) and MNQ (micro Nasdaq-100).**

- They are the most liquid products Topstep offers, and costs are tiny:
  - Topstep lists micro commission at $0.25 per side.
  - Its own MNQ example puts the all-in round trip at about $1.22 including exchange fees.
- With 1 tick of slippage each way, one MES round trip costs about $3.75 on roughly
  $30,000 of exposure. That is about 1 bp, against 10 bp for stocks and 70 bp for crypto.
- They can be sold short as easily as bought, so a model can trade both directions.
- The SPY/QQQ ideas carry over, because MES and MNQ track the same indexes.

**Score models on what Topstep pays for, not on ROI.**

- A Topstep rules simulator replays a model's day-by-day results through the Combine
  and the Express Funded account. That covers:
  - the profit target and the trailing loss limit, including intraday dips
  - the best-day consistency rule
  - the flat-by-3:10 PM CT rule and contract limits
  - payout rules, caps and the profit split
  - the monthly Combine fee
- The simulator answers how often the model passes, how long that takes, and the
  expected payout minus fees.

**Compare every model with luck.**

- With a $3,000 target and a $2,000 loss limit, a strategy with no edge at all still
  passes about 40% of the time (2,000 / 5,000) before the trailing limit cuts that down.
- So a pass rate alone means little. Each model is also run as a "coin-flip twin":
  same entry and exit times, random direction.
- A model is only interesting when it clearly beats its twin, and when its funded-stage
  expected payout is positive. That stage needs a real edge, because luck runs out over
  many payouts.

**Size positions with the simulator.**

- Under a trailing loss limit, the number of contracts matters as much as the strategy.
- The simulator tries 1 to N micros for each model and reports the size with the best
  expected payout.
- Sizing never goes above the size where the model's worst historical stretch would use
  more than half the loss limit.

## 3. Four sets of prices, used once each

Futures history is split by date. Nothing later in the list is ever seen by anything
earlier.

| Set | Share | Used by | Opened |
|---|---|---|---|
| Training | first ~60% | model search only | always |
| Held-out | next ~25% | ranking on the Models screen | once per found model |
| Lockbox | last ~15% | the "Final check" button | once per model, result stored forever, never re-run |
| Forward | live prices after today | shadow trading | at least 20 trading days |

A model is shown as "ready for a Combine" only when all of these hold. The exact
thresholds go in `config/topstep.toml`.

- **Held-out:**
  - expected payout above zero
  - pass rate clearly above its coin-flip twin's
  - profitable at double slippage
- **Lockbox:** profitable, and the pass rate is not far below the held-out one.
- **Shadow trading:**
  - at least 20 trading days on live prices
  - daily results inside the range the backtest predicted
  - profitable overall

## 4. Changes, in build order

Stages 1 to 5 are research only and place no real orders. Stage 6 is optional and only
starts after the owner says so.

### Stage 1: futures prices

- **Data source: Databento**, CME Globex, dataset `GLBX.MDP3`, schema `ohlcv-1m`.
  - It is pay-as-you-go. The coordinator asks Databento for the cost before every
    download and refuses anything above `max_download_usd` in `config/topstep.toml`.
  - The key `DATABENTO_API_KEY` lives in `.env` on box1, used only by the coordinator,
    like the Alpaca keys.
- **Symbols:** MES and MNQ, from their launch in May 2019 (about 7 years).
  - ES and NQ are an optional add for earlier years. Only their price movement would be
    used; P&L is always computed with micro contract sizes.
- **Hours:** regular-session bars only (8:30 to 15:00 CT) to start. The overnight
  session can be added later.
- **Contract rolls:** store the contract (instrument id) on every bar.
  - Models are flat every night, so each day simply uses that day's front contract.
  - A feature that compares prices across days (an overnight gap, say) only compares
    bars of the same contract, and is skipped on a roll day.
- **Free stand-in while there is no Databento key:** SPY and QQQ 1-minute bars from
  Alpaca, labelled "proxy" everywhere.
  - Good for building and testing.
  - Never good enough for a "ready for a Combine" verdict.
- **Storage:**
  - The `bars.timeframe` check gains `'1Min'`, and `models.market` gains `'futures'`.
  - Seven years of regular-session 1-minute bars for two symbols is about 1.4M rows,
    which Postgres handles fine.
- **Sending prices to workers:** gzip JSON is too big at this size.
  - Futures bars are served as a compressed numpy file (`.npz`) with an ETag.
  - Workers cache it in memory and only fetch again when the ETag changes.
  - Stocks and crypto keep today's format.
- **Sessions:** a small module that knows CME hours in America/Chicago time.
  - Regular session 8:30 to 15:00 CT, holidays, half days.
  - Maps every bar to its trading day.

### Stage 2: a futures backtester

This is a new file beside the current backtester, which is left alone.

- **Positions:** whole contracts, long or short, for each symbol.
- **Model interface (futures only):**
  - Each model returns a target per symbol from -1 (full short) to +1 (full long).
  - The backtester turns that into contracts using the size being tested.
- **Fills:** at the open of the next 1-minute bar, plus slippage.
  - Slippage is in ticks: default 1 per side.
  - Commission and fees are in dollars per side per contract.
  - All come from `config/topstep.toml`.
- **Stops and targets:** optional settings handled by the backtester, using each bar's
  high and low.
  - Assume the worst: if a stop and a target both fall inside one bar, the stop fills.
  - A stop that gaps fills at the bar's open, not at the stop price.
- **Flat every day:**
  - No new trades after a cut-off (default 14:50 CT).
  - Everything is sold or bought back at 15:00 CT, well before Topstep's 15:10 CT.
- **Bar sizes:** models decide on 1-, 3-, 5- or 15-minute bars built from 1-minute bars.
  Bar size is a search setting. Fills still happen on the next 1-minute bar.
- **Speed:**
  - Models compute their signals for a whole array of bars at once with numpy, instead
    of being called bar by bar.
  - The lookahead guarantee becomes a test. For random cut-off points, signals computed
    on prices cut off at bar t must equal the full-data signals up to t. Every futures
    model must pass it.
- **Results:**
  - One record per trading day: P&L in dollars, worst intraday dip, trades, minutes
    held.
  - Plus a trade list and the existing summary numbers.

### Stage 3: the Topstep rules simulator

- **Rules file:** all rules live in `config/topstep.toml`.
  - Each value has a comment with its source and the date the owner last checked it.
  - Topstep has changed these rules several times in 2025 and 2026, so nothing is
    hard-coded.
- **Starting values** for the 50K account, from third-party summaries in October 2026.
  The owner must check every one at help.topstep.com.

| Rule | Value | Note |
|---|---|---|
| Profit target | $3,000 | |
| Maximum Loss Limit | $2,000 | Trails the end-of-day balance high, enforced live including open trades, stops trailing at the starting balance |
| Daily loss limit | none | Optional add-on |
| Consistency | best day ≤ 50% | Sources disagree: 50% of the target or 55% of total profit. Both kinds are supported by setting |
| Max size | 5 minis or 50 micros | |
| Flat by | 3:10 PM CT | |
| Express Funded payout | 5 days of $150+ net profit | Payout up to 50% of balance, capped by account size and path |
| Profit split | 90/10 | |
| After a payout | loss limit moves to the starting balance | |
| Combine fee, activation fee | owner fills in | The simulator shows "set the fee" until then |

- **What it replays:** a model's daily results, at a given contract size, starting from
  every possible day in a period.
  - It plays the Combine. If the Combine passes, it plays the Express Funded account
    for up to 120 trading days.
  - Live enforcement of the loss limit uses each day's worst intraday dip, not just the
    closing number.
- **What it reports:**
  - pass rate and fail rate
  - median days to pass
  - expected payouts, expected fees, and expected net dollars per attempt
  - the same numbers for the coin-flip twin
  - the best contract size

### Stage 4: model files built for the day

New files under `fleet2/models/futures/`. Each one gets:
- plain-English `DESCRIPTION` and `HOW_IT_WORKS` text, like today's models
- a `SEARCH_SPACE` that includes bar size, and stop and target settings
- the cut-off test

| File | What it does |
|---|---|
| `opening_range.py` | Trades a break above or below the first 15 to 60 minutes' range |
| `vwap_revert.py` | Fades a stretch far from the day's volume-weighted average price, only on quiet days |
| `trend_day.py` | The first hour's direction predicts the last hour's: enter mid-afternoon in that direction |
| `gap_fade.py` | Bets that part of a large overnight gap closes in the first hours |
| `pullback.py` | In a strong intraday trend, buys short dips, or sells short rallies |

These are well-known ideas. That is deliberate: they are honest starting points.
Whether any of them still works after costs is exactly what the fleet finds out.

### Stage 5: model search and the Models screen for futures

**Search score.** It is computed on training data only.

1. Split training into 4 back-to-back periods.
2. In each one, compute the Sharpe ratio of daily P&L for 1 contract at **double
   slippage**.
3. The score is the **worst** of the four.
   - A setting that only worked in one stretch scores low.
   - Daily Sharpe already accounts for how many days there are, so the old
     trades/(trades+100) bonus for trading more goes away.
4. **Gates:**
   - trades on at least 100 different days in training
   - positive P&L at normal and at double slippage
   - average hold under the day (always true by construction, but checked)

**Learning between rounds.**
- Half of each round's candidates are drawn at random.
- The other half are small changes to the models already kept, which the job fetches
  from the coordinator at the start of each round.
- The seed text records the parent model, so every candidate can still be re-created.

**Robustness check before keeping a model.**
- Re-score the winner with each setting moved ±10% (two neighbours per setting).
- Drop it if the median neighbour scores below half the winner. That is a sharp peak,
  which usually means luck.

**Keep a varied set.**
- At most 5 kept models per file.
- A new find replaces its closest kept model when it scores better, rather than always
  replacing the weakest.
- Finds whose daily P&L is more than 90% correlated with a kept model are dropped.

**Count tries.**
- The number of settings tried per file is stored.
- The Models screen shows a "chance this is luck" figure from the deflated Sharpe
  ratio (Bailey and López de Prado), which uses that count.

**Speed.**
- A search job uses every core on the worker, through a process pool over candidates,
  and still stops promptly on Stop.
- Prices are cached by ETag instead of being fetched again every round.

**Models screen for futures.**
- Ranked by held-out expected net payout per Combine attempt.
- Only models that beat their coin-flip twin get a rank.
- The detail panel shows:
  - pass rate and the twin's pass rate
  - median days to pass
  - expected payout, at the chosen contract size
  - worst day and best-day share
  - trades per day and average hold in minutes
  - the "chance this is luck" figure
  - a Growth chart of daily P&L
- A **Final check** button opens the lockbox once per model and stores the result.

### Stage 6 (later, only with the owner's OK): shadow trading, then Topstep

**Shadow trading.**
- A paper job runs a futures model on live 1-minute prices and records the trades it
  would have made, with the same costs.
- No orders are placed anywhere.
- It needs live futures prices: Databento live (paid), or TopstepX's own market data
  once the owner has the API.

**Topstep execution.** A `TopstepBroker` beside the Alpaca one, through the TopstepX
(ProjectX) API. To check first:
- Topstep's help center says API bots are allowed on Combine and Express Funded
  accounts, are not available on Live Funded accounts, and that high-frequency trading
  is banned.
- Third-party summaries say orders must come from your own device, not a VPS. box1 at
  home is probably fine, but confirm it.
- The API is a separate paid subscription.
- Topstep's own safety rules would be added to `coordinator/safety.py` for that account
  only:
  - stop trading at 80% of the loss limit
  - flat by 15:00 CT
  - the contract limit

**Manual alternative.** The dashboard shows each model's live signals and the owner
places the trades by hand on TopstepX.

## 5. What it costs

- **Databento:** pay-as-you-go. The cost check runs before every download.
- **Topstep:** the Combine monthly fee and, later, the API subscription. Neither is
  needed until stage 6.
- **Compute:** existing workers. Stages 2 and 5 are designed so one 4-core worker can
  try thousands of settings a day.

## 6. Open decisions for the owner

1. Buy Databento data now, or build on the free SPY/QQQ stand-in first? Recommended:
   build on the stand-in, then buy before any verdict counts.
2. Account size to target: 50K, 100K or 150K. Recommended: 50K, the cheapest test.
3. Trade by hand from the dashboard's signals, or automate through the TopstepX API
   later?
4. Check every value in `config/topstep.toml` against help.topstep.com, and fill in
   the fees.
