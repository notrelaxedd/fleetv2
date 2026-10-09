# Claude Haiku in the fleet: plan

Status: written 2026-10-09. Stage A in progress (see the bottom of this file).

Goal: use Claude Haiku 5.5 to give model search new ideas to test, and to explain what
it finds, without weakening the one promise this software makes: a model only reaches
a paid Topstep Combine after it has made money on prices it was never tuned on.

No AI can promise a profitable model either. Setups that are said to "make thousands
with only AI" are usually shown without the many that lost money, or are selling
something, and almost none are tested on prices the AI never saw. The aim here is
narrower and honest: more and better ideas going into the same tests as before.

---

## 1. What the search can and cannot do today

Model search only changes the settings inside five ideas written by hand (opening
range, VWAP revert, trend day, gap fade, pullback): how wide a stop is, which minute to
start, and so on. It can never come up with a different idea, such as "buy a break of
the first 30 minutes' high, but only after a gap down, and get out at the day's
average price". After a few hours it has tried most of what those five ideas can do.

## 2. Ground rules for anything AI

1. **The key stays on box1.** `ANTHROPIC_API_KEY` goes in `.env` on box1 and only the
   coordinator reads it, like the Alpaca, Databento and TopstepX keys. Workers never
   see it.
2. **A spending cap.** Every call's cost is counted. Once a month's spend reaches the
   cap in `config/ai.toml`, the AI features stop until next month and the dashboard
   says so.
3. **AI writes recipes, never code.** An idea is a recipe: a choice of building blocks
   (signals, filters, exits) that the fleet's own code already knows. Nothing an AI
   writes is ever run as code on a worker. Each building block is proved once never to
   look at a later price (the cut-off test), and that covers every recipe made of them.
4. **The AI only sees training results.** It learns from how its earlier ideas did on
   the training years. Held-out and lockbox results never reach it, so it can never
   tune its ideas to the tests that judge them.
5. **Every idea counts.** Every recipe and every setting tried is counted in the
   "chance this is luck" figure. More ideas tried means a winner needs more proof.
6. **No AI ever trades.** AI output never places an order, starts or stops a model, or
   changes a setting. It goes through the same backtest, Final check and Alpaca paper
   trading as everything else.

### Why an AI cannot simply be backtested on the news

Claude Haiku 5.5 learned from text up to its training date, which includes much of
2019 to 2026: crashes, rate decisions and what markets did next. Ask it about a 2022
headline and it may "remember" what happened after. So any test of AI decisions on past
news would be fooled by the AI's memory. AI judgment about news or markets can only be
tested going forward, on Alpaca paper, day by day. Recipes do not have this problem:
the AI never sees a date or a price, and the backtester judges them on prices.

## 3. Stages

### Stage A: building blocks and random recipes (no key, no cost)

- A library of building blocks, each a small numpy function of the bars so far:
  - signals: opening-range break, stretch from the day's average price (VWAP), short
    and long average cross, gap from yesterday's close, move from today's open, new
    high or low of the day, momentum over the last few bars;
  - each signal can be followed ("buy strength") or faded ("sell strength");
  - filters: quiet day, busy day, on the right side of VWAP, with the day's move,
    after a gap, after no gap;
  - exits: hold to the stop, target or close; leave on a cross of VWAP; take profit
    at VWAP; leave on the opposite signal; leave after a number of bars;
  - first signal of the day only, or every signal; long only, short only, or both.
- A recipe is a choice of one signal, up to two filters, one exit and those options.
  It gets its name from its contents, so the same recipe always has the same name.
- Each recipe works like a model file: it has its settings, and model search tunes
  them. Its plain-English description is written from its blocks.
- Model search mixes a few new random recipes into every round, next to the five
  files, and keeps tuning the recipes that produced kept models.
- Up to 15 recipe models are kept at a time; a new find replaces a weaker one. The
  "same idea twice" rule (over 90% alike day to day) applies across everything.
- All recipe tries are counted together under "recipe" for the luck figure.

### Stage B: Haiku writes recipes (needs the key)

- While a futures search runs, the coordinator keeps a short queue of new recipes
  written by Claude Haiku 5.5 (model `claude-haiku-5-5`). Search takes them from the
  queue at the start of each round, ahead of random ones.
- Haiku is given the building blocks, what each does, and a table of earlier recipes
  with their training results only: best score, days traded, whether they made money
  at double costs. It is asked for recipes unlike those already tried, each with one
  sentence on the idea behind it.
- Its answer must match the recipe format exactly (structured output); anything that
  does not pass the same checks as a random recipe is dropped and counted.
- The Models screen marks recipe models "by Haiku" or "random", with Haiku's sentence.
- Cost: about 6,000 tokens in and 1,500 out per batch of recipes, about $0.0014 at
  $0.10 / $0.50 per million tokens. A hundred batches a day is about 15 cents.

### Stage C: Haiku reviews models (needs the key)

- On a model's page, "Ask for a review" sends its numbers to Haiku: training,
  held-out, Final check and paper days, plus the luck figure. It writes a short plain
  review: what looks like luck, what is fragile, what to watch on paper.
- Advice to the owner only. Reviews are never shown to the idea writer (rule 4).

### Stage D, an experiment: a daily market note (needs the key)

- Before each session, Haiku reads the day's headlines from Alpaca's news feed (the
  existing Alpaca keys) and writes a short note: scheduled events (Fed decision, CPI,
  jobs report), and whether the news looks unusually heavy.
- Tested only going forward (see "Why an AI cannot simply be backtested on the
  news"): after 40 or more paper days, compare each paper model's results on days the
  note flagged with the other days. Only if flagged days are clearly worse would a
  "skip flagged days" switch be offered, and only for Alpaca paper first.
- Scheduled-event dates themselves are plain facts, not AI judgment, and can be tested
  on the past honestly as an ordinary filter. That part may come first.

### Several agents at once

Stages B, C and D are separate small jobs on the coordinator, each with its own share
of the spending cap, not a swarm talking to each other. More agents only help when
each has a job that can be checked; an idea writer, a reviewer and a news note each
can be. A "market scanner" that picks trades by itself cannot be tested honestly on
the past (see above), so it is not planned.

## 4. What it costs

| Use | Rough cost |
|---|---|
| Stage A | nothing |
| Stage B, 100 batches of recipes a day | about $4 a month |
| Stage C, a review | well under a cent |
| Stage D, one note a day | about 30 cents a month |

The default cap is $5 a month, set in `config/ai.toml`.

## 5. Build order and status

1. Stage A: building blocks, recipes, cut-off test, recipes in model search. In
   progress.
2. Stage B: the Haiku idea writer, with the cap and the spend count.
3. Stage C: reviews.
4. Stage D: the market note, as an experiment.
