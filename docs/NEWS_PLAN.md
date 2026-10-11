# News research engine: plan

Status: approved by the owner on 2026-10-11 (option B: one shared Alpaca paper account).
This is the owner-approved exception to rule 6 of docs/AI_PLAN.md ("No AI ever trades").

Claude reads the market news, predicts how each stock or coin it names will move, and
every prediction is saved and later scored against what the price really did. A news
strategy only gets paper money once its saved predictions beat a coin-flip twin. Every
order still goes through `coordinator.safety.approve_and_place()`.

---

## 1. Honest testing

Claude already knows how most past news turned out (it was in its training data), so
a backtest on old news would look brilliant and mean nothing. Two sources of evidence
count, and nothing else:

- **Post-cutoff replay.** Articles published after the model's training cutoff plus a
  margin (`no_replay_before` in `config/limits.toml`, default 2026-08-01 for
  Claude Opus 5.5) are replayed with prices known at the article's time only.
- **Forward shadow mode.** Live predictions are saved with a timestamp, and scored once
  their horizon has passed.

**Coin-flip twin.** Every prediction is scored next to a twin with the same entry and
exit but a random direction (seeded by the prediction id, so it never changes). A
market is tradeable only when its predictions beat the twin by more than two standard
errors over at least `min_scored_predictions` scored predictions, and their own mean
return after costs is positive.

**Entry and exit.** Entry is the open of the first bar that starts after the decision
(stocks: daily bars, so the next session's open, pessimistic on purpose; crypto: the
next hourly open). Exit is the close of the bar `horizon_days` trading days later
(crypto: `horizon_days * 24` hourly bars). Costs: stocks 5 bp slippage each way plus a
borrow fee on shorts (`short_borrow_fee_pct` a year, 3% until checked with Alpaca);
crypto 10 bp slippage plus Alpaca's 25 bp taker fee each way.

## 2. Markets

| | Stocks (the 22 in `fleet2/universe.py`, less SPY and QQQ for trading) | Crypto (the 8) |
|---|---|---|
| Up | buy | buy |
| Down | **short** (whole shares, shortable and easy to borrow only) | sell / stay out (Alpaca cannot short crypto) |
| Bars | daily (IEX) | hourly |
| Minimum predicted move | `cost_multiple` x round-trip cost | same |

There are two news strategies, `news-stocks` and `news-crypto`, each a `models` row
with `origin = 'news'` and its own book, because every book trades one market.

## 3. One shared Alpaca account (option B)

All books share one Alpaca account, and Alpaca holds one net position per symbol. So:

- A news short of S is refused while any other book holds S long or has a buy of S in flight.
- Any buy of S by another book is refused while a news book is short S.
- A sell larger than what a book holds is refused (no flipping long to short in one order).

The futures paper books already trade SPY and QQQ shares long *and short* on the same
account, from their own tables (futures_books, futures_positions). So the news
strategies never trade SPY or QQQ at all (`NEWS_EXCLUDED` in coordinator/news_trading.py;
predictions about them are still made and scored). With those rules, whenever a news
book is short S the account's position in S is exactly that short.

## 4. Safety

- The article is untrusted input: Claude gets no tools, the answer is forced into a
  JSON schema and validated again in code, and symbols outside the article's own
  tagged universe symbols are dropped.
- `daily_cap_usd` in `config/ai.toml` [news] caps the news engine's Anthropic spend per
  UTC day; every call is also recorded in `ai_calls` (feature "news") and counts against
  `monthly_cap_usd`, the one cap for all AI features. Over either cap, articles are
  marked `over_budget` and not analysed. Trading is not affected.
- `max_trades_per_day` caps new news positions per strategy per day.
- Shorts: `max_per_short_position` per position, `max_short_per_model` per strategy,
  and gross exposure (longs + |shorts|) within `max_per_model`.
- Every news position has a stop (`stop_loss_pct` from the entry price), watched by the
  coordinator against Alpaca's latest trade price every loop pass. (Not a resting order
  at Alpaca: fractional and crypto positions cannot all carry one, and a resting order
  would hold the book's "order in flight" lock forever. If box1 is down, stops are not
  watched; the daily loss limit is not either.)
- Pause all trading and the 2% daily loss limit apply unchanged.
- Paper only; real money uses the same steps as today.

## 5. Claude

- Settings in `config/ai.toml` [news]: model `claude-opus-5-5`, effort medium, max
  tokens, daily cap, prices. Adaptive thinking, structured output, server-side refusal
  fallback (`fallbacks: "default"`); the model that actually answered is saved.
- Key: `ANTHROPIC_API_KEY` in `.env` on box1 (already used by the Haiku features), read
  only by `coordinator/config.py`.
- About 4 cents an article; $2-5 a day expected with crypto news, capped at $5 a day.
  The monthly cap in `config/ai.toml` (default $5) must be raised to fit.

## 6. Module interfaces

These are the contracts between the parts; each part may add private helpers.

### Tables (`coordinator/migrations/0009_news.sql`)
`news`, `news_analyses`, `news_predictions` (scores are columns on the prediction),
`news_positions`; `models.origin` gains `'news'`.

### `coordinator/news_feed.py` (ingest)
```python
@dataclass(frozen=True)
class Article:
    id: int; headline: str; summary: str; content: str; author: str; source: str; url: str
    symbols: tuple[str, ...]            # as Alpaca tags them, e.g. ("AAPL", "BTCUSD")
    created_at: datetime; updated_at: datetime | None

class NewsSource(Protocol):
    connected: bool; problem: str | None
    def fetch(self, start: datetime, end: datetime, symbols: list[str],
              page_token: str | None) -> tuple[list[Article], str | None]: ...

AlpacaNewsSource(config), FakeNewsSource(seed=0), NoNewsSource(problem)
make_news_source(config) -> NewsSource
to_universe(alpaca_symbol) -> str | None      # "BTCUSD" -> "BTC/USD", "AAPL" -> "AAPL", else None
from_universe(symbol) -> str                  # "BTC/USD" -> "BTCUSD"
store_articles(conn, articles, status="pending") -> int     # upsert; only articles naming our symbols
poll_step(conn, source, now) -> dict          # {"added": int, "error": str | None}
backfill(conn, source, start, end) -> int     # stores with status "backfill"
```

### `coordinator/research.py` (Claude analyst)
```python
@dataclass(frozen=True)
class Prediction:
    symbol: str; direction: str  # "up" | "down" | "none"
    expected_move_pct: float; horizon_days: int; confidence: float
    already_priced_in: bool; rationale: str

@dataclass(frozen=True)
class AnalysisResult:
    predictions: list[Prediction]; requested_model: str; served_model: str
    prompt_version: str; input_tokens: int; output_tokens: int
    cache_read_tokens: int; cache_write_tokens: int; cost_usd: float
    stop_reason: str | None; raw: dict

class Analyst(Protocol):
    connected: bool; problem: str | None; model: str
    def analyse(self, article: dict, context: dict) -> AnalysisResult: ...

ClaudeAnalyst(api_key, settings: haiku.AiSettings, client=None), FakeAnalyst(seed=0), NoAnalyst(problem)
make_analyst(config, settings: haiku.AiSettings) -> Analyst
PriceNow = Callable[[list[str]], dict[str, float]]          # latest trade prices
build_context(conn, article_row, now, price_now: PriceNow | None) -> dict
spent_today(conn, now) -> float                    # news calls in ai_calls this UTC day
analyse_article(conn, analyst, article_row, context, now, source="live") -> list[dict]
analyse_pending(conn, analyst, settings, now, price_now) -> dict
```

### `coordinator/news_scoring.py`
```python
score_pending(conn, limits, now) -> int
stats(conn, market=None, source=None, model=None) -> dict
gate(conn, market, limits) -> tuple[bool, str]     # (tradeable, reason in plain words)
```

### `coordinator/data.py` addition
```python
BarSource.latest_prices(symbols: list[str]) -> dict[str, float]   # Alpaca latest trade
```

### `coordinator/news_trading.py`
```python
NEWS_MODELS = {"stocks": "news-stocks", "crypto": "news-crypto"}
ensure_models(conn) -> None
start(conn, market, mode, limits) -> dict          # refuses until gate() passes
stop(conn, market) -> dict                         # closes every position, then the book
news_pass(conn, status, limits, price_now, now) -> int
open_positions(conn, market=None) -> list[dict]
book_summary(conn) -> dict[str, dict]              # per market, for the dashboard
```

## 7. Stages

1. Ingest and the read-only News feed (fake news source in tests).
2. Claude analyst in shadow mode, spend cap and output checks.
3. Scoring, post-cutoff replay, coin-flip twin, stats.
4. Long trades through the existing safety path, gated on the scores.
5. Shorting with the extra guards; a fresh reviewer checks the money and safety code.
