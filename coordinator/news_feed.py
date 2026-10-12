"""News ingest: where articles come from and how they are kept (docs/NEWS_PLAN.md, section 6).

The coordinator reads Alpaca's news feed with the paper keys already in .env (the same
feed the daily market note reads), keeps every article that names a symbol we trade in
the `news` table, and leaves the rest alone. Nothing here calls Claude and nothing here
trades: the analyst (coordinator/research.py) reads the rows this module marks "pending".

Alpaca tags crypto without a slash ("BTCUSD"); our universe writes "BTC/USD". The two
helpers to_universe / from_universe translate, and an article is kept only when at least
one of its tags maps into fleet2/universe.py. The article text arrives as HTML; it is
stored as plain text, capped at 20,000 characters, because that is what Claude reads.

Live polling (poll_step) keeps a cursor in the settings table. Alpaca publishes some
articles late, so every poll looks 10 minutes behind the cursor; articles seen twice are
upserts, and an article's analysis_status is never reset by seeing it again. A backfill
stores old articles with status "backfill": they are for replay only and are never sent
to the live analyst.

Alpaca allows about 200 data calls a minute across everything on the coordinator (price
refreshes included). Every request here goes through the shared coordinator.data
ALPACA_LIMITER, and a poll makes at most 10 requests (one when it is quiet).
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Jsonb

from coordinator import data
from fleet2 import universe

log = logging.getLogger(__name__)

CONTENT_LIMIT = 20_000          # characters of article text kept
PAGE_SIZE = 50                  # Alpaca's largest page for the news endpoint
CURSOR_KEY = "news_cursor"      # settings key: ISO time of the newest article seen
FIRST_LOOKBACK = timedelta(hours=1)   # a first poll starts this far back
OVERLAP = timedelta(minutes=10)       # articles get published late: look this far behind the cursor
POLL_MAX_PAGES = 10
BACKFILL_MAX_PAGES = 500

MISSING_KEYS_MESSAGE = "Add your Alpaca paper keys to .env on box1 (the news feed needs them)"


@dataclass(frozen=True)
class Article:
    """One news article as Alpaca gave it (symbols are Alpaca's tags, e.g. "BTCUSD")."""

    id: int
    headline: str
    summary: str
    content: str
    author: str
    source: str
    url: str
    symbols: tuple[str, ...]
    created_at: datetime
    updated_at: datetime | None


class NewsSource(Protocol):
    """Where articles come from. fetch returns one page, oldest first, for
    start <= created_at <= end, plus the token of the next page (None on the last)."""

    connected: bool
    problem: str | None

    def fetch(self, start: datetime, end: datetime, symbols: list[str],
              page_token: str | None) -> tuple[list[Article], str | None]: ...


# ------------------------------------------------------------------ symbols

_CRYPTO_BY_ALPACA = {c.replace("/", ""): c for c in universe.CRYPTO}


def to_universe(alpaca_symbol: str) -> str | None:
    """Our symbol for an Alpaca tag ("BTCUSD" -> "BTC/USD", "AAPL" -> "AAPL"), or None."""
    s = str(alpaca_symbol or "").strip().upper()
    if s in universe.STOCKS:
        return s
    if s in universe.CRYPTO:
        return s
    return _CRYPTO_BY_ALPACA.get(s)


def from_universe(symbol: str) -> str:
    """Alpaca's tag for one of our symbols ("BTC/USD" -> "BTCUSD")."""
    return symbol.replace("/", "")


def universe_tags() -> list[str]:
    """Every symbol we trade, as Alpaca tags them (what a poll asks the feed about)."""
    return [from_universe(s) for s in (*universe.STOCKS, *universe.CRYPTO)]


# ------------------------------------------------------------------ text

_BLOCK_TAGS = frozenset({"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table",
                         "blockquote", "pre", "section", "article", "header", "footer", "hr"})
_LINE_TAGS = frozenset({"br", "li", "tr"})
_SKIP_TAGS = frozenset({"script", "style", "head", "title"})


class _TextExtractor(HTMLParser):
    """HTML to text: paragraphs become blank-line separated, breaks and list items new lines."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")
        elif tag in _LINE_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _SKIP_TAGS:  # <br/> and friends; a self-closed <script/> holds nothing
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(re.sub(r"\s+", " ", data))


def html_to_text(html: str | None) -> str:
    """Plain text of an HTML fragment: entities unescaped, paragraph breaks kept, no markup."""
    if not html:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(str(html))
        parser.close()
    except Exception:  # noqa: BLE001 - odd markup must not lose the article
        log.warning("news: could not parse an article's HTML; kept it as text")
        parser.parts = [re.sub(r"<[^>]*>", " ", str(html))]
    text = "".join(parser.parts).replace("\xa0", " ").replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _one_line(html: str | None) -> str:
    return re.sub(r"\s+", " ", html_to_text(html)).strip()


def _clean(value: Any) -> str:
    """A plain string with no NUL characters (Postgres text cannot hold them)."""
    return "" if value is None else str(value).replace("\x00", "")


# ------------------------------------------------------------------ time


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime | None:
    """An RFC 3339 time (Alpaca sometimes sends nanoseconds) as UTC, or None."""
    if isinstance(value, datetime):
        return _utc(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"(\.\d{6})\d+", r"\1", value.strip())
    try:
        return _utc(datetime.fromisoformat(text))
    except ValueError:
        return None


# ------------------------------------------------------------------ Alpaca


def parse_news_response(raw: Any) -> tuple[list[Article], str | None]:
    """Articles and next page token from one raw /v1beta1/news response.

    The response is {"news": [{"id", "headline", "summary", "content", "author", "source",
    "url", "symbols", "created_at", "updated_at", ...}], "next_page_token": str | null}.
    An item without an id, headline or publish time is dropped (it cannot be stored).
    """
    if not isinstance(raw, dict):
        return [], None
    articles: list[Article] = []
    for item in raw.get("news") or []:
        if not isinstance(item, dict):
            continue
        created = _parse_time(item.get("created_at"))
        try:
            article_id = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        headline = _one_line(_clean(item.get("headline")))
        if created is None or not headline:
            continue
        symbols = tuple(str(s).strip() for s in (item.get("symbols") or []) if str(s).strip())
        articles.append(Article(
            id=article_id,
            headline=headline,
            summary=html_to_text(_clean(item.get("summary"))),
            content=html_to_text(_clean(item.get("content")))[:CONTENT_LIMIT],
            author=_one_line(_clean(item.get("author"))),
            source=_one_line(_clean(item.get("source"))),
            url=_clean(item.get("url")).strip(),
            symbols=symbols,
            created_at=created,
            updated_at=_parse_time(item.get("updated_at")),
        ))
    token = raw.get("next_page_token")
    return articles, (str(token) if token else None)


class AlpacaNewsSource:
    """Alpaca's news feed through alpaca-py's NewsClient(key, secret, raw_data=True).

    One fetch is one request, so the page token is in our hands. NewsClient.get_news would
    page through everything itself and drop the token (its raw result is only
    {"news": [...]}), so this calls the client's own GET on the same path and parameters
    (NewsRequest.to_request_fields(), "/news"), which returns the whole JSON body,
    next_page_token included. Every request goes through the shared limiter.
    """

    connected = True
    problem: str | None = None

    def __init__(self, key_id: str, secret: str, client: Any = None) -> None:
        if client is None:
            from alpaca.data.historical.news import NewsClient

            client = NewsClient(key_id, secret, raw_data=True)
        self._client = client

    def fetch(self, start: datetime, end: datetime, symbols: list[str],
              page_token: str | None) -> tuple[list[Article], str | None]:
        from alpaca.data.requests import NewsRequest

        request = NewsRequest(start=_utc(start), end=_utc(end), symbols=",".join(symbols) or None, sort="asc",
                              limit=PAGE_SIZE, include_content=True, exclude_contentless=False,
                              page_token=page_token)
        try:
            data.ALPACA_LIMITER.acquire()
            raw = self._client.get(path="/news", data=request.to_request_fields())
        except Exception as exc:  # noqa: BLE001 - turned into one plain message
            raise RuntimeError(_explain(exc)) from None
        return parse_news_response(raw)


def _explain(exc: Exception) -> str:
    """A plain-words message for a failed news request."""
    from alpaca.common.exceptions import APIError

    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            return "Alpaca refused the keys in .env on box1: check ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY"
        if status == 429:
            return "Alpaca says too many requests: the news feed will be tried again in a minute"
        return f"Alpaca news error: {exc}"
    return f"Could not read Alpaca's news feed: {str(exc) or type(exc).__name__}"


class NoNewsSource:
    """Stands in when the Alpaca keys are missing: not connected, and fetching says why."""

    connected = False

    def __init__(self, problem: str) -> None:
        self.problem: str | None = problem

    def fetch(self, start: datetime, end: datetime, symbols: list[str],
              page_token: str | None) -> tuple[list[Article], str | None]:
        raise RuntimeError(self.problem)


# ------------------------------------------------------------------ fake

_DEMO_STOCK_HEADLINES = (
    "Demo: {s} beats quarterly estimates",
    "Demo: {s} misses on revenue, shares slip",
    "Demo: analyst raises {s} price target",
    "Demo: {s} announces a new buyback",
    "Demo: regulators open a made-up inquiry into {s}",
    "Demo: {s} unveils a pretend product line",
)
_DEMO_CRYPTO_HEADLINES = (
    "Demo: {s} rallies as pretend buyers return",
    "Demo: {s} drops after a made-up exchange outage",
    "Demo: large made-up wallet moves {s}",
    "Demo: {s} network upgrade goes live (not real)",
)
_DEMO_OTHER = ("TSLA", "NFLX", "AMD", "DIS", "BA")


class FakeNewsSource:
    """Deterministic made-up articles for tests and FLEET_FAKE_BROKER demos.

    A few articles every hour on a fixed grid, so any window is a slice of one endless
    list: ids, times and text depend only on (seed, hour). Most name one or two of our
    symbols; some also name a symbol we do not trade (TSLA...), and some name only that
    (the ingest must skip those). Crypto appears as "BTCUSD". Every headline starts with
    "Demo:" so nobody mistakes it for news.
    """

    connected = True
    problem: str | None = None

    def __init__(self, seed: int = 0, page_size: int = PAGE_SIZE) -> None:
        self.seed = int(seed)
        self.page_size = max(1, int(page_size))

    def _hour(self, hour: int) -> list[Article]:
        rng = random.Random(f"fleet2-news:{self.seed}:{hour}")
        out: list[Article] = []
        tags = universe_tags()
        for j in range(rng.randint(2, 4)):
            created = datetime.fromtimestamp(hour * 3600 + rng.randint(0, 3599), timezone.utc)
            roll = rng.random()
            if roll < 0.12:
                symbols = [rng.choice(_DEMO_OTHER)]
            else:
                symbols = rng.sample(tags, rng.choice((1, 1, 2)))
                if roll < 0.25:
                    symbols.append(rng.choice(_DEMO_OTHER))
            lead = symbols[0]
            crypto = to_universe(lead) in universe.CRYPTO
            headline = rng.choice(_DEMO_CRYPTO_HEADLINES if crypto else _DEMO_STOCK_HEADLINES).format(s=lead)
            article_id = 9_000_000_000_000 + hour * 10 + j
            out.append(Article(
                id=article_id,
                headline=headline,
                summary=f"Made-up demo article about {', '.join(symbols)}.",
                content=(f"<p>{headline}.</p><p>This is a made-up demo article about {' and '.join(symbols)} "
                         f"&amp; nothing in it is real.</p>"),
                author="Demo Desk", source="demo", url=f"https://example.invalid/demo/{article_id}",
                symbols=tuple(symbols), created_at=created, updated_at=created,
            ))
        return out

    def fetch(self, start: datetime, end: datetime, symbols: list[str],
              page_token: str | None) -> tuple[list[Article], str | None]:
        start, end = _utc(start), _utc(end)
        try:
            offset = int(page_token) if page_token else 0
            if offset < 0:
                raise ValueError
        except ValueError:
            raise RuntimeError(f"bad page token {page_token!r}") from None
        wanted = {s.upper() for s in symbols}
        matching: list[Article] = []
        for hour in range(int(start.timestamp()) // 3600, int(end.timestamp()) // 3600 + 1):
            for a in self._hour(hour):
                if start <= a.created_at <= end and (not wanted or wanted.intersection(a.symbols)):
                    matching.append(a)
        matching.sort(key=lambda a: (a.created_at, a.id))
        page = matching[offset:offset + self.page_size]
        more = offset + self.page_size < len(matching)
        return page, (str(offset + self.page_size) if more else None)


def make_news_source(config: Any) -> NewsSource:
    """Made-up articles in demo mode, Alpaca with the paper keys, else a source that explains."""
    if getattr(config, "fake_broker", False):
        return FakeNewsSource()
    if config.alpaca_paper_key_id and config.alpaca_paper_secret:
        return AlpacaNewsSource(config.alpaca_paper_key_id, config.alpaca_paper_secret)
    return NoNewsSource(MISSING_KEYS_MESSAGE)


# ------------------------------------------------------------------ storing

_UPSERT = """
INSERT INTO news (id, headline, summary, content, author, source, url, symbols, ours,
                  created_at, updated_at, analysis_status)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
  headline = EXCLUDED.headline,
  summary = EXCLUDED.summary,
  content = CASE WHEN EXCLUDED.content = '' THEN news.content ELSE EXCLUDED.content END,
  author = EXCLUDED.author,
  source = EXCLUDED.source,
  url = EXCLUDED.url,
  symbols = EXCLUDED.symbols,
  ours = EXCLUDED.ours,
  updated_at = EXCLUDED.updated_at
RETURNING (xmax = 0) AS inserted
"""


def ours_of(article: Article) -> list[str]:
    """The article's tags that are in our universe, in our form, deduplicated and sorted."""
    return sorted({u for u in (to_universe(s) for s in article.symbols) if u is not None})


def store_articles(conn: psycopg.Connection, articles: list[Article], status: str = "pending") -> int:
    """Upsert articles that name a symbol we trade; returns how many rows are new.

    An existing row gets its text and symbols refreshed, but its analysis_status (and
    note) is never touched, so seeing an article again cannot queue it for analysis twice.
    """
    new = 0
    with conn.transaction():
        for a in articles:
            ours = ours_of(a)
            if not ours:
                continue
            row = conn.execute(_UPSERT, (
                a.id, _clean(a.headline), _clean(a.summary), _clean(a.content)[:CONTENT_LIMIT],
                _clean(a.author), _clean(a.source), _clean(a.url), list(a.symbols), ours,
                a.created_at, a.updated_at, status,
            )).fetchone()
            new += 1 if row["inserted"] else 0
    return new


# ------------------------------------------------------------------ polling


def _read_cursor(conn: psycopg.Connection) -> datetime | None:
    row = conn.execute("SELECT value FROM settings WHERE key = %s", (CURSOR_KEY,)).fetchone()
    return _parse_time(row["value"]) if row is not None else None


def _write_cursor(conn: psycopg.Connection, cursor: datetime) -> None:
    # Straight to the table, no audit row: this changes every minute.
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (%s, %s, now())"
        " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        (CURSOR_KEY, Jsonb(_iso(cursor))),
    )


def poll_step(conn: psycopg.Connection, source: NewsSource, now: datetime) -> dict[str, Any]:
    """One live poll: new articles since the cursor (10 minutes of overlap), stored "pending".

    Returns {"added": rows newly stored, "error": message or None}. The cursor moves to the
    newest publish time seen (never backwards, never past `now`); after an error it stays
    where it was, so the next poll covers the same window again.
    """
    now = _utc(now)
    if not source.connected:
        return {"added": 0, "error": source.problem or "The news feed is not connected"}
    cursor = _read_cursor(conn)
    start = cursor - OVERLAP if cursor is not None else now - FIRST_LOOKBACK
    tags = universe_tags()
    added = 0
    newest: datetime | None = None
    token: str | None = None
    try:
        for page in range(POLL_MAX_PAGES):
            articles, token = source.fetch(start, now, tags, token)
            added += store_articles(conn, articles, "pending")
            for a in articles:
                if newest is None or a.created_at > newest:
                    newest = a.created_at
            if token is None:
                break
        else:
            log.info("news poll stopped at %d pages; the next poll carries on from the cursor", POLL_MAX_PAGES)
    except Exception as exc:  # noqa: BLE001 - shown on the dashboard, tried again next pass
        message = str(exc) or type(exc).__name__
        log.warning("news poll failed: %s", message)
        return {"added": added, "error": message}
    if newest is not None:
        new_cursor = min(max(_utc(newest), cursor) if cursor is not None else _utc(newest), now)
        if cursor is None or new_cursor > cursor:
            _write_cursor(conn, new_cursor)
    return {"added": added, "error": None}


def backfill(conn: psycopg.Connection, source: NewsSource, start: datetime, end: datetime,
             max_pages: int = BACKFILL_MAX_PAGES) -> int:
    """Store every article of a window with status "backfill" (replay only); returns rows new.

    Reads at most `max_pages` pages (50 articles each); a source error is raised. Existing
    rows keep their status, so a backfill cannot re-queue an article already analysed.
    """
    tags = universe_tags()
    added = 0
    token: str | None = None
    for _ in range(max_pages):
        articles, token = source.fetch(_utc(start), _utc(end), tags, token)
        added += store_articles(conn, articles, "backfill")
        if token is None:
            break
    else:
        log.warning("news backfill stopped at %d pages; the window is not fully covered", max_pages)
    return added


# ------------------------------------------------------------------ reading


def recent(conn: psycopg.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """The newest stored articles, newest first, without the full text (for the dashboard)."""
    limit = max(1, min(int(limit), 500))
    rows = conn.execute(
        "SELECT id, headline, summary, author, source, url, symbols, ours, created_at, fetched_at, "
        "analysis_status, analysis_note FROM news ORDER BY created_at DESC, id DESC LIMIT %s",
        (limit,),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["created_at"] = _iso(r["created_at"])
        d["fetched_at"] = _iso(r["fetched_at"])
        out.append(d)
    return out
