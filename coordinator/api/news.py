"""News routes: the owner starts and stops paper trading for a news strategy (stocks or
crypto). The strategy only starts once its saved predictions beat the coin-flip twin
(news_scoring.gate, checked inside news_trading.start)."""
from __future__ import annotations

from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from coordinator.api.deps import DB, require_owner
from coordinator.api.serialize import jsonable
from coordinator.api.trading import LIVE_PHRASE
from coordinator.broker import LIVE
from coordinator.errors import BadRequest, NotFound

MARKETS = ("stocks", "crypto")

router = APIRouter(prefix="/api/news", tags=["news"], dependencies=[Depends(require_owner)])


class StartBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    confirm: str | None = Field(default=None, max_length=64)


def _market(market: str) -> str:
    if market not in MARKETS:
        raise NotFound(f"There is no news market called {market!r}: use stocks or crypto")
    return market


def _message(result: Any, default: str) -> str:
    text = result.get("message") if isinstance(result, dict) else None
    return text if isinstance(text, str) and text else default


@router.post("/{market}/start", status_code=201)
def start(market: str, request: Request, body: StartBody | None = None, conn: psycopg.Connection = DB) -> dict[str, Any]:
    """Open the news strategy's book for this market (refused until its predictions beat the coin flip)."""
    from coordinator import news_trading  # the trading module is imported late: the screen works without it
    market = _market(market)
    mode = request.app.state.broker_status.broker.mode
    if mode == LIVE and (body or StartBody()).confirm != LIVE_PHRASE:
        raise BadRequest(f"The coordinator is in live mode: type {LIVE_PHRASE} to trade the news strategy with real money")
    result = news_trading.start(conn, market, mode, request.app.state.limits)
    word = "live" if mode == LIVE else "paper"
    return {**jsonable(result if isinstance(result, dict) else {}),
            "message": _message(result, f"News {market} strategy is {word} trading")}


@router.post("/{market}/stop")
def stop(market: str, conn: psycopg.Connection = DB) -> dict[str, Any]:
    """Close every position of the news strategy for this market, then its book."""
    from coordinator import news_trading
    market = _market(market)
    result = news_trading.stop(conn, market)
    return {**jsonable(result if isinstance(result, dict) else {}),
            "message": _message(result, f"News {market} strategy stopped; its positions will be sold")}
