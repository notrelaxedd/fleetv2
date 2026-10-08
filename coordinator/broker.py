"""The coordinator's only door to Alpaca (alpaca-py). Nothing else imports alpaca.

It reads the market clock (are stocks open, when do they close) and the account
(value, today's change), and places market orders. Every order reaches place_order()
only through coordinator.safety.approve_and_place(), which checks the pause switch,
the daily loss limit and the money limits first and logs the order.

Mode: paper, always, unless ALPACA_LIVE=true is set in .env AND the owner confirmed
live in the dashboard AND live keys are present. The broker never switches mode on its
own: the mode is fixed when the coordinator builds it.

Requests are kept few: BrokerStatus refreshes clock and account at most every
STATUS_TTL_S seconds from the background loop, and every page reads that cache, so
the dashboard never spends Alpaca's 200-requests-a-minute budget.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from coordinator.config import Config

log = logging.getLogger(__name__)

STATUS_TTL_S = 30.0
PAPER = "paper"
LIVE = "live"


@dataclass(frozen=True)
class ClockInfo:
    """Alpaca's market clock for US stocks."""

    is_open: bool
    next_open: datetime
    next_close: datetime
    timestamp: datetime


@dataclass(frozen=True)
class AccountInfo:
    """The parts of the Alpaca account the dashboard and the safety checks use."""

    equity: float
    last_equity: float
    cash: float
    buying_power: float

    @property
    def day_change(self) -> float:
        """Today's profit or loss in dollars (equity now minus at the last close)."""
        return self.equity - self.last_equity

    @property
    def day_change_pct(self) -> float:
        """Today's change as a percent of the last close (0 when there is no history)."""
        return 100.0 * self.day_change / self.last_equity if self.last_equity else 0.0


@dataclass(frozen=True)
class OrderState:
    """Where an order stands at the broker, in our own words."""

    status: str  # submitted, partially_filled, filled, cancelled, rejected
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    broker_order_id: str | None = None
    error: str | None = None


# Alpaca order statuses -> ours.
STATUS_MAP = {
    "filled": "filled",
    "partially_filled": "partially_filled",
    "canceled": "cancelled", "expired": "cancelled", "replaced": "cancelled", "stopped": "cancelled",
    "rejected": "rejected", "suspended": "rejected",
}


NEVER_REACHED = "the order never reached Alpaca"


def _http_status(exc: Exception) -> int | None:
    """The HTTP status of an alpaca-py APIError, None for transport errors."""
    try:
        code = getattr(exc, "status_code", None)
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


class Broker(Protocol):
    """What the rest of the coordinator may ask of Alpaca."""

    mode: str
    connected: bool
    problem: str | None

    def clock(self) -> ClockInfo: ...

    def account(self) -> AccountInfo: ...

    def place_order(self, client_order_id: str, symbol: str, side: str, notional: float | None = None,
                    qty: float | None = None) -> OrderState: ...

    def get_order(self, client_order_id: str) -> OrderState: ...


class NoBroker:
    """No keys in .env: nothing is called, and the dashboard says what to add."""

    connected = False

    def __init__(self, mode: str = PAPER, problem: str = "Add your Alpaca paper keys to .env on box1") -> None:
        self.mode = mode
        self.problem = problem

    def clock(self) -> ClockInfo:
        raise BrokerUnavailable(self.problem)

    def account(self) -> AccountInfo:
        raise BrokerUnavailable(self.problem)

    def place_order(self, client_order_id: str, symbol: str, side: str, notional: float | None = None,
                    qty: float | None = None) -> OrderState:
        raise BrokerUnavailable(self.problem)

    def get_order(self, client_order_id: str) -> OrderState:
        raise BrokerUnavailable(self.problem)


class BrokerUnavailable(RuntimeError):
    """Alpaca could not be asked (no keys, or the call failed)."""


class AlpacaBroker:
    """alpaca-py TradingClient, paper unless built with mode LIVE."""

    connected = True
    problem: str | None = None

    def __init__(self, key_id: str, secret: str, mode: str = PAPER) -> None:
        from alpaca.trading.client import TradingClient

        self.mode = mode
        self._client = TradingClient(key_id, secret, paper=(mode != LIVE))

    def clock(self) -> ClockInfo:
        try:
            c = self._client.get_clock()
        except Exception as exc:  # noqa: BLE001 - alpaca raises several types; the message is what matters
            raise BrokerUnavailable(f"Alpaca clock: {exc}") from None
        return ClockInfo(bool(c.is_open), c.next_open, c.next_close, c.timestamp)

    def account(self) -> AccountInfo:
        try:
            a = self._client.get_account()
        except Exception as exc:  # noqa: BLE001
            raise BrokerUnavailable(f"Alpaca account: {exc}") from None
        return AccountInfo(
            equity=float(a.equity or 0),
            last_equity=float(a.last_equity or 0),
            cash=float(a.cash or 0),
            buying_power=float(a.buying_power or 0),
        )

    def place_order(self, client_order_id: str, symbol: str, side: str, notional: float | None = None,
                    qty: float | None = None) -> OrderState:
        """A market order: buys by dollar amount (notional), sells by quantity. Stocks use
        time in force "day" (Alpaca requires it for fractional orders), crypto "gtc"."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        crypto = "/" in symbol
        request = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2) if notional is not None else None,
            qty=round(qty, 9) if qty is not None else None,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if crypto else TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            order = self._client.submit_order(request)
        except Exception as exc:  # noqa: BLE001 - sorted into "refused" or "unknown" below
            if _http_status(exc) is not None and 400 <= _http_status(exc) < 500:
                return OrderState("rejected", error=f"Alpaca refused the order: {exc}")
            # No clear answer (timeout, reset, 5xx): the order may exist at Alpaca. It
            # stays open and the fill poller asks Alpaca by its client order id.
            return OrderState("submitted", error=f"no clear answer from Alpaca ({exc}); checking")
        return self._state(order)

    def get_order(self, client_order_id: str) -> OrderState:
        try:
            order = self._client.get_order_by_client_id(client_order_id)
        except Exception as exc:  # noqa: BLE001
            if _http_status(exc) == 404:
                return OrderState("rejected", error=NEVER_REACHED)
            raise BrokerUnavailable(f"Alpaca order lookup: {exc}") from None
        return self._state(order)

    @staticmethod
    def _state(order: Any) -> OrderState:
        raw = getattr(order.status, "value", order.status)
        return OrderState(
            status=STATUS_MAP.get(str(raw), "submitted"),
            filled_qty=float(order.filled_qty or 0),
            filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            broker_order_id=str(order.id),
        )


class FakeBroker:
    """A stand-in for tests and FLEET_FAKE_BROKER=1 demos; never used with real keys.

    The dashboard labels its numbers "(demo data)" so a screenshot cannot pass for a
    real account.
    """

    connected = True
    problem: str | None = None
    fake = True

    def __init__(self, equity: float = 100_000.0, last_equity: float = 100_000.0, is_open: bool = True,
                 now: datetime | None = None) -> None:
        self.mode = PAPER
        self.equity = equity
        self.last_equity = last_equity
        self.is_open = is_open
        self._now = now

    def clock(self) -> ClockInfo:
        now = self._now or datetime.now(timezone.utc)
        close = now.replace(hour=20, minute=0, second=0, microsecond=0)
        if close <= now:
            close += timedelta(days=1)
        return ClockInfo(self.is_open, close + timedelta(hours=17, minutes=30), close, now)

    def account(self) -> AccountInfo:
        return AccountInfo(self.equity, self.last_equity, self.equity, self.equity)

    # Orders fill at once at `price_of(symbol)` with 5 bp (stocks) or 10 bp (crypto)
    # slippage. price_of is set by the app (latest cached close) or by a test.
    price_of: Any = None
    refuse: str | None = None

    def place_order(self, client_order_id: str, symbol: str, side: str, notional: float | None = None,
                    qty: float | None = None) -> OrderState:
        if not hasattr(self, "_orders"):
            self._orders: dict[str, OrderState] = {}
        if self.refuse:
            state = OrderState("rejected", error=self.refuse)
        else:
            price = float(self.price_of(symbol)) if self.price_of else 100.0
            slip = (0.0010 if "/" in symbol else 0.0005) * (1 if side == "buy" else -1)
            fill = price * (1 + slip)
            filled = (notional / fill) if notional is not None else float(qty or 0)
            state = OrderState("filled", filled, fill, f"fake-{client_order_id[:8]}")
        self._orders[client_order_id] = state
        return state

    def get_order(self, client_order_id: str) -> OrderState:
        return getattr(self, "_orders", {}).get(client_order_id) or OrderState("rejected", error=NEVER_REACHED)


def key_fingerprint(key_id: str) -> str:
    """A short, one-way fingerprint of a live key id (never the key itself)."""
    import hashlib

    return hashlib.sha256(key_id.encode("utf-8")).hexdigest()[:16] if key_id else ""


def confirmation_for(config: Config) -> dict[str, str]:
    """What the owner's live confirmation stores: the fingerprint of the live keys it
    was given for, so new or changed keys need a new confirmation."""
    return {"key": key_fingerprint(config.alpaca_live_key_id)}


def choose_mode(config: Config, live_confirmed: Any) -> str:
    """LIVE only with ALPACA_LIVE=true, live keys, and the owner's confirmation given
    for exactly those keys; PAPER otherwise. Decided once, when the coordinator starts."""
    if not (config.alpaca_live_allowed and config.alpaca_live_key_id and config.alpaca_live_secret):
        return PAPER
    if not isinstance(live_confirmed, dict) or live_confirmed.get("key") != key_fingerprint(config.alpaca_live_key_id):
        return PAPER
    return LIVE


def make_broker(config: Config, live_confirmed: Any = None) -> Broker:
    """The broker for this run of the coordinator."""
    if config.fake_broker:
        return FakeBroker()
    mode = choose_mode(config, live_confirmed)
    if mode == LIVE:
        return AlpacaBroker(config.alpaca_live_key_id, config.alpaca_live_secret, LIVE)
    if not (config.alpaca_paper_key_id and config.alpaca_paper_secret):
        return NoBroker()
    return AlpacaBroker(config.alpaca_paper_key_id, config.alpaca_paper_secret, PAPER)


class BrokerStatus:
    """Cached clock and account, refreshed at most every STATUS_TTL_S seconds."""

    def __init__(self, broker: Broker, ttl: float = STATUS_TTL_S, clock: Any = time.monotonic) -> None:
        self.broker = broker
        self.ttl = ttl
        self._mono = clock
        self._lock = threading.Lock()
        self._at: float | None = None
        self.clock_info: ClockInfo | None = None
        self.account_info: AccountInfo | None = None
        self.error: str | None = broker.problem

    def age(self) -> float:
        """Seconds since the last refresh (infinite before the first)."""
        with self._lock:
            return float("inf") if self._at is None else self._mono() - self._at

    def refresh(self, force: bool = False) -> None:
        """Ask Alpaca again when the cache is older than ttl (or force)."""
        if not self.broker.connected:
            return
        with self._lock:
            now = self._mono()
            if not force and self._at is not None and now - self._at < self.ttl:
                return
            self._at = now
        errors = []
        try:
            clock_info = self.broker.clock()
        except BrokerUnavailable as exc:
            clock_info, errors = None, errors + [str(exc)]
        try:
            account_info = self.broker.account()
        except BrokerUnavailable as exc:
            account_info, errors = None, errors + [str(exc)]
        with self._lock:
            # A failed reading clears the old one: safety checks must never trade on
            # a stale account value or a stale "market open".
            self.clock_info = clock_info
            self.account_info = account_info
            self.error = "; ".join(errors) or None
        if errors:
            log.warning("broker status: %s", self.error)
