"""The coordinator's only door to Alpaca (alpaca-py). Nothing else imports alpaca.

Stage 1 is read-only: the market clock (are stocks open, when do they close) and the
account (value, today's change). Orders arrive in stage 4 and go through
coordinator.safety first.

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


class Broker(Protocol):
    """What the rest of the coordinator may ask of Alpaca."""

    mode: str
    connected: bool
    problem: str | None

    def clock(self) -> ClockInfo: ...

    def account(self) -> AccountInfo: ...


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


def choose_mode(config: Config, live_confirmed: bool) -> str:
    """LIVE only with ALPACA_LIVE=true, the owner's confirmation and live keys; else PAPER."""
    if config.alpaca_live_allowed and live_confirmed and config.alpaca_live_key_id and config.alpaca_live_secret:
        return LIVE
    return PAPER


def make_broker(config: Config, live_confirmed: bool = False) -> Broker:
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
            if clock_info is not None:
                self.clock_info = clock_info
            if account_info is not None:
                self.account_info = account_info
            self.error = "; ".join(errors) or None
        if errors:
            log.warning("broker status: %s", self.error)
