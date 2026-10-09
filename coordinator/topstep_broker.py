"""The coordinator's only door to Topstep: TopstepX, through the ProjectX Gateway API.
Nothing else talks to Topstep. Every order reaches it only through
coordinator.safety.approve_futures_order(), which checks the pause switches, Topstep's
loss limit, the time of day and the contract limits first and logs the order.

Calls used (ProjectX Gateway API; POST with JSON, a bearer token after login; every
answer carries "success", "errorCode" and "errorMessage"):
  /api/Auth/loginKey        {"userName", "apiKey"} -> {"token"}  (valid 24 hours)
  /api/Auth/validate        -> {"newToken"}
  /api/Account/search       {"onlyActiveAccounts": true} -> {"accounts": [{"id", "name", "balance", "canTrade"}]}
  /api/Contract/search      {"searchText": "MES", "live": false} -> {"contracts": [{"id", "name", "activeContract", ...}]}
  /api/History/retrieveBars {"contractId", "live", "startTime", "endTime", "unit": 2 (minute), "unitNumber": 1,
                             "limit", "includePartialBar": false} -> {"bars": [{"t", "o", "h", "l", "c", "v"}]}
  /api/Order/place          {"accountId", "contractId", "type": 2 (market), "side": 0 buy | 1 sell, "size",
                             "customTag"} -> {"orderId"}
  /api/Order/search         {"accountId", "startTimestamp"} -> {"orders": [{"id", "status", "fillVolume", "filledPrice"}]}
  /api/Position/searchOpen  {"accountId"} -> {"positions": [{"contractId", "type": 1 long | 2 short, "size", "averagePrice"}]}
These were taken from Topstep's documentation as quoted by two independent client
libraries (Python and Rust); this sandbox could not reach the API itself, so the first
real call happens on box1. "live": false asks for the data feed that Combine and
Express Funded (simulated) accounts trade on.

Rate limits (ProjectX): bars 50 requests per 30 seconds, everything else 200 per minute.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from coordinator.broker import NEVER_REACHED, OrderState
from coordinator.config import Config
from coordinator.data import RateLimiter

log = logging.getLogger(__name__)

TOKEN_HOURS = 23
MARKET, BUY, SELL = 2, 0, 1
# ProjectX order status -> ours (1 open, 2 filled, 3 cancelled, 4 expired, 5 rejected, 6 pending).
STATUS_MAP = {1: "submitted", 2: "filled", 3: "cancelled", 4: "cancelled", 5: "rejected", 6: "submitted"}
CONFIRM_PHRASE = "TRADE ON TOPSTEP"
NOT_SET = "Add TOPSTEPX_USERNAME, TOPSTEPX_API_KEY and TOPSTEPX_ACCOUNT to .env on box1 to trade on Topstep"


class TopstepError(RuntimeError):
    """TopstepX said no, or could not be reached."""


@dataclass(frozen=True)
class TopstepAccountInfo:
    id: str
    name: str
    balance: float
    can_trade: bool


def instrument_id(contract_id: str) -> int:
    """A whole number standing for a TopstepX contract id (for the roll checks)."""
    return zlib.crc32(contract_id.encode()) & 0x7FFFFFFF


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TopstepX:
    """A logged-in TopstepX client for one account. Thread-safe; tokens renew themselves."""

    fake = False

    def __init__(self, config: Config, opener: Callable[..., Any] | None = None, clock: Callable[[], float] = time.time) -> None:
        self.base = config.topstepx_api_url
        self._user, self._key = config.topstepx_username, config.topstepx_api_key
        self.account_name = config.topstepx_account
        self._open = opener or urllib.request.urlopen
        self._clock = clock
        self._lock = threading.Lock()
        self._token: str | None = None
        self._token_at = 0.0
        self._account: TopstepAccountInfo | None = None
        self._contracts: dict[tuple[str, str], str] = {}
        self.bars_limiter = RateLimiter(50 * 2 * 0.8)  # 50 per 30 s, with room to spare
        self.limiter = RateLimiter(200 * 0.8)

    # -------------------------------------------------------------- plumbing

    def _send(self, path: str, body: dict[str, Any], token: str | None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "fleet-v2"}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with self._open(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise TopstepError("401") from None
            raise TopstepError(f"TopstepX answered {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise TopstepError(f"Cannot reach TopstepX: {exc}") from None
        try:
            data = json.loads(raw)
        except ValueError:
            raise TopstepError(f"TopstepX sent something unreadable: {raw[:100]!r}") from None
        if not data.get("success", False) or int(data.get("errorCode") or 0) != 0:
            raise TopstepError(f"TopstepX refused ({data.get('errorCode')}): {data.get('errorMessage') or 'no reason given'}")
        return data

    def _login(self) -> str:
        if not (self._user and self._key):
            raise TopstepError(NOT_SET)
        try:
            data = self._send("/api/Auth/loginKey", {"userName": self._user, "apiKey": self._key}, None)
        except TopstepError as exc:
            if str(exc) == "401":
                raise TopstepError("TopstepX refused TOPSTEPX_USERNAME / TOPSTEPX_API_KEY in .env on box1") from None
            raise
        self._token, self._token_at = str(data["token"]), self._clock()
        return self._token

    def call(self, path: str, body: dict[str, Any], bars: bool = False) -> dict[str, Any]:
        (self.bars_limiter if bars else self.limiter).acquire()
        with self._lock:
            token = self._token
            if token is None or self._clock() - self._token_at > TOKEN_HOURS * 3600:
                token = self._login()
        try:
            return self._send(path, body, token)
        except TopstepError as exc:
            if str(exc) != "401":
                raise
            with self._lock:
                token = self._login()
            return self._send(path, body, token)

    # -------------------------------------------------------------- account and contracts

    def account(self, refresh: bool = False) -> TopstepAccountInfo:
        """The account named in TOPSTEPX_ACCOUNT (by name or id), with its balance now."""
        if self._account is not None and not refresh:
            return self._account
        accounts = self.call("/api/Account/search", {"onlyActiveAccounts": True}).get("accounts") or []
        wanted = self.account_name
        if not wanted:
            raise TopstepError("Set TOPSTEPX_ACCOUNT in .env to the account to trade (one of: "
                               + ", ".join(str(a.get("name")) for a in accounts) + ")")
        for a in accounts:
            if str(a.get("id")) == wanted or str(a.get("name")) == wanted:
                self._account = TopstepAccountInfo(str(a["id"]), str(a.get("name") or a["id"]),
                                                   float(a.get("balance") or 0.0), bool(a.get("canTrade")))
                return self._account
        raise TopstepError(f"No active TopstepX account named {wanted!r}")

    def contract(self, symbol: str, day: str | None = None) -> str:
        """Today's active contract id for MES or MNQ, e.g. CON.F.US.MES.Z26."""
        day = day or datetime.now(timezone.utc).date().isoformat()
        key = (symbol, day)
        if key not in self._contracts:
            found = self.call("/api/Contract/search", {"searchText": symbol, "live": False}).get("contracts") or []
            active = [c for c in found if c.get("activeContract") and str(c.get("name", "")).upper().startswith(symbol)]
            if not active:
                raise TopstepError(f"TopstepX lists no active {symbol} contract")
            self._contracts[key] = str(active[0]["id"])
        return self._contracts[key]

    # -------------------------------------------------------------- prices

    def bars(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """1-minute bars of today's active contract, start <= t < end (in pieces of at most
        20,000 bars, TopstepX's limit for one request), ascending."""
        contract = self.contract(symbol)
        iid = instrument_id(contract)
        out: list[dict[str, Any]] = []
        piece = timedelta(days=10)
        t = start
        while t < end:
            stop = min(t + piece, end)
            raw = self.call("/api/History/retrieveBars", {
                "contractId": contract, "live": False, "startTime": _iso(t), "endTime": _iso(stop),
                "unit": 2, "unitNumber": 1, "limit": 20000, "includePartialBar": False}, bars=True).get("bars") or []
            for b in raw:
                ts = int(datetime.fromisoformat(str(b["t"]).replace("Z", "+00:00")).timestamp())
                if int(t.timestamp()) <= ts < int(stop.timestamp()):
                    out.append({"t": ts, "o": float(b["o"]), "h": float(b["h"]), "l": float(b["l"]),
                                "c": float(b["c"]), "v": float(b.get("v") or 0), "iid": iid})
            t = stop
        out.sort(key=lambda b: b["t"])
        return out

    def cost(self, symbol: str, start: datetime, end: datetime) -> float:
        return 0.0

    feed = "topstepx"

    def fetch(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return self.bars(symbol, start, end)

    # -------------------------------------------------------------- orders and positions

    def positions(self) -> dict[str, tuple[int, float]]:
        """{contract id: (contracts, + long / - short, average price)}."""
        acc = self.account()
        rows = self.call("/api/Position/searchOpen", {"accountId": int(acc.id)}).get("positions") or []
        out = {}
        for p in rows:
            size = int(p.get("size") or 0) * (1 if int(p.get("type") or 1) == 1 else -1)
            out[str(p["contractId"])] = (size, float(p.get("averagePrice") or 0.0))
        return out

    def place_order(self, client_order_id: str, contract_id: str, side: str, qty: int) -> OrderState:
        """A market order for whole contracts, tagged with our order id."""
        acc = self.account()
        try:
            data = self.call("/api/Order/place", {"accountId": int(acc.id), "contractId": contract_id, "type": MARKET,
                                                  "side": BUY if side == "buy" else SELL, "size": int(qty),
                                                  "customTag": client_order_id})
        except TopstepError as exc:
            text = str(exc)
            if text.startswith("TopstepX refused"):
                return OrderState("rejected", error=text)
            return OrderState("submitted", error=f"no clear answer from TopstepX ({text}); checking")
        return OrderState("submitted", broker_order_id=str(data.get("orderId")))

    def get_order(self, client_order_id: str, broker_order_id: str | None = None) -> OrderState:
        acc = self.account()
        since = _iso(datetime.now(timezone.utc) - timedelta(days=2))
        orders = self.call("/api/Order/search", {"accountId": int(acc.id), "startTimestamp": since}).get("orders") or []
        for o in orders:
            if (broker_order_id and str(o.get("id")) == str(broker_order_id)) or o.get("customTag") == client_order_id:
                status = STATUS_MAP.get(int(o.get("status") or 1), "submitted")
                filled = float(o.get("fillVolume") or 0)
                price = o.get("filledPrice")
                return OrderState(status, filled, float(price) if price is not None else None, str(o.get("id")))
        if broker_order_id is None:
            return OrderState("rejected", error=NEVER_REACHED)
        return OrderState("submitted", broker_order_id=broker_order_id)


class FakeTopstep:
    """A stand-in for tests and FLEET_FAKE_BROKER=1 demos; never used with real keys.
    Orders fill at once at `price_of(symbol)`, one tick worse."""

    fake = True
    feed = "synthetic"

    def __init__(self, balance: float = 50_000.0, price_of: Callable[[str], float] | None = None) -> None:
        self.balance = balance
        self.price_of = price_of
        self._orders: dict[str, OrderState] = {}
        self._positions: dict[str, tuple[int, float]] = {}
        self.refuse: str | None = None
        self.placed: list[tuple[str, str, int]] = []

    def account(self, refresh: bool = False) -> TopstepAccountInfo:
        return TopstepAccountInfo("1", "Demo Combine", self.balance, True)

    def contract(self, symbol: str, day: str | None = None) -> str:
        return f"CON.F.US.{symbol}.DEMO"

    def positions(self) -> dict[str, tuple[int, float]]:
        return dict(self._positions)

    def place_order(self, client_order_id: str, contract_id: str, side: str, qty: int) -> OrderState:
        if self.refuse:
            state = OrderState("rejected", error=self.refuse)
        else:
            symbol = contract_id.split(".")[3]
            price = float(self.price_of(symbol)) if self.price_of else 5000.0
            price += 0.25 if side == "buy" else -0.25
            signed = int(qty) if side == "buy" else -int(qty)
            held, avg = self._positions.get(contract_id, (0, 0.0))
            new = held + signed
            self._positions[contract_id] = (new, price if held == 0 or (new and (new > 0) != (held > 0)) else avg)
            if new == 0:
                self._positions.pop(contract_id)
            self.placed.append((contract_id, side, int(qty)))
            state = OrderState("filled", float(qty), price, f"fake-{client_order_id[:8]}")
        self._orders[client_order_id] = state
        return state

    def get_order(self, client_order_id: str, broker_order_id: str | None = None) -> OrderState:
        return self._orders.get(client_order_id) or OrderState("rejected", error=NEVER_REACHED)


def fingerprint(config: Config) -> str:
    """One-way fingerprint of the TopstepX login and account (never the key itself)."""
    raw = f"{config.topstepx_username}|{config.topstepx_api_key}|{config.topstepx_account}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16] if config.topstepx_api_key else ""


@dataclass
class TopstepLink:
    """Whether Topstep trading is on for this run of the coordinator, and why not.
    On only with the keys in .env AND the owner's typed confirmation for exactly those
    keys, read when the coordinator starts (it never switches on by itself)."""

    client: Any = None
    problem: str | None = NOT_SET

    @property
    def on(self) -> bool:
        return self.client is not None and self.problem is None


def make_topstep(config: Config, confirmed: Any) -> TopstepLink:
    if config.fake_broker:
        return TopstepLink(FakeTopstep(), None)
    if not (config.topstepx_username and config.topstepx_api_key and config.topstepx_account):
        return TopstepLink(None, NOT_SET)
    client = TopstepX(config)
    if not isinstance(confirmed, dict) or confirmed.get("key") != fingerprint(config):
        return TopstepLink(client, f"Type {CONFIRM_PHRASE} on the Models screen (Futures) and restart the coordinator "
                                   "to trade on Topstep")
    return TopstepLink(client, None)
