"""What the fleet trades and how far back its price history goes.

Shared by the coordinator (what to download) and the models (what to trade), so the
two can never disagree. Stocks use daily bars from Alpaca's free IEX feed; crypto uses
hourly bars (crypto trades 24/7). The benchmarks are SPY for stock models and BTC for
crypto models, as the owner asked.
"""
from __future__ import annotations

STOCKS = (
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "V", "MA",
    "KO", "PEP", "XOM", "CVX", "HD", "LOW", "WMT", "COST", "UNH", "JNJ", "PG",
)
CRYPTO = ("BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "LINK/USD", "AVAX/USD", "DOGE/USD", "BCH/USD")

MARKETS = {
    "stocks": {"symbols": STOCKS, "timeframe": "1Day", "start": "2016-01-01", "benchmark": "SPY",
               "bars_per_year": 252},
    "crypto": {"symbols": CRYPTO, "timeframe": "1Hour", "start": "2021-01-01", "benchmark": "BTC/USD",
               "bars_per_year": 24 * 365},
}

# The most recent quarter of each market's history is held out: model search never
# sees it, and models are ranked only on it.
HELD_OUT_FRACTION = 0.25


def market_of(symbol: str) -> str:
    """"stocks" or "crypto" for a symbol in the universe; KeyError otherwise."""
    for market, spec in MARKETS.items():
        if symbol in spec["symbols"]:
            return market
    raise KeyError(symbol)


# Futures (Topstep day trading) sit beside MARKETS, not inside it, so nothing that loops
# over the stock and crypto markets ever meets them. Regular-session 1-minute bars of the
# CME micro index futures, from their launch in May 2019. Contract sizes are CME's:
# MES pays $5 per index point and moves in 0.25 point ticks ($1.25), MNQ pays $2 per
# point with 0.25 point ticks ($0.50). Without a Databento key, SPY and QQQ 1-minute
# bars from Alpaca stand in for them ("proxy" prices), scaled to roughly index points
# (S&P 500 is about 10 x SPY, Nasdaq-100 about 41 x QQQ) and rounded to whole ticks.
FUTURES = {"symbols": ("MES", "MNQ"), "timeframe": "1Min", "start": "2019-05-06", "dataset": "GLBX.MDP3",
           "schema": "ohlcv-1m"}
CONTRACTS = {
    "MES": {"point_value": 5.0, "tick": 0.25, "databento": "MES.v.0", "proxy": "SPY", "proxy_scale": 10.0},
    "MNQ": {"point_value": 2.0, "tick": 0.25, "databento": "MNQ.v.0", "proxy": "QQQ", "proxy_scale": 41.0},
}
# Futures history is split by trading day into three back-to-back periods, fixed once.
# Training is for model search only, held-out for ranking, the lockbox for one Final
# check per model. Nothing later in the list is ever read by anything earlier.
FUTURES_PERIODS = (("train", 0.60), ("held_out", 0.25), ("lockbox", 0.15))
