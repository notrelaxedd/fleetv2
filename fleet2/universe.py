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
