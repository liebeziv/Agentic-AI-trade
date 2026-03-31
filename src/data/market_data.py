"""Market data fetcher — supports yfinance (history) and ccxt (crypto)."""
from __future__ import annotations
import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

import pandas as pd
import polars as pl
import yfinance as yf

from src.types import Bar
from src.utils.logger import get_logger

log = get_logger(__name__)

# How instrument types map to data sources
SOURCE_MAP: dict[str, list[str]] = {
    "forex":     ["yfinance", "mt5"],
    "futures":   ["yfinance", "ib"],
    "us_stocks": ["yfinance", "alpaca"],
    "hk_stocks": ["yfinance", "futu"],
    "crypto":    ["ccxt"],
}

# yfinance ticker aliases for non-standard symbols
YFINANCE_ALIASES: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "ES":     "ES=F",
    "NQ":     "NQ=F",
    "GC":     "GC=F",
    "CL":     "CL=F",
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",
}

# Map timeframe strings to yfinance interval strings
TF_TO_YF: dict[str, str] = {
    "1m": "1m",  "5m": "5m",  "15m": "15m",
    "30m": "30m", "1h": "1h",  "4h": "4h",
    "1d": "1d",
}

# Max lookback yfinance supports for each interval
YF_MAX_DAYS: dict[str, int] = {
    "1m": 7, "5m": 60, "15m": 60, "30m": 60,
    "1h": 730, "4h": 730, "1d": 9999,
}


def _infer_instrument_type(instrument: str) -> str:
    if "/" in instrument:
        return "crypto"
    if instrument.endswith("USD") or instrument.endswith("JPY") or instrument.endswith("EUR"):
        return "forex"
    if instrument in {"ES", "NQ", "GC", "CL", "SI", "ZB"}:
        return "futures"
    if instrument.endswith(".HK"):
        return "hk_stocks"
    return "us_stocks"


def _yf_to_polars(df_pd, instrument: str, timeframe: str, source: str = "yfinance") -> pl.DataFrame:
    """Convert a yfinance pandas DataFrame to a polars DataFrame matching our schema."""
    _empty = pl.DataFrame(schema={
        "timestamp": pl.Datetime, "open": pl.Float64, "high": pl.Float64,
        "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
        "instrument": pl.Utf8, "timeframe": pl.Utf8, "source": pl.Utf8,
    })
    if df_pd is None or df_pd.empty:
        return _empty

    df_pd = df_pd.copy()

    # yfinance ≥1.0 returns MultiIndex columns: ('Close', 'AAPL') etc.
    if isinstance(df_pd.columns, pd.MultiIndex):
        # Flatten: keep only the first level (OHLCV field name)
        df_pd.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                         for c in df_pd.columns]
        # Drop duplicate columns (can happen with multi-ticker downloads)
        df_pd = df_pd.loc[:, ~df_pd.columns.duplicated()]
    else:
        df_pd.columns = [str(c).lower().replace(" ", "_") for c in df_pd.columns]

    df_pd.index.name = "timestamp"
    df_pd = df_pd.reset_index()

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(df_pd.columns)):
        return _empty

    df_pd = df_pd[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df_pd["instrument"] = instrument
    df_pd["timeframe"] = timeframe
    df_pd["source"] = source
    df_pd = df_pd.dropna(subset=["open", "high", "low", "close"])

    return pl.from_pandas(df_pd).with_columns([
        pl.col("timestamp").cast(pl.Datetime),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    ])


class MarketDataFetcher:
    def __init__(self, config: dict, store=None) -> None:
        self.config = config
        self.store = store
        self._subscribers: dict[str, list[Callable]] = {}
        self._ccxt_exchanges: dict[str, object] = {}

    def get_instrument_type(self, instrument: str) -> str:
        return _infer_instrument_type(instrument)

    def _yf_ticker(self, instrument: str) -> str:
        return YFINANCE_ALIASES.get(instrument, instrument)

    async def get_historical(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV bars. Checks cache first."""
        # Check cache
        if self.store:
            cached = self.store.get_bars(instrument, timeframe, start, end)
            if not cached.is_empty():
                return cached

        instrument_type = self.get_instrument_type(instrument)

        if instrument_type == "crypto":
            df = await self._fetch_ccxt_historical(instrument, timeframe, start, end)
        else:
            df = await self._fetch_yf_historical(instrument, timeframe, start, end)

        if self.store and not df.is_empty():
            self.store.save_bars(df)

        return df

    async def get_latest_bars(
        self, instrument: str, timeframe: str, count: int = 200
    ) -> pl.DataFrame:
        """Fetch the most recent N bars."""
        yf_interval = TF_TO_YF.get(timeframe, "5m")
        max_days = YF_MAX_DAYS.get(timeframe, 60)
        end = datetime.utcnow()
        # Request enough history to get `count` bars
        start = end - timedelta(days=min(max_days, count * 2))

        instrument_type = self.get_instrument_type(instrument)

        if instrument_type == "crypto":
            df = await self._fetch_ccxt_historical(instrument, timeframe, start, end)
        else:
            df = await self._fetch_yf_historical(instrument, timeframe, start, end)

        # Return last `count` rows
        if len(df) > count:
            df = df.tail(count)

        return df

    async def _fetch_yf_historical(
        self, instrument: str, timeframe: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        ticker_sym = self._yf_ticker(instrument)
        yf_interval = TF_TO_YF.get(timeframe, "5m")

        loop = asyncio.get_event_loop()
        try:
            df_pd = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    ticker_sym,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=True,
                )
            )
        except Exception as exc:
            log.warning("yfinance fetch failed", instrument=instrument, error=str(exc))
            return pl.DataFrame()

        return _yf_to_polars(df_pd, instrument, timeframe)

    async def _fetch_ccxt_historical(
        self, instrument: str, timeframe: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        try:
            import ccxt.async_support as ccxt_async
        except ImportError:
            log.warning("ccxt not installed, falling back to yfinance for crypto")
            return await self._fetch_yf_historical(instrument, timeframe, start, end)

        exchange_id = "binance"
        if exchange_id not in self._ccxt_exchanges:
            self._ccxt_exchanges[exchange_id] = ccxt_async.binance({"enableRateLimit": True})

        exchange = self._ccxt_exchanges[exchange_id]
        since_ms = int(start.timestamp() * 1000)

        try:
            ohlcv = await exchange.fetch_ohlcv(instrument, timeframe, since=since_ms, limit=1000)
        except Exception as exc:
            log.warning("ccxt fetch failed, falling back to yfinance", instrument=instrument, error=str(exc))
            return await self._fetch_yf_historical(instrument, timeframe, start, end)

        if not ohlcv:
            return pl.DataFrame()

        rows = [
            {
                "timestamp": datetime.utcfromtimestamp(row[0] / 1000),
                "open": float(row[1]), "high": float(row[2]),
                "low": float(row[3]), "close": float(row[4]),
                "volume": float(row[5]),
                "instrument": instrument, "timeframe": timeframe, "source": "ccxt",
            }
            for row in ohlcv
        ]
        return pl.DataFrame(rows)

    async def close(self) -> None:
        for ex in self._ccxt_exchanges.values():
            try:
                await ex.close()
            except Exception:
                pass
