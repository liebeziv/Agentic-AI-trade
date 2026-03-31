"""Technical indicator engine — 15+ indicators, signal generation, multi-timeframe."""
from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import polars as pl

from src.types import TechnicalSnapshot
from src.utils.logger import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------ helpers

def _to_pd(df: pl.DataFrame) -> pd.DataFrame:
    """Convert polars DataFrame to pandas for ta library."""
    pdf = df.to_pandas()
    if "timestamp" in pdf.columns:
        pdf = pdf.set_index("timestamp")
    return pdf


def _safe(series, default: float = 0.0) -> float:
    """Extract last non-NaN value from a pandas Series."""
    if series is None:
        return default
    arr = series.dropna()
    if arr.empty:
        return default
    return float(arr.iloc[-1])


# --------------------------------------------------------------- indicators

def compute_indicators(df: pl.DataFrame) -> dict[str, float]:
    """Compute all indicators from an OHLCV DataFrame. Returns flat dict."""
    if len(df) < 50:
        return {}

    pdf = _to_pd(df)
    close = pdf["close"]
    high = pdf["high"]
    low = pdf["low"]
    volume = pdf["volume"] if "volume" in pdf.columns else pd.Series(np.ones(len(close)), index=close.index)

    ind: dict[str, float] = {}
    ind["close"] = _safe(close)

    # --- Trend ---
    ind["sma_20"]  = _safe(ta.trend.sma_indicator(close, window=20))
    ind["sma_50"]  = _safe(ta.trend.sma_indicator(close, window=50))
    ind["sma_200"] = _safe(ta.trend.sma_indicator(close, window=200))
    ind["ema_9"]   = _safe(ta.trend.ema_indicator(close, window=9))
    ind["ema_21"]  = _safe(ta.trend.ema_indicator(close, window=21))

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    ind["macd"]        = _safe(macd.macd())
    ind["macd_signal"] = _safe(macd.macd_signal())
    ind["macd_hist"]   = _safe(macd.macd_diff())

    adx = ta.trend.ADXIndicator(high, low, close, window=14)
    ind["adx_14"] = _safe(adx.adx())
    ind["dmp_14"] = _safe(adx.adx_pos())
    ind["dmn_14"] = _safe(adx.adx_neg())

    # --- Momentum ---
    ind["rsi_14"] = _safe(ta.momentum.RSIIndicator(close, window=14).rsi())

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    ind["stoch_k"] = _safe(stoch.stoch())
    ind["stoch_d"] = _safe(stoch.stoch_signal())

    ind["cci_20"]   = _safe(ta.trend.CCIIndicator(high, low, close, window=20).cci())
    ind["willr_14"] = _safe(ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r())
    ind["mfi_14"]   = _safe(ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index())

    # --- Volatility ---
    ind["atr_14"] = _safe(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range())

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    ind["bbands_upper"] = _safe(bb.bollinger_hband())
    ind["bbands_mid"]   = _safe(bb.bollinger_mavg())
    ind["bbands_lower"] = _safe(bb.bollinger_lband())
    bbm = ind["bbands_mid"] or 1
    ind["bbands_width"] = (ind["bbands_upper"] - ind["bbands_lower"]) / bbm if bbm != 0 else 0

    kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
    ind["kc_upper"] = _safe(kc.keltner_channel_hband())
    ind["kc_lower"] = _safe(kc.keltner_channel_lband())

    # --- Volume ---
    obv_series = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    ind["obv"] = _safe(obv_series)
    if len(obv_series.dropna()) >= 5:
        last5 = obv_series.dropna().iloc[-5:].values
        slope = float(np.polyfit(np.arange(5), last5, 1)[0])
        ind["obv_slope"] = slope
    else:
        ind["obv_slope"] = 0.0

    # VWAP (approximate: cumulative (typical price × volume) / cumulative volume)
    tp = (high + low + close) / 3
    cum_tpv = (tp * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    ind["vwap"] = _safe(cum_tpv / cum_vol)

    # --- Structure ---
    ind["donchian_high"] = float(high.iloc[-20:].max()) if len(high) >= 20 else float(high.max())
    ind["donchian_low"]  = float(low.iloc[-20:].min()) if len(low) >= 20 else float(low.min())

    prev_h = float(high.iloc[-2]) if len(high) >= 2 else float(high.iloc[-1])
    prev_l = float(low.iloc[-2]) if len(low) >= 2 else float(low.iloc[-1])
    prev_c = float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])
    pivot = (prev_h + prev_l + prev_c) / 3
    ind["pivot"]   = pivot
    ind["pivot_r1"] = 2 * pivot - prev_l
    ind["pivot_s1"] = 2 * pivot - prev_h

    return ind


# ----------------------------------------------------------- signal rules

def generate_signals(indicators: dict[str, float]) -> dict[str, str]:
    signals: dict[str, str] = {}

    # RSI
    rsi = indicators.get("rsi_14", 50)
    if rsi > 70:   signals["rsi"] = "overbought"
    elif rsi < 30: signals["rsi"] = "oversold"
    else:          signals["rsi"] = "neutral"

    # MACD
    macd_val  = indicators.get("macd", 0)
    macd_sig  = indicators.get("macd_signal", 0)
    macd_hist = indicators.get("macd_hist", 0)
    if macd_val > macd_sig and macd_hist > 0:   signals["macd"] = "bullish"
    elif macd_val < macd_sig and macd_hist < 0: signals["macd"] = "bearish"
    else:                                        signals["macd"] = "neutral"

    # MA alignment
    sma20  = indicators.get("sma_20", 0)
    sma50  = indicators.get("sma_50", 0)
    sma200 = indicators.get("sma_200", 0)
    close  = indicators.get("close", 0)
    if sma20 > sma50 > sma200:   signals["ma_alignment"] = "strong_bullish"
    elif sma20 < sma50 < sma200: signals["ma_alignment"] = "strong_bearish"
    elif sma20 > sma50:          signals["ma_alignment"] = "bullish"
    elif sma20 < sma50:          signals["ma_alignment"] = "bearish"
    else:                         signals["ma_alignment"] = "mixed"

    # Bollinger Bands
    bb_upper = indicators.get("bbands_upper", 0)
    bb_lower = indicators.get("bbands_lower", 0)
    if bb_upper > 0 and close > bb_upper:   signals["bbands"] = "overbought"
    elif bb_lower > 0 and close < bb_lower: signals["bbands"] = "oversold"
    else:                                    signals["bbands"] = "neutral"

    # ADX
    adx = indicators.get("adx_14", 0)
    if adx > 25:   signals["adx"] = "trending"
    elif adx > 15: signals["adx"] = "weak_trend"
    else:          signals["adx"] = "ranging"

    # OBV slope
    obv_slope = indicators.get("obv_slope", 0)
    if obv_slope > 0:   signals["volume"] = "confirming_up"
    elif obv_slope < 0: signals["volume"] = "confirming_down"
    else:               signals["volume"] = "neutral"

    # Stochastic
    stoch_k = indicators.get("stoch_k", 50)
    stoch_d = indicators.get("stoch_d", 50)
    if stoch_k > 80 and stoch_d > 80:    signals["stoch"] = "overbought"
    elif stoch_k < 20 and stoch_d < 20:  signals["stoch"] = "oversold"
    elif stoch_k > stoch_d:              signals["stoch"] = "bullish"
    elif stoch_k < stoch_d:              signals["stoch"] = "bearish"
    else:                                 signals["stoch"] = "neutral"

    return signals


def compute_trend_bias(signals: dict[str, str]) -> str:
    bull_tags = {"bullish", "strong_bullish", "oversold", "confirming_up"}
    bear_tags = {"bearish", "strong_bearish", "overbought", "confirming_down"}
    bullish = sum(1 for v in signals.values() if v in bull_tags)
    bearish = sum(1 for v in signals.values() if v in bear_tags)
    total = len(signals) or 1
    if bullish / total > 0.6: return "bullish"
    if bearish / total > 0.6: return "bearish"
    return "neutral"


def compute_volatility_regime(indicators: dict[str, float]) -> str:
    atr   = indicators.get("atr_14", 0)
    close = indicators.get("close", 1) or 1
    atr_pct = atr / close * 100
    bb_width = indicators.get("bbands_width", 0)
    if atr_pct > 2.0 or bb_width > 0.06: return "high"
    if atr_pct < 0.3 or bb_width < 0.01: return "low"
    return "normal"


def detect_patterns(indicators: dict[str, float], signals: dict[str, str]) -> list[str]:
    patterns: list[str] = []
    if signals.get("rsi") == "oversold" and signals.get("ma_alignment") in ("bearish", "strong_bearish"):
        patterns.append("oversold_in_downtrend")
    if signals.get("rsi") == "overbought" and signals.get("ma_alignment") in ("bullish", "strong_bullish"):
        patterns.append("overbought_in_uptrend")
    bb_upper = indicators.get("bbands_upper", 0)
    bb_lower = indicators.get("bbands_lower", 0)
    kc_upper = indicators.get("kc_upper", 0)
    kc_lower = indicators.get("kc_lower", 0)
    if kc_upper > 0 and kc_lower > 0 and bb_upper < kc_upper and bb_lower > kc_lower:
        patterns.append("bollinger_squeeze")
    sma20 = indicators.get("sma_20", 0)
    sma50 = indicators.get("sma_50", 0)
    if sma20 > sma50 * 1.001:  patterns.append("golden_cross_zone")
    elif sma20 < sma50 * 0.999: patterns.append("death_cross_zone")
    return patterns


# ------------------------------------------------------------------ engine

class TechnicalEngine:
    def __init__(self) -> None:
        pass

    def compute_all(
        self, df: pl.DataFrame, instrument: str, timeframe: str
    ) -> TechnicalSnapshot:
        if df.is_empty() or len(df) < 20:
            return TechnicalSnapshot(
                instrument=instrument, timestamp=datetime.utcnow(), timeframe=timeframe,
            )
        indicators = compute_indicators(df)
        signals    = generate_signals(indicators)
        trend      = compute_trend_bias(signals)
        vol_regime = compute_volatility_regime(indicators)
        patterns   = detect_patterns(indicators, signals)
        return TechnicalSnapshot(
            instrument=instrument, timestamp=datetime.utcnow(), timeframe=timeframe,
            indicators=indicators, signals=signals,
            trend_bias=trend, volatility_regime=vol_regime, patterns=patterns,
        )

    async def multi_timeframe_analysis(
        self, instrument: str, data_fetcher, timeframes: list[str] | None = None,
    ) -> dict[str, TechnicalSnapshot]:
        import asyncio
        timeframes = timeframes or ["5m", "15m", "1h"]

        async def fetch_and_compute(tf: str) -> tuple[str, TechnicalSnapshot]:
            df = await data_fetcher.get_latest_bars(instrument, tf, count=250)
            return tf, self.compute_all(df, instrument, tf)

        results = await asyncio.gather(*[fetch_and_compute(tf) for tf in timeframes])
        return dict(results)
