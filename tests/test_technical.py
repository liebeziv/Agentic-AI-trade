"""Tests for the technical indicator engine."""
import pytest
from datetime import datetime, timedelta
import polars as pl
import numpy as np

from src.analysis.technical import (
    TechnicalEngine, compute_indicators, generate_signals,
    compute_trend_bias, compute_volatility_regime,
)
from src.types import TechnicalSnapshot


def make_bars(n: int = 250, trend: str = "up") -> pl.DataFrame:
    np.random.seed(42)
    close = [100.0]
    for _ in range(n - 1):
        if trend == "up":
            close.append(close[-1] * (1 + np.random.normal(0.001, 0.01)))
        elif trend == "down":
            close.append(close[-1] * (1 + np.random.normal(-0.001, 0.01)))
        else:
            close.append(close[-1] * (1 + np.random.normal(0, 0.005)))

    now = datetime.utcnow()
    timestamps = [now - timedelta(minutes=(n - i) * 15) for i in range(n)]

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": [c * 0.999 for c in close],
        "high": [c * 1.005 for c in close],
        "low": [c * 0.995 for c in close],
        "close": close,
        "volume": np.random.randint(1000, 10000, n).tolist(),
        "instrument": ["TEST"] * n,
        "timeframe": ["15m"] * n,
        "source": ["test"] * n,
    })


def test_compute_indicators_returns_dict():
    df = make_bars(250)
    result = compute_indicators(df)
    assert isinstance(result, dict)
    assert "rsi_14" in result
    assert "macd" in result
    assert "sma_20" in result
    assert "atr_14" in result


def test_compute_indicators_values_in_range():
    df = make_bars(250)
    result = compute_indicators(df)
    rsi = result.get("rsi_14", 50)
    assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"


def test_generate_signals_returns_dict():
    df = make_bars(250)
    indicators = compute_indicators(df)
    signals = generate_signals(indicators)
    assert isinstance(signals, dict)
    assert "rsi" in signals
    assert "macd" in signals
    assert "ma_alignment" in signals


def test_trend_bias_bullish():
    signals = {
        "rsi": "oversold", "macd": "bullish", "ma_alignment": "strong_bullish",
        "bbands": "oversold", "adx": "trending", "volume": "confirming_up",
        "stoch": "oversold",
    }
    assert compute_trend_bias(signals) == "bullish"


def test_trend_bias_bearish():
    signals = {
        "rsi": "overbought", "macd": "bearish", "ma_alignment": "strong_bearish",
        "bbands": "overbought", "adx": "trending", "volume": "confirming_down",
        "stoch": "overbought",
    }
    assert compute_trend_bias(signals) == "bearish"


def test_technical_engine_compute_all():
    engine = TechnicalEngine()
    df = make_bars(250)
    snap = engine.compute_all(df, "TEST", "15m")
    assert isinstance(snap, TechnicalSnapshot)
    assert snap.instrument == "TEST"
    assert snap.trend_bias in ("bullish", "bearish", "neutral")
    assert snap.volatility_regime in ("low", "normal", "high")


def test_technical_engine_insufficient_data():
    engine = TechnicalEngine()
    df = make_bars(10)  # Too few bars
    snap = engine.compute_all(df, "TEST", "15m")
    assert isinstance(snap, TechnicalSnapshot)
    assert snap.indicators == {}
