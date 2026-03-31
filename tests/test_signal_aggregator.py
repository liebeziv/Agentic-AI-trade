"""Tests for the signal aggregator."""
import pytest
from datetime import datetime

from src.strategy.signal_aggregator import (
    SignalAggregator, compute_technical_score, compute_composite,
    compute_regime_score,
)
from src.types import MarketRegime, TechnicalSnapshot, TradeAction


def make_snap(signals: dict) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        instrument="TEST", timestamp=datetime.utcnow(),
        timeframe="15m", signals=signals,
    )


def test_technical_score_bullish():
    snap = make_snap({
        "rsi": "oversold", "macd": "bullish", "ma_alignment": "strong_bullish",
        "bbands": "neutral", "adx": "trending", "volume": "confirming_up",
        "stoch": "bullish",
    })
    score = compute_technical_score(snap)
    assert score > 0.3, f"Expected bullish score > 0.3, got {score}"


def test_technical_score_bearish():
    snap = make_snap({
        "rsi": "overbought", "macd": "bearish", "ma_alignment": "strong_bearish",
        "bbands": "overbought", "adx": "trending", "volume": "confirming_down",
        "stoch": "bearish",
    })
    score = compute_technical_score(snap)
    assert score < -0.3, f"Expected bearish score < -0.3, got {score}"


def test_technical_score_neutral():
    snap = make_snap({
        "rsi": "neutral", "macd": "neutral", "ma_alignment": "mixed",
        "bbands": "neutral", "adx": "ranging", "volume": "neutral",
        "stoch": "neutral",
    })
    score = compute_technical_score(snap)
    assert -0.2 <= score <= 0.2, f"Expected neutral score, got {score}"


def test_composite_strong_buy():
    composite, action = compute_composite(0.8, 0.7, 0.8, 0.9)
    assert action in (TradeAction.BUY, TradeAction.STRONG_BUY)


def test_composite_strong_sell():
    composite, action = compute_composite(-0.8, -0.7, -0.8, 0.1)
    assert action in (TradeAction.SELL, TradeAction.STRONG_SELL)


def test_composite_neutral():
    composite, action = compute_composite(0.0, 0.0, 0.0, 0.5)
    assert action == TradeAction.NEUTRAL


def test_regime_score_crisis():
    score = compute_regime_score(MarketRegime.CRISIS)
    assert score < 0, "Crisis regime should have bearish bias"


def test_regime_score_trend_up():
    score = compute_regime_score(MarketRegime.STRONG_TREND_UP)
    assert score > 0.5, "Strong uptrend should have strong bullish bias"


def test_regime_score_trend_down():
    score = compute_regime_score(MarketRegime.STRONG_TREND_DOWN)
    assert score < -0.5, "Strong downtrend should have strong bearish bias"


def test_aggregator_build_signal():
    agg = SignalAggregator()
    snap = make_snap({
        "rsi": "oversold", "macd": "bullish", "ma_alignment": "bullish",
        "bbands": "neutral", "adx": "trending", "volume": "confirming_up",
        "stoch": "bullish",
    })
    signal = agg.build_signal(
        instrument="TEST", technical_snap=snap,
        regime=MarketRegime.STRONG_TREND_UP,
        claude_confidence=0.75, sentiment_score=0.5,
    )
    assert signal.instrument == "TEST"
    assert signal.action != TradeAction.NO_TRADE
    assert -1.0 <= signal.composite_score <= 1.0
