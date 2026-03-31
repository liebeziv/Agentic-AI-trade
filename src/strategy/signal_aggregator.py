"""Multi-factor signal aggregator."""
from __future__ import annotations

from src.types import MarketRegime, SignalScore, TechnicalSnapshot, TradeAction
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_WEIGHTS = {
    "technical": 0.35,
    "sentiment": 0.25,
    "regime": 0.20,
    "claude_confidence": 0.20,
    "thresholds": {
        "strong_buy": 0.55,
        "buy": 0.25,
        "sell": -0.25,
        "strong_sell": -0.55,
    },
}

SIGNAL_VALUES: dict[str, float] = {
    "strong_bullish": 1.0,
    "bullish": 0.6,
    "oversold": 0.5,
    "confirming_up": 0.4,
    "trending": 0.2,
    "weak_trend": 0.1,
    "neutral": 0.0,
    "mixed": 0.0,
    "ranging": -0.1,
    "weak_trend_down": -0.1,
    "confirming_down": -0.4,
    "overbought": -0.5,
    "bearish": -0.6,
    "strong_bearish": -1.0,
}

SIGNAL_WEIGHTS = {
    "rsi": 0.15,
    "macd": 0.20,
    "ma_alignment": 0.25,
    "bbands": 0.10,
    "adx": 0.10,
    "volume": 0.10,
    "stoch": 0.10,
}

# Directional regime bias: positive = bullish, negative = bearish
REGIME_DIRECTION: dict[MarketRegime, float] = {
    MarketRegime.STRONG_TREND_UP:   +0.8,
    MarketRegime.STRONG_TREND_DOWN: -0.8,
    MarketRegime.WEAK_TREND:        +0.2,   # slight bullish lean
    MarketRegime.RANGE_BOUND:        0.0,
    MarketRegime.BREAKOUT:          +0.3,   # momentum bias; direction unknown → small positive
    MarketRegime.CRISIS:            -0.7,   # risk-off
}


def compute_technical_score(snapshot: TechnicalSnapshot) -> float:
    """Convert technical signals to -1..+1 score."""
    score = 0.0
    for signal_name, weight in SIGNAL_WEIGHTS.items():
        val_str = snapshot.signals.get(signal_name, "neutral")
        score += SIGNAL_VALUES.get(val_str, 0.0) * weight
    return max(-1.0, min(1.0, score))


def compute_regime_score(regime: MarketRegime, action: str = "") -> float:
    """Directional regime bias (-1..+1). Positive = regime favours longs."""
    return REGIME_DIRECTION.get(regime, 0.0)


def compute_composite(
    technical_score: float,
    sentiment_score: float,
    regime_score: float,
    claude_confidence: float,
    weights: dict | None = None,
) -> tuple[float, TradeAction]:
    w = weights or DEFAULT_WEIGHTS
    composite = (
        technical_score * w["technical"]
        + sentiment_score * w["sentiment"]
        + regime_score * w["regime"]
        + (claude_confidence - 0.5) * 2 * w["claude_confidence"]
    )
    composite = max(-1.0, min(1.0, composite))

    thresholds = w["thresholds"]
    if composite >= thresholds["strong_buy"]:
        action = TradeAction.STRONG_BUY
    elif composite >= thresholds["buy"]:
        action = TradeAction.BUY
    elif composite <= thresholds["strong_sell"]:
        action = TradeAction.STRONG_SELL
    elif composite <= thresholds["sell"]:
        action = TradeAction.SELL
    else:
        action = TradeAction.NEUTRAL

    return composite, action


class SignalAggregator:
    def __init__(self, weights: dict | None = None) -> None:
        self.weights = weights or DEFAULT_WEIGHTS

    def compute_technical_score(self, snapshot: TechnicalSnapshot) -> float:
        return compute_technical_score(snapshot)

    def compute_regime_score(self, regime: MarketRegime, action: str) -> float:
        return compute_regime_score(regime, action)

    def compute_composite(
        self,
        technical_score: float,
        sentiment_score: float,
        regime_score: float,
        claude_confidence: float,
    ) -> tuple[float, TradeAction]:
        return compute_composite(
            technical_score, sentiment_score, regime_score, claude_confidence, self.weights
        )

    def build_signal(
        self,
        instrument: str,
        technical_snap: TechnicalSnapshot,
        regime: MarketRegime,
        claude_confidence: float,
        sentiment_score: float,
        recommendation=None,
    ) -> SignalScore:
        from datetime import datetime

        tech_score = self.compute_technical_score(technical_snap)
        action_str = recommendation.action.value if recommendation else "NO_TRADE"
        reg_score = self.compute_regime_score(regime, action_str)
        composite, action = self.compute_composite(
            tech_score, sentiment_score, reg_score, claude_confidence
        )

        log.debug(
            "Signal computed",
            instrument=instrument,
            tech=round(tech_score, 3),
            sentiment=round(sentiment_score, 3),
            regime=round(reg_score, 3),
            claude_conf=round(claude_confidence, 3),
            composite=round(composite, 3),
            action=action.value,
        )

        return SignalScore(
            instrument=instrument,
            timestamp=datetime.utcnow(),
            technical_score=tech_score,
            sentiment_score=sentiment_score,
            regime_score=reg_score,
            claude_confidence=claude_confidence,
            composite_score=composite,
            action=action,
            recommendation=recommendation,
        )
