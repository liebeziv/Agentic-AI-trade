"""Market regime classifier — rule-based with feature engineering."""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.stats

from src.types import MarketRegime, TechnicalSnapshot
from src.utils.logger import get_logger

log = get_logger(__name__)


def _compute_atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 0.0
    tr = np.maximum(high[1:] - low[1:], np.maximum(
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ))
    atr = np.mean(tr[-period:])
    return float(atr / close[-1] * 100) if close[-1] != 0 else 0.0


def _compute_ma_alignment_score(close: np.ndarray) -> float:
    """Normalized MA alignment: +1 = fully bullish, -1 = fully bearish."""
    if len(close) < 200:
        return 0.0
    sma20 = np.mean(close[-20:])
    sma50 = np.mean(close[-50:])
    sma200 = np.mean(close[-200:])
    price = close[-1]

    # Count bullish factors (0-5), map linearly to -1..+1
    bullish = sum([
        price > sma20,
        price > sma50,
        price > sma200,
        sma20 > sma50,
        sma50 > sma200,
    ])
    return (bullish - 2.5) / 2.5


def _compute_consecutive_direction(returns: np.ndarray) -> int:
    """Count consecutive positive or negative returns from the end."""
    if len(returns) == 0:
        return 0
    sign = 1 if returns[-1] > 0 else -1
    count = 0
    for r in reversed(returns):
        if r * sign > 0:
            count += 1
        else:
            break
    return sign * count


def compute_regime_features(bars: pl.DataFrame) -> dict[str, float]:
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    volume = bars["volume"].to_numpy() if "volume" in bars.columns else np.ones(len(close))

    if len(close) < 30:
        return {}

    returns = np.diff(np.log(close + 1e-10))

    # ATR as % of price
    atr_pct = _compute_atr_pct(high, low, close)

    # ADX approximation
    period = 14
    if len(close) > period + 1:
        tr = np.maximum(high[1:] - low[1:], np.maximum(
            np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])
        ))
        atr = np.mean(tr[-period:])
        dx_vals = []
        for i in range(1, min(period, len(close) - 1)):
            dm_plus = max(high[-i] - high[-i-1], 0) if high[-i] - high[-i-1] > low[-i-1] - low[-i] else 0
            dm_minus = max(low[-i-1] - low[-i], 0) if low[-i-1] - low[-i] > high[-i] - high[-i-1] else 0
            if atr > 0:
                di_plus = dm_plus / atr * 100
                di_minus = dm_minus / atr * 100
                if di_plus + di_minus > 0:
                    dx_vals.append(abs(di_plus - di_minus) / (di_plus + di_minus) * 100)
        adx = float(np.mean(dx_vals)) if dx_vals else 20.0
    else:
        adx = 20.0

    # Bollinger Band width
    if len(close) >= 20:
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        bb_width = (2 * 2 * std20) / ma20 if ma20 > 0 else 0.0
    else:
        bb_width = 0.0

    # Volume ratio (current vs 20-bar avg)
    vol_ratio = float(volume[-1] / np.mean(volume[-20:])) if len(volume) >= 20 and np.mean(volume[-20:]) > 0 else 1.0

    # Return skew / kurtosis
    sample = returns[-20:] if len(returns) >= 20 else returns
    skew = float(scipy.stats.skew(sample)) if len(sample) >= 4 else 0.0
    kurt = float(scipy.stats.kurtosis(sample)) if len(sample) >= 4 else 0.0

    return {
        "atr_pct": atr_pct,
        "adx": adx,
        "bb_width": bb_width,
        "volume_ratio": vol_ratio,
        "return_skew": skew,
        "return_kurt": kurt,
        "ma_alignment_score": _compute_ma_alignment_score(close),
        "consecutive_direction": float(_compute_consecutive_direction(returns)),
    }


def classify_regime(features: dict[str, float]) -> MarketRegime:
    if not features:
        return MarketRegime.RANGE_BOUND

    atr_pct = features.get("atr_pct", 0)
    adx = features.get("adx", 20)
    bb_width = features.get("bb_width", 0.03)
    vol_ratio = features.get("volume_ratio", 1.0)
    skew = features.get("return_skew", 0)
    ma_score = features.get("ma_alignment_score", 0)

    # 1. Crisis — high volatility + extreme skew
    if atr_pct > 3.0 and abs(skew) > 2.0:
        return MarketRegime.CRISIS

    # 2. Breakout — tight bands + volume surge
    if bb_width < 0.02 and vol_ratio > 2.0:
        return MarketRegime.BREAKOUT

    # 3. Strong trend (ADX-driven)
    if adx > 25:
        if ma_score > 0.3:
            return MarketRegime.STRONG_TREND_UP
        elif ma_score < -0.3:
            return MarketRegime.STRONG_TREND_DOWN

    # 4. Weak trend
    if adx > 15:
        return MarketRegime.WEAK_TREND

    # 5. Default: range
    return MarketRegime.RANGE_BOUND


class RegimeDetector:
    def classify(self, bars: pl.DataFrame | None) -> MarketRegime:
        if bars is None or bars.is_empty():
            return MarketRegime.RANGE_BOUND
        features = compute_regime_features(bars)
        regime = classify_regime(features)
        log.debug("Regime classified", regime=regime.value, features=features)
        return regime

    def classify_from_snapshot(self, snap: TechnicalSnapshot) -> MarketRegime:
        """Classify from an already-computed TechnicalSnapshot."""
        adx = snap.indicators.get("adx_14", 20)
        bb_width = snap.indicators.get("bbands_width", 0.03)
        atr = snap.indicators.get("atr_14", 0)
        close = snap.indicators.get("close", 1) or 1
        atr_pct = atr / close * 100

        sma20 = snap.indicators.get("sma_20", 0)
        sma50 = snap.indicators.get("sma_50", 0)
        sma200 = snap.indicators.get("sma_200", 0)

        ma_score = 0.0
        if close > sma20 > sma50 > sma200:
            ma_score = 0.8
        elif close < sma20 < sma50 < sma200:
            ma_score = -0.8

        features = {
            "atr_pct": atr_pct,
            "adx": adx,
            "bb_width": bb_width,
            "volume_ratio": 1.0,
            "return_skew": 0.0,
            "ma_alignment_score": ma_score,
        }
        return classify_regime(features)
