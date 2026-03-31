"""Event-driven strategy agent — trades news events using Claude API."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from src.strategy.strategies.base_agent import BaseStrategyAgent
from src.strategy.signal_aggregator import compute_composite, compute_regime_score
from src.types import (
    MarketRegime, NewsItem, SignalScore, TechnicalSnapshot, TradeAction,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_PARAMS = {
    "min_news_relevance": 0.7,
    "min_claude_confidence": 0.60,   # 0.0–1.0 scale
    "max_news_age_hours": 2,
    "atr_stop_multiple": 2.5,
    "atr_tp_multiple": 4.0,
    "require_technical_alignment": True,
}


class EventDrivenAgent(BaseStrategyAgent):
    """News/event specialist. Uses Claude to assess sentiment-driven moves."""

    def __init__(self, agent_id: str = "event_driven", **kwargs) -> None:
        super().__init__(agent_id, **kwargs)
        self._params = dict(_DEFAULT_PARAMS)

    @property
    def name(self) -> str:
        return "Event Driven"

    @property
    def description(self) -> str:
        return "Claude-powered news analysis: trades high-relevance events with technical confirmation"

    @property
    def preferred_regimes(self) -> list[MarketRegime]:
        return [MarketRegime.BREAKOUT, MarketRegime.STRONG_TREND_UP,
                MarketRegime.STRONG_TREND_DOWN]

    @property
    def target_instruments(self) -> list[str]:
        cfg = self.config.get("instruments", {})
        return (cfg.get("us_stocks", []) + cfg.get("crypto", []) +
                cfg.get("futures", []))

    def get_parameters(self) -> dict:
        return dict(self._params)

    def set_parameters(self, params: dict) -> None:
        self._params.update(params)

    async def generate_signals(
        self,
        market_data: dict[str, pl.DataFrame],
        technical: dict[str, TechnicalSnapshot],
        news: list[NewsItem],
        regime: MarketRegime,
    ) -> list[SignalScore]:
        """Generate event-driven signals from high-relevance news."""
        if not news:
            return []

        # Filter: recent and relevant
        now = datetime.now(tz=timezone.utc)
        max_age_s = self._params["max_news_age_hours"] * 3600
        relevant = [
            n for n in news
            if getattr(n, "relevance_score", 0.0) >= self._params["min_news_relevance"]
            and (now - n.timestamp.replace(tzinfo=timezone.utc)
                 if n.timestamp.tzinfo is None
                 else now - n.timestamp).total_seconds() < max_age_s
        ]
        if not relevant:
            return []

        # Group news by instrument
        instrument_news: dict[str, list[NewsItem]] = {}
        for item in relevant:
            for inst in getattr(item, "instruments", []):
                instrument_news.setdefault(inst, []).append(item)

        signals: list[SignalScore] = []

        for instrument, inst_news in instrument_news.items():
            snap = technical.get(instrument)
            if not snap:
                continue

            # Use Claude to analyse — fall back to aggregate sentiment if no Claude
            recommendation = None
            if self.claude:
                try:
                    recommendation = await self.claude.analyze_market(
                        instrument=instrument,
                        technical={"15m": snap},
                        news=inst_news,
                        regime=regime,
                        recent_trades=self.store.get_trades(limit=5) if self.store else [],
                        portfolio_state=None,
                    )
                except Exception as exc:
                    log.warning("EventDriven: Claude call failed",
                                instrument=instrument, error=str(exc))

            if recommendation is None:
                # Fallback: aggregate raw sentiment scores
                avg_sentiment = sum(
                    getattr(n, "sentiment_score", 0.0) for n in inst_news
                ) / len(inst_news)
                if abs(avg_sentiment) < 0.3:
                    continue
                action = (TradeAction.BUY if avg_sentiment > 0 else TradeAction.SELL)
                sentiment_score = avg_sentiment
                claude_confidence = 0.50
            else:
                action = recommendation.action
                if action in (TradeAction.NO_TRADE, TradeAction.NEUTRAL):
                    continue
                claude_confidence = recommendation.confidence / 100.0
                if claude_confidence < self._params["min_claude_confidence"]:
                    continue
                sentiment_score = claude_confidence * (
                    1.0 if action in (TradeAction.BUY, TradeAction.STRONG_BUY) else -1.0
                )

            # Optional: require technical alignment
            if self._params["require_technical_alignment"]:
                trend_bias = getattr(snap, "trend_bias", "neutral")
                if action in (TradeAction.BUY, TradeAction.STRONG_BUY) and trend_bias == "bearish":
                    log.debug("EventDriven: skipping BUY — bearish trend", instrument=instrument)
                    continue
                if action in (TradeAction.SELL, TradeAction.STRONG_SELL) and trend_bias == "bullish":
                    log.debug("EventDriven: skipping SELL — bullish trend", instrument=instrument)
                    continue

            regime_score = compute_regime_score(regime)
            # Technical score 0 (we're driven by news, not technicals here)
            composite, resolved_action = compute_composite(
                0.0, sentiment_score, regime_score, claude_confidence
            )

            sig = SignalScore(
                instrument=instrument,
                timestamp=datetime.utcnow(),
                technical_score=0.0,
                sentiment_score=sentiment_score,
                regime_score=regime_score,
                claude_confidence=claude_confidence,
                composite_score=composite,
                action=resolved_action,
            )
            signals.append(sig)
            log.info("EventDriven signal",
                     instrument=instrument, action=resolved_action.value,
                     news_count=len(inst_news), composite=round(composite, 3))

        return signals
