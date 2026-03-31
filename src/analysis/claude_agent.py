"""Claude API agent — market analysis and trade reflection."""
from __future__ import annotations
import json
from datetime import datetime

import anthropic

from src.types import (
    MarketRegime, NewsItem, PortfolioState, TradeAction,
    TradeRecommendation, TradeRecord, TechnicalSnapshot,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert intraday trader operating an automated trading system.

ROLE: Analyze market data and provide structured trading recommendations.

INPUT YOU RECEIVE:
- Technical indicator snapshot (RSI, MACD, MA alignment, ATR, Bollinger Bands, etc.)
- Multi-timeframe analysis (5m, 15m, 1h)
- Recent news headlines with relevance scores
- Current market regime (trend/range/breakout/crisis)
- Portfolio state (open positions, daily P&L, exposure)
- Recent trade history (last 10 trades with outcomes)

OUTPUT FORMAT (strict JSON):
{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "confidence": 0-100,
  "entry_price": <float>,
  "stop_loss": <float>,
  "take_profit_1": <float>,
  "take_profit_2": <float or null>,
  "risk_reward_ratio": <float>,
  "position_size_pct": <float 0.5-5.0>,
  "reasoning": "<2-3 sentence explanation>",
  "key_factors": ["factor1", "factor2", "factor3"],
  "time_horizon": "scalp" | "intraday" | "swing",
  "regime_assessment": "<current regime and whether it favors this trade>",
  "risk_warnings": ["warning1", "warning2"]
}

RULES:
1. CAPITAL PRESERVATION IS PRIORITY #1. If uncertain, output NO_TRADE.
2. Never recommend risk:reward below 1:1.5.
3. Always specify a stop_loss — no exceptions.
4. Consider upcoming economic events.
5. If 3+ recent trades were losses, be MORE conservative (lower confidence, smaller size).
6. Confidence > 80 means "high conviction" — use sparingly.
7. Account for current portfolio exposure — don't overload one direction.
8. Output ONLY valid JSON, no markdown, no code blocks.
"""

REFLECTION_PROMPT = """You are reviewing a completed trade. Analyze what happened and extract lessons.

TRADE DETAILS:
{trade_json}

MARKET CONTEXT AT ENTRY:
{entry_context}

Provide your analysis as JSON:
{{
  "performance_grade": "A" | "B" | "C" | "D" | "F",
  "entry_timing": "excellent" | "good" | "fair" | "poor",
  "exit_timing": "excellent" | "good" | "fair" | "poor",
  "sizing_assessment": "appropriate" | "too_large" | "too_small",
  "what_worked": ["factor1", "factor2"],
  "what_failed": ["factor1", "factor2"],
  "lessons": "<1-2 sentence actionable lesson>",
  "factor_attribution": {{
    "technical": <0-1>,
    "sentiment": <0-1>,
    "regime": <0-1>,
    "timing": <0-1>
  }}
}}
Output ONLY valid JSON.
"""


class CostTracker:
    def __init__(self, monthly_budget_usd: float = 200.0) -> None:
        self.monthly_budget = monthly_budget_usd
        self.month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    @property
    def estimated_cost(self) -> float:
        # claude-sonnet-4-6 pricing (approximate per million tokens)
        input_cost = self.total_input_tokens / 1_000_000 * 3.0
        output_cost = self.total_output_tokens / 1_000_000 * 15.0
        return input_cost + output_cost

    @property
    def budget_remaining_pct(self) -> float:
        return max(0.0, (1 - self.estimated_cost / self.monthly_budget) * 100)

    def can_make_call(self, estimated_tokens: int = 3000) -> bool:
        return self.estimated_cost < self.monthly_budget * 0.95

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1


def _build_analysis_prompt(
    instrument: str,
    technical: dict[str, TechnicalSnapshot],
    news: list[NewsItem],
    regime: MarketRegime,
    portfolio: PortfolioState,
    recent_trades: list[TradeRecord],
) -> str:
    prompt = f"""INSTRUMENT: {instrument}
CURRENT TIME: {datetime.utcnow().isoformat()}

=== TECHNICAL ANALYSIS ==="""

    for tf, snap in technical.items():
        prompt += f"\n[{tf}] Trend: {snap.trend_bias} | Vol: {snap.volatility_regime}\n"
        if snap.indicators:
            key_inds = {k: round(v, 5) for k, v in snap.indicators.items()
                        if k in ("rsi_14", "macd", "macd_signal", "adx_14",
                                  "sma_20", "sma_50", "sma_200", "atr_14",
                                  "bbands_upper", "bbands_lower", "close")}
            prompt += f"  Key Indicators: {json.dumps(key_inds)}\n"
        if snap.signals:
            prompt += f"  Signals: {json.dumps(snap.signals)}\n"
        if snap.patterns:
            prompt += f"  Patterns: {', '.join(snap.patterns)}\n"

    prompt += f"""
=== MARKET REGIME ===
Current: {regime.value}

=== RECENT NEWS ({len(news)} items) ===
"""
    for n in news[:10]:
        prompt += f"- [{n.timestamp.strftime('%H:%M')}] {n.title} (relevance: {n.relevance_score:.1f})\n"

    prompt += f"""
=== PORTFOLIO STATE ===
Equity: ${portfolio.equity:,.2f}
Daily P&L: ${portfolio.daily_pnl:,.2f} ({portfolio.daily_pnl/max(portfolio.equity, 1)*100:.1f}%)
Open positions: {len(portfolio.positions)}
Portfolio heat: {portfolio.portfolio_heat_pct:.1f}%
Current drawdown: {portfolio.max_drawdown_current:.1f}%

=== RECENT TRADES (last 5) ===
"""
    for t in recent_trades[-5:]:
        sign = "+" if t.pnl > 0 else ""
        prompt += f"- {t.instrument} {t.side.value}: {sign}{t.pnl_pct:.1f}% ({t.exit_reason.value})\n"

    prompt += "\nBased on all the above, provide your trading recommendation as JSON.\n"
    return prompt


def _parse_recommendation(
    raw: str, instrument: str
) -> TradeRecommendation | None:
    try:
        # Strip potential markdown code fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)
        action_str = data.get("action", "NO_TRADE").upper()
        try:
            action = TradeAction(action_str)
        except ValueError:
            action = TradeAction.NO_TRADE

        return TradeRecommendation(
            action=action,
            instrument=instrument,
            confidence=int(data.get("confidence", 50)),
            entry_price=float(data.get("entry_price", 0)),
            stop_loss=float(data.get("stop_loss", 0)),
            take_profit_1=float(data.get("take_profit_1", 0)),
            take_profit_2=float(data["take_profit_2"]) if data.get("take_profit_2") else None,
            position_size_suggestion_pct=float(data.get("position_size_pct", 2.0)),
            risk_reward_ratio=float(data.get("risk_reward_ratio", 0)),
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            time_horizon=data.get("time_horizon", "intraday"),
            regime_assessment=data.get("regime_assessment", ""),
        )
    except Exception as exc:
        log.warning("Failed to parse Claude recommendation", error=str(exc), raw=raw[:200])
        return None


class ClaudeAgent:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = anthropic.Anthropic()
        self.model = config.get("default_model", "claude-sonnet-4-6")
        self.premium_model = config.get("premium_model", "claude-opus-4-6")
        self.escalation_threshold = config.get("escalation_threshold", 0.8)
        self.max_tokens = config.get("max_tokens_analysis", 2000)
        self.temperature = config.get("temperature", 0.3)
        self.cost_tracker = CostTracker(
            monthly_budget_usd=config.get("monthly_budget_usd", 200.0)
        )

    async def analyze_market(
        self,
        instrument: str,
        technical: dict[str, TechnicalSnapshot],
        news: list[NewsItem],
        regime: MarketRegime,
        recent_trades: list[TradeRecord],
        portfolio_state: PortfolioState,
    ) -> TradeRecommendation | None:
        if not self.cost_tracker.can_make_call():
            log.warning("Claude budget exhausted, skipping analysis")
            return None

        prompt = _build_analysis_prompt(
            instrument, technical, news, regime, portfolio_state, recent_trades
        )

        # Use premium model if portfolio heat is high
        model = (
            self.premium_model
            if portfolio_state.portfolio_heat_pct > self.escalation_threshold * 6
            else self.model
        )

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            )

            self.cost_tracker.record_usage(
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            raw = response.content[0].text
            rec = _parse_recommendation(raw, instrument)

            if rec:
                log.info(
                    "Claude analysis complete",
                    instrument=instrument,
                    action=rec.action.value,
                    confidence=rec.confidence,
                    cost_usd=f"{self.cost_tracker.estimated_cost:.4f}",
                )
            return rec

        except Exception as exc:
            log.error("Claude API error", error=str(exc))
            return None

    async def reflect_on_trade(
        self, trade: TradeRecord, entry_context: str = ""
    ) -> dict:
        if not self.cost_tracker.can_make_call():
            return {}

        import dataclasses
        trade_json = json.dumps(dataclasses.asdict(trade), default=str)
        prompt = REFLECTION_PROMPT.format(
            trade_json=trade_json,
            entry_context=entry_context,
        )

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=self.config.get("max_tokens_reflection", 1500),
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
            )

            self.cost_tracker.record_usage(
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            return json.loads(raw)

        except Exception as exc:
            log.warning("Claude reflection failed", error=str(exc))
            return {}
