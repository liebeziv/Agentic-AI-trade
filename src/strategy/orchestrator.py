"""Multi-agent orchestrator — capital allocation, conflict resolution, debate."""
from __future__ import annotations

from datetime import datetime

from src.strategy.strategies.base_agent import BaseStrategyAgent
from src.types import MarketRegime, SignalScore, TradeAction
from src.utils.logger import get_logger

log = get_logger(__name__)

# Minimum fitness score for an agent to receive capital
MIN_FITNESS_THRESHOLD = 0.2
# Maximum fraction of total capital any single agent can hold
MAX_AGENT_CAPITAL_FRACTION = 0.40


class Orchestrator:
    """
    Coordinates multiple strategy agents:
    - Allocates capital proportional to rolling fitness
    - Resolves conflicting signals (same instrument, different directions)
    - Optionally runs a Claude-powered debate for high-stakes conflicts
    """

    def __init__(self, agents: list[BaseStrategyAgent], total_capital: float,
                 claude_agent=None) -> None:
        self.agents = agents
        self.total_capital = total_capital
        self.claude = claude_agent
        self._last_allocation: dict[str, float] = {}

    # --------------------------------------------------------- Capital

    async def allocate_capital(self, regime: MarketRegime) -> dict[str, float]:
        """
        Distribute capital among agents weighted by fitness score.
        Returns agent_id → allocated USD amount.
        """
        fitnesses: dict[str, float] = {}
        for agent in self.agents:
            if not agent.is_active:
                fitnesses[agent.agent_id] = 0.0
                continue
            fit = await agent.evaluate_fitness(regime)
            fitnesses[agent.agent_id] = fit if fit >= MIN_FITNESS_THRESHOLD else 0.0

        total_fitness = sum(fitnesses.values())
        if total_fitness == 0:
            # Equal weight fallback
            n_active = sum(1 for f in fitnesses.values() if f > 0) or len(self.agents)
            allocation = {a.agent_id: self.total_capital / n_active for a in self.agents}
        else:
            allocation = {
                aid: min(
                    self.total_capital * fit / total_fitness,
                    self.total_capital * MAX_AGENT_CAPITAL_FRACTION,
                )
                for aid, fit in fitnesses.items()
            }

        # Update agents
        for agent in self.agents:
            agent.allocated_capital = allocation.get(agent.agent_id, 0.0)

        self._last_allocation = allocation
        log.info("Capital allocated", regime=regime.value,
                 allocation={k: f"${v:,.0f}" for k, v in allocation.items()})
        return allocation

    # ------------------------------------------------------ Conflict resolution

    def resolve_conflicts(self, all_signals: list[SignalScore]) -> list[SignalScore]:
        """
        When multiple agents signal the same instrument in opposite directions,
        apply conflict resolution rules and return the reconciled signal list.
        """
        by_instrument: dict[str, list[SignalScore]] = {}
        for sig in all_signals:
            by_instrument.setdefault(sig.instrument, []).append(sig)

        resolved: list[SignalScore] = []
        for instrument, sigs in by_instrument.items():
            if len(sigs) == 1:
                resolved.append(sigs[0])
                continue

            buys = [s for s in sigs if s.action in (TradeAction.BUY, TradeAction.STRONG_BUY)]
            sells = [s for s in sigs if s.action in (TradeAction.SELL, TradeAction.STRONG_SELL)]

            # No conflict
            if not buys or not sells:
                # Use highest-composite signal
                best = max(sigs, key=lambda s: abs(s.composite_score))
                resolved.append(best)
                continue

            # Conflict — compare composite magnitudes
            buy_score = max(s.composite_score for s in buys)
            sell_score = min(s.composite_score for s in sells)  # most negative

            margin = abs(buy_score - abs(sell_score))
            if margin < 0.15:
                # Too close to call → skip (NEUTRAL)
                log.info("Signal conflict unresolved — skipping", instrument=instrument,
                         buy=round(buy_score, 3), sell=round(sell_score, 3))
                continue

            # Use the stronger signal
            if buy_score > abs(sell_score):
                winner = max(buys, key=lambda s: s.composite_score)
            else:
                winner = min(sells, key=lambda s: s.composite_score)

            log.info("Signal conflict resolved", instrument=instrument,
                     winner=winner.action.value, margin=round(margin, 3))
            resolved.append(winner)

        return resolved

    # -------------------------------------------------- Claude debate

    async def claude_debate(
        self, instrument: str, signals: list[SignalScore], regime: MarketRegime
    ) -> SignalScore | None:
        """
        For high-stakes conflicting signals, ask Claude to arbitrate.
        Returns the winning signal or None (skip trade).
        """
        if not self.claude:
            return None

        prompt = (
            f"You are arbitrating a trading signal conflict for {instrument}.\n"
            f"Market regime: {regime.value}\n\n"
            f"Agent signals:\n"
        )
        for sig in signals:
            prompt += (
                f"- Action: {sig.action.value}, Composite: {sig.composite_score:.3f}, "
                f"Tech: {sig.technical_score:.2f}, Regime: {sig.regime_score:.2f}\n"
            )
        prompt += (
            "\nWhich action is more appropriate given the regime and signal strengths? "
            "Reply with ONLY one of: BUY, SELL, NEUTRAL"
        )

        try:
            # Use claude agent's raw call if available
            import anthropic, os
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = response.content[0].text.strip().upper()
            log.info("Claude debate verdict", instrument=instrument, verdict=verdict)

            if "BUY" in verdict:
                return max(signals, key=lambda s: s.composite_score)
            elif "SELL" in verdict:
                return min(signals, key=lambda s: s.composite_score)
            return None
        except Exception as exc:
            log.warning("Claude debate failed", error=str(exc))
            return None

    # -------------------------------------------------- Aggregate signals

    async def aggregate(
        self, regime: MarketRegime
    ) -> tuple[dict[str, float], list[SignalScore]]:
        """
        Collect signals from all active agents, allocate capital, resolve conflicts.
        Returns (allocation, resolved_signals).
        """
        allocation = await self.allocate_capital(regime)
        all_signals: list[SignalScore] = []

        for agent in self.agents:
            if not agent.is_active or allocation.get(agent.agent_id, 0) <= 0:
                continue
            # Signals are generated by the caller (run_multiagent.py) and passed in
            # This method is called after signals are collected
        # (signal generation happens in run_multiagent.py)

        resolved = self.resolve_conflicts(all_signals)
        return allocation, resolved

    def agent_status(self) -> list[dict]:
        return [a.status_dict() for a in self.agents]
