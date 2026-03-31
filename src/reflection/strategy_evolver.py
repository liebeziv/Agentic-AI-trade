"""Strategy evolution loop — weekly Claude-powered parameter review."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from src.utils.logger import get_logger

log = get_logger(__name__)

# Minimum Sharpe improvement to apply suggested params (10%)
IMPROVEMENT_THRESHOLD = 0.10
# Minimum trades in review window before evolution runs
MIN_TRADES_FOR_REVIEW = 5


class StrategyEvolver:
    """
    Weekly strategy review cycle.

    For each agent:
    1. Fetches last 7 days of trades from the store.
    2. Asks Claude to propose parameter improvements.
    3. If Claude suggests changes and Sharpe improves ≥ 10%, applies them.
    4. Deactivates agents with Sharpe < -0.5 and > 20 trades.
    5. Sends a Telegram summary via notifier.
    """

    def __init__(self, agents: list, claude, store, notifier=None) -> None:
        self.agents = agents
        self.claude = claude
        self.store = store
        self.notifier = notifier

    # ------------------------------------------------------------------

    async def weekly_review(self) -> dict:
        """Run weekly strategy evolution. Returns report dict."""
        report: dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": {},
        }

        for agent in self.agents:
            agent_id = agent.agent_id
            start = datetime.utcnow() - timedelta(days=7)

            # Fetch recent trades
            try:
                trades = (
                    self.store.get_trades(limit=200)
                    if self.store else []
                )
                # Filter to this agent's trades if agent_id attribute exists
                trades = [
                    t for t in trades
                    if getattr(t, "agent_id", agent_id) == agent_id
                    and t.open_time >= start
                ]
            except Exception as exc:
                log.warning("Evolver: failed to fetch trades",
                            agent=agent_id, error=str(exc))
                trades = []

            if len(trades) < MIN_TRADES_FOR_REVIEW:
                report["agents"][agent_id] = {
                    "status": "insufficient_trades",
                    "trade_count": len(trades),
                }
                log.info("Evolver: skipping agent — insufficient trades",
                         agent=agent_id, trades=len(trades))
                continue

            # Ask Claude to review
            analysis = await self._claude_review(agent, trades)
            current_sharpe = agent.performance.rolling_sharpe_30d

            if not analysis.get("suggested_params"):
                report["agents"][agent_id] = {
                    "status": "no_changes_suggested",
                    "current_sharpe": current_sharpe,
                    "claude_reasoning": analysis.get("reasoning", ""),
                }
                continue

            # Simple forward check: does new sharpe beat old?
            new_sharpe = analysis.get("estimated_sharpe", current_sharpe)
            improvement = (new_sharpe - current_sharpe) / max(abs(current_sharpe), 0.01)

            if improvement >= IMPROVEMENT_THRESHOLD:
                old_params = agent.get_parameters()
                agent.set_parameters(analysis["suggested_params"])
                log.info("Evolver: parameters updated",
                         agent=agent_id,
                         old_sharpe=round(current_sharpe, 3),
                         new_sharpe=round(new_sharpe, 3),
                         changes=analysis["suggested_params"])
                report["agents"][agent_id] = {
                    "status": "parameters_updated",
                    "old_sharpe": current_sharpe,
                    "estimated_new_sharpe": new_sharpe,
                    "changes": analysis["suggested_params"],
                    "old_params": old_params,
                    "reasoning": analysis.get("reasoning", ""),
                }
            else:
                report["agents"][agent_id] = {
                    "status": "no_improvement",
                    "current_sharpe": current_sharpe,
                    "estimated_sharpe": new_sharpe,
                    "reasoning": analysis.get("reasoning", ""),
                }

            # Deactivate chronic underperformers
            if (current_sharpe < -0.5
                    and agent.performance.total_trades > 20
                    and agent.is_active):
                agent.is_active = False
                log.warning("Evolver: deactivated underperformer", agent=agent_id,
                            sharpe=round(current_sharpe, 3))
                report["agents"][agent_id]["deactivated"] = True

        # Send summary
        if self.notifier:
            await self._send_report(report)

        return report

    # ------------------------------------------------------------------

    async def _claude_review(self, agent, trades: list) -> dict:
        """Ask Claude to analyse recent trades and suggest parameter changes."""
        if not self.claude:
            return {}

        wins = [t for t in trades if getattr(t, "pnl", 0) > 0]
        losses = [t for t in trades if getattr(t, "pnl", 0) <= 0]
        total_pnl = sum(getattr(t, "pnl", 0) for t in trades)
        win_rate = len(wins) / len(trades) if trades else 0

        prompt = (
            f"You are a senior quant reviewing the trading strategy '{agent.name}'.\n\n"
            f"Strategy description: {agent.description}\n"
            f"Current parameters: {json.dumps(agent.get_parameters(), indent=2)}\n\n"
            f"Last 7 days performance:\n"
            f"  Trades: {len(trades)}\n"
            f"  Win rate: {win_rate:.0%}\n"
            f"  Total P&L: ${total_pnl:,.2f}\n"
            f"  Sharpe 30d: {agent.performance.rolling_sharpe_30d:.2f}\n"
            f"  Max DD: {agent.performance.max_drawdown_pct:.1f}%\n\n"
            f"Common losing patterns:\n"
        )
        for t in losses[:5]:
            lessons = getattr(t, "lessons", [])
            if lessons:
                prompt += f"  - {'; '.join(lessons[:2])}\n"

        prompt += (
            "\nAnalyse the performance and suggest parameter improvements if warranted.\n"
            "Respond ONLY with valid JSON in this exact format:\n"
            "{\n"
            '  "suggested_params": {},\n'
            '  "estimated_sharpe": <float>,\n'
            '  "reasoning": "<string>"\n'
            "}\n"
            "If no changes are needed, set suggested_params to {} and explain why.\n"
            "Only suggest parameters that exist in the current parameter set."
        )

        try:
            import anthropic
            import os
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as exc:
            log.warning("Evolver: Claude review failed",
                        agent=agent.agent_id, error=str(exc))
            return {}

    async def _send_report(self, report: dict) -> None:
        """Send weekly evolution summary via notifier."""
        lines = ["📊 *Weekly Strategy Evolution Report*"]
        for agent_id, result in report.get("agents", {}).items():
            status = result.get("status", "unknown")
            if status == "parameters_updated":
                lines.append(
                    f"✅ {agent_id}: params updated "
                    f"(Sharpe {result.get('old_sharpe', 0):.2f} → "
                    f"{result.get('estimated_new_sharpe', 0):.2f})"
                )
            elif status == "no_improvement":
                lines.append(f"➖ {agent_id}: no improvement")
            elif status == "insufficient_trades":
                lines.append(f"⏳ {agent_id}: insufficient trades")
            else:
                lines.append(f"ℹ️ {agent_id}: {status}")
            if result.get("deactivated"):
                lines.append(f"🔴 {agent_id}: DEACTIVATED (chronic underperformance)")

        try:
            await self.notifier.send("\n".join(lines))
        except Exception as exc:
            log.warning("Evolver: notifier send failed", error=str(exc))
