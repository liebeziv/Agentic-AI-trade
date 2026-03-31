"""Global risk manager — portfolio-level risk checks across all agents."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.types import PortfolioState, Position, SignalScore, Side
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RiskCheckResult:
    approved: bool
    reason: str = ""
    checks_passed: list[str] = field(default_factory=list)


class GlobalRiskManager:
    """
    Portfolio-level risk management that sits above individual agent risk checks.

    Enforces:
    - Total portfolio drawdown ceiling
    - Max exposure per instrument across all agents
    - Net directional concentration limit
    - Per-agent risk budget
    - Cross-agent correlated position limit
    """

    def __init__(self, config: dict) -> None:
        self.limits = config
        # agent_id → max risk budget (USD at risk via stop distances)
        self.agent_risk_budgets: dict[str, float] = {}

    # ------------------------------------------------------------------

    def check_portfolio_signal(
        self,
        signal: SignalScore,
        agent_id: str,
        all_positions: list[Position],
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """
        Check a proposed signal against portfolio-level risk limits.
        Returns RiskCheckResult.approved=True if the signal may proceed.
        """
        checks: list[str] = []

        # 1. Portfolio drawdown
        max_dd = self.limits.get("max_drawdown_pct", 15.0)
        current_dd = getattr(portfolio, "max_drawdown_current", 0.0)
        if current_dd >= max_dd:
            return RiskCheckResult(
                approved=False,
                reason=f"Portfolio drawdown {current_dd:.1f}% ≥ limit {max_dd:.1f}%",
            )
        checks.append("portfolio_dd_ok")

        # 2. Cross-agent exposure per instrument (max 2 agents in same instrument)
        max_instrument_agents = self.limits.get("max_agents_per_instrument", 2)
        same_instrument = [p for p in all_positions if p.instrument == signal.instrument]
        if len(same_instrument) >= max_instrument_agents:
            return RiskCheckResult(
                approved=False,
                reason=f"Max {max_instrument_agents} agents already in {signal.instrument}",
            )
        checks.append("instrument_exposure_ok")

        # 3. Net directional concentration
        if portfolio.equity > 0:
            long_exp = sum(
                p.quantity * p.current_price
                for p in all_positions
                if getattr(p, "side", None) == Side.BUY
            )
            short_exp = sum(
                p.quantity * p.current_price
                for p in all_positions
                if getattr(p, "side", None) == Side.SELL
            )
            net_pct = abs(long_exp - short_exp) / portfolio.equity * 100
            max_net = self.limits.get("max_net_exposure_pct", 80.0)
            if net_pct > max_net:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Net exposure {net_pct:.0f}% > limit {max_net:.0f}%",
                )
        checks.append("directional_ok")

        # 4. Agent risk budget
        budget = self.agent_risk_budgets.get(
            agent_id, portfolio.equity * self.limits.get("agent_budget_pct", 0.25)
        )
        agent_positions = [
            p for p in all_positions
            if getattr(p, "agent_id", None) == agent_id
        ]
        agent_risk_used = sum(
            abs(p.entry_price - (p.stop_loss or p.entry_price)) * p.quantity
            for p in agent_positions
        )
        if agent_risk_used >= budget:
            return RiskCheckResult(
                approved=False,
                reason=f"Agent {agent_id} risk budget exhausted ({agent_risk_used:.0f}/{budget:.0f})",
            )
        checks.append("agent_budget_ok")

        # 5. Correlated position count (simple heuristic: same sector/asset-class)
        max_corr = self.limits.get("max_correlated_cross_agent", 4)
        correlated_count = self._count_correlated_positions(signal.instrument, all_positions)
        if correlated_count >= max_corr:
            return RiskCheckResult(
                approved=False,
                reason=f"Correlated position count {correlated_count} ≥ limit {max_corr}",
            )
        checks.append("correlation_ok")

        log.debug("GlobalRisk: approved", agent=agent_id,
                  instrument=signal.instrument, checks=checks)
        return RiskCheckResult(approved=True, checks_passed=checks)

    # ------------------------------------------------------------------

    def set_agent_budget(self, agent_id: str, budget_usd: float) -> None:
        self.agent_risk_budgets[agent_id] = budget_usd

    def _count_correlated_positions(
        self, instrument: str, positions: list[Position]
    ) -> int:
        """
        Simple sector-based correlation proxy.
        Instruments sharing a sector tag count as correlated.
        """
        sector = _instrument_sector(instrument)
        count = 0
        for p in positions:
            if _instrument_sector(p.instrument) == sector and p.instrument != instrument:
                count += 1
        return count


# ---------------------------------------------------------------------------
# Minimal sector mapping (extendable)
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech",
    "META": "tech", "AMZN": "tech", "TSLA": "tech",
    "SPY": "broad_equity", "QQQ": "broad_equity",
    "GC": "commodities", "SLV": "commodities", "CL": "commodities",
    "ES": "futures_equity", "NQ": "futures_equity",
    "EURUSD": "forex", "GBPUSD": "forex", "USDJPY": "forex",
    "BTC/USDT": "crypto", "ETH/USDT": "crypto",
}


def _instrument_sector(instrument: str) -> str:
    return _SECTOR_MAP.get(instrument, "other")
