"""AgentRegistry — lifecycle management for all specialist strategy agents."""
from __future__ import annotations

from typing import Iterator, Optional

from src.strategy.strategies.base_agent import BaseStrategyAgent
from src.utils.logger import get_logger

log = get_logger(__name__)


class AgentRegistry:
    """
    Central registry that owns the full lifecycle of strategy agents.

    Responsibilities
    ----------------
    - Register / deactivate / reactivate agents at runtime
    - Expose filtered views (active only, all, by id)
    - Produce a structured summary used by the dashboard and orchestrator
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseStrategyAgent] = {}

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def register(self, agent: BaseStrategyAgent) -> None:
        """Add *agent* to the registry (idempotent on agent_id)."""
        if agent.agent_id in self._agents:
            log.warning(
                "Agent already registered — overwriting",
                agent_id=agent.agent_id,
                name=agent.name,
            )
        self._agents[agent.agent_id] = agent
        log.info(
            "Agent registered",
            agent_id=agent.agent_id,
            name=agent.name,
            is_active=agent.is_active,
        )

    def deactivate(self, agent_id: str) -> None:
        """Mark agent as inactive so it is excluded from signal generation."""
        agent = self._agents.get(agent_id)
        if agent is None:
            log.warning("deactivate called on unknown agent_id", agent_id=agent_id)
            return
        agent.is_active = False
        log.info("Agent deactivated", agent_id=agent_id, name=agent.name)

    def activate(self, agent_id: str) -> None:
        """Re-enable a previously deactivated agent."""
        agent = self._agents.get(agent_id)
        if agent is None:
            log.warning("activate called on unknown agent_id", agent_id=agent_id)
            return
        agent.is_active = True
        log.info("Agent activated", agent_id=agent_id, name=agent.name)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_active_agents(self) -> list[BaseStrategyAgent]:
        """Return all agents whose ``is_active`` flag is ``True``."""
        return [a for a in self._agents.values() if a.is_active]

    def get_agent(self, agent_id: str) -> Optional[BaseStrategyAgent]:
        """Return the agent with the given id, or ``None`` if not found."""
        return self._agents.get(agent_id)

    def get_all_agents(self) -> list[BaseStrategyAgent]:
        """Return every registered agent regardless of active status."""
        return list(self._agents.values())

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary(self) -> list[dict]:
        """
        Return a list of dicts — one per agent — suitable for dashboard display
        or JSON serialisation.

        Keys per entry
        --------------
        id, name, active, capital, sharpe_30d, total_pnl, win_rate, fitness
        """
        rows: list[dict] = []
        for agent in self._agents.values():
            p = agent.performance
            rows.append(
                {
                    "id": agent.agent_id,
                    "name": agent.name,
                    "active": agent.is_active,
                    "capital": agent.allocated_capital,
                    "sharpe_30d": round(p.rolling_sharpe_30d, 4),
                    "total_pnl": round(p.total_pnl, 2),
                    "win_rate": round(p.win_rate_pct, 2),
                    "fitness": round(p.fitness, 4),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[BaseStrategyAgent]:
        return iter(self._agents.values())

    def __repr__(self) -> str:  # pragma: no cover
        active = sum(1 for a in self._agents.values() if a.is_active)
        return (
            f"AgentRegistry(total={len(self._agents)}, active={active})"
        )
