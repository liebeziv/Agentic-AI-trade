"""Correlation monitor — tracks inter-agent and inter-instrument correlations."""
from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class CorrelationMonitor:
    """
    Monitors return correlations between agents.

    Used by the Orchestrator to:
    - Penalise agents whose returns are highly correlated (reduces diversification)
    - Alert when two agents are effectively duplicating each other
    - Report a portfolio diversification score
    """

    # Agents with |corr| > this threshold trigger an alert
    ALERT_THRESHOLD = 0.70

    def __init__(self, agents: list) -> None:
        """
        Parameters
        ----------
        agents : list[BaseStrategyAgent]
        """
        self.agents = agents

    # ------------------------------------------------------------------

    def compute_correlation_matrix(self, lookback_days: int = 30) -> dict[str, dict[str, float]]:
        """
        Return nested dict corr[agent_a][agent_b] for agents that have
        at least `lookback_days` daily returns recorded.
        """
        returns_data: dict[str, list[float]] = {}
        for agent in self.agents:
            if len(agent.performance.daily_returns) >= lookback_days:
                returns_data[agent.agent_id] = list(
                    agent.performance.daily_returns[-lookback_days:]
                )

        if len(returns_data) < 2:
            return {}

        ids = list(returns_data.keys())
        matrix: dict[str, dict[str, float]] = {a: {} for a in ids}
        for i, aid_a in enumerate(ids):
            for j, aid_b in enumerate(ids):
                if i == j:
                    matrix[aid_a][aid_b] = 1.0
                    continue
                arr_a = np.array(returns_data[aid_a])
                arr_b = np.array(returns_data[aid_b])
                min_len = min(len(arr_a), len(arr_b))
                if min_len < 5:
                    matrix[aid_a][aid_b] = 0.0
                    continue
                corr = float(np.corrcoef(arr_a[-min_len:], arr_b[-min_len:])[0, 1])
                matrix[aid_a][aid_b] = round(corr, 4)

        return matrix

    def get_diversification_score(self, lookback_days: int = 30) -> float:
        """
        Portfolio diversification score in [0, 1].
        1.0 = fully uncorrelated agents (maximum diversification).
        0.0 = perfectly correlated agents (no diversification benefit).
        """
        matrix = self.compute_correlation_matrix(lookback_days)
        if not matrix:
            return 0.5  # unknown — return neutral score

        ids = list(matrix.keys())
        n = len(ids)
        if n < 2:
            return 0.5

        off_diag = [
            abs(matrix[ids[i]][ids[j]])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        avg_corr = float(np.mean(off_diag)) if off_diag else 0.0
        return round(1.0 - avg_corr, 4)

    def get_correlation_penalties(self, lookback_days: int = 30) -> dict[str, float]:
        """
        Return per-agent penalty in [0, 1]: the maximum |correlation| with any
        other agent.  Used by the Orchestrator to down-weight redundant agents.
        """
        matrix = self.compute_correlation_matrix(lookback_days)
        penalties: dict[str, float] = {}
        for agent in self.agents:
            aid = agent.agent_id
            if aid not in matrix:
                penalties[aid] = 0.0
                continue
            others = [abs(v) for k, v in matrix[aid].items() if k != aid]
            penalties[aid] = round(max(others), 4) if others else 0.0
        return penalties

    def check_alerts(self, lookback_days: int = 30) -> list[str]:
        """
        Return human-readable alerts for agent pairs with |corr| > ALERT_THRESHOLD.
        """
        matrix = self.compute_correlation_matrix(lookback_days)
        alerts: list[str] = []
        ids = list(matrix.keys())
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(matrix[ids[i]][ids[j]])
                if corr > self.ALERT_THRESHOLD:
                    alerts.append(
                        f"High correlation ({corr:.2f}) between "
                        f"{ids[i]} and {ids[j]} — consider deactivating one"
                    )
        if alerts:
            log.warning("Correlation alerts", count=len(alerts))
        return alerts

    def summary(self, lookback_days: int = 30) -> dict:
        """Return a dict suitable for dashboard display."""
        return {
            "diversification_score": self.get_diversification_score(lookback_days),
            "alerts": self.check_alerts(lookback_days),
            "correlation_matrix": self.compute_correlation_matrix(lookback_days),
            "penalties": self.get_correlation_penalties(lookback_days),
        }
