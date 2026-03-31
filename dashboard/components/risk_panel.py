"""Risk monitoring components — exposure gauges, limit utilization, kill switch status."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def exposure_gauge(current_pct: float, limit_pct: float = 80.0,
                   title: str = "Net Exposure") -> go.Figure:
    """Gauge chart for net directional exposure."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_pct,
        delta={"reference": limit_pct, "decreasing": {"color": "#00c853"},
               "increasing": {"color": "#d32f2f"}},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#1565c0"},
            "steps": [
                {"range": [0, 50],        "color": "#e8f5e9"},
                {"range": [50, limit_pct * 0.9], "color": "#fff9c4"},
                {"range": [limit_pct * 0.9, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "#d32f2f", "width": 4},
                "thickness": 0.75,
                "value": limit_pct,
            },
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20))
    return fig


def drawdown_gauge(current_dd: float, limit_dd: float = 15.0) -> go.Figure:
    """Gauge chart for portfolio drawdown."""
    warning = limit_dd * 0.7
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_dd,
        title={"text": "Current Drawdown"},
        gauge={
            "axis": {"range": [0, limit_dd * 1.5], "ticksuffix": "%"},
            "bar": {"color": "#ef5350"},
            "steps": [
                {"range": [0, warning],   "color": "#e8f5e9"},
                {"range": [warning, limit_dd], "color": "#fff9c4"},
                {"range": [limit_dd, limit_dd * 1.5], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "#b71c1c", "width": 4},
                "thickness": 0.75,
                "value": limit_dd,
            },
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20))
    return fig


def risk_limits_table(limits: dict[str, dict]) -> None:
    """Render risk limits table: metric / limit / current / status."""
    if not limits:
        st.info("No risk limits configured.")
        return

    import pandas as pd
    rows = []
    for metric, vals in limits.items():
        ok = vals.get("ok", True)
        rows.append({
            "Metric":  metric.replace("_", " ").title(),
            "Limit":   vals.get("limit", "—"),
            "Current": vals.get("current", "—"),
            "Status":  "✓" if ok else "✗",
        })
    df = pd.DataFrame(rows)

    def _color_status(val):
        return "color: #2e7d32; font-weight:bold" if val == "✓" \
               else "color: #c62828; font-weight:bold"

    styled = df.style.applymap(_color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def position_concentration_chart(positions: list[dict]) -> go.Figure:
    """Treemap of position concentration by instrument."""
    if not positions:
        fig = go.Figure()
        fig.update_layout(title="Position Concentration — No Positions")
        return fig

    instruments = [p.get("instrument", "?") for p in positions]
    notionals   = [abs(float(p.get("notional_value", 0))) for p in positions]
    sides       = [p.get("side", "BUY") for p in positions]
    colors      = ["#66bb6a" if s == "BUY" else "#ef5350" for s in sides]

    fig = px.treemap(
        names=instruments,
        parents=[""] * len(instruments),
        values=notionals,
        color=colors,
        title="Position Concentration",
    )
    fig.update_layout(height=400)
    return fig


def kill_switch_status(active: bool, reason: str = "") -> None:
    """Render kill switch status indicator."""
    if active:
        msg = f"🔴 KILL SWITCH ACTIVE"
        if reason:
            msg += f" — {reason}"
        st.error(msg)
    else:
        st.success("🟢 System Running — All checks nominal")


def correlation_heatmap(matrix: dict[str, dict[str, float]]) -> go.Figure:
    """Square heatmap of inter-agent correlations."""
    if not matrix:
        fig = go.Figure()
        fig.update_layout(title="Inter-Agent Correlation — Insufficient Data")
        return fig

    agents = list(matrix.keys())
    z = [[matrix[a].get(b, 0.0) for b in agents] for a in agents]
    text = [[f"{matrix[a].get(b, 0.0):.2f}" for b in agents] for a in agents]

    fig = go.Figure(go.Heatmap(
        z=z, x=agents, y=agents,
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=text,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(
        title="Inter-Agent Correlation",
        xaxis_title="Agent",
        yaxis_title="Agent",
        height=400,
    )
    return fig
