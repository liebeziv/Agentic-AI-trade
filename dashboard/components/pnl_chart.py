"""P&L visualization components — equity curve, drawdown, returns distribution."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
import plotly.express as px


def _empty_figure(title: str = "") -> go.Figure:
    """Return a blank figure with a 'No data' annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    if title:
        fig.update_layout(title=title)
    return fig


def equity_curve(
    equity: list[float],
    timestamps: Optional[list[datetime]] = None,
    initial_capital: float = 100_000,
    title: str = "Equity Curve",
) -> go.Figure:
    """Line chart of equity over time with shaded area and initial capital baseline.

    Args:
        equity: Sequence of portfolio equity values.
        timestamps: Optional x-axis datetimes; uses integer index if None.
        initial_capital: Reference level drawn as a horizontal dashed line.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    if not equity:
        return _empty_figure(title)

    x = timestamps if timestamps else list(range(len(equity)))
    current = equity[-1]
    line_color = "#26a69a" if current >= initial_capital else "#ef5350"
    fill_color = "rgba(38,166,154,0.15)" if current >= initial_capital else "rgba(239,83,80,0.15)"

    fig = go.Figure()

    # Shaded area beneath the line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=equity,
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            name="Equity",
            hovertemplate="%{x}<br>Equity: $%{y:,.2f}<extra></extra>",
        )
    )

    # Initial capital reference line
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text=f"Initial: ${initial_capital:,.0f}",
        annotation_position="bottom right",
        annotation_font_color="rgba(255,255,255,0.6)",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=50),
        showlegend=False,
    )
    return fig


def drawdown_chart(
    equity: list[float],
    timestamps: Optional[list[datetime]] = None,
) -> go.Figure:
    """Filled area chart of running drawdown from peak.

    Args:
        equity: Sequence of portfolio equity values.
        timestamps: Optional x-axis datetimes.

    Returns:
        Plotly Figure with y-axis inverted.
    """
    title = "Drawdown %"
    if not equity:
        return _empty_figure(title)

    # Compute running max and drawdown percentage
    running_max: list[float] = []
    peak = equity[0]
    for val in equity:
        if val > peak:
            peak = val
        running_max.append(peak)

    drawdown = [
        (e - m) / m * 100 if m != 0 else 0.0
        for e, m in zip(equity, running_max)
    ]

    x = timestamps if timestamps else list(range(len(equity)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=drawdown,
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.3)",
            line=dict(color="#ef5350", width=1.5),
            name="Drawdown %",
            hovertemplate="%{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    max_dd = min(drawdown)
    fig.add_hline(
        y=max_dd,
        line_dash="dot",
        line_color="rgba(239,83,80,0.7)",
        annotation_text=f"Max DD: {max_dd:.2f}%",
        annotation_position="bottom right",
        annotation_font_color="rgba(239,83,80,0.9)",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        yaxis_autorange="reversed",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=50),
        showlegend=False,
    )
    return fig


def returns_histogram(
    pnl_list: list[float],
    bins: int = 40,
) -> go.Figure:
    """Histogram of trade P&L values split into green/red bars.

    Args:
        pnl_list: List of individual trade P&L values.
        bins: Number of histogram bins.

    Returns:
        Plotly Figure.
    """
    title = "P&L Distribution"
    if not pnl_list:
        return _empty_figure(title)

    positive = [v for v in pnl_list if v >= 0]
    negative = [v for v in pnl_list if v < 0]
    mean_val = sum(pnl_list) / len(pnl_list)

    # Determine shared bin edges so both traces align
    min_val = min(pnl_list)
    max_val = max(pnl_list)
    bin_size = (max_val - min_val) / bins if max_val != min_val else 1.0

    fig = go.Figure()

    if positive:
        fig.add_trace(
            go.Histogram(
                x=positive,
                xbins=dict(start=0, end=max_val + bin_size, size=bin_size),
                marker_color="#26a69a",
                opacity=0.8,
                name="Profit",
                hovertemplate="P&L: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )

    if negative:
        fig.add_trace(
            go.Histogram(
                x=negative,
                xbins=dict(start=min_val - bin_size, end=0, size=bin_size),
                marker_color="#ef5350",
                opacity=0.8,
                name="Loss",
                hovertemplate="P&L: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )

    # Zero line
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="rgba(255,255,255,0.5)",
        line_width=1.5,
        annotation_text="0",
        annotation_position="top",
        annotation_font_color="rgba(255,255,255,0.6)",
    )

    # Mean line
    mean_color = "#26a69a" if mean_val >= 0 else "#ef5350"
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=mean_color,
        line_width=1.5,
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right",
        annotation_font_color=mean_color,
    )

    fig.update_layout(
        title=title,
        xaxis_title="P&L ($)",
        yaxis_title="Frequency",
        barmode="overlay",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def monthly_returns_heatmap(
    trades: list[dict],
    initial_capital: float = 100_000,
) -> go.Figure:
    """Heatmap of monthly returns grouped by year (rows) and month (cols).

    Args:
        trades: List of dicts with keys ``exit_time`` (datetime) and ``pnl`` (float).
        initial_capital: Used to normalise returns as a percentage.

    Returns:
        Plotly Figure heatmap.
    """
    title = "Monthly Returns Heatmap"
    if not trades:
        return _empty_figure(title)

    # Aggregate P&L by (year, month)
    monthly: dict[tuple[int, int], float] = {}
    for trade in trades:
        exit_time = trade.get("exit_time")
        pnl = float(trade.get("pnl", 0))
        if exit_time is None:
            continue
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)
        key = (exit_time.year, exit_time.month)
        monthly[key] = monthly.get(key, 0.0) + pnl

    if not monthly:
        return _empty_figure(title)

    years = sorted({k[0] for k in monthly})
    months = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Build 2-D matrix: rows = years (oldest at top), cols = months
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for year in years:
        row: list[float | None] = []
        row_text: list[str] = []
        for month in months:
            pnl = monthly.get((year, month))
            if pnl is not None:
                ret_pct = pnl / initial_capital * 100
                row.append(ret_pct)
                row_text.append(f"{ret_pct:+.2f}%")
            else:
                row.append(None)
                row_text.append("")
        z.append(row)
        text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=month_labels,
            y=[str(y) for y in years],
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="Return %"),
            hoverongaps=False,
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def multi_agent_pnl(
    agent_pnls: dict[str, list[float]],
    timestamps: Optional[list[datetime]] = None,
) -> go.Figure:
    """Multiple cumulative P&L lines, one per agent.

    Args:
        agent_pnls: Mapping of agent_id → list of cumulative P&L values.
        timestamps: Shared x-axis datetimes; uses integer index if None.

    Returns:
        Plotly Figure.
    """
    title = "Per-Agent Cumulative P&L"
    if not agent_pnls:
        return _empty_figure(title)

    fig = go.Figure()

    for agent_id, pnl_series in agent_pnls.items():
        if not pnl_series:
            continue
        x = timestamps if timestamps else list(range(len(pnl_series)))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pnl_series,
                mode="lines",
                name=agent_id,
                hovertemplate=f"{agent_id}<br>%{{x}}<br>P&L: $%{{y:,.2f}}<extra></extra>",
            )
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        line_width=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Cumulative P&L ($)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
