"""Signal monitoring components — live signal feed, regime display, agent heatmap."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


_SIGNAL_KEYS = [
    "timestamp",
    "instrument",
    "action",
    "composite_score",
    "technical_score",
    "sentiment_score",
    "claude_confidence",
]

_REGIME_COLORS: dict[str, str] = {
    "STRONG_TREND_UP": "#26a69a",       # green
    "STRONG_TREND_DOWN": "#ef5350",     # red
    "BREAKOUT": "#42a5f5",              # blue
    "RANGE_BOUND": "#ffa726",           # orange
    "WEAK_TREND": "#ffee58",            # yellow
    "CRISIS": "#b71c1c",                # dark red
    "UNKNOWN": "#9e9e9e",               # grey
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def signal_table(signals: list[dict]) -> None:
    """Render a Streamlit dataframe for the live signal feed.

    Args:
        signals: List of signal dicts. Missing keys are filled with defaults.
    """
    if not signals:
        st.info("No signals available.")
        return

    # Normalise rows — fill missing keys
    rows: list[dict] = []
    for raw in signals:
        row = {
            "timestamp": raw.get("timestamp", ""),
            "instrument": raw.get("instrument", ""),
            "action": raw.get("action", ""),
            "composite_score": float(raw.get("composite_score", 0)),
            "technical_score": float(raw.get("technical_score", 0)),
            "sentiment_score": float(raw.get("sentiment_score", 0)),
            "claude_confidence": float(raw.get("claude_confidence", 0)),
        }
        rows.append(row)

    # Try to import pandas for Styler; fall back to plain dataframe
    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=_SIGNAL_KEYS)

        def _color_composite(val: float) -> str:
            if val > 0.3:
                return "color: #26a69a; font-weight: bold"
            if val < -0.3:
                return "color: #ef5350; font-weight: bold"
            return "color: #ffee58"

        styled = df.style.applymap(_color_composite, subset=["composite_score"])
        st.dataframe(styled, use_container_width=True)
    except ImportError:
        # No pandas — plain list-of-dicts display
        st.dataframe(rows, use_container_width=True)


def regime_badge(regime: str) -> None:
    """Render a colored badge showing the current market regime.

    Args:
        regime: One of the known regime strings (e.g. "STRONG_TREND_UP").
    """
    regime = (regime or "UNKNOWN").strip().upper()
    color = _REGIME_COLORS.get(regime, _REGIME_COLORS["UNKNOWN"])

    label_map = {
        "STRONG_TREND_UP": "Strong Trend Up",
        "STRONG_TREND_DOWN": "Strong Trend Down",
        "BREAKOUT": "Breakout",
        "RANGE_BOUND": "Range Bound",
        "WEAK_TREND": "Weak Trend",
        "CRISIS": "Crisis",
        "UNKNOWN": "Unknown",
    }
    label = label_map.get(regime, regime)

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background-color: {color};
            color: #fff;
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.4);
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def agent_signal_heatmap(
    agent_signals: dict[str, dict[str, float]],
) -> go.Figure:
    """Heatmap of composite scores: rows = agents, cols = instruments.

    Args:
        agent_signals: ``{agent_id: {instrument: composite_score}}``.

    Returns:
        Plotly Figure.
    """
    title = "Agent Signal Heatmap"
    if not agent_signals:
        return _empty_figure(title)

    agents = sorted(agent_signals.keys())
    instruments: list[str] = sorted(
        {inst for scores in agent_signals.values() for inst in scores}
    )

    if not instruments:
        return _empty_figure(title)

    z: list[list[float]] = []
    text: list[list[str]] = []
    for agent in agents:
        row: list[float] = []
        row_text: list[str] = []
        for inst in instruments:
            score = agent_signals[agent].get(inst, 0.0)
            row.append(score)
            row_text.append(f"{score:+.2f}")
        z.append(row)
        text.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=instruments,
            y=agents,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            zmid=0,
            colorbar=dict(title="Score"),
            hovertemplate="Agent: %{y}<br>Instrument: %{x}<br>Score: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Instrument",
        yaxis_title="Agent",
        template="plotly_dark",
        margin=dict(l=100, r=20, t=50, b=80),
    )
    return fig


def signal_history_chart(
    signals: list[dict],
    instrument: str,
) -> go.Figure:
    """Scatter of composite_score over time for a single instrument.

    Args:
        signals: List of signal dicts.
        instrument: Instrument to filter on.

    Returns:
        Plotly Figure.
    """
    title = f"Signal History — {instrument}"
    if not signals:
        return _empty_figure(title)

    filtered = [s for s in signals if s.get("instrument") == instrument]
    if not filtered:
        return _empty_figure(title)

    _ACTION_COLOR = {"BUY": "#26a69a", "SELL": "#ef5350", "NEUTRAL": "#9e9e9e"}

    x_vals: list = []
    y_vals: list[float] = []
    colors: list[str] = []
    hover_texts: list[str] = []

    for sig in filtered:
        ts = sig.get("timestamp", "")
        score = float(sig.get("composite_score", 0))
        action = str(sig.get("action", "NEUTRAL")).upper()
        x_vals.append(ts)
        y_vals.append(score)
        colors.append(_ACTION_COLOR.get(action, "#9e9e9e"))
        hover_texts.append(
            f"Time: {ts}<br>Score: {score:+.3f}<br>Action: {action}"
        )

    fig = go.Figure()

    # Plot each action group separately so they appear in the legend
    for action_key, color in _ACTION_COLOR.items():
        idxs = [i for i, s in enumerate(filtered)
                if str(s.get("action", "NEUTRAL")).upper() == action_key]
        if not idxs:
            continue
        fig.add_trace(
            go.Scatter(
                x=[x_vals[i] for i in idxs],
                y=[y_vals[i] for i in idxs],
                mode="markers",
                marker=dict(color=color, size=8, symbol="circle"),
                name=action_key,
                hovertext=[hover_texts[i] for i in idxs],
                hoverinfo="text",
            )
        )

    # Entry threshold lines at ±0.25
    for threshold, label in [(0.25, "+0.25 threshold"), (-0.25, "-0.25 threshold")]:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="rgba(255,255,255,0.35)",
            line_width=1,
            annotation_text=label,
            annotation_position="right",
            annotation_font_color="rgba(255,255,255,0.45)",
            annotation_font_size=10,
        )

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.15)", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Composite Score",
        template="plotly_dark",
        hovermode="closest",
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def news_sentiment_bar(
    news_items: list[dict],
) -> go.Figure:
    """Horizontal bar chart of average news sentiment per instrument.

    Args:
        news_items: List of dicts with ``instrument``, ``sentiment_score``, ``title``.

    Returns:
        Plotly Figure.
    """
    title = "News Sentiment by Instrument"
    if not news_items:
        return _empty_figure(title)

    # Aggregate average sentiment per instrument
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in news_items:
        inst = str(item.get("instrument", "Unknown"))
        score = float(item.get("sentiment_score", 0))
        totals[inst] = totals.get(inst, 0.0) + score
        counts[inst] = counts.get(inst, 0) + 1

    if not totals:
        return _empty_figure(title)

    instruments = sorted(totals.keys())
    avg_scores = [totals[i] / counts[i] for i in instruments]
    bar_colors = ["#26a69a" if s >= 0 else "#ef5350" for s in avg_scores]

    fig = go.Figure(
        go.Bar(
            x=avg_scores,
            y=instruments,
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="Instrument: %{y}<br>Avg Sentiment: %{x:.3f}<extra></extra>",
            name="",
        )
    )

    fig.add_vline(x=0, line_dash="solid", line_color="rgba(255,255,255,0.4)", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Avg Sentiment Score",
        yaxis_title="Instrument",
        template="plotly_dark",
        margin=dict(l=100, r=20, t=50, b=50),
        showlegend=False,
    )
    return fig
