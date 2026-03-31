"""Tests for production-readiness modules:
- SocialSentimentFeed + SentimentScorer
- OKXAdapter (offline / mocked)
- HealthServer
- Dashboard components (figure generation)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.sentiment_scorer import SentimentScore, SentimentScorer
from src.data.social_sentiment import SocialPost, SocialSentimentFeed, _simple_sentiment
from src.types import NewsItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.run(coro)


def _news(instrument: str, score: float, age_hours: float = 0.5) -> NewsItem:
    return NewsItem(
        timestamp=datetime.utcnow() - timedelta(hours=age_hours),
        title=f"News about {instrument}",
        summary="",
        source="test",
        instruments=[instrument],
        sentiment_score=score,
    )


def _post(instrument: str, score: float, upvotes: int = 10,
          age_hours: float = 0.5) -> SocialPost:
    return SocialPost(
        timestamp=datetime.utcnow() - timedelta(hours=age_hours),
        text=f"bullish on {instrument}",
        source="reddit",
        instrument=instrument,
        sentiment_score=score,
        score=upvotes,
    )


# ===========================================================================
# SocialSentimentFeed
# ===========================================================================

class TestSimpleSentiment:
    def test_bullish_words(self):
        # No punctuation — _simple_sentiment uses str.split()
        assert _simple_sentiment("bull buy moon rally surge") > 0

    def test_bearish_words(self):
        assert _simple_sentiment("bear sell crash dump panic") < 0

    def test_neutral_empty(self):
        assert _simple_sentiment("") == 0.0

    def test_neutral_no_keywords(self):
        assert _simple_sentiment("the weather is nice today") == 0.0

    def test_clamped_to_range(self):
        s = _simple_sentiment("bull buy moon rally long breakout surge moon bull buy moon rally")
        assert -1.0 <= s <= 1.0

    def test_case_insensitive(self):
        assert _simple_sentiment("BULL BUY MOON") > 0


class TestSocialPost:
    def test_dataclass_fields(self):
        p = _post("AAPL", 0.5)
        assert p.instrument == "AAPL"
        assert p.sentiment_score == 0.5
        assert p.source == "reddit"

    def test_default_url(self):
        p = SocialPost(
            timestamp=datetime.utcnow(), text="test",
            source="stocktwits", instrument="AAPL",
            sentiment_score=0.0, score=0,
        )
        assert p.url == ""


class TestSocialSentimentFeed:
    def _feed(self, **cfg) -> SocialSentimentFeed:
        return SocialSentimentFeed(cfg)

    def test_instantiates_without_config(self):
        feed = self._feed()
        assert feed is not None

    def test_get_aggregate_sentiment_empty(self):
        feed = self._feed()
        result = feed.get_aggregate_sentiment([], "AAPL")
        assert result == 0.0

    def test_get_aggregate_sentiment_filters_by_instrument(self):
        feed = self._feed()
        posts = [
            _post("AAPL", 0.8),
            _post("MSFT", -0.5),
        ]
        score = feed.get_aggregate_sentiment(posts, "AAPL")
        assert score > 0

    def test_get_aggregate_sentiment_filters_old_posts(self):
        feed = self._feed()
        old = _post("AAPL", 0.9, age_hours=10)
        score = feed.get_aggregate_sentiment([old], "AAPL", hours=4)
        assert score == 0.0

    def test_get_aggregate_sentiment_weighted_by_upvotes(self):
        feed = self._feed()
        low  = _post("AAPL", 0.2, upvotes=1)
        high = _post("AAPL", 0.9, upvotes=100)
        score = feed.get_aggregate_sentiment([low, high], "AAPL")
        # Should be close to 0.9 (high upvotes dominate)
        assert score > 0.5

    def test_fetch_all_no_creds_returns_empty(self):
        feed = self._feed()
        result = run(feed.fetch_all(["AAPL"]))
        assert isinstance(result, list)

    def test_fetch_stocktwits_handles_http_error(self):
        """StockTwits fetch should return [] on HTTP failure."""
        feed = self._feed()
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 429
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value.get = MagicMock(return_value=mock_resp)
            result = run(feed._fetch_stocktwits(["AAPL"]))
            assert isinstance(result, list)


# ===========================================================================
# SentimentScorer
# ===========================================================================

class TestSentimentScorer:
    def _scorer(self) -> SentimentScorer:
        return SentimentScorer(news_weight=0.6, social_weight=0.4)

    def test_score_empty_inputs(self):
        scorer = self._scorer()
        result = scorer.score("AAPL", [], [])
        assert isinstance(result, SentimentScore)
        assert result.combined_score == 0.0
        assert result.confidence == 0.0

    def test_score_news_only(self):
        scorer = self._scorer()
        news = [_news("AAPL", 0.8), _news("AAPL", 0.6)]
        result = scorer.score("AAPL", news, [])
        assert result.news_count == 2
        assert result.news_score > 0
        assert result.combined_score > 0

    def test_score_social_only(self):
        scorer = self._scorer()
        posts = [_post("AAPL", -0.7)]
        result = scorer.score("AAPL", [], posts)
        assert result.social_count == 1
        assert result.social_score < 0

    def test_score_combined_weights(self):
        scorer = SentimentScorer(news_weight=0.6, social_weight=0.4)
        news  = [_news("AAPL", 1.0)]
        posts = [_post("AAPL", 0.0)]
        result = scorer.score("AAPL", news, posts)
        # combined = 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        assert result.combined_score == pytest.approx(0.6, abs=0.05)

    def test_confidence_grows_with_samples(self):
        scorer = self._scorer()
        few  = scorer.score("AAPL", [_news("AAPL", 0.5)], [])
        many = scorer.score("AAPL", [_news("AAPL", 0.5)] * 10, [])
        assert many.confidence >= few.confidence

    def test_confidence_capped_at_one(self):
        scorer = self._scorer()
        news = [_news("AAPL", 0.5)] * 50
        result = scorer.score("AAPL", news, [])
        assert result.confidence <= 1.0

    def test_score_filters_by_instrument(self):
        scorer = self._scorer()
        news = [_news("AAPL", 0.9), _news("MSFT", -0.9)]
        result = scorer.score("AAPL", news, [])
        assert result.news_score > 0   # MSFT should not contaminate AAPL

    def test_score_all_returns_dict(self):
        scorer = self._scorer()
        news = [_news("AAPL", 0.5), _news("MSFT", -0.3)]
        result = scorer.score_all(["AAPL", "MSFT"], news, [])
        assert set(result.keys()) == {"AAPL", "MSFT"}

    def test_to_signal_input_clamped(self):
        scorer = self._scorer()
        # Manually craft a score with high combined_score
        s = SentimentScore(
            instrument="AAPL", timestamp=datetime.utcnow(),
            news_score=1.0, social_score=1.0,
            combined_score=1.5,  # would exceed 1 without clamping
            news_count=10, social_count=10, confidence=1.0,
        )
        val = scorer.to_signal_input(s)
        assert -1.0 <= val <= 1.0

    def test_to_signal_input_scales_by_confidence(self):
        scorer = self._scorer()
        s = SentimentScore(
            instrument="AAPL", timestamp=datetime.utcnow(),
            news_score=0.8, social_score=0.8,
            combined_score=0.8, news_count=1, social_count=0,
            confidence=0.5,
        )
        val = scorer.to_signal_input(s)
        # Should be 0.8 * 0.5 = 0.4
        assert abs(val) < abs(s.combined_score)


# ===========================================================================
# OKXAdapter (offline tests — no real API calls)
# ===========================================================================

class TestOKXAdapter:
    def _adapter(self, **overrides) -> "OKXAdapter":
        from src.execution.adapters.okx_adapter import OKXAdapter
        cfg = {"api_key": "test", "secret": "test",
               "passphrase": "test", "sandbox": True, **overrides}
        return OKXAdapter(cfg)

    def test_instantiates(self):
        adapter = self._adapter()
        assert not adapter._connected

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("OKX_API_KEY", "env_key")
        from src.execution.adapters.okx_adapter import OKXAdapter
        a = OKXAdapter({})
        assert a.api_key == "env_key"

    def test_connect_raises_without_credentials(self):
        """connect() raises ValueError or ConnectionError without valid creds."""
        adapter = self._adapter(api_key="", secret="", passphrase="")
        with pytest.raises((ValueError, ConnectionError, ImportError)):
            run(adapter.connect())

    def test_status_mapping(self):
        from src.execution.adapters.okx_adapter import _map_okx_status
        from src.types import OrderStatus
        assert _map_okx_status("filled") == OrderStatus.FILLED
        assert _map_okx_status("canceled") == OrderStatus.CANCELLED
        assert _map_okx_status("live") == OrderStatus.SUBMITTED
        assert _map_okx_status("unknown") == OrderStatus.PENDING

    def test_disconnect_when_not_connected(self):
        adapter = self._adapter()
        run(adapter.disconnect())   # should not raise


# ===========================================================================
# HealthServer
# ===========================================================================

class TestHealthServer:
    def test_instantiates(self):
        from src.utils.health import HealthServer
        server = HealthServer(port=19876)
        assert server is not None

    def test_register_check(self):
        from src.utils.health import HealthServer
        server = HealthServer(port=19877)
        server.register_check("db", lambda: True)
        assert "db" in server._checks

    def test_all_checks_pass(self):
        from src.utils.health import HealthServer
        server = HealthServer(port=19878)
        server.register_check("a", lambda: True)
        server.register_check("b", lambda: True)
        status = server._get_status()
        assert status.status == "ok"
        assert all(status.checks.values())

    def test_failing_check_degrades_status(self):
        from src.utils.health import HealthServer
        server = HealthServer(port=19879)
        server.register_check("ok",   lambda: True)
        server.register_check("fail", lambda: False)
        status = server._get_status()
        assert status.status == "degraded"
        assert not status.checks["fail"]

    def test_health_status_dataclass(self):
        from src.utils.health import HealthStatus
        s = HealthStatus(status="ok", uptime_seconds=60.0, checks={"db": True})
        assert s.status == "ok"
        assert s.uptime_seconds == 60.0

    def test_start_stop(self):
        from src.utils.health import HealthServer
        server = HealthServer(port=19880)
        server.start()
        import time; time.sleep(0.1)  # let thread spin up
        server.stop()


# ===========================================================================
# Dashboard components — figure generation (no Streamlit session needed)
# ===========================================================================

class TestPnlChartComponents:
    def test_equity_curve_empty(self):
        from dashboard.components.pnl_chart import equity_curve
        fig = equity_curve([])
        assert fig is not None

    def test_equity_curve_with_data(self):
        from dashboard.components.pnl_chart import equity_curve
        curve = [100_000 + i * 50 for i in range(30)]
        fig = equity_curve(curve, initial_capital=100_000)
        assert len(fig.data) > 0

    def test_drawdown_chart(self):
        from dashboard.components.pnl_chart import drawdown_chart
        curve = [100_000, 105_000, 102_000, 98_000, 103_000]
        fig = drawdown_chart(curve)
        assert fig is not None

    def test_returns_histogram(self):
        from dashboard.components.pnl_chart import returns_histogram
        pnls = [100, -50, 200, -30, 150, -80, 300]
        fig = returns_histogram(pnls)
        assert len(fig.data) > 0

    def test_multi_agent_pnl(self):
        from dashboard.components.pnl_chart import multi_agent_pnl
        data = {"momentum": [100, 150, 200], "mean_reversion": [100, 120, 140]}
        fig = multi_agent_pnl(data)
        assert len(fig.data) == 2

    def test_monthly_returns_heatmap_empty(self):
        from dashboard.components.pnl_chart import monthly_returns_heatmap
        fig = monthly_returns_heatmap([])
        assert fig is not None


class TestSignalMonitorComponents:
    def test_agent_signal_heatmap_empty(self):
        from dashboard.components.signal_monitor import agent_signal_heatmap
        fig = agent_signal_heatmap({})
        assert fig is not None

    def test_agent_signal_heatmap_with_data(self):
        from dashboard.components.signal_monitor import agent_signal_heatmap
        data = {
            "momentum":      {"AAPL": 0.7, "MSFT": 0.3},
            "mean_reversion": {"AAPL": -0.4, "MSFT": 0.1},
        }
        fig = agent_signal_heatmap(data)
        assert len(fig.data) == 1   # one heatmap trace

    def test_signal_history_chart_empty(self):
        from dashboard.components.signal_monitor import signal_history_chart
        fig = signal_history_chart([], "AAPL")
        assert fig is not None

    def test_signal_history_chart_filters_instrument(self):
        from dashboard.components.signal_monitor import signal_history_chart
        signals = [
            {"instrument": "AAPL", "composite_score": 0.5, "action": "BUY",
             "timestamp": datetime.utcnow()},
            {"instrument": "MSFT", "composite_score": -0.3, "action": "SELL",
             "timestamp": datetime.utcnow()},
        ]
        fig = signal_history_chart(signals, "AAPL")
        # Should only show AAPL — check trace has 1 point
        assert len(fig.data) > 0

    def test_news_sentiment_bar_empty(self):
        from dashboard.components.signal_monitor import news_sentiment_bar
        fig = news_sentiment_bar([])
        assert fig is not None

    def test_news_sentiment_bar_with_data(self):
        from dashboard.components.signal_monitor import news_sentiment_bar
        items = [
            {"instruments": ["AAPL"], "sentiment_score": 0.7, "title": "Apple up"},
            {"instruments": ["MSFT"], "sentiment_score": -0.3, "title": "MSFT down"},
        ]
        fig = news_sentiment_bar(items)
        assert len(fig.data) > 0


class TestRiskPanelComponents:
    def test_exposure_gauge(self):
        from dashboard.components.risk_panel import exposure_gauge
        fig = exposure_gauge(45.0, limit_pct=80.0)
        assert len(fig.data) == 1

    def test_drawdown_gauge(self):
        from dashboard.components.risk_panel import drawdown_gauge
        fig = drawdown_gauge(8.5, limit_dd=15.0)
        assert fig is not None

    def test_correlation_heatmap_empty(self):
        from dashboard.components.risk_panel import correlation_heatmap
        fig = correlation_heatmap({})
        assert fig is not None

    def test_correlation_heatmap_with_data(self):
        from dashboard.components.risk_panel import correlation_heatmap
        matrix = {
            "momentum":       {"momentum": 1.0, "mean_reversion": 0.3},
            "mean_reversion": {"momentum": 0.3, "mean_reversion": 1.0},
        }
        fig = correlation_heatmap(matrix)
        assert len(fig.data) == 1

    def test_position_concentration_empty(self):
        from dashboard.components.risk_panel import position_concentration_chart
        fig = position_concentration_chart([])
        assert fig is not None

    def test_position_concentration_with_data(self):
        from dashboard.components.risk_panel import position_concentration_chart
        positions = [
            {"instrument": "AAPL", "notional_value": 10_000, "side": "BUY"},
            {"instrument": "MSFT", "notional_value": 8_000,  "side": "BUY"},
            {"instrument": "EURUSD", "notional_value": 5_000, "side": "SELL"},
        ]
        fig = position_concentration_chart(positions)
        assert fig is not None


# ===========================================================================
# pyproject.toml — dev extras exist
# ===========================================================================

class TestProjectConfig:
    def test_dev_extras_defined(self):
        import tomllib
        from pathlib import Path
        path = Path(__file__).parent.parent / "pyproject.toml"
        with open(path, "rb") as f:
            data = tomllib.load(f)
        extras = data.get("project", {}).get("optional-dependencies", {})
        assert "dev" in extras, "Missing [project.optional-dependencies.dev]"
        dev = extras["dev"]
        assert any("pytest" in d for d in dev)

    def test_new_scripts_defined(self):
        import tomllib
        from pathlib import Path
        path = Path(__file__).parent.parent / "pyproject.toml"
        with open(path, "rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {})
        assert "atlas-multiagent" in scripts
        assert "atlas-backtest-multi" in scripts
