"""Tests for the attribution engine and enhanced trade journal."""
import pytest
from datetime import datetime

from src.data.data_store import DataStore
from src.reflection.attribution import AttributionEngine
from src.types import ExitReason, Side, TradeRecord


def make_trade(pnl: float, side: Side = Side.BUY, regime: str = "STRONG_TREND_UP",
               trend_bias: str = "bullish", signal_score: float = 0.6,
               duration_min: float = 45.0, atr: float = 2.0) -> TradeRecord:
    entry = 100.0
    exit_p = entry + pnl / 10  # 10 units
    return TradeRecord(
        id=f"TEST-{id(pnl)}",
        instrument="TEST",
        side=side,
        entry_price=entry,
        exit_price=exit_p,
        quantity=10.0,
        pnl=pnl,
        pnl_pct=pnl / 1000 * 100,
        commission_total=0.0,
        entry_time=datetime(2026, 1, 1),
        exit_time=datetime(2026, 1, 1, 1),
        duration_minutes=duration_min,
        exit_reason=ExitReason.TAKE_PROFIT if pnl > 0 else ExitReason.STOP_LOSS,
        signal_score=signal_score,
        regime_at_entry=regime,
        technical_at_entry={"trend_bias": trend_bias, "atr_14": atr},
    )


@pytest.fixture
def store(tmp_path):
    db = DataStore(str(tmp_path / "test.duckdb"))
    yield db
    db.close()


@pytest.fixture
def engine(store):
    return AttributionEngine(store)


# ---------------------------------------------------------------- Unit tests

def test_winning_trade_positive_attribution(engine):
    trade = make_trade(pnl=100.0, trend_bias="bullish", regime="STRONG_TREND_UP")
    attr = engine.attribute_trade(trade)
    total = sum(attr.values())
    assert total > 0, "Winning trade should have net positive attribution"


def test_losing_trade_negative_attribution(engine):
    trade = make_trade(pnl=-50.0, trend_bias="bearish", regime="STRONG_TREND_DOWN",
                       side=Side.BUY)  # bought in downtrend — should be negative
    attr = engine.attribute_trade(trade)
    total = sum(attr.values())
    assert total < 0, "Losing trade against trend should have net negative attribution"


def test_attribution_factors_present(engine):
    trade = make_trade(pnl=80.0)
    attr = engine.attribute_trade(trade)
    expected_factors = {"technical", "sentiment", "regime", "timing", "sizing"}
    assert expected_factors == set(attr.keys())


def test_attribution_magnitudes_sum_to_one(engine):
    trade = make_trade(pnl=50.0)
    attr = engine.attribute_trade(trade)
    total_mag = sum(abs(v) for v in attr.values())
    assert abs(total_mag - 1.0) < 1e-9, f"Magnitudes should sum to 1.0, got {total_mag}"


def test_misaligned_regime_penalised(engine):
    """Buy in a downtrend and lose — regime factor should be strongly negative."""
    trade = make_trade(pnl=-60.0, side=Side.BUY,
                       regime="STRONG_TREND_DOWN", trend_bias="bearish")
    attr = engine.attribute_trade(trade)
    assert attr["regime"] < 0, "Regime should be negative when trading against trend"


def test_high_confidence_wrong_penalised(engine):
    """High signal_score that led to a loss should give negative sentiment."""
    trade = make_trade(pnl=-80.0, signal_score=0.8)
    attr = engine.attribute_trade(trade)
    assert attr["sentiment"] < 0


def test_quick_win_good_timing(engine):
    """A short-duration winning trade should score positively on timing."""
    trade = make_trade(pnl=50.0, duration_min=10.0)
    attr = engine.attribute_trade(trade)
    assert attr["timing"] > 0


def test_report_empty_period(engine, store):
    from datetime import timedelta
    now = datetime(2026, 1, 1)
    report = engine.generate_report(now, now + timedelta(days=1))
    assert "error" in report


def test_report_with_trades(engine, store):
    from datetime import timedelta
    trades = [
        make_trade(100.0, regime="STRONG_TREND_UP", trend_bias="bullish"),
        make_trade(-30.0, regime="WEAK_TREND", trend_bias="bearish", side=Side.BUY),
        make_trade(60.0, regime="STRONG_TREND_UP", trend_bias="bullish"),
    ]
    for t in trades:
        t.entry_time = datetime(2026, 2, 1)
        t.exit_time = datetime(2026, 2, 1, 2)
        store.save_trade(t)

    report = engine.generate_report(
        datetime(2026, 2, 1), datetime(2026, 2, 2)
    )
    assert report["total_trades"] == 3
    assert report["winning_trades"] == 2
    assert report["total_pnl"] == pytest.approx(130.0)
    assert "technical" in report["avg_attribution"]
    assert report["best_factor"] != ""
    assert report["worst_factor"] != ""


def test_report_instrument_breakdown(engine, store):
    from datetime import timedelta
    t1 = make_trade(50.0)
    t1.instrument = "AAPL"
    t1.entry_time = t1.exit_time = datetime(2026, 3, 1)
    t2 = make_trade(-20.0)
    t2.instrument = "NVDA"
    t2.entry_time = t2.exit_time = datetime(2026, 3, 1)
    store.save_trade(t1)
    store.save_trade(t2)

    report = engine.generate_report(datetime(2026, 3, 1), datetime(2026, 3, 2))
    assert "AAPL" in report["instrument_breakdown"]
    assert "NVDA" in report["instrument_breakdown"]
    assert report["instrument_breakdown"]["AAPL"]["pnl"] == pytest.approx(50.0)
