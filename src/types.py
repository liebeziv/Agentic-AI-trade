"""Shared data types for atlas-trader."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TradeAction(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_TRADE = "NO_TRADE"


class MarketRegime(str, Enum):
    STRONG_TREND_UP = "STRONG_TREND_UP"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
    WEAK_TREND = "WEAK_TREND"
    RANGE_BOUND = "RANGE_BOUND"
    BREAKOUT = "BREAKOUT"
    CRISIS = "CRISIS"


class ExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    TIME_EXIT = "TIME_EXIT"
    RISK_MANAGER = "RISK_MANAGER"
    MANUAL = "MANUAL"
    SESSION_END = "SESSION_END"


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument: str
    timeframe: str
    source: str = ""


@dataclass
class NewsItem:
    timestamp: datetime
    title: str
    summary: str
    source: str
    url: str = ""
    relevance_score: float = 0.0
    instruments: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0   # -1..+1, parsed from sentiment_raw
    sentiment_raw: str = ""


@dataclass
class EconomicEvent:
    timestamp: datetime
    name: str
    country: str
    impact: str              # "HIGH" | "MEDIUM" | "LOW"
    forecast: str = ""
    previous: str = ""
    actual: str = ""


@dataclass
class TechnicalSnapshot:
    instrument: str
    timestamp: datetime
    timeframe: str
    indicators: dict[str, float] = field(default_factory=dict)
    signals: dict[str, str] = field(default_factory=dict)
    trend_bias: str = "neutral"
    volatility_regime: str = "normal"
    patterns: list[str] = field(default_factory=list)


@dataclass
class TradeRecommendation:
    action: TradeAction
    instrument: str
    confidence: int              # 0-100
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None = None
    position_size_suggestion_pct: float = 2.0
    risk_reward_ratio: float = 0.0
    reasoning: str = ""
    key_factors: list[str] = field(default_factory=list)
    time_horizon: str = "intraday"
    regime_assessment: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SignalScore:
    instrument: str
    timestamp: datetime
    technical_score: float = 0.0    # -1 to +1
    sentiment_score: float = 0.0    # -1 to +1
    regime_score: float = 0.0       # -1 to +1
    claude_confidence: float = 0.0  # 0 to 1
    composite_score: float = 0.0
    action: TradeAction = TradeAction.NO_TRADE
    recommendation: TradeRecommendation | None = None


@dataclass
class PositionSize:
    instrument: str
    units: float
    notional_value: float
    risk_amount: float
    risk_pct_of_equity: float
    method_used: str
    kelly_fraction: float | None = None


@dataclass
class Order:
    id: str
    instrument: str
    side: Side
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    time_in_force: str = "GTC"
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    fill_price: float | None = None
    fill_quantity: float = 0.0
    slippage: float | None = None
    commission: float = 0.0
    signal_id: str = ""
    claude_reasoning: str = ""


@dataclass
class Position:
    instrument: str
    side: Side
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    order_ids: list[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    equity: float
    cash: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    positions: list[Position] = field(default_factory=list)
    open_orders: list[Order] = field(default_factory=list)
    daily_pnl: float = 0.0
    max_drawdown_current: float = 0.0
    portfolio_heat_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeRecord:
    id: str
    instrument: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission_total: float
    entry_time: datetime
    exit_time: datetime
    duration_minutes: float
    exit_reason: ExitReason
    # Context
    signal_score: float = 0.0
    claude_reasoning_entry: str = ""
    claude_reasoning_exit: str = ""
    technical_at_entry: dict = field(default_factory=dict)
    regime_at_entry: str = ""
    news_at_entry: list[str] = field(default_factory=list)
    # Attribution
    factor_attribution: dict[str, float] = field(default_factory=dict)
    lessons_learned: str = ""


@dataclass
class RiskCheckResult:
    approved: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    adjusted_size: float | None = None
    reason: str = ""


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    margin_used: float = 0.0
    currency: str = "USD"


@dataclass
class OrderResult:
    order_id: str
    status: OrderStatus
    fill_price: float | None = None
    fill_quantity: float = 0.0
    commission: float = 0.0
    message: str = ""
