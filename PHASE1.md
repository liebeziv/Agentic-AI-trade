# Phase 1: Simulation + Claude API + Signal Fusion

> **Duration**: Weeks 1-8
> **Mode**: Paper trading on real market data
> **Exit Criteria**: 3-month sim → Sharpe > 1.5, Max DD < 15%, > 200 trades

---

## 1. Implementation Order

```
Step 1:  Project scaffolding (structure, deps, config)
Step 2:  Data layer — market data fetcher + data store
Step 3:  Data layer — news feed + economic calendar
Step 4:  Analysis — technical indicator engine
Step 5:  Analysis — Claude API agent
Step 6:  Analysis — regime detector
Step 7:  Strategy — signal aggregator
Step 8:  Strategy — risk manager
Step 9:  Strategy — position sizer
Step 10: Execution — paper trading engine
Step 11: Reflection — trade journal + basic attribution
Step 12: Pipeline — main trading loop (connect all modules)
Step 13: Backtest — backtesting framework
Step 14: Dashboard — Streamlit monitoring UI
Step 15: Integration testing + parameter optimization
```

---

## 2. Step 1: Project Scaffolding

### 2.1 pyproject.toml

```toml
[project]
name = "atlas-trader"
version = "0.1.0"
description = "Agentic AI Trading System with Claude API"
requires-python = ">=3.12"
dependencies = [
    # Core
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "structlog>=24.0",
    "python-dotenv>=1.0",
    # Data
    "polars>=1.0",
    "pandas>=2.2",
    "duckdb>=1.0",
    "ccxt>=4.0",
    "yfinance>=0.2",
    "MetaTrader5>=5.0; sys_platform == 'win32'",
    "newsapi-python>=0.2",
    "feedparser>=6.0",
    "aiohttp>=3.9",
    "websockets>=12.0",
    # Analysis
    "pandas-ta>=0.3",
    "numpy>=1.26",
    "scipy>=1.12",
    "hmmlearn>=0.3",
    # AI
    "anthropic>=0.40",
    # Execution & Monitoring
    "apscheduler>=3.10",
    "streamlit>=1.38",
    "plotly>=5.22",
    "python-telegram-bot>=21.0",
    # Testing
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]

[project.scripts]
atlas-backtest = "scripts.run_backtest:main"
atlas-paper = "scripts.run_paper:main"
atlas-live = "scripts.run_live:main"
atlas-dashboard = "dashboard.app:main"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 2.2 Config Files

**config/settings.yaml**:
```yaml
system:
  name: "atlas-trader"
  phase: 1
  mode: "paper"            # paper | live | backtest
  timezone: "UTC"
  log_level: "INFO"
  log_dir: "./data/logs"

trading:
  default_timeframes: ["5m", "15m", "1h"]
  analysis_interval_seconds: 300
  max_instruments_parallel: 5
  trading_hours:
    forex: { start: "22:00", end: "21:59", days: [0,1,2,3,4] }  # Sun-Fri UTC
    us_stocks: { start: "13:30", end: "20:00", days: [0,1,2,3,4] }
    hk_stocks: { start: "01:30", end: "08:00", days: [0,1,2,3,4] }
    crypto: { start: "00:00", end: "23:59", days: [0,1,2,3,4,5,6] }
    futures: { start: "23:00", end: "22:00", days: [0,1,2,3,4] }

claude:
  default_model: "claude-sonnet-4-20250514"
  premium_model: "claude-opus-4-6"
  escalation_threshold: 0.8
  max_tokens_analysis: 2000
  max_tokens_reflection: 1500
  temperature: 0.3
  cache_ttl_seconds: 300
  monthly_budget_usd: 200
  batch_interval_seconds: 300

backtest:
  initial_capital: 100000
  commission_pct: 0.1
  slippage_pct: 0.05
  walk_forward_window_days: 90
  walk_forward_step_days: 30
```

**config/instruments.yaml**:
```yaml
instruments:
  forex:
    - symbol: "EURUSD"
      pip_value: 0.0001
      spread_typical: 0.00012
      lot_size: 100000
      margin_pct: 3.33
    - symbol: "GBPUSD"
      pip_value: 0.0001
      spread_typical: 0.00015
      lot_size: 100000
      margin_pct: 3.33
    - symbol: "USDJPY"
      pip_value: 0.01
      spread_typical: 0.012
      lot_size: 100000
      margin_pct: 3.33
    - symbol: "AUDUSD"
      pip_value: 0.0001
      spread_typical: 0.00014
      lot_size: 100000
      margin_pct: 5.0

  futures:
    - symbol: "ES"
      tick_size: 0.25
      tick_value: 12.50
      margin: 15000
      exchange: "CME"
    - symbol: "NQ"
      tick_size: 0.25
      tick_value: 5.00
      margin: 18000
      exchange: "CME"
    - symbol: "GC"
      tick_size: 0.10
      tick_value: 10.00
      margin: 10000
      exchange: "COMEX"
    - symbol: "CL"
      tick_size: 0.01
      tick_value: 10.00
      margin: 8000
      exchange: "NYMEX"

  us_stocks:
    - symbol: "AAPL"
      min_qty: 1
      commission_per_share: 0.005
    - symbol: "NVDA"
      min_qty: 1
      commission_per_share: 0.005
    - symbol: "MSFT"
      min_qty: 1
      commission_per_share: 0.005
    - symbol: "TSLA"
      min_qty: 1
      commission_per_share: 0.005
    - symbol: "AMZN"
      min_qty: 1
      commission_per_share: 0.005
    - symbol: "META"
      min_qty: 1
      commission_per_share: 0.005
    - symbol: "GOOGL"
      min_qty: 1
      commission_per_share: 0.005
    - symbol: "SPY"
      min_qty: 1
      commission_per_share: 0.005

  crypto:
    - symbol: "BTC/USDT"
      min_qty: 0.0001
      commission_pct: 0.1
      exchange: "binance"
    - symbol: "ETH/USDT"
      min_qty: 0.001
      commission_pct: 0.1
      exchange: "binance"
    - symbol: "SOL/USDT"
      min_qty: 0.01
      commission_pct: 0.1
      exchange: "binance"
```

**config/risk_params.yaml**:
```yaml
hard_limits:
  max_position_size_pct: 5.0
  max_portfolio_heat_pct: 6.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 15.0
  max_weekly_loss_pct: 5.0
  max_correlated_positions: 3
  max_trades_per_day_per_instrument: 10
  max_open_positions: 8
  min_risk_reward_ratio: 1.5
  no_trade_before_news_minutes: 15
  no_trade_session_end_minutes: 5
  forced_close_session_end: true

soft_limits:
  preferred_position_size_pct: 2.0
  preferred_portfolio_heat_pct: 4.0
  max_consecutive_losses_alert: 3
  max_consecutive_losses_reduce: 5
  reduce_size_factor: 0.5
  correlation_warning_threshold: 0.6
  min_minutes_between_trades: 5

position_sizing:
  method: "half_kelly"        # kelly | half_kelly | fixed_fractional | atr_based
  fixed_fraction_pct: 2.0     # Used when method = fixed_fractional
  kelly_lookback_trades: 50   # Trades to compute Kelly from
  atr_risk_multiple: 2.0      # Stop = entry ± ATR * multiple
  min_position_size_usd: 100
  max_position_size_usd: 50000
```

### 2.3 Shared Types

Create `src/types.py` — all shared data classes:

```python
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
```

---

## 3. Step 2: Data Layer — Market Data

### 3.1 Market Data Fetcher (`src/data/market_data.py`)

**Requirements**:
- Unified interface for fetching OHLCV bars from multiple sources
- Support both historical (backfill) and real-time (streaming) modes
- Automatic source selection based on instrument type
- Data normalization to `Bar` dataclass
- Rate limiting and error retry with exponential backoff
- Data gap detection and fill

**Source Routing**:
```python
SOURCE_MAP = {
    "forex":     ["yfinance", "mt5"],        # yfinance for history, MT5 for live
    "futures":   ["yfinance", "ib"],
    "us_stocks": ["yfinance", "alpaca"],
    "hk_stocks": ["yfinance", "futu"],
    "crypto":    ["ccxt"],                    # ccxt wraps Binance/OKX
}
```

**Key Implementation Details**:
- Use `asyncio` for all I/O operations
- `yfinance` for historical data (free, reliable for daily/intraday)
- `ccxt` for crypto (async mode: `ccxt.async_support`)
- Cache fetched data in DuckDB automatically
- Return `polars.DataFrame` for performance, with method to convert to pandas

**Interface**:
```python
class MarketDataFetcher:
    def __init__(self, config: dict, store: DataStore)
    
    async def get_historical(
        self, instrument: str, timeframe: str,
        start: datetime, end: datetime
    ) -> pl.DataFrame
    
    async def get_latest_bars(
        self, instrument: str, timeframe: str, count: int = 100
    ) -> pl.DataFrame
    
    async def subscribe(
        self, instrument: str, timeframe: str,
        callback: Callable[[Bar], Awaitable[None]]
    ) -> None
    
    async def unsubscribe(self, instrument: str, timeframe: str) -> None
    
    def get_instrument_type(self, instrument: str) -> str
    # Returns "forex" | "futures" | "us_stocks" | "hk_stocks" | "crypto"
```

### 3.2 Data Store (`src/data/data_store.py`)

**Technology**: DuckDB (embedded analytical DB)

**Schema**:
```sql
CREATE TABLE IF NOT EXISTS bars (
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    instrument VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    source VARCHAR,
    PRIMARY KEY (instrument, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS news (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    title VARCHAR NOT NULL,
    summary TEXT,
    source VARCHAR,
    url VARCHAR,
    relevance_score DOUBLE,
    instruments VARCHAR[],      -- Array of affected instruments
    sentiment_score DOUBLE,     -- Computed score (-1 to 1)
    raw_text TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    instrument VARCHAR NOT NULL,
    technical_score DOUBLE,
    sentiment_score DOUBLE,
    regime_score DOUBLE,
    claude_confidence DOUBLE,
    composite_score DOUBLE,
    action VARCHAR,
    recommendation_json TEXT    -- Full TradeRecommendation as JSON
);

CREATE TABLE IF NOT EXISTS trades (
    id VARCHAR PRIMARY KEY,
    instrument VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    entry_price DOUBLE,
    exit_price DOUBLE,
    quantity DOUBLE,
    pnl DOUBLE,
    pnl_pct DOUBLE,
    commission_total DOUBLE,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    duration_minutes DOUBLE,
    exit_reason VARCHAR,
    signal_score DOUBLE,
    claude_reasoning_entry TEXT,
    claude_reasoning_exit TEXT,
    technical_at_entry TEXT,     -- JSON
    regime_at_entry VARCHAR,
    news_at_entry TEXT,          -- JSON array
    factor_attribution TEXT,     -- JSON
    lessons_learned TEXT
);

CREATE TABLE IF NOT EXISTS economic_events (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    name VARCHAR NOT NULL,
    country VARCHAR,
    impact VARCHAR,
    forecast VARCHAR,
    previous VARCHAR,
    actual VARCHAR
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    timestamp TIMESTAMP PRIMARY KEY,
    equity DOUBLE,
    cash DOUBLE,
    unrealized_pnl DOUBLE,
    realized_pnl DOUBLE,
    daily_pnl DOUBLE,
    max_drawdown DOUBLE,
    positions_json TEXT          -- JSON of all positions
);
```

**Interface**:
```python
class DataStore:
    def __init__(self, db_path: str = "./data/atlas.duckdb")
    
    # Bars
    def save_bars(self, df: pl.DataFrame) -> int
    def get_bars(self, instrument, timeframe, start, end) -> pl.DataFrame
    def get_latest_bars(self, instrument, timeframe, count) -> pl.DataFrame
    
    # News
    def save_news(self, items: list[NewsItem]) -> int
    def get_recent_news(self, instrument, hours=24) -> list[NewsItem]
    
    # Signals
    def save_signal(self, signal: SignalScore) -> None
    def get_signals(self, instrument, start, end) -> list[SignalScore]
    
    # Trades
    def save_trade(self, trade: TradeRecord) -> None
    def get_trades(self, start=None, end=None, instrument=None) -> list[TradeRecord]
    
    # Portfolio
    def save_portfolio_snapshot(self, state: PortfolioState) -> None
    def get_portfolio_history(self, start, end) -> pl.DataFrame
    
    # Cleanup
    def vacuum(self) -> None
```

---

## 4. Step 3: News Feed + Economic Calendar

### 4.1 News Feed (`src/data/news_feed.py`)

**Sources**:
1. **NewsAPI** — `/v2/everything` endpoint, financial keywords
2. **RSS Feeds** — Reuters, Bloomberg, CNBC, FT (via feedparser)
3. **Alpha Vantage** — Market news sentiment endpoint (free tier)

**Fetching Logic**:
- Fetch every `analysis_interval_seconds` (default 5 min)
- Deduplicate by title similarity (>90% fuzzy match = duplicate)
- Pre-filter by relevance: title must contain instrument name or related keyword
- Store raw text for Claude analysis

**Keyword Map** (for relevance filtering):
```python
INSTRUMENT_KEYWORDS = {
    "EURUSD": ["euro", "ecb", "eurozone", "eur/usd", "eurusd"],
    "GBPUSD": ["pound", "sterling", "boe", "bank of england", "gbp"],
    "USDJPY": ["yen", "boj", "bank of japan", "jpy"],
    "ES": ["s&p", "sp500", "s&p 500", "us stocks", "wall street"],
    "NQ": ["nasdaq", "tech stocks", "qqq"],
    "GC": ["gold", "xau", "precious metals", "gold price"],
    "CL": ["crude", "oil", "wti", "brent", "opec"],
    "BTC/USDT": ["bitcoin", "btc", "crypto"],
    "ETH/USDT": ["ethereum", "eth"],
    # ... extend for all instruments
}
```

### 4.2 Economic Calendar (`src/data/economic_calendar.py`)

**Source**: Free API — `https://nfs.faireconomy.media/ff_calendar_thisweek.json` (ForexFactory data) or Trading Economics RSS.

**Usage**: Pre-fetch weekly calendar. Before every trade decision, check if a HIGH-impact event is within `no_trade_before_news_minutes`.

---

## 5. Step 4: Technical Indicator Engine

### 5.1 Implementation (`src/analysis/technical.py`)

**Full Indicator List**:

```python
INDICATOR_CONFIG = {
    # Trend
    "sma_20":    {"func": "sma", "period": 20},
    "sma_50":    {"func": "sma", "period": 50},
    "sma_200":   {"func": "sma", "period": 200},
    "ema_9":     {"func": "ema", "period": 9},
    "ema_21":    {"func": "ema", "period": 21},
    "macd":      {"func": "macd", "fast": 12, "slow": 26, "signal": 9},
    "adx_14":    {"func": "adx", "period": 14},
    
    # Momentum
    "rsi_14":    {"func": "rsi", "period": 14},
    "stoch":     {"func": "stoch", "k": 14, "d": 3, "smooth": 3},
    "cci_20":    {"func": "cci", "period": 20},
    "mfi_14":    {"func": "mfi", "period": 14},
    "willr_14":  {"func": "willr", "period": 14},
    
    # Volatility
    "atr_14":    {"func": "atr", "period": 14},
    "bbands":    {"func": "bbands", "period": 20, "std": 2},
    "keltner":   {"func": "kc", "period": 20, "atr_period": 10},
    
    # Volume
    "vwap":      {"func": "vwap"},
    "obv":       {"func": "obv"},
    
    # Structure
    "pivots":    {"func": "pivot_points"},
    "donchian":  {"func": "donchian", "period": 20},
}
```

**Signal Generation Rules**:
```python
def generate_signals(indicators: dict) -> dict[str, str]:
    signals = {}
    
    # RSI
    rsi = indicators.get("rsi_14", 50)
    if rsi > 70: signals["rsi"] = "overbought"
    elif rsi < 30: signals["rsi"] = "oversold"
    else: signals["rsi"] = "neutral"
    
    # MACD
    macd_val = indicators.get("macd", 0)
    macd_signal = indicators.get("macd_signal", 0)
    macd_hist = indicators.get("macd_hist", 0)
    if macd_val > macd_signal and macd_hist > 0:
        signals["macd"] = "bullish"
    elif macd_val < macd_signal and macd_hist < 0:
        signals["macd"] = "bearish"
    else:
        signals["macd"] = "neutral"
    
    # Moving Average alignment
    sma20 = indicators.get("sma_20", 0)
    sma50 = indicators.get("sma_50", 0)
    sma200 = indicators.get("sma_200", 0)
    if sma20 > sma50 > sma200:
        signals["ma_alignment"] = "strong_bullish"
    elif sma20 < sma50 < sma200:
        signals["ma_alignment"] = "strong_bearish"
    else:
        signals["ma_alignment"] = "mixed"
    
    # Bollinger Bands
    close = indicators.get("close", 0)
    bb_upper = indicators.get("bbands_upper", 0)
    bb_lower = indicators.get("bbands_lower", 0)
    bb_mid = indicators.get("bbands_mid", 0)
    if close > bb_upper:
        signals["bbands"] = "overbought"
    elif close < bb_lower:
        signals["bbands"] = "oversold"
    else:
        signals["bbands"] = "neutral"
    
    # ADX trend strength
    adx = indicators.get("adx_14", 0)
    if adx > 25: signals["adx"] = "trending"
    elif adx > 15: signals["adx"] = "weak_trend"
    else: signals["adx"] = "ranging"
    
    # Volume confirmation
    obv_slope = indicators.get("obv_slope", 0)  # computed as 5-bar OBV SMA slope
    if obv_slope > 0: signals["volume"] = "confirming_up"
    elif obv_slope < 0: signals["volume"] = "confirming_down"
    else: signals["volume"] = "neutral"
    
    return signals
```

**Trend Bias Calculation**:
```python
def compute_trend_bias(signals: dict) -> str:
    bullish_count = sum(1 for v in signals.values() if "bullish" in v or "oversold" in v or "confirming_up" in v)
    bearish_count = sum(1 for v in signals.values() if "bearish" in v or "overbought" in v or "confirming_down" in v)
    total = len(signals)
    
    if bullish_count / total > 0.6: return "bullish"
    elif bearish_count / total > 0.6: return "bearish"
    else: return "neutral"
```

**Multi-Timeframe Analysis**:
```python
async def multi_timeframe_analysis(
    self, instrument: str, timeframes: list[str] = ["5m", "15m", "1h"]
) -> dict[str, TechnicalSnapshot]:
    """Compute indicators on all timeframes. Higher TF takes precedence for trend."""
    results = {}
    for tf in timeframes:
        bars = await self.data_fetcher.get_latest_bars(instrument, tf, count=250)
        results[tf] = self.compute_all(bars, instrument, tf)
    return results
```

---

## 6. Step 5: Claude API Agent

### 6.1 Implementation (`src/analysis/claude_agent.py`)

**System Prompt**:
```python
SYSTEM_PROMPT = """You are an expert intraday trader operating an automated trading system.

ROLE: Analyze market data and provide structured trading recommendations.

INPUT YOU RECEIVE:
- Technical indicator snapshot (RSI, MACD, MA alignment, ATR, Bollinger Bands, etc.)
- Multi-timeframe analysis (5m, 15m, 1h)
- Recent news headlines with relevance scores
- Current market regime (trend/range/breakout/crisis)
- Portfolio state (open positions, daily P&L, exposure)
- Recent trade history (last 10 trades with outcomes)

OUTPUT FORMAT (strict JSON):
{
  "action": "BUY" | "SELL" | "NO_TRADE",
  "confidence": 0-100,
  "entry_price": <float>,
  "stop_loss": <float>,
  "take_profit_1": <float>,
  "take_profit_2": <float or null>,
  "risk_reward_ratio": <float>,
  "position_size_pct": <float 0.5-5.0>,
  "reasoning": "<2-3 sentence explanation>",
  "key_factors": ["factor1", "factor2", "factor3"],
  "time_horizon": "scalp" | "intraday" | "swing",
  "regime_assessment": "<current regime and whether it favors this trade>",
  "risk_warnings": ["warning1", "warning2"]
}

RULES:
1. CAPITAL PRESERVATION IS PRIORITY #1. If uncertain, output NO_TRADE.
2. Never recommend risk:reward below 1:1.5.
3. Always specify a stop_loss — no exceptions.
4. Consider upcoming economic events.
5. If 3+ recent trades were losses, be MORE conservative (lower confidence, smaller size).
6. Confidence > 80 means "high conviction" — use sparingly.
7. Account for current portfolio exposure — don't overload one direction.
8. Output ONLY valid JSON, no markdown, no code blocks.
"""
```

**Analysis Prompt Builder**:
```python
def build_analysis_prompt(
    instrument: str,
    technical: dict[str, TechnicalSnapshot],   # multi-TF
    news: list[NewsItem],
    regime: MarketRegime,
    portfolio: PortfolioState,
    recent_trades: list[TradeRecord],
) -> str:
    prompt = f"""
INSTRUMENT: {instrument}
CURRENT TIME: {datetime.utcnow().isoformat()}

=== TECHNICAL ANALYSIS ===
"""
    for tf, snap in technical.items():
        prompt += f"\n[{tf}] Trend: {snap.trend_bias} | Vol: {snap.volatility_regime}\n"
        prompt += f"  Indicators: {json.dumps({k: round(v, 5) for k, v in snap.indicators.items()})}\n"
        prompt += f"  Signals: {json.dumps(snap.signals)}\n"
        if snap.patterns:
            prompt += f"  Patterns: {', '.join(snap.patterns)}\n"

    prompt += f"""
=== MARKET REGIME ===
Current: {regime.value}

=== RECENT NEWS ({len(news)} items) ===
"""
    for n in news[:10]:
        prompt += f"- [{n.timestamp.strftime('%H:%M')}] {n.title} (relevance: {n.relevance_score:.1f})\n"

    prompt += f"""
=== PORTFOLIO STATE ===
Equity: ${portfolio.equity:,.2f}
Daily P&L: ${portfolio.daily_pnl:,.2f} ({portfolio.daily_pnl/portfolio.equity*100:.1f}%)
Open positions: {len(portfolio.positions)}
Portfolio heat: {portfolio.portfolio_heat_pct:.1f}%
Current drawdown: {portfolio.max_drawdown_current:.1f}%

=== RECENT TRADES (last 5) ===
"""
    for t in recent_trades[-5:]:
        prompt += f"- {t.instrument} {t.side.value}: {'+' if t.pnl > 0 else ''}{t.pnl_pct:.1f}% ({t.exit_reason.value})\n"

    prompt += """
Based on all the above, provide your trading recommendation as JSON.
"""
    return prompt
```

**Reflection Prompt**:
```python
REFLECTION_PROMPT = """You are reviewing a completed trade. Analyze what happened and extract lessons.

TRADE DETAILS:
{trade_json}

MARKET CONTEXT AT ENTRY:
{entry_context}

MARKET CONTEXT AT EXIT:
{exit_context}

Provide your analysis as JSON:
{{
  "performance_grade": "A" | "B" | "C" | "D" | "F",
  "entry_timing": "excellent" | "good" | "fair" | "poor",
  "exit_timing": "excellent" | "good" | "fair" | "poor",
  "sizing_assessment": "appropriate" | "too_large" | "too_small",
  "what_worked": ["factor1", "factor2"],
  "what_failed": ["factor1", "factor2"],
  "lessons": "<1-2 sentence actionable lesson>",
  "factor_attribution": {{
    "technical": <0-1 float, how much technical analysis contributed>,
    "sentiment": <0-1 float>,
    "regime": <0-1 float>,
    "timing": <0-1 float>
  }}
}}
Output ONLY valid JSON.
"""
```

**Cost Control Implementation**:
```python
class CostTracker:
    def __init__(self, monthly_budget_usd: float):
        self.monthly_budget = monthly_budget_usd
        self.month_start = datetime.utcnow().replace(day=1)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    @property
    def estimated_cost(self) -> float:
        # Sonnet pricing (approximate)
        input_cost = self.total_input_tokens / 1_000_000 * 3.0
        output_cost = self.total_output_tokens / 1_000_000 * 15.0
        return input_cost + output_cost
    
    @property
    def budget_remaining_pct(self) -> float:
        return max(0, (1 - self.estimated_cost / self.monthly_budget) * 100)
    
    def can_make_call(self, estimated_tokens: int = 3000) -> bool:
        return self.estimated_cost < self.monthly_budget * 0.95
    
    def record_usage(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
```

---

## 7. Step 6: Regime Detector

### 7.1 Implementation (`src/analysis/regime_detector.py`)

**Feature Engineering**:
```python
def compute_regime_features(bars: pl.DataFrame) -> dict:
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    volume = bars["volume"].to_numpy()
    
    returns = np.diff(np.log(close))
    
    return {
        "atr_pct": compute_atr_pct(high, low, close, period=14),
        "adx": compute_adx(high, low, close, period=14),
        "volume_ratio": volume[-1] / np.mean(volume[-20:]),
        "return_skew": scipy.stats.skew(returns[-20:]),
        "return_kurt": scipy.stats.kurtosis(returns[-20:]),
        "bb_width": compute_bb_width(close, period=20, std=2),
        "ma_alignment_score": compute_ma_alignment(close),
        "consecutive_direction": compute_consecutive(returns),
    }
```

**Classification** (Rule-based first, upgrade to HMM in optimization):
```python
def classify_regime(features: dict) -> MarketRegime:
    adx = features["adx"]
    bb_width = features["bb_width"]
    volume_ratio = features["volume_ratio"]
    
    # Crisis detection (highest priority)
    if features["atr_pct"] > 3.0 and abs(features["return_skew"]) > 2.0:
        return MarketRegime.CRISIS
    
    # Breakout
    if bb_width < 0.02 and volume_ratio > 2.0:
        return MarketRegime.BREAKOUT
    
    # Strong trend
    if adx > 25:
        if features["ma_alignment_score"] > 0.5:
            return MarketRegime.STRONG_TREND_UP
        elif features["ma_alignment_score"] < -0.5:
            return MarketRegime.STRONG_TREND_DOWN
    
    # Weak trend
    if adx > 15:
        return MarketRegime.WEAK_TREND
    
    # Range
    return MarketRegime.RANGE_BOUND
```

---

## 8. Step 7: Signal Aggregator

### 8.1 Implementation (`src/strategy/signal_aggregator.py`)

**Technical Score Computation**:
```python
def compute_technical_score(snapshot: TechnicalSnapshot) -> float:
    """Convert technical signals to -1 to +1 score."""
    score = 0.0
    weights = {
        "rsi": 0.15, "macd": 0.20, "ma_alignment": 0.25,
        "bbands": 0.10, "adx": 0.10, "volume": 0.10,
        "stoch": 0.10
    }
    
    signal_values = {
        "strong_bullish": 1.0, "bullish": 0.6, "oversold": 0.5,
        "confirming_up": 0.4, "trending": 0.2, "neutral": 0.0,
        "ranging": -0.1, "confirming_down": -0.4, "overbought": -0.5,
        "bearish": -0.6, "strong_bearish": -1.0,
    }
    
    for signal_name, weight in weights.items():
        signal_val = snapshot.signals.get(signal_name, "neutral")
        score += signal_values.get(signal_val, 0.0) * weight
    
    return max(-1.0, min(1.0, score))
```

**Composite Score**:
```python
def compute_composite(
    technical_score: float,      # -1 to +1
    sentiment_score: float,      # -1 to +1 (from Claude)
    regime_score: float,         # -1 to +1
    claude_confidence: float,    # 0 to 1
    weights: dict
) -> tuple[float, TradeAction]:
    composite = (
        technical_score * weights["technical"] +
        sentiment_score * weights["sentiment"] +
        regime_score * weights["regime"] +
        (claude_confidence - 0.5) * 2 * weights["claude_confidence"]  # normalize to -1..+1
    )
    
    thresholds = weights["thresholds"]
    if composite >= thresholds["strong_buy"]:
        action = TradeAction.STRONG_BUY
    elif composite >= thresholds["buy"]:
        action = TradeAction.BUY
    elif composite <= thresholds["strong_sell"]:
        action = TradeAction.STRONG_SELL
    elif composite <= thresholds["sell"]:
        action = TradeAction.SELL
    else:
        action = TradeAction.NEUTRAL
    
    return composite, action
```

**Regime Score Logic**:
```python
def compute_regime_score(regime: MarketRegime, action: str) -> float:
    """How well does the proposed action align with the current regime?"""
    alignment = {
        MarketRegime.STRONG_TREND_UP:   {"BUY": 0.8, "SELL": -0.6, "NO_TRADE": 0.0},
        MarketRegime.STRONG_TREND_DOWN: {"BUY": -0.6, "SELL": 0.8, "NO_TRADE": 0.0},
        MarketRegime.WEAK_TREND:        {"BUY": 0.3, "SELL": 0.3, "NO_TRADE": 0.2},
        MarketRegime.RANGE_BOUND:       {"BUY": 0.4, "SELL": 0.4, "NO_TRADE": 0.1},
        MarketRegime.BREAKOUT:          {"BUY": 0.6, "SELL": 0.6, "NO_TRADE": -0.2},
        MarketRegime.CRISIS:            {"BUY": -0.8, "SELL": 0.3, "NO_TRADE": 0.9},
    }
    return alignment.get(regime, {}).get(action, 0.0)
```

---

## 9. Steps 8-9: Risk Manager + Position Sizer

### 9.1 Risk Manager (`src/strategy/risk_manager.py`)

**Pipeline**:
```python
class RiskManager:
    def check_signal(
        self, signal: SignalScore, portfolio: PortfolioState,
        calendar: list[EconomicEvent]
    ) -> RiskCheckResult:
        
        checks = []
        
        # 1. Daily loss check
        if abs(portfolio.daily_pnl / portfolio.equity) >= self.limits.max_daily_loss_pct / 100:
            return RiskCheckResult(approved=False, reason="Daily loss limit reached")
        checks.append("daily_loss_ok")
        
        # 2. Max drawdown check
        if portfolio.max_drawdown_current >= self.limits.max_drawdown_pct:
            return RiskCheckResult(approved=False, reason="Max drawdown kill switch")
        checks.append("drawdown_ok")
        
        # 3. Portfolio heat check
        if portfolio.portfolio_heat_pct >= self.limits.max_portfolio_heat_pct:
            return RiskCheckResult(approved=False, reason="Portfolio heat limit reached")
        checks.append("heat_ok")
        
        # 4. Max open positions
        if len(portfolio.positions) >= self.limits.max_open_positions:
            return RiskCheckResult(approved=False, reason="Max open positions reached")
        checks.append("positions_ok")
        
        # 5. Correlation check
        if self._has_correlated_position(signal.instrument, portfolio.positions):
            return RiskCheckResult(approved=False, reason="Correlated position exists")
        checks.append("correlation_ok")
        
        # 6. News embargo check
        if self._near_high_impact_event(calendar):
            return RiskCheckResult(approved=False, reason="High-impact event imminent")
        checks.append("news_ok")
        
        # 7. Risk:Reward check
        if signal.recommendation:
            rr = signal.recommendation.risk_reward_ratio
            if rr < self.limits.min_risk_reward_ratio:
                return RiskCheckResult(approved=False, reason=f"R:R {rr:.1f} below min {self.limits.min_risk_reward_ratio}")
        checks.append("rr_ok")
        
        # 8. Consecutive loss check (soft — reduce size)
        adjusted_size = None
        if self._consecutive_losses() >= self.soft_limits.max_consecutive_losses_reduce:
            adjusted_size = self.soft_limits.reduce_size_factor
        
        return RiskCheckResult(
            approved=True,
            checks_passed=checks,
            adjusted_size=adjusted_size
        )
```

### 9.2 Position Sizer (`src/strategy/position_sizer.py`)

```python
class PositionSizer:
    def calculate(
        self, signal: SignalScore, portfolio: PortfolioState,
        instrument_config: dict, current_atr: float
    ) -> PositionSize:
        
        method = self.config.method
        equity = portfolio.equity
        rec = signal.recommendation
        
        if method == "half_kelly":
            trades = self.store.get_trades(limit=self.config.kelly_lookback)
            wins = [t for t in trades if t.pnl > 0]
            if len(trades) < 10:
                risk_pct = self.config.fixed_fraction_pct / 100
            else:
                win_rate = len(wins) / len(trades)
                avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
                avg_loss = abs(np.mean([t.pnl_pct for t in trades if t.pnl <= 0])) or 1
                b = avg_win / avg_loss
                kelly = (b * win_rate - (1 - win_rate)) / b
                risk_pct = max(0.005, min(kelly * 0.5, self.limits.max_position_size_pct / 100))
        
        elif method == "atr_based":
            stop_distance = current_atr * self.config.atr_risk_multiple
            risk_amount = equity * (self.config.fixed_fraction_pct / 100)
            units = risk_amount / stop_distance
            risk_pct = (units * rec.entry_price) / equity
        
        else:  # fixed_fractional
            risk_pct = self.config.fixed_fraction_pct / 100
        
        # Apply hard limits
        risk_pct = min(risk_pct, self.limits.max_position_size_pct / 100)
        risk_amount = equity * risk_pct
        
        # Convert to units
        if rec and rec.stop_loss and rec.entry_price:
            stop_distance = abs(rec.entry_price - rec.stop_loss)
            units = risk_amount / stop_distance if stop_distance > 0 else 0
        else:
            units = risk_amount / rec.entry_price if rec else 0
        
        notional = units * (rec.entry_price if rec else 0)
        
        return PositionSize(
            instrument=signal.instrument,
            units=round(units, 4),
            notional_value=round(notional, 2),
            risk_amount=round(risk_amount, 2),
            risk_pct_of_equity=round(risk_pct * 100, 2),
            method_used=method,
            kelly_fraction=risk_pct if method == "half_kelly" else None
        )
```

---

## 10. Step 10: Paper Trading Engine

### 10.1 Implementation (`src/execution/paper_engine.py`)

```python
class PaperTradingEngine:
    """Simulates realistic order execution for paper trading."""
    
    def __init__(self, config: dict, store: DataStore):
        self.slippage_mean = config.get("slippage_mean_pct", 0.02)
        self.slippage_std = config.get("slippage_std_pct", 0.01)
        self.latency_ms = config.get("latency_ms", 100)
        self.partial_fill_prob = config.get("partial_fill_prob", 0.1)
        self.portfolio = PortfolioState(
            equity=config["initial_capital"],
            cash=config["initial_capital"],
            total_unrealized_pnl=0,
            total_realized_pnl=0,
        )
        self.equity_high_water = config["initial_capital"]
    
    async def execute_order(self, order: Order, current_bar: Bar) -> Order:
        """Simulate order execution with realistic fill."""
        await asyncio.sleep(self.latency_ms / 1000)  # Simulate latency
        
        if order.order_type == OrderType.MARKET:
            slippage = np.random.normal(self.slippage_mean, self.slippage_std) / 100
            if order.side == Side.BUY:
                fill_price = current_bar.close * (1 + slippage)
            else:
                fill_price = current_bar.close * (1 - slippage)
            
            order.fill_price = round(fill_price, 6)
            order.slippage = round(abs(fill_price - current_bar.close), 6)
            order.filled_at = datetime.utcnow()
            order.fill_quantity = order.quantity
            order.status = OrderStatus.FILLED
            order.commission = self._calc_commission(order)
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders fill only if price reaches limit
            if order.side == Side.BUY and current_bar.low <= order.price:
                order.fill_price = order.price
                order.status = OrderStatus.FILLED
            elif order.side == Side.SELL and current_bar.high >= order.price:
                order.fill_price = order.price
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PENDING
                return order
            
            order.filled_at = datetime.utcnow()
            order.fill_quantity = order.quantity
            order.slippage = 0.0
            order.commission = self._calc_commission(order)
        
        # Update portfolio
        if order.status == OrderStatus.FILLED:
            self._update_portfolio(order)
        
        return order
    
    def check_stops(self, current_bar: Bar) -> list[Order]:
        """Check all open positions for stop-loss / take-profit hits."""
        closed_orders = []
        for pos in self.portfolio.positions[:]:
            # Stop loss
            if pos.stop_loss:
                if pos.side == Side.BUY and current_bar.low <= pos.stop_loss:
                    order = self._create_close_order(pos, pos.stop_loss, ExitReason.STOP_LOSS)
                    closed_orders.append(order)
                elif pos.side == Side.SELL and current_bar.high >= pos.stop_loss:
                    order = self._create_close_order(pos, pos.stop_loss, ExitReason.STOP_LOSS)
                    closed_orders.append(order)
            
            # Take profit
            if pos.take_profit:
                if pos.side == Side.BUY and current_bar.high >= pos.take_profit:
                    order = self._create_close_order(pos, pos.take_profit, ExitReason.TAKE_PROFIT)
                    closed_orders.append(order)
                elif pos.side == Side.SELL and current_bar.low <= pos.take_profit:
                    order = self._create_close_order(pos, pos.take_profit, ExitReason.TAKE_PROFIT)
                    closed_orders.append(order)
        
        return closed_orders
    
    def update_mark_to_market(self, current_prices: dict[str, float]):
        """Update unrealized P&L for all positions."""
        total_unrealized = 0.0
        for pos in self.portfolio.positions:
            price = current_prices.get(pos.instrument, pos.current_price)
            pos.current_price = price
            if pos.side == Side.BUY:
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity
            total_unrealized += pos.unrealized_pnl
        
        self.portfolio.total_unrealized_pnl = total_unrealized
        self.portfolio.equity = self.portfolio.cash + total_unrealized
        
        # Update drawdown
        if self.portfolio.equity > self.equity_high_water:
            self.equity_high_water = self.portfolio.equity
        dd = (self.equity_high_water - self.portfolio.equity) / self.equity_high_water * 100
        self.portfolio.max_drawdown_current = dd
        
        # Update heat
        total_risk = sum(
            abs(p.entry_price - p.stop_loss) * p.quantity
            for p in self.portfolio.positions if p.stop_loss
        )
        self.portfolio.portfolio_heat_pct = total_risk / self.portfolio.equity * 100 if self.portfolio.equity > 0 else 0
```

---

## 11. Step 12: Main Trading Loop

### 11.1 Implementation (`scripts/run_paper.py`)

```python
async def trading_loop(config):
    """Main event loop for paper trading."""
    
    # Initialize modules
    store = DataStore(config["db_path"])
    data_fetcher = MarketDataFetcher(config, store)
    technical = TechnicalEngine()
    claude_agent = ClaudeAgent(config["claude"])
    regime_detector = RegimeDetector()
    signal_agg = SignalAggregator(config["signal_weights"])
    risk_mgr = RiskManager(config["risk_params"])
    position_sizer = PositionSizer(config["position_sizing"], risk_mgr.limits, store)
    paper_engine = PaperTradingEngine(config["paper"], store)
    journal = TradeJournal(store, claude_agent)
    news_feed = NewsFeed(config["news"])
    calendar = EconomicCalendar()
    
    instruments = config["instruments"]
    interval = config["trading"]["analysis_interval_seconds"]
    
    log.info("Atlas Trader starting", mode="paper", instruments=instruments)
    
    while True:
        cycle_start = datetime.utcnow()
        
        for instrument in instruments:
            try:
                # 1. PERCEIVE — Gather data
                bars_mt = await technical.multi_timeframe_analysis_async(
                    instrument, data_fetcher, ["5m", "15m", "1h"]
                )
                news = await news_feed.get_recent(instrument, hours=4)
                events = calendar.upcoming_events(hours=2)
                current_bar = await data_fetcher.get_latest_bars(instrument, "5m", count=1)
                
                # 2. ANALYZE — Compute regime + Claude analysis
                regime = regime_detector.classify(bars_mt["1h"])
                
                # Claude analysis (if budget allows)
                recommendation = None
                if claude_agent.cost_tracker.can_make_call():
                    recommendation = await claude_agent.analyze_market(
                        instrument=instrument,
                        technical=bars_mt,
                        news=news,
                        regime=regime,
                        recent_trades=store.get_trades(limit=10),
                        portfolio_state=paper_engine.portfolio,
                    )
                
                # 3. DECIDE — Score and aggregate signals
                tech_score = signal_agg.compute_technical_score(bars_mt["15m"])
                sentiment_score = recommendation.confidence / 100 * (
                    1 if recommendation.action in [TradeAction.BUY, TradeAction.STRONG_BUY] else -1
                ) if recommendation else 0
                
                regime_action = recommendation.action.value if recommendation else "NO_TRADE"
                regime_score = signal_agg.compute_regime_score(regime, regime_action)
                claude_conf = recommendation.confidence / 100 if recommendation else 0.5
                
                composite, action = signal_agg.compute_composite(
                    tech_score, sentiment_score, regime_score, claude_conf
                )
                
                signal = SignalScore(
                    instrument=instrument,
                    timestamp=datetime.utcnow(),
                    technical_score=tech_score,
                    sentiment_score=sentiment_score,
                    regime_score=regime_score,
                    claude_confidence=claude_conf,
                    composite_score=composite,
                    action=action,
                    recommendation=recommendation,
                )
                store.save_signal(signal)
                
                # Skip if no actionable signal
                if action in [TradeAction.NEUTRAL, TradeAction.NO_TRADE]:
                    continue
                
                # 4. RISK CHECK
                risk_result = risk_mgr.check_signal(
                    signal, paper_engine.portfolio, events
                )
                if not risk_result.approved:
                    log.info("Signal vetoed", instrument=instrument, reason=risk_result.reason)
                    continue
                
                # 5. SIZE POSITION
                atr = bars_mt["15m"].indicators.get("atr_14", 0)
                size = position_sizer.calculate(
                    signal, paper_engine.portfolio, {}, atr
                )
                
                if risk_result.adjusted_size:
                    size.units *= risk_result.adjusted_size
                    size.risk_amount *= risk_result.adjusted_size
                
                # 6. EXECUTE (paper)
                order = Order(
                    id=f"ORD-{uuid4().hex[:8]}",
                    instrument=instrument,
                    side=Side.BUY if action in [TradeAction.BUY, TradeAction.STRONG_BUY] else Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=size.units,
                    stop_loss=recommendation.stop_loss if recommendation else None,
                    take_profit=recommendation.take_profit_1 if recommendation else None,
                    signal_id=f"SIG-{signal.timestamp.strftime('%H%M%S')}",
                    claude_reasoning=recommendation.reasoning if recommendation else "",
                )
                
                result = await paper_engine.execute_order(order, current_bar.row(0))
                log.info("Order executed", order_id=result.id, status=result.status.value)
                
            except Exception as e:
                log.error("Error in trading loop", instrument=instrument, error=str(e))
        
        # Check stops for all positions
        for instrument in set(p.instrument for p in paper_engine.portfolio.positions):
            bar = await data_fetcher.get_latest_bars(instrument, "5m", count=1)
            closed = paper_engine.check_stops(bar.row(0))
            for order in closed:
                # 7. REFLECT — Journal closed trades
                await journal.record_trade(order, paper_engine.portfolio)
        
        # Update mark-to-market
        prices = {}
        for instrument in instruments:
            bar = await data_fetcher.get_latest_bars(instrument, "5m", count=1)
            prices[instrument] = bar["close"][0]
        paper_engine.update_mark_to_market(prices)
        
        # Save portfolio snapshot
        store.save_portfolio_snapshot(paper_engine.portfolio)
        
        # Wait for next cycle
        elapsed = (datetime.utcnow() - cycle_start).total_seconds()
        sleep_time = max(0, interval - elapsed)
        await asyncio.sleep(sleep_time)
```

---

## 12. Step 13: Backtesting Framework

### 12.1 Implementation (`scripts/run_backtest.py`)

**Walk-Forward Optimization**:
```
|--- Train Window (90 days) ---|--- Test Window (30 days) ---|
                               |--- Train (90) ---|--- Test (30) ---|
                                                   |--- Train (90) ---|--- Test (30) ---|
```

The backtester replays historical bars through the same pipeline (technical → Claude mock → signal → risk → paper engine) but uses cached/mocked Claude responses to avoid API costs during backtesting.

**Claude Mock for Backtesting**:
- Option A: Use cached responses from similar market contexts
- Option B: Use a lightweight local model (e.g., rule-based sentiment)
- Option C: Run Claude on a sample (10%) and interpolate

**Metrics Computed**:
```python
@dataclass
class BacktestResults:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float        # gross profit / gross loss
    sharpe_ratio: float         # annualized
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: float
    calmar_ratio: float         # annual return / max DD
    avg_trade_duration_min: float
    avg_risk_reward_achieved: float
    monthly_returns: list[float]
    equity_curve: list[float]
    trade_log: list[TradeRecord]
```

---

## 13. Step 14: Dashboard

### 13.1 Streamlit Dashboard (`dashboard/app.py`)

**Pages**:
1. **Overview** — Equity curve, key metrics (Sharpe, DD, win rate, profit factor), daily P&L bar chart
2. **Signals** — Live signal scores for each instrument, factor breakdown radar chart
3. **Positions** — Open positions table, unrealized P&L, heat gauge
4. **Risk** — Portfolio heat meter, drawdown chart, daily loss tracker
5. **Journal** — Trade log with Claude reasoning, filterable by instrument/outcome
6. **Backtest** — Backtest results, parameter comparison, walk-forward chart

**Auto-refresh**: Every 60 seconds during paper trading

---

## 14. Testing Checklist

```
[ ] Technical engine: RSI(14) of known series matches expected value
[ ] Technical engine: MACD crossover detection on synthetic data
[ ] Claude agent: Parses valid JSON from mock response
[ ] Claude agent: Handles API errors gracefully (timeout, rate limit)
[ ] Claude agent: Cost tracker blocks calls when budget exceeded
[ ] Regime detector: Correctly classifies trending vs ranging synthetic data
[ ] Signal aggregator: Composite score within [-1, +1] range
[ ] Signal aggregator: STRONG_BUY only when composite > threshold
[ ] Risk manager: Rejects trade when daily loss limit reached
[ ] Risk manager: Rejects trade when max drawdown hit
[ ] Risk manager: Reduces size after consecutive losses
[ ] Risk manager: Blocks trade before high-impact news
[ ] Position sizer: Half-Kelly output between 0.5% and max_position_size
[ ] Position sizer: ATR-based size inversely proportional to volatility
[ ] Paper engine: Market order fills with realistic slippage
[ ] Paper engine: Limit order only fills when price crosses
[ ] Paper engine: Stop-loss triggered correctly on bar low/high
[ ] Paper engine: Portfolio equity tracks correctly across trades
[ ] Paper engine: Drawdown calculation correct
[ ] Full pipeline: Signal → Risk → Size → Execute → Journal → Snapshot
[ ] Backtest: Walk-forward produces per-window Sharpe ratios
[ ] Backtest: Metrics match manual calculation on small dataset
```
