# Agentic AI Trading System — Project Specification

> **Codename**: `atlas-trader`
> **Version**: 0.1.0
> **Last Updated**: 2026-03-31
> **Designed for**: Claude Code implementation

---

## 1. Project Overview

### 1.1 Vision

Build an Agentic AI automated trading system that uses Claude API as the cognitive core, combining technical indicators and news sentiment analysis for multi-asset intraday trading. The system evolves across three phases: simulation → live execution → multi-agent orchestration.

### 1.2 First Principles

The system is grounded in three market axioms:

- **Price Discovery**: Prices reflect supply/demand equilibrium. The system seeks to understand supply/demand shifts faster than the market.
- **Information Asymmetry**: Profitability comes from synthesizing multi-source information (technical, fundamental, sentiment) into a unified edge.
- **Non-linear Risk**: Tail events dominate P&L. Risk management is not optional — it is the first module built and the last to be overridden.

### 1.3 Agent Cognitive Loop

Every trading decision follows a 5-step loop:

```
Perceive → Analyze → Decide → Execute → Reflect
    ↑                                        |
    └────────── feedback loop ───────────────┘
```

### 1.4 Target Markets

| Market       | Instruments             | Data Source           | Broker/Exchange     |
|--------------|-------------------------|-----------------------|---------------------|
| Forex        | Major & cross pairs     | MT5 API / OANDA       | MT5 / cTrader       |
| Futures      | Indices, commodities    | IB API / CQG          | Interactive Brokers  |
| US Stocks    | Large-cap equities      | Alpaca / IB           | Alpaca / IB          |
| HK Stocks    | HSI components          | IB / Futu             | IB / Futu OpenD      |
| Crypto       | BTC, ETH, major alts    | ccxt (Binance/OKX)    | Binance / OKX        |

### 1.5 Trading Frequency

- **Timeframe**: Intraday (minutes to hours)
- **Holding period**: 5 minutes to 8 hours
- **Max trades per day per instrument**: 10
- **Trading sessions**: 24/5 for Forex, market hours for stocks

---

## 2. Three-Phase Roadmap

### Phase 1: Simulation + Claude API + Signal Fusion (Weeks 1-8)

**Goal**: Prove positive expected value on real data without risking capital.

**Deliverables**:
- Real-time data pipeline (price, volume, news)
- Technical indicator engine (15+ indicators)
- Claude API integration for market analysis and sentiment
- Multi-factor signal aggregator with weighted scoring
- Paper trading engine with realistic simulation (slippage, fees)
- Backtesting framework with walk-forward optimization
- Performance dashboard (Sharpe, drawdown, win rate, P&L curve)

**Exit Criteria**: 3-month simulation, Sharpe Ratio > 1.5, Max Drawdown < 15%

### Phase 2: Live Execution + Trade Journal + Attribution (Weeks 9-16)

**Goal**: Transition to real money with full audit trail.

**Deliverables**:
- Unified exchange adapter layer (single interface, multiple brokers)
- Smart order execution engine (limit-first, slippage control)
- Automated trade journal (every trade with entry reason, screenshots, indicators)
- Attribution analysis engine (what factor drove each trade's P&L)
- Incremental position sizing (start small, scale on consistency)
- Real-time alerting (Telegram/Discord/Email)

**Exit Criteria**: 3-month live profit, Max Drawdown < 15%, > 100 trades logged

### Phase 3: Multi-Agent Orchestration (Weeks 17+)

**Goal**: Run multiple independent strategies with unified risk management.

**Deliverables**:
- Specialist agents (Momentum, Mean Reversion, Event-Driven, Stat Arb)
- Orchestrator agent (capital allocation, conflict resolution)
- Cross-strategy correlation monitoring
- Global risk manager (portfolio-level exposure limits)
- Strategy evolution loop (weekly performance review + parameter tuning)
- Full monitoring dashboard (per-agent P&L, system health, alerts)

**Exit Criteria**: Portfolio Sharpe > 2.0, strategies show low correlation (< 0.3)

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                            │
│  Market Data │ News Feed │ Econ Calendar │ Social Sent.  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 PROCESSING LAYER                         │
│  Technical Engine │ Claude API Agent │ Regime Detector   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 DECISION LAYER                            │
│  Signal Aggregator → Position Sizer → Risk Filter        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 EXECUTION LAYER                           │
│  Order Manager → Exchange Adapter → Fill Tracker         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 REFLECTION LAYER                          │
│  Trade Journal → Attribution Engine → Strategy Evolver   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
atlas-trader/
├── README.md
├── PROJECT.md                    # This file
├── pyproject.toml                # Project config + dependencies
├── .env.example                  # Environment variables template
│
├── config/
│   ├── settings.yaml             # Global settings (timeframes, limits)
│   ├── instruments.yaml          # Tradeable instruments config
│   ├── risk_params.yaml          # Risk management parameters
│   └── api_keys.yaml.example     # API keys template (gitignored)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                     # DATA LAYER
│   │   ├── __init__.py
│   │   ├── market_data.py        # Price/volume data fetcher
│   │   ├── news_feed.py          # News API integration
│   │   ├── economic_calendar.py  # Econ events fetcher
│   │   ├── social_sentiment.py   # Social media sentiment
│   │   └── data_store.py         # Local data cache (SQLite/DuckDB)
│   │
│   ├── analysis/                 # PROCESSING LAYER
│   │   ├── __init__.py
│   │   ├── technical.py          # Technical indicator engine
│   │   ├── claude_agent.py       # Claude API integration
│   │   ├── regime_detector.py    # Market regime classification
│   │   └── sentiment_scorer.py   # Unified sentiment scoring
│   │
│   ├── strategy/                 # DECISION LAYER
│   │   ├── __init__.py
│   │   ├── signal_aggregator.py  # Multi-factor signal scoring
│   │   ├── position_sizer.py     # Kelly criterion + vol adjustment
│   │   ├── risk_manager.py       # Risk rules enforcement
│   │   └── strategies/           # Phase 3: Individual strategies
│   │       ├── __init__.py
│   │       ├── momentum.py
│   │       ├── mean_reversion.py
│   │       ├── event_driven.py
│   │       └── stat_arb.py
│   │
│   ├── execution/                # EXECUTION LAYER
│   │   ├── __init__.py
│   │   ├── order_manager.py      # Order lifecycle management
│   │   ├── paper_engine.py       # Paper trading simulator
│   │   ├── exchange_adapter.py   # Abstract exchange interface
│   │   └── adapters/             # Concrete exchange adapters
│   │       ├── __init__.py
│   │       ├── alpaca_adapter.py
│   │       ├── ib_adapter.py
│   │       ├── mt5_adapter.py
│   │       ├── binance_adapter.py
│   │       └── okx_adapter.py
│   │
│   ├── reflection/               # REFLECTION LAYER
│   │   ├── __init__.py
│   │   ├── trade_journal.py      # Trade logging + snapshots
│   │   ├── attribution.py        # P&L factor attribution
│   │   └── strategy_evolver.py   # Parameter optimization loop
│   │
│   ├── orchestrator/             # Phase 3: MULTI-AGENT
│   │   ├── __init__.py
│   │   ├── orchestrator.py       # Capital allocation + conflict
│   │   ├── agent_registry.py     # Agent lifecycle management
│   │   └── correlation_monitor.py # Cross-strategy correlation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py             # Structured logging
│       ├── notifier.py           # Telegram/Discord alerts
│       ├── metrics.py            # Performance metric calculators
│       └── time_utils.py         # Timezone + market hours
│
├── dashboard/                    # Web dashboard (Phase 1+)
│   ├── app.py                    # Streamlit/Dash app
│   └── components/
│       ├── pnl_chart.py
│       ├── signal_monitor.py
│       └── risk_panel.py
│
├── tests/
│   ├── test_technical.py
│   ├── test_claude_agent.py
│   ├── test_signal_aggregator.py
│   ├── test_risk_manager.py
│   ├── test_paper_engine.py
│   └── test_attribution.py
│
├── scripts/
│   ├── run_backtest.py           # Run historical backtest
│   ├── run_paper.py              # Run paper trading
│   ├── run_live.py               # Run live trading (Phase 2)
│   └── run_multi_agent.py        # Run multi-agent (Phase 3)
│
└── data/
    ├── historical/               # Cached historical data
    ├── logs/                     # Trade logs + system logs
    └── journal/                  # Trade journal entries
```

---

## 4. Module Specifications

### 4.1 Data Layer

#### 4.1.1 Market Data (`src/data/market_data.py`)

**Responsibilities**:
- Fetch real-time OHLCV data via WebSocket (streaming) and REST (historical)
- Support multiple timeframes: 1m, 5m, 15m, 1h, 4h, 1d
- Normalize data format across all exchanges into unified DataFrame
- Handle reconnection, rate limiting, data gaps

**Unified Data Schema**:
```python
@dataclass
class Bar:
    timestamp: datetime      # UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument: str          # e.g., "EURUSD", "AAPL", "BTCUSDT"
    timeframe: str           # e.g., "5m", "1h"
    source: str              # e.g., "alpaca", "binance"
```

**Key Methods**:
```python
class MarketDataFetcher:
    async def get_historical(instrument, timeframe, start, end) -> pd.DataFrame
    async def subscribe_realtime(instrument, timeframe, callback) -> None
    async def get_latest_bar(instrument, timeframe) -> Bar
    async def get_orderbook(instrument, depth=10) -> OrderBook
```

#### 4.1.2 News Feed (`src/data/news_feed.py`)

**Sources** (in priority order):
1. NewsAPI.org — broad financial news
2. Alpha Vantage News — market-specific
3. Reddit/Twitter API — social sentiment (rate-limited)
4. RSS feeds — major financial outlets

**Output Schema**:
```python
@dataclass
class NewsItem:
    timestamp: datetime
    title: str
    summary: str
    source: str
    relevance_score: float   # 0-1, pre-filtered
    instruments: list[str]   # Affected instruments
    sentiment_raw: str       # Raw text for Claude analysis
```

#### 4.1.3 Data Store (`src/data/data_store.py`)

**Technology**: DuckDB (fast analytical queries, embedded, zero-config)

**Tables**:
- `bars` — OHLCV data, partitioned by instrument + timeframe
- `news` — News items with sentiment scores
- `signals` — Generated trading signals
- `trades` — Executed trades (paper + live)
- `journal` — Trade journal entries with Claude reasoning

### 4.2 Processing Layer

#### 4.2.1 Technical Indicator Engine (`src/analysis/technical.py`)

**Indicators** (all computed on multiple timeframes):

| Category       | Indicators                                  |
|----------------|---------------------------------------------|
| Trend          | SMA(20,50,200), EMA(9,21), MACD, ADX       |
| Momentum       | RSI(14), Stochastic(14,3,3), CCI(20), MFI  |
| Volatility     | ATR(14), Bollinger Bands(20,2), Keltner     |
| Volume         | VWAP, OBV, Volume Profile                   |
| Support/Resist | Pivot Points, Donchian Channels(20)         |

**Output Schema**:
```python
@dataclass
class TechnicalSnapshot:
    instrument: str
    timestamp: datetime
    timeframe: str
    indicators: dict[str, float]  # e.g., {"rsi_14": 65.3, "macd_signal": 0.0012}
    signals: dict[str, str]       # e.g., {"rsi": "overbought", "macd": "bullish_cross"}
    trend_bias: str               # "bullish" | "bearish" | "neutral"
    volatility_regime: str        # "low" | "normal" | "high" | "extreme"
```

**Key Methods**:
```python
class TechnicalEngine:
    def compute_all(bars: pd.DataFrame) -> TechnicalSnapshot
    def compute_indicator(bars: pd.DataFrame, name: str, params: dict) -> pd.Series
    def detect_patterns(bars: pd.DataFrame) -> list[CandlePattern]
    def multi_timeframe_analysis(instrument: str) -> dict[str, TechnicalSnapshot]
```

#### 4.2.2 Claude API Agent (`src/analysis/claude_agent.py`)

**This is the cognitive core of the system.**

**Responsibilities**:
- Analyze market context by synthesizing technical data + news + sentiment
- Provide trade recommendations with confidence scores and reasoning
- Detect regime changes and anomalies
- Generate post-trade reflections for the journal

**Prompt Architecture**:

```python
SYSTEM_PROMPT = """
You are a professional intraday trader with 20 years of experience.
You analyze markets across forex, futures, stocks, and crypto.

Your role:
1. Synthesize technical indicators, news sentiment, and market context
2. Identify high-probability trade setups
3. Provide specific entry/exit levels with risk:reward ratios
4. Rate your confidence level (0-100)
5. Explain your reasoning transparently

Rules:
- Never recommend a trade with risk:reward below 1:1.5
- Always specify a stop-loss level
- Consider the current market regime (trend/range/crisis)
- Account for upcoming economic events that may cause volatility
- If unsure, recommend "NO TRADE" — capital preservation is priority #1
"""
```

**API Call Structure**:
```python
class ClaudeAgent:
    async def analyze_market(
        instrument: str,
        technical: TechnicalSnapshot,
        news: list[NewsItem],
        regime: MarketRegime,
        recent_trades: list[Trade],     # For reflection
        portfolio_state: PortfolioState  # Current exposure
    ) -> TradeRecommendation

    async def reflect_on_trade(
        trade: Trade,
        market_context_at_entry: dict,
        market_context_at_exit: dict
    ) -> TradeReflection
```

**Output Schema**:
```python
@dataclass
class TradeRecommendation:
    action: str              # "BUY" | "SELL" | "NO_TRADE"
    instrument: str
    confidence: int          # 0-100
    entry_price: float
    stop_loss: float
    take_profit_1: float     # Conservative target
    take_profit_2: float     # Aggressive target
    position_size_pct: float # % of capital (suggestion, risk mgr overrides)
    risk_reward_ratio: float
    reasoning: str           # Natural language explanation
    key_factors: list[str]   # e.g., ["RSI oversold", "Bullish news catalyst"]
    time_horizon: str        # "scalp" | "intraday" | "swing"
    regime_assessment: str   # Current market regime
```

**Cost Management**:
- Use `claude-sonnet-4-20250514` for routine analysis (cost-efficient)
- Escalate to `claude-opus-4-6` for high-stakes decisions (large positions, anomalies)
- Batch non-urgent analyses into scheduled intervals (every 5 min during active hours)
- Cache similar market contexts to avoid redundant API calls
- Monthly API budget cap with alerting at 80% usage

#### 4.2.3 Regime Detector (`src/analysis/regime_detector.py`)

**Regimes**:
| Regime       | Characteristics                    | Strategy Implication        |
|--------------|------------------------------------|-----------------------------|
| Strong Trend | ADX > 25, aligned MAs              | Momentum, trend following   |
| Weak Trend   | ADX 15-25, mixed signals            | Reduced size, tight stops   |
| Range Bound  | ADX < 15, price in Bollinger range  | Mean reversion, fade        |
| Breakout     | Squeeze release, volume spike       | Breakout entries            |
| Crisis       | VIX spike, correlation breakdown    | Reduce exposure, hedge      |

**Implementation**: Hidden Markov Model (HMM) with 5 states, trained on rolling 60-day windows. Features: ATR %, ADX, volume ratio, return distribution skew.

### 4.3 Decision Layer

#### 4.3.1 Signal Aggregator (`src/strategy/signal_aggregator.py`)

**Multi-Factor Scoring System**:

```python
@dataclass
class SignalScore:
    instrument: str
    timestamp: datetime
    technical_score: float    # -1 to +1 (from technical engine)
    sentiment_score: float    # -1 to +1 (from Claude + news)
    regime_score: float       # -1 to +1 (regime alignment)
    claude_confidence: float  # 0 to 1 (Claude's self-rated confidence)
    composite_score: float    # Weighted combination
    action: str               # "STRONG_BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG_SELL"
```

**Scoring Weights** (configurable, optimized via walk-forward):
```yaml
signal_weights:
  technical: 0.30
  sentiment: 0.25
  regime: 0.20
  claude_confidence: 0.25

thresholds:
  strong_buy: 0.7
  buy: 0.4
  neutral: -0.4 to 0.4
  sell: -0.7
  strong_sell: -0.7
```

#### 4.3.2 Position Sizer (`src/strategy/position_sizer.py`)

**Methods**:
1. **Kelly Criterion** (primary): `f* = (bp - q) / b` where b=win/loss ratio, p=win rate, q=1-p
2. **Half-Kelly** (default): Use 50% of Kelly to reduce variance
3. **ATR-based adjustment**: Scale position inversely with current ATR
4. **Portfolio heat limit**: Total open risk never exceeds configured max (default 6% of equity)

```python
class PositionSizer:
    def calculate_size(
        signal: SignalScore,
        account_equity: float,
        current_atr: float,
        stop_distance: float,
        existing_exposure: float,
        risk_params: RiskParams
    ) -> PositionSize
```

#### 4.3.3 Risk Manager (`src/strategy/risk_manager.py`)

**Hard Rules (NEVER overridden)**:

```yaml
risk_limits:
  max_position_size_pct: 5         # Max 5% of equity per position
  max_portfolio_heat_pct: 6        # Max 6% total open risk
  max_daily_loss_pct: 3            # Stop trading if daily loss > 3%
  max_drawdown_pct: 15             # Kill switch at 15% drawdown
  max_correlated_positions: 3      # Max 3 positions in same sector/correlated assets
  max_trades_per_day: 10           # Per instrument
  min_risk_reward: 1.5             # Reject trades below 1:1.5 RR
  no_trade_before_news: 15         # Minutes before major news events
  no_trade_session_boundary: 5     # Minutes before market close
```

**Soft Rules (warnings, can be adjusted)**:
```yaml
soft_limits:
  preferred_position_size_pct: 2   # Target 2% per position
  preferred_portfolio_heat_pct: 4  # Target 4% total risk
  max_consecutive_losses: 3        # Alert after 3 losses, reduce size
  correlation_threshold: 0.6       # Warn on correlated entries
```

**Risk Check Pipeline**:
```
Signal → [Size Check] → [Heat Check] → [Drawdown Check] → [Correlation Check]
       → [News Check] → [Session Check] → [RR Check] → APPROVED / REJECTED
```

### 4.4 Execution Layer

#### 4.4.1 Order Manager (`src/execution/order_manager.py`)

**Order Lifecycle**:
```
PENDING → SUBMITTED → PARTIAL_FILL → FILLED → CLOSED
                    → REJECTED
                    → CANCELLED
                    → EXPIRED
```

**Order Types**:
```python
@dataclass
class Order:
    id: str
    instrument: str
    side: str                # "BUY" | "SELL"
    order_type: str          # "LIMIT" | "MARKET" | "STOP" | "STOP_LIMIT"
    quantity: float
    price: float | None      # None for market orders
    stop_loss: float
    take_profit: float | None
    time_in_force: str       # "GTC" | "IOC" | "FOK" | "DAY"
    status: str
    created_at: datetime
    filled_at: datetime | None
    fill_price: float | None
    slippage: float | None   # Calculated post-fill
    signal_id: str           # Link back to the signal that created this
    claude_reasoning: str    # Claude's analysis stored with the order
```

#### 4.4.2 Paper Trading Engine (`src/execution/paper_engine.py`)

**Realistic Simulation**:
- **Slippage model**: Normal distribution, mean=0.5 * spread, std=0.3 * spread
- **Fill probability**: Limit orders fill only when price crosses, with partial fill simulation
- **Commission model**: Configurable per exchange (e.g., 0.1% for crypto, $0.005/share for stocks)
- **Latency model**: Simulated 50-200ms execution delay

#### 4.4.3 Exchange Adapter (`src/execution/exchange_adapter.py`)

**Abstract Interface** (all adapters implement this):
```python
class ExchangeAdapter(ABC):
    async def connect() -> None
    async def disconnect() -> None
    async def get_account_info() -> AccountInfo
    async def place_order(order: Order) -> OrderResult
    async def cancel_order(order_id: str) -> bool
    async def get_order_status(order_id: str) -> OrderStatus
    async def get_positions() -> list[Position]
    async def get_balance() -> Balance
    async def subscribe_fills(callback) -> None
    async def subscribe_orderbook(instrument, callback) -> None
```

### 4.5 Reflection Layer

#### 4.5.1 Trade Journal (`src/reflection/trade_journal.py`)

**Every trade records**:
```python
@dataclass
class JournalEntry:
    trade_id: str
    timestamp_entry: datetime
    timestamp_exit: datetime
    instrument: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    # Context at entry
    signal_score: float
    claude_reasoning_entry: str
    technical_snapshot_entry: TechnicalSnapshot
    regime_at_entry: str
    news_at_entry: list[str]
    # Context at exit
    exit_reason: str         # "take_profit" | "stop_loss" | "signal_reversal" | "time_exit"
    claude_reflection: str   # Post-trade analysis by Claude
    # Attribution
    factor_attribution: dict # {"technical": 0.4, "sentiment": 0.3, "regime": 0.3}
    lessons_learned: str     # Claude's takeaway for future improvement
```

#### 4.5.2 Attribution Engine (`src/reflection/attribution.py`)

**Attribution Dimensions**:
1. **Factor Attribution**: Which signal component (technical/sentiment/regime) contributed most?
2. **Timing Attribution**: Was entry timing good? Exit timing?
3. **Sizing Attribution**: Was position size appropriate given the outcome?
4. **Regime Attribution**: Did the market regime match the strategy's sweet spot?

#### 4.5.3 Strategy Evolver (`src/reflection/strategy_evolver.py`)

**Weekly Optimization Loop**:
1. Pull last 7 days of trade journal entries
2. Feed to Claude for pattern analysis: "What's working? What's failing?"
3. Propose parameter adjustments (signal weights, thresholds, indicators)
4. Run 30-day walk-forward backtest on proposed changes
5. If improved Sharpe + reduced drawdown → apply changes
6. Log all changes with reasoning for audit trail

### 4.6 Orchestrator (Phase 3)

#### 4.6.1 Orchestrator Agent (`src/orchestrator/orchestrator.py`)

**Responsibilities**:
- Allocate capital across specialist agents based on performance + regime fit
- Resolve conflicting signals (Agent A says BUY, Agent B says SELL same instrument)
- Enforce global portfolio-level risk limits
- Rebalance allocation weights weekly based on rolling Sharpe

**Capital Allocation**:
```python
# Risk-parity with performance overlay
allocation_weight[agent] = (
    base_weight                          # Equal weight baseline
    * sharpe_multiplier[agent]           # Scale by rolling 30-day Sharpe
    * regime_fit_score[agent]            # Scale by regime alignment
    * (1 - correlation_penalty[agent])   # Penalize correlated agents
)
```

---

## 5. Tech Stack

### 5.1 Core

| Component          | Technology          | Reason                                    |
|--------------------|---------------------|-------------------------------------------|
| Language           | Python 3.12+        | Ecosystem, async support, TA libraries    |
| AI Engine          | Claude API (Anthropic) | Reasoning quality, structured output    |
| Data Store         | DuckDB              | Fast analytics, embedded, zero-config     |
| Task Queue         | asyncio + APScheduler | Lightweight, no infrastructure overhead |
| Config             | YAML + Pydantic     | Type-safe, human-readable                 |
| Logging            | structlog            | Structured JSON logs                      |

### 5.2 Data & Analysis

| Component          | Technology          | Reason                                    |
|--------------------|---------------------|-------------------------------------------|
| Technical Analysis | pandas-ta / TA-Lib  | Comprehensive indicators, fast            |
| DataFrames         | Polars (preferred) / Pandas | Speed for large datasets          |
| Market Data        | ccxt, yfinance, MetaTrader5 | Multi-exchange coverage          |
| News               | NewsAPI, feedparser | Broad + RSS support                       |

### 5.3 Execution & Monitoring

| Component          | Technology          | Reason                                    |
|--------------------|---------------------|-------------------------------------------|
| Exchange APIs      | ccxt, alpaca-py, ib_insync | Standard connectors              |
| Dashboard          | Streamlit           | Rapid prototyping, Python-native          |
| Alerts             | python-telegram-bot | Real-time mobile alerts                   |
| Backtesting        | Custom + vectorbt   | Flexibility + speed                       |

### 5.4 Development

| Component          | Technology          | Reason                                    |
|--------------------|---------------------|-------------------------------------------|
| Package Manager    | uv                  | Fast, modern Python packaging             |
| Testing            | pytest + pytest-asyncio | Async test support                   |
| Type Checking      | mypy                | Catch bugs early                          |
| Linting            | ruff                | Fast, comprehensive                       |
| Pre-commit         | pre-commit          | Automated code quality                    |

---

## 6. Configuration

### 6.1 Environment Variables (`.env`)

```bash
# Claude API
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL_DEFAULT=claude-sonnet-4-20250514
CLAUDE_MODEL_PREMIUM=claude-opus-4-6
CLAUDE_MONTHLY_BUDGET_USD=200

# Exchange APIs (Phase 2)
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper first!
IB_HOST=127.0.0.1
IB_PORT=7497
BINANCE_API_KEY=...
BINANCE_SECRET=...

# Data
NEWSAPI_KEY=...
ALPHAVANTAGE_KEY=...

# Notifications
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Database
DB_PATH=./data/atlas.duckdb
```

### 6.2 Global Settings (`config/settings.yaml`)

```yaml
system:
  name: "atlas-trader"
  phase: 1                     # Current phase (1/2/3)
  mode: "paper"                # "paper" | "live" | "backtest"
  timezone: "UTC"
  log_level: "INFO"

trading:
  default_timeframes: ["5m", "15m", "1h"]
  analysis_interval_seconds: 300   # Run analysis every 5 min
  max_instruments_parallel: 5      # Analyze max 5 instruments at once

claude:
  analysis_model: "claude-sonnet-4-20250514"
  reflection_model: "claude-sonnet-4-20250514"
  escalation_model: "claude-opus-4-6"
  escalation_threshold: 0.8        # Use premium model when signal > 0.8
  max_tokens_per_analysis: 2000
  temperature: 0.3                 # Low temp for consistent analysis
  cache_ttl_seconds: 300           # Cache similar contexts for 5 min

backtest:
  start_date: "2024-01-01"
  end_date: "2025-12-31"
  initial_capital: 100000
  commission_pct: 0.1
  slippage_pct: 0.05
  walk_forward_window_days: 90
  walk_forward_step_days: 30
```

---

## 7. Key Design Decisions

### 7.1 Why Claude API as the Core (Not Pure ML)

| Consideration         | Pure ML                          | Claude API (Our Choice)           |
|-----------------------|----------------------------------|-----------------------------------|
| Development time      | Months of training               | Days of prompt engineering        |
| Data requirements     | 10K+ labeled examples            | Zero-shot / few-shot              |
| Adaptability          | Retrain on new data              | Adjust prompts, instant           |
| Interpretability      | Black box                        | Natural language reasoning        |
| News understanding    | Requires NLP pipeline            | Native capability                 |
| Multi-asset transfer  | Separate models per asset        | Single model, universal           |
| Cost at scale         | GPU infrastructure               | Pay-per-call, predictable         |
| Risk                  | Overfitting on historical data   | May hallucinate confidence        |

**Mitigation for Claude risks**: Always validate Claude's output against hard technical rules. Claude suggests, risk manager validates.

### 7.2 Why Not High-Frequency

- Claude API latency (~1-3 seconds) makes sub-second strategies impossible
- Retail infrastructure can't compete with institutional HFT
- Intraday (5min-8hr) is the sweet spot: enough time for Claude to reason, enough frequency for meaningful signal generation

### 7.3 Risk-First Architecture

Every module has a "risk veto" — the Risk Manager can block any action at any point in the pipeline. This is implemented as middleware:

```python
# Pseudocode
signal = signal_aggregator.score(market_data)
if not risk_manager.pre_check(signal):
    log("Signal vetoed by risk manager", signal=signal)
    return

size = position_sizer.calculate(signal, account)
if not risk_manager.size_check(size):
    size = risk_manager.adjust_size(size)

order = order_manager.create(signal, size)
if not risk_manager.order_check(order):
    log("Order vetoed by risk manager", order=order)
    return

result = exchange.execute(order)
risk_manager.post_execution_check(result)
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

- Each module independently testable with mock data
- Technical indicators validated against known values (e.g., RSI of specific series)
- Risk manager tested with edge cases (max drawdown hit, correlated positions)
- Claude agent tested with mocked API responses

### 8.2 Integration Tests

- Full pipeline test: data → analysis → signal → risk check → paper execution
- Exchange adapter tests against sandbox/testnet endpoints
- Multi-instrument concurrent analysis test

### 8.3 Backtest Validation

- Walk-forward optimization on 2 years of data
- Out-of-sample testing (train on 2024, test on 2025)
- Monte Carlo simulation (1000 random entry shuffles → distribution of outcomes)
- Regime-specific backtests (backtest only during trending / ranging / crisis periods)

---

## 9. Monitoring & Alerts

### 9.1 Real-Time Alerts (Telegram)

| Event                        | Severity | Alert                            |
|------------------------------|----------|----------------------------------|
| Trade opened                 | INFO     | Instrument, direction, size, SL  |
| Trade closed (profit)        | INFO     | P&L, duration, factors           |
| Trade closed (loss)          | WARNING  | P&L, SL hit, analysis            |
| Daily loss limit approached  | CRITICAL | Current loss %, threshold        |
| Max drawdown approached      | CRITICAL | Current DD %, kill switch status  |
| System error                 | CRITICAL | Error details, module            |
| Claude API budget 80%        | WARNING  | Spend to date, days remaining    |
| Exchange connection lost     | CRITICAL | Exchange, reconnection attempts  |

### 9.2 Dashboard Panels

1. **P&L Curve** — Equity line, drawdown area, benchmark comparison
2. **Signal Monitor** — Live signal scores, factor breakdown
3. **Position Map** — Open positions, unrealized P&L, heat gauge
4. **Risk Gauge** — Portfolio heat, drawdown meter, daily loss tracker
5. **Trade Log** — Recent trades with Claude reasoning excerpts
6. **Agent Health** (Phase 3) — Per-agent status, performance, allocation

---

## 10. Development Workflow with Claude Code

### 10.1 Implementation Order

```
Phase 1 Implementation:
  1. Project scaffolding (pyproject.toml, directory structure, config)
  2. Data layer (market_data.py, data_store.py)
  3. Technical engine (technical.py)
  4. Claude agent integration (claude_agent.py)
  5. Regime detector (regime_detector.py)
  6. Signal aggregator (signal_aggregator.py)
  7. Risk manager (risk_manager.py)
  8. Position sizer (position_sizer.py)
  9. Paper trading engine (paper_engine.py)
  10. Trade journal (trade_journal.py)
  11. Backtest runner (run_backtest.py)
  12. Dashboard (basic P&L + signal view)
  13. Integration testing + optimization
```

### 10.2 Claude Code Commands

```bash
# Initialize project
claude "Read PROJECT.md and scaffold the atlas-trader project structure with all directories, pyproject.toml, and config files"

# Build module by module
claude "Implement src/data/market_data.py following the spec in PROJECT.md section 4.1.1"
claude "Implement src/analysis/technical.py following the spec in PROJECT.md section 4.2.1"
claude "Implement src/analysis/claude_agent.py following the spec in PROJECT.md section 4.2.2"

# Test
claude "Write and run tests for the technical indicator engine"
claude "Run the full Phase 1 integration test"

# Backtest
claude "Run a backtest on EURUSD 2024-2025 data and show results"
```

---

## 11. Risk Disclaimer

This system is for research and educational purposes. Trading involves significant risk of loss. Key warnings:

- Past backtest performance does not guarantee future results
- AI-generated trading signals can be wrong
- Always start with paper trading before risking real capital
- Never risk more than you can afford to lose
- The system includes multiple safety mechanisms, but no system is foolproof
- Regulatory requirements vary by jurisdiction — ensure compliance

---

## 12. Success Metrics

| Metric                 | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|------------------------|----------------|----------------|----------------|
| Sharpe Ratio           | > 1.5          | > 1.2          | > 2.0          |
| Max Drawdown           | < 15%          | < 15%          | < 12%          |
| Win Rate               | > 55%          | > 52%          | > 55%          |
| Profit Factor          | > 1.5          | > 1.3          | > 1.8          |
| Avg Risk:Reward        | > 1:1.5        | > 1:1.5        | > 1:2.0        |
| Monthly Return (median)| +3-5%          | +2-4%          | +3-6%          |
| Strategy Correlation   | N/A            | N/A            | < 0.3          |
| System Uptime          | N/A            | > 99.5%        | > 99.9%        |

---

*This document is the single source of truth for the atlas-trader project. All implementation should reference this spec. Update this document when architecture decisions change.*
