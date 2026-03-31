# Dashboard: React Web App — Full Specification

> **Type**: React SPA + FastAPI Backend
> **Features**: All 8 panels, real-time WebSocket, mobile responsive
> **Codename**: `atlas-dashboard`

---

## 1. Tech Stack

### 1.1 Frontend

| Technology | Version | Purpose |
|---|---|---|
| React | 18+ | UI framework |
| TypeScript | 5.x | Type safety |
| Vite | 5.x | Build tool (fast HMR) |
| Tailwind CSS | 3.x | Utility-first styling |
| shadcn/ui | latest | Component library (Radix-based) |
| Recharts | 2.x | Charts (P&L, equity curve, bar charts) |
| D3.js | 7.x | Heatmap, correlation matrix, custom viz |
| Lucide React | latest | Icons |
| TanStack Query | 5.x | Server state management, auto-refresh |
| TanStack Table | 8.x | Trade log, data tables |
| Zustand | 4.x | Client state (filters, preferences) |
| React Router | 6.x | Page routing |
| Socket.IO Client | 4.x | WebSocket real-time data |
| date-fns | 3.x | Date formatting |
| numeral.js | 2.x | Number formatting ($, %, commas) |

### 1.2 Backend (API Server)

| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.110+ | REST API + WebSocket |
| uvicorn | 0.29+ | ASGI server |
| python-socketio | 5.x | WebSocket server |
| DuckDB | 1.x | Query data store (same DB as trading engine) |
| Pydantic | 2.x | Request/response schemas |
| orjson | 3.x | Fast JSON serialization |

### 1.3 Project Structure

```
atlas-trader/
├── dashboard/
│   ├── backend/                    # FastAPI server
│   │   ├── main.py                 # App entry, CORS, mount routes
│   │   ├── config.py               # Settings
│   │   ├── deps.py                 # Dependencies (DB connection)
│   │   ├── ws.py                   # WebSocket manager
│   │   ├── routes/
│   │   │   ├── portfolio.py        # /api/portfolio/*
│   │   │   ├── signals.py          # /api/signals/*
│   │   │   ├── trades.py           # /api/trades/*
│   │   │   ├── risk.py             # /api/risk/*
│   │   │   ├── backtest.py         # /api/backtest/*
│   │   │   ├── agents.py           # /api/agents/*
│   │   │   └── execution.py        # /api/execution/*
│   │   └── schemas.py              # Pydantic response models
│   │
│   └── frontend/                   # React app
│       ├── index.html
│       ├── package.json
│       ├── tailwind.config.js
│       ├── tsconfig.json
│       ├── vite.config.ts
│       └── src/
│           ├── main.tsx
│           ├── App.tsx
│           ├── api/                # API client layer
│           │   ├── client.ts       # Axios/fetch base
│           │   ├── hooks.ts        # TanStack Query hooks
│           │   └── websocket.ts    # Socket.IO client
│           ├── store/
│           │   └── useStore.ts     # Zustand store
│           ├── types/
│           │   └── index.ts        # TypeScript interfaces
│           ├── lib/
│           │   ├── utils.ts        # Helpers
│           │   └── format.ts       # Number/date formatters
│           ├── components/
│           │   ├── ui/             # shadcn components
│           │   ├── layout/
│           │   │   ├── AppShell.tsx
│           │   │   ├── Sidebar.tsx
│           │   │   ├── Header.tsx
│           │   │   ├── MobileNav.tsx
│           │   │   └── StatusBar.tsx
│           │   ├── charts/
│           │   │   ├── EquityCurve.tsx
│           │   │   ├── DrawdownChart.tsx
│           │   │   ├── DailyPnLBars.tsx
│           │   │   ├── SignalRadar.tsx
│           │   │   ├── CorrelationMatrix.tsx
│           │   │   └── HeatMap.tsx
│           │   ├── panels/
│           │   │   ├── MetricCard.tsx
│           │   │   ├── RiskGauge.tsx
│           │   │   ├── PositionCard.tsx
│           │   │   ├── TradeRow.tsx
│           │   │   ├── AgentCard.tsx
│           │   │   └── ClaudeReasoning.tsx
│           │   └── execution/
│           │       ├── OrderForm.tsx
│           │       ├── QuickActions.tsx
│           │       └── KillSwitch.tsx
│           └── pages/
│               ├── DashboardPage.tsx    # Overview (default)
│               ├── SignalsPage.tsx       # Signal analysis
│               ├── PositionsPage.tsx     # Positions + risk
│               ├── JournalPage.tsx       # Trade journal
│               ├── BacktestPage.tsx      # Backtest results
│               ├── AgentsPage.tsx        # Multi-agent view
│               └── ExecutionPage.tsx     # Manual trading
```

---

## 2. API Endpoints

### 2.1 Portfolio

```
GET  /api/portfolio/summary        → PortfolioSummary
GET  /api/portfolio/equity-curve    → EquityCurveData (array of {timestamp, equity, drawdown_pct})
GET  /api/portfolio/daily-pnl       → DailyPnLData (array of {date, pnl, pnl_pct})
GET  /api/portfolio/monthly-returns → MonthlyReturns (12-month grid)
```

**PortfolioSummary Schema**:
```typescript
interface PortfolioSummary {
  equity: number
  cash: number
  unrealized_pnl: number
  realized_pnl_today: number
  daily_pnl: number
  daily_pnl_pct: number
  max_drawdown_current: number
  portfolio_heat_pct: number
  open_positions: number
  sharpe_30d: number
  sharpe_90d: number
  win_rate_30d: number
  profit_factor_30d: number
  total_trades_today: number
  system_status: 'running' | 'paused' | 'error'
  last_signal_time: string
  claude_budget_remaining_pct: number
}
```

### 2.2 Signals

```
GET  /api/signals/current           → CurrentSignals (per instrument)
GET  /api/signals/history?hours=24  → SignalHistory
GET  /api/signals/radar/:instrument → RadarData (factor breakdown)
```

**RadarData Schema**:
```typescript
interface RadarData {
  instrument: string
  timestamp: string
  factors: {
    name: string          // "Technical" | "Sentiment" | "Regime" | "Claude" | "Volume"
    score: number         // -1 to +1
    details: string       // e.g., "RSI oversold, MACD bullish cross"
  }[]
  composite_score: number
  action: string
  confidence: number
}
```

### 2.3 Trades

```
GET  /api/trades?page=1&limit=50&instrument=&side=&outcome=  → PaginatedTrades
GET  /api/trades/:id                                         → TradeDetail (full journal entry)
GET  /api/trades/:id/reasoning                               → ClaudeReasoning (entry + exit)
GET  /api/trades/stats?period=30d                             → TradeStats
```

### 2.4 Risk

```
GET  /api/risk/current              → RiskSnapshot
GET  /api/risk/limits               → RiskLimits (hard + soft)
GET  /api/risk/exposure             → ExposureBreakdown (by asset, sector, direction)
GET  /api/risk/var                  → ValueAtRisk (95% VaR estimate)
```

**RiskSnapshot Schema**:
```typescript
interface RiskSnapshot {
  portfolio_heat_pct: number
  max_heat_limit: number
  daily_loss_pct: number
  max_daily_loss_limit: number
  drawdown_pct: number
  max_drawdown_limit: number
  open_risk_usd: number
  long_exposure_pct: number
  short_exposure_pct: number
  net_exposure_pct: number
  correlated_positions: number
  max_correlated_limit: number
  consecutive_losses: number
  risk_level: 'low' | 'moderate' | 'high' | 'critical'
}
```

### 2.5 Backtest

```
GET  /api/backtest/results                    → BacktestList
GET  /api/backtest/results/:id                → BacktestDetail
GET  /api/backtest/compare?ids=id1,id2,id3    → ComparisonData
POST /api/backtest/run                        → BacktestJob (async, returns job_id)
GET  /api/backtest/jobs/:id/status            → JobStatus
```

### 2.6 Agents (Phase 3)

```
GET  /api/agents                              → AgentList
GET  /api/agents/:id                          → AgentDetail
GET  /api/agents/:id/performance              → AgentPerformance
GET  /api/agents/allocation                   → AllocationBreakdown
GET  /api/agents/correlation                  → CorrelationMatrix (NxN)
POST /api/agents/:id/activate                 → OK
POST /api/agents/:id/deactivate               → OK
```

### 2.7 Execution

```
GET  /api/execution/positions                 → OpenPositions (live from exchange)
POST /api/execution/close/:instrument         → CloseResult
POST /api/execution/close-all                 → CloseAllResult
POST /api/execution/pause                     → PauseTrading
POST /api/execution/resume                    → ResumeTrading
POST /api/execution/kill-switch               → KillSwitchResult (close all + pause)
GET  /api/execution/orders/open               → OpenOrders
POST /api/execution/orders/:id/cancel         → CancelResult
POST /api/execution/manual-order              → ManualOrderResult
```

### 2.8 WebSocket Events

```
Connection: ws://localhost:8001/ws

Server → Client events:
  portfolio:update     → PortfolioSummary (every 5s)
  position:update      → Position[]       (on change)
  signal:new           → SignalScore       (on generation)
  trade:open           → TradeRecord       (on fill)
  trade:close          → TradeRecord       (on exit)
  risk:alert           → RiskAlert         (on threshold breach)
  agent:status         → AgentStatus       (on state change)
  system:heartbeat     → { timestamp }     (every 10s)

Client → Server events:
  subscribe:instrument → Subscribe to instrument-specific updates
  unsubscribe          → Unsubscribe
```

---

## 3. Page Specifications

### 3.1 Page 1: Dashboard Overview (`DashboardPage.tsx`)

**Layout (Desktop)**: 3-column grid at top, full-width chart middle, 2-column bottom

```
┌─────────────────────────────────────────────────────────────┐
│ Header: Atlas Trader  │ Status: ● Running │ Phase 1 │ 14:32 │
├───────────┬───────────┬───────────┬─────────────────────────┤
│  Equity   │ Daily P&L │  Sharpe   │     Risk Level          │
│ $104,230  │  +$1,340  │   1.82    │   ██░░ Moderate         │
│  +4.2%    │  +1.3%    │  30-day   │   Heat: 3.8% / 6%      │
├───────────┴───────────┴───────────┴─────────────────────────┤
│                                                             │
│  ┌─ Equity Curve ──────────────────────────────────────┐    │
│  │  [Line chart: equity over time]                     │    │
│  │  [Area chart: drawdown below x-axis, red fill]      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
├─────────────────────────────┬───────────────────────────────┤
│  Daily P&L Bars             │  Recent Trades (last 10)      │
│  [Bar chart: green/red]     │  EURUSD  BUY  +$230  ✅       │
│  [Hover: date + amount]     │  AAPL    SELL -$85   ❌       │
│                             │  BTC     BUY  +$410  ✅       │
│                             │  ...                          │
├─────────────────────────────┴───────────────────────────────┤
│  Monthly Returns Heatmap (12 months × years)                │
│  [Grid: green shades = profit, red = loss, number overlay]  │
└─────────────────────────────────────────────────────────────┘
```

**Layout (Mobile)**: Single column, stacked vertically, cards

```
┌─────────────────┐
│ ≡ Atlas Trader   │
├─────────────────┤
│ $104,230  +4.2% │
│ Today: +$1,340  │
├─────────────────┤
│ [Mini equity     │
│  curve, 7 days]  │
├─────────────────┤
│ Sharpe: 1.82     │
│ DD: -2.1%        │
│ Heat: 3.8%       │
├─────────────────┤
│ Recent Trades ▼  │
│  EURUSD +$230    │
│  AAPL  -$85      │
├─────────────────┤
│ 📊 🎯 📋 🧪 🤖 ⚡ │
│    Bottom Nav     │
└─────────────────┘
```

**Metric Cards Component**:
```typescript
interface MetricCardProps {
  title: string
  value: string | number
  change?: number           // +/- percentage
  changeLabel?: string      // "today" | "30d"
  icon?: LucideIcon
  trend?: 'up' | 'down' | 'flat'
  severity?: 'normal' | 'warning' | 'critical'
}

// Examples:
// { title: "Equity", value: "$104,230", change: 4.2, changeLabel: "all-time", trend: "up" }
// { title: "Drawdown", value: "-2.1%", severity: "normal" }
// { title: "Daily P&L", value: "+$1,340", change: 1.3, trend: "up" }
```

**Equity Curve Component**:
```typescript
interface EquityCurveProps {
  data: { timestamp: string; equity: number; drawdown_pct: number }[]
  benchmark?: { timestamp: string; value: number }[]  // Buy & hold comparison
  timeRange: '1W' | '1M' | '3M' | '6M' | '1Y' | 'ALL'
  showDrawdown: boolean      // Toggle drawdown area
  showBenchmark: boolean
}

// Implementation:
// - Recharts ComposedChart
// - Line for equity (primary color, 2px)
// - Area for drawdown (red, below x-axis, 20% opacity)
// - Optional Line for benchmark (gray dashed)
// - Time range selector: button group at top-right
// - Crosshair on hover: show date, equity, DD, benchmark
// - Responsive: on mobile, hide benchmark, reduce margins
```

### 3.2 Page 2: Signals (`SignalsPage.tsx`)

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Instrument Selector: [EURUSD] [AAPL] [BTC] [ES] [+Add]   │
├──────────────────────────────┬──────────────────────────────┤
│                              │                              │
│  Signal Radar Chart          │  Factor Details              │
│                              │                              │
│      Technical               │  Technical:  0.65 ████████░  │
│       /    \                 │   RSI: 38 (oversold)         │
│  Claude --- Sentiment        │   MACD: bullish cross        │
│       \    /                 │   MA: aligned bullish        │
│       Regime                 │                              │
│                              │  Sentiment:  0.42 █████░░░░  │
│  Score: 0.68 → BUY           │   Claude: "Positive EUR      │
│  Confidence: 72%             │   outlook post-ECB..."       │
│                              │                              │
│                              │  Regime: 0.71 ████████░░     │
│                              │   Strong Trend Up            │
│                              │   ADX: 31                    │
├──────────────────────────────┴──────────────────────────────┤
│  Signal History Timeline                                     │
│  [Scatter plot: x=time, y=score, color=action, size=conf]   │
│  [Click dot → show full signal detail]                       │
├─────────────────────────────────────────────────────────────┤
│  Multi-Timeframe View                                        │
│  ┌── 5m ──┬── 15m ─┬── 1h ──┐                               │
│  │Bearish  │Bullish │Bullish │  ← Trend bias per TF         │
│  │RSI: 42  │RSI: 38 │RSI: 55 │                               │
│  └────────┴────────┴────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

**Radar Chart Component**:
```typescript
interface SignalRadarProps {
  factors: {
    name: string       // axis label
    value: number      // -1 to +1
    fullMark: number   // always 1
  }[]
  fillColor: string    // based on action (green=buy, red=sell, gray=neutral)
  instrument: string
}

// Implementation:
// - Recharts RadarChart with PolarGrid
// - 5 axes: Technical, Sentiment, Regime, Claude Confidence, Volume
// - Fill area with 30% opacity
// - Animate on data change
// - Click axis label → expand details in side panel
```

### 3.3 Page 3: Positions + Risk (`PositionsPage.tsx`)

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Risk Dashboard                                              │
│  ┌──────────┬──────────┬──────────┬──────────┐              │
│  │ Heat     │ Daily DD │ Max DD   │ Net Exp  │              │
│  │ [Gauge]  │ [Gauge]  │ [Gauge]  │ [Gauge]  │              │
│  │ 3.8/6%   │ 0.4/3%   │ 2.1/15%  │ +32%     │              │
│  │  🟢       │  🟢       │  🟢       │  🟡       │              │
│  └──────────┴──────────┴──────────┴──────────┘              │
├─────────────────────────────────────────────────────────────┤
│  Open Positions                                              │
│  ┌───────┬──────┬───────┬────────┬────────┬────────┬──────┐ │
│  │Instr. │ Side │ Size  │ Entry  │ Current│ P&L    │ Risk │ │
│  ├───────┼──────┼───────┼────────┼────────┼────────┼──────┤ │
│  │EURUSD │ BUY  │ 0.5L  │ 1.0842 │ 1.0867 │ +$125  │ 1.2%│ │
│  │AAPL   │ SELL │ 50sh  │ 198.30 │ 197.10 │ +$60   │ 0.8%│ │
│  │BTC    │ BUY  │ 0.02  │ 67,200 │ 67,850 │ +$13   │ 1.1%│ │
│  └───────┴──────┴───────┴────────┴────────┴────────┴──────┘ │
│  Each row: click → expand to show SL/TP lines, Claude reason │
│  Each row: [Close] button on right                           │
├──────────────────────────────┬──────────────────────────────┤
│  Position Heatmap            │  Exposure Breakdown           │
│  [Treemap: size = notional,  │  [Stacked bar: long vs short │
│   color = P&L pct,           │   per asset class]            │
│   label = instrument]        │                              │
│                              │  Forex:   ████████ $12,000    │
│                              │  Stocks:  █████    $8,000     │
│                              │  Crypto:  ██       $3,000     │
└──────────────────────────────┴──────────────────────────────┘
```

**Risk Gauge Component**:
```typescript
interface RiskGaugeProps {
  label: string
  value: number
  max: number
  warningThreshold: number     // % of max → yellow
  criticalThreshold: number    // % of max → red
  unit: string                 // "%" | "$"
  size?: 'sm' | 'md' | 'lg'
}

// Implementation:
// - SVG semi-circle gauge (180°)
// - Green zone: 0 to warning
// - Yellow zone: warning to critical
// - Red zone: critical to max
// - Needle animation on value change
// - Center text: current value
// - Below: "of {max}{unit}" label
// - Responsive: sm for mobile, lg for desktop
```

**Position Heatmap Component**:
```typescript
interface HeatMapProps {
  positions: {
    instrument: string
    notional_value: number     // Size of rectangle
    pnl_pct: number            // Color: green (profit) → red (loss)
    side: 'BUY' | 'SELL'
  }[]
}

// Implementation:
// - D3 treemap layout
// - Color scale: diverging RdYlGn (red → yellow → green)
// - Label: instrument name + P&L %
// - Click → navigate to position detail
// - Tooltip on hover: full position info
```

### 3.4 Page 4: Trade Journal (`JournalPage.tsx`)

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Filters: [All/Wins/Losses] [Instrument ▼] [Date Range]    │
│           [Sort: newest ▼]  [Search Claude reasoning...]    │
├─────────────────────────────────────────────────────────────┤
│  Stats Bar: 247 trades │ 58% win │ PF 1.72 │ Avg +0.8%     │
├─────────────────────────────────────────────────────────────┤
│  Trade List (TanStack Table, virtualized for performance)   │
│  ┌──────┬──────┬──────┬────────┬────────┬──────┬──────────┐ │
│  │ Time │Instr │ Side │  P&L   │  P&L%  │ Exit │ Duration │ │
│  ├──────┼──────┼──────┼────────┼────────┼──────┼──────────┤ │
│  │14:20 │EURUSD│ BUY  │ +$230  │ +1.2%  │  TP  │  42min   │ │
│  │      │                                                  │ │
│  │  ▼ EXPAND ─────────────────────────────────────────── │ │
│  │  ┌─ Claude Reasoning ────────────────────────────────┐ │ │
│  │  │ Entry: "Bullish setup identified. RSI recovering  │ │ │
│  │  │  from oversold with MACD bullish cross on 15m.    │ │ │
│  │  │  ECB dovish stance supports EUR strength.         │ │ │
│  │  │  R:R = 1:2.3, targeting 1.0880."                  │ │ │
│  │  ├───────────────────────────────────────────────────┤ │ │
│  │  │ Exit Reflection: "Trade executed well. Entry       │ │ │
│  │  │  timing was good (B+ grade). TP hit within        │ │ │
│  │  │  expected time. Lesson: ECB-related trades have   │ │ │
│  │  │  higher success rate in trending regime."          │ │ │
│  │  ├───────────────────────────────────────────────────┤ │ │
│  │  │ Attribution: Tech 40% │ Sentiment 30% │ Regime 30%│ │ │
│  │  │ [Horizontal stacked bar]                          │ │ │
│  │  └───────────────────────────────────────────────────┘ │ │
│  │                                                       │ │
│  ├──────┼──────┼──────┼────────┼────────┼──────┼─────────┤ │
│  │11:05 │AAPL  │ SELL │  -$85  │ -0.4%  │  SL  │  18min  │ │
│  └──────┴──────┴──────┴────────┴────────┴──────┴─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Claude Reasoning Component**:
```typescript
interface ClaudeReasoningProps {
  entryReasoning: string
  exitReflection: string | null
  keyFactors: string[]
  confidence: number
  grade?: string                  // "A" | "B" | "C" | "D" | "F"
  factorAttribution: {
    factor: string
    weight: number               // 0-1
  }[]
}

// Implementation:
// - Collapsible card within trade row
// - Entry reasoning in a quote-style block (left blue border)
// - Exit reflection in a separate block (left green/red border based on P&L)
// - Key factors as tags/badges
// - Confidence as progress bar
// - Attribution as horizontal stacked bar chart (inline)
// - Grade badge: A=green, B=blue, C=yellow, D=orange, F=red
```

### 3.5 Page 5: Backtest Results (`BacktestPage.tsx`)

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Backtest Runs: [Run #12 ▼] vs [Run #10 ▼] vs [Benchmark]  │
│  [+ New Backtest]                                            │
├──────────────────────────────┬──────────────────────────────┤
│  Equity Curves (overlaid)    │  Key Metrics Comparison       │
│                              │                              │
│  [Line chart: 3 lines]       │  Metric    #12    #10   B&H  │
│  ── Run #12 (blue)           │  Return   +32%  +21%  +14%   │
│  -- Run #10 (orange dashed)  │  Sharpe    1.82  1.45  0.91  │
│  ·· Benchmark (gray dotted)  │  MaxDD    -8.2% -12%  -18%  │
│                              │  WinRate   58%   54%   N/A   │
│                              │  PF        1.72  1.38  N/A   │
│                              │  Trades    247   198   N/A   │
│                              │  AvgDur.   35m   42m   N/A   │
├──────────────────────────────┴──────────────────────────────┤
│  Walk-Forward Analysis (if applicable)                       │
│  [Bar chart: Sharpe per window, x=period, y=Sharpe]          │
│  Window 1 (Jan-Mar): 2.1                                    │
│  Window 2 (Apr-Jun): 1.4                                    │
│  Window 3 (Jul-Sep): 1.9                                    │
│  Window 4 (Oct-Dec): 1.6                                    │
├─────────────────────────────────────────────────────────────┤
│  Monthly Returns Grid                                        │
│  [Table: months vs years, cell = return %, color coded]      │
├─────────────────────────────────────────────────────────────┤
│  Parameters Used                                             │
│  [JSON viewer / diff view between two runs]                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.6 Page 6: Multi-Agent (`AgentsPage.tsx`)

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  Capital Allocation                                          │
│  [Donut chart: segments per agent, center = total capital]   │
│                                                              │
│  Momentum: 32% ($32,000)                                    │
│  Mean Rev: 28% ($28,000)                                    │
│  Event:    22% ($22,000)                                    │
│  Stat Arb: 18% ($18,000)                                    │
├─────────────────────────────────────────────────────────────┤
│  Agent Cards (grid)                                          │
│  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │ 🟢 Momentum          │  │ 🟢 Mean Reversion   │           │
│  │ Sharpe: 2.1          │  │ Sharpe: 1.4          │           │
│  │ P&L: +$4,200         │  │ P&L: +$1,800         │           │
│  │ Win: 62%  Trades: 84 │  │ Win: 58%  Trades: 67 │           │
│  │ Capital: $32,000     │  │ Capital: $28,000     │           │
│  │ [Pause] [Details]    │  │ [Pause] [Details]    │           │
│  │ [Mini equity curve]  │  │ [Mini equity curve]  │           │
│  └─────────────────────┘  └─────────────────────┘           │
│  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │ 🟡 Event Driven      │  │ 🔴 Stat Arb (weak)  │           │
│  │ Sharpe: 0.9          │  │ Sharpe: 0.3          │           │
│  │ ...                  │  │ ...                  │           │
│  └─────────────────────┘  └─────────────────────┘           │
├──────────────────────────────┬──────────────────────────────┤
│  Correlation Matrix           │  Diversification Score       │
│  [D3 heatmap: NxN agents]     │                              │
│                               │  Score: 0.72 / 1.0           │
│      Mom  MR   Evt  SA        │  [Progress ring]             │
│  Mom  1   .21  .35  .08       │                              │
│  MR   .21  1   .18  .42       │  Target: > 0.7  ✅            │
│  Evt  .35 .18   1   .11       │                              │
│  SA   .08 .42  .11   1        │  Highest pair:               │
│                               │  MR ↔ SA: 0.42              │
│  Color: blue=low, red=high    │                              │
└──────────────────────────────┴──────────────────────────────┘
```

**Correlation Matrix Component**:
```typescript
interface CorrelationMatrixProps {
  agents: string[]             // Agent names
  matrix: number[][]           // NxN correlation values (-1 to +1)
  threshold: number            // Highlight cells above this (e.g., 0.5)
}

// Implementation:
// - D3.js or custom SVG
// - Cell size proportional to container
// - Color scale: diverging blue (negative) → white (0) → red (positive)
// - Diagonal = 1.0 (gray)
// - Hover cell → tooltip with exact correlation value
// - Click cell → show return scatter plot of the two agents
// - Responsive: on mobile, rotate labels 45°
```

**Agent Card Component**:
```typescript
interface AgentCardProps {
  agent: {
    id: string
    name: string
    status: 'active' | 'paused' | 'deactivated'
    sharpe_30d: number
    total_pnl: number
    win_rate: number
    total_trades: number
    allocated_capital: number
    allocation_pct: number
    preferred_regimes: string[]
    equity_curve_mini: { timestamp: string; equity: number }[]  // Last 30 days
  }
  onPause: (id: string) => void
  onActivate: (id: string) => void
  onViewDetails: (id: string) => void
}

// Implementation:
// - Card with status indicator (green dot = active, yellow = paused, red = deactivated)
// - Mini sparkline equity curve (Recharts, 80px height, no axes)
// - Action buttons: Pause/Resume, View Details
// - Click card → navigate to agent detail page
// - Highlight card border when Sharpe < 0.5 (warning) or < 0 (danger)
```

### 3.7 Page 7: Execution Panel (`ExecutionPage.tsx`)

**Layout**:
```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️  EXECUTION PANEL — Manual actions affect live portfolio  │
├─────────────────────────────────────────────────────────────┤
│  Quick Actions                                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐     │
│  │ ⏸ PAUSE  │  │ ▶ RESUME │  │ 🛑 KILL SWITCH         │     │
│  │ Trading  │  │ Trading  │  │ (Close ALL + Pause)    │     │
│  └──────────┘  └──────────┘  └────────────────────────┘     │
├──────────────────────────────┬──────────────────────────────┤
│  Open Positions               │  Manual Order Entry          │
│  (with individual close btns) │                              │
│                               │  Instrument: [EURUSD ▼]     │
│  EURUSD BUY 0.5L +$125      │  Side: [BUY] [SELL]          │
│    [Close 50%] [Close 100%]  │  Type: [Market] [Limit]      │
│                               │  Quantity: [_______]         │
│  AAPL SELL 50sh +$60         │  Price: [_______ ] (if limit)│
│    [Close 50%] [Close 100%]  │  Stop Loss: [_______]        │
│                               │  Take Profit: [_______]      │
│  BTC BUY 0.02 +$13          │                              │
│    [Close 50%] [Close 100%]  │  [Submit Order]              │
│                               │                              │
├──────────────────────────────┴──────────────────────────────┤
│  Open Orders (pending)                                       │
│  ┌───────┬──────┬───────┬────────┬─────────┬───────────┐    │
│  │Instr. │ Side │ Type  │ Price  │ Status  │ Action    │    │
│  │GBPUSD │ BUY  │ LIMIT │ 1.2650 │ PENDING │ [Cancel]  │    │
│  └───────┴──────┴───────┴────────┴─────────┴───────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Activity Log (real-time, scrolling)                         │
│  14:32:05  Order submitted: BUY EURUSD 0.5L @ MARKET       │
│  14:32:06  Fill: EURUSD 0.5L @ 1.0842, slippage: 0.3 pips │
│  14:35:12  Signal: AAPL SELL, confidence 72%, score 0.68    │
│  14:35:14  Risk check: APPROVED                              │
│  14:35:15  Order submitted: SELL AAPL 50sh @ MARKET          │
└─────────────────────────────────────────────────────────────┘
```

**Kill Switch Component**:
```typescript
interface KillSwitchProps {
  isActive: boolean
  onActivate: () => void
  confirmationRequired: boolean   // Always true — requires double-click or hold
}

// Implementation:
// - Large red button
// - First click → shows confirmation modal:
//   "This will CLOSE ALL positions and PAUSE trading. Are you sure?"
//   [Cancel] [Confirm — Hold 3 seconds]
// - Confirm requires holding button for 3 seconds (progress animation)
// - After activation: button turns gray, shows "Kill switch active — Resume?"
// - WebSocket broadcast to all connected clients
```

---

## 4. Responsive Design

### 4.1 Breakpoints

```css
/* Tailwind defaults */
sm:  640px   /* Mobile landscape */
md:  768px   /* Tablet */
lg:  1024px  /* Desktop */
xl:  1280px  /* Wide desktop */
2xl: 1536px  /* Ultra-wide */
```

### 4.2 Mobile Navigation

```typescript
// Bottom tab bar on mobile (< 768px)
const MOBILE_TABS = [
  { icon: LayoutDashboard, label: "Overview", path: "/" },
  { icon: Target, label: "Signals", path: "/signals" },
  { icon: Briefcase, label: "Positions", path: "/positions" },
  { icon: BookOpen, label: "Journal", path: "/journal" },
  { icon: Zap, label: "Execute", path: "/execution" },
]

// Side drawer for Agents and Backtest (less frequent access on mobile)
```

### 4.3 Mobile-Specific Adaptations

| Component | Desktop | Mobile |
|---|---|---|
| Metric cards | 4-column grid | 2-column grid, smaller text |
| Equity curve | Full width, 400px height | Full width, 200px height, simplified |
| Trade table | Full columns | 3 columns: instrument, P&L, exit |
| Radar chart | 300x300 | 250x250, smaller labels |
| Risk gauges | 4 gauges in row | 2x2 grid |
| Correlation matrix | Full NxN with labels | Simplified heatmap, rotated labels |
| Execution panel | Side-by-side panels | Stacked, positions above order form |
| Kill switch | Inline button | Full-width sticky bottom bar |

### 4.4 Touch Interactions

- **Swipe left on position** → Reveal close buttons
- **Pull to refresh** → Refresh portfolio data
- **Long press on trade** → Show Claude reasoning popup
- **Pinch zoom on charts** → Zoom time range

---

## 5. Real-Time Data Flow

### 5.1 WebSocket Architecture

```
Trading Engine (Python)
    │
    ├─── writes to DuckDB ───┐
    │                         │
    ├─── emits events ────── Socket.IO Server (FastAPI)
                              │
                              ├──── ws://localhost:8001/ws ──── Browser Client
                              │                                     │
                              │                                     ├── usePortfolio() hook
                              │                                     ├── usePositions() hook
                              │                                     ├── useSignals() hook
                              │                                     └── useAlerts() hook
```

### 5.2 React Hooks for Real-Time Data

```typescript
// api/websocket.ts
import { io, Socket } from 'socket.io-client'

let socket: Socket | null = null

export function getSocket(): Socket {
  if (!socket) {
    socket = io('ws://localhost:8001', {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 10,
    })
  }
  return socket
}

// api/hooks.ts
export function usePortfolioLive(): PortfolioSummary | null {
  const [data, setData] = useState<PortfolioSummary | null>(null)
  
  useEffect(() => {
    const socket = getSocket()
    socket.on('portfolio:update', (payload: PortfolioSummary) => {
      setData(payload)
    })
    return () => { socket.off('portfolio:update') }
  }, [])
  
  // Also fetch initial via REST
  const { data: initial } = useQuery({
    queryKey: ['portfolio', 'summary'],
    queryFn: () => api.get('/api/portfolio/summary'),
    refetchInterval: 30_000,  // Fallback polling
  })
  
  return data ?? initial ?? null
}

export function usePositionsLive(): Position[] {
  const [positions, setPositions] = useState<Position[]>([])
  
  useEffect(() => {
    const socket = getSocket()
    socket.on('position:update', setPositions)
    return () => { socket.off('position:update') }
  }, [])
  
  return positions
}

export function useRiskAlerts(): RiskAlert[] {
  const [alerts, setAlerts] = useState<RiskAlert[]>([])
  
  useEffect(() => {
    const socket = getSocket()
    socket.on('risk:alert', (alert: RiskAlert) => {
      setAlerts(prev => [alert, ...prev].slice(0, 50))
      // Also show toast notification
      toast.warning(alert.message)
    })
    return () => { socket.off('risk:alert') }
  }, [])
  
  return alerts
}
```

### 5.3 Backend WebSocket Manager

```python
# dashboard/backend/ws.py
import socketio

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

class DashboardBroadcaster:
    """Broadcast trading engine events to all connected dashboard clients."""
    
    async def emit_portfolio_update(self, summary: dict):
        await sio.emit('portfolio:update', summary)
    
    async def emit_position_update(self, positions: list[dict]):
        await sio.emit('position:update', positions)
    
    async def emit_signal(self, signal: dict):
        await sio.emit('signal:new', signal)
    
    async def emit_trade_open(self, trade: dict):
        await sio.emit('trade:open', trade)
    
    async def emit_trade_close(self, trade: dict):
        await sio.emit('trade:close', trade)
    
    async def emit_risk_alert(self, alert: dict):
        await sio.emit('risk:alert', alert)
    
    async def emit_agent_status(self, status: dict):
        await sio.emit('agent:status', status)

broadcaster = DashboardBroadcaster()
```

---

## 6. Theme & Design System

### 6.1 Color Palette

```typescript
const colors = {
  // P&L colors
  profit: '#10B981',        // Emerald 500
  loss: '#EF4444',          // Red 500
  profitBg: '#10B98120',    // 12% opacity
  lossBg: '#EF444420',
  
  // Risk levels
  riskLow: '#10B981',       // Green
  riskModerate: '#F59E0B',  // Amber
  riskHigh: '#EF4444',      // Red
  riskCritical: '#991B1B',  // Red 800
  
  // Agent colors (consistent across all views)
  momentum: '#3B82F6',      // Blue
  meanReversion: '#8B5CF6', // Purple
  eventDriven: '#F59E0B',   // Amber
  statArb: '#06B6D4',       // Cyan
  
  // Chart colors
  equity: '#3B82F6',        // Blue
  drawdown: '#EF444440',    // Red 25%
  benchmark: '#9CA3AF',     // Gray 400
}
```

### 6.2 Chart Defaults

```typescript
const chartDefaults = {
  font: "'Inter', system-ui, sans-serif",
  fontSize: 12,
  gridColor: 'var(--border)',          // Tailwind CSS variable
  axisColor: 'var(--muted-foreground)',
  tooltipBg: 'var(--popover)',
  tooltipBorder: 'var(--border)',
  animation: { duration: 300, easing: 'ease-out' },
  responsive: true,
}
```

### 6.3 Dark Mode

- Use Tailwind `dark:` classes throughout
- Store preference in `localStorage`
- System preference detection via `prefers-color-scheme`
- All charts use CSS variables for colors → auto-adapt

---

## 7. Claude Code Implementation Commands

```bash
# Backend setup
Read DASHBOARD.md section 2 and implement the FastAPI backend with all route files, Pydantic schemas, and WebSocket manager. The backend reads from the same DuckDB used by the trading engine.

# Frontend scaffolding
Read DASHBOARD.md section 1.3 and scaffold the React frontend with Vite, TypeScript, Tailwind, shadcn/ui, and all directories/files listed.

# Type definitions
Implement src/types/index.ts with all TypeScript interfaces from DASHBOARD.md section 2 (API schemas).

# API layer
Implement src/api/client.ts, src/api/hooks.ts, and src/api/websocket.ts following DASHBOARD.md section 5.

# Layout components
Implement AppShell, Sidebar, Header, MobileNav, and StatusBar following the responsive specs in sections 3 and 4.

# Page 1: Dashboard Overview
Implement DashboardPage.tsx with MetricCard, EquityCurve, DrawdownChart, DailyPnLBars, and MonthlyReturnsHeatmap following section 3.1.

# Page 2: Signals
Implement SignalsPage.tsx with SignalRadar (Recharts RadarChart), factor detail panel, signal history timeline, and multi-timeframe view following section 3.2.

# Page 3: Positions + Risk
Implement PositionsPage.tsx with RiskGauge (SVG), position table, position HeatMap (D3 treemap), and exposure breakdown following section 3.3.

# Page 4: Trade Journal
Implement JournalPage.tsx with filterable TanStack Table, expandable ClaudeReasoning component, and attribution bars following section 3.4.

# Page 5: Backtest
Implement BacktestPage.tsx with overlaid equity curves, metrics comparison table, walk-forward analysis, and parameter diff view following section 3.5.

# Page 6: Multi-Agent
Implement AgentsPage.tsx with donut allocation chart, AgentCard grid, CorrelationMatrix (D3 heatmap), and diversification score following section 3.6.

# Page 7: Execution
Implement ExecutionPage.tsx with QuickActions, KillSwitch (with 3-second hold confirmation), manual OrderForm, and real-time activity log following section 3.7.

# Mobile optimization
Review all pages and apply mobile adaptations from section 4. Ensure bottom tab navigation works, charts resize properly, and touch interactions are implemented.

# Dark mode
Implement dark mode toggle in Header, store preference, and verify all components render correctly in both themes.
```

---

## 8. Testing Checklist

```
BACKEND:
[ ] All REST endpoints return correct schemas
[ ] WebSocket connects and receives portfolio:update
[ ] WebSocket reconnects after disconnect
[ ] DuckDB queries are read-only (no writes from dashboard)
[ ] CORS configured correctly for frontend dev server
[ ] Kill switch endpoint actually calls trading engine close_all

FRONTEND:
[ ] Dashboard page loads with mock data
[ ] Equity curve renders with time range selector
[ ] Radar chart animates on instrument change
[ ] Risk gauges show correct color zones
[ ] Trade journal table supports pagination, filtering, sorting
[ ] Claude reasoning expands/collapses in trade rows
[ ] Backtest comparison overlays multiple equity curves
[ ] Agent cards show mini sparkline equity curves
[ ] Correlation matrix renders NxN cells with correct colors
[ ] Kill switch requires 3-second hold confirmation
[ ] Manual order form validates inputs before submit

RESPONSIVE:
[ ] All pages render correctly on 375px width (iPhone SE)
[ ] Bottom tab navigation works on mobile
[ ] Charts resize without overflow
[ ] Tables scroll horizontally on narrow screens
[ ] Touch: swipe on position reveals close buttons
[ ] Touch: long press on trade shows reasoning

REAL-TIME:
[ ] Portfolio summary updates every 5 seconds via WebSocket
[ ] New trade notification appears as toast
[ ] Risk alert shows warning banner
[ ] Position P&L updates live (not on refresh)
[ ] System heartbeat reconnects if missed for 30 seconds

DARK MODE:
[ ] All text is readable in dark mode
[ ] Chart gridlines adapt to dark background
[ ] Profit/loss colors remain distinguishable
[ ] Modal overlays are not invisible
```
