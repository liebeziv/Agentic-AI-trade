# Claude Code Quick Start Guide

> 使用 Claude Code 构建 atlas-trader 项目的操作指南

---

## 文件说明

| 文件 | 内容 | 行数 |
|------|------|------|
| `PROJECT.md` | 项目总览：愿景、架构、技术栈、目录结构、设计决策 | 931 |
| `PHASE1.md` | 阶段1详细实现：模拟交易 + Claude API + 技术指标 + 信号融合 | 1618 |
| `PHASE2.md` | 阶段2详细实现：实盘执行 + 交易日志 + 归因分析 | 917 |
| `PHASE3.md` | 阶段3详细实现：多Agent协同 + 策略编排 + 全局风控 | 1079 |
| `DASHBOARD.md` | React Web App 全栈仪表盘：7 页面 + WebSocket + 移动端 | 1070 |

## 使用方法

### 1. 初始化项目

将四个 MD 文件放入项目根目录，然后在终端中运行 Claude Code：

```bash
# 创建项目目录
mkdir atlas-trader && cd atlas-trader

# 把四个 MD 文件复制到此目录
# PROJECT.md, PHASE1.md, PHASE2.md, PHASE3.md

# 启动 Claude Code，开始搭建
claude
```

### 2. Phase 1 — 逐步构建命令

在 Claude Code 中按顺序执行以下指令：

```
# Step 1: 项目脚手架
Read PROJECT.md and PHASE1.md, then scaffold the complete project structure including pyproject.toml, all config files, and src/types.py with shared data classes.

# Step 2: 数据层
Implement src/data/data_store.py and src/data/market_data.py following PHASE1.md sections 3.1-3.2. Include DuckDB schema creation and yfinance/ccxt data fetching.

# Step 3: 新闻 + 日历
Implement src/data/news_feed.py and src/data/economic_calendar.py following PHASE1.md section 4.

# Step 4: 技术指标引擎
Implement src/analysis/technical.py following PHASE1.md section 5. Include all 15+ indicators, signal generation, trend bias computation, and multi-timeframe analysis.

# Step 5: Claude API Agent
Implement src/analysis/claude_agent.py following PHASE1.md section 6. Include system prompt, analysis prompt builder, reflection prompt, JSON parsing, cost tracking, and error handling.

# Step 6: 市场状态检测
Implement src/analysis/regime_detector.py following PHASE1.md section 7. Include feature engineering and rule-based classification.

# Step 7: 信号聚合器
Implement src/strategy/signal_aggregator.py following PHASE1.md section 8. Include technical score, regime score, composite calculation.

# Step 8: 风控管理器
Implement src/strategy/risk_manager.py following PHASE1.md section 9.1. Include all hard rules and soft rules with the full check pipeline.

# Step 9: 仓位计算器
Implement src/strategy/position_sizer.py following PHASE1.md section 9.2. Include Half-Kelly, ATR-based, and fixed fractional methods.

# Step 10: 模拟交易引擎
Implement src/execution/paper_engine.py following PHASE1.md section 10. Include realistic slippage, fill simulation, stop-loss checking, and mark-to-market.

# Step 11: 交易日志
Implement src/reflection/trade_journal.py following PHASE1.md step 11 and the TradeRecord schema in types.py.

# Step 12: 主交易循环
Implement scripts/run_paper.py following PHASE1.md section 11. Wire all modules together in the async trading loop.

# Step 13: 回测框架
Implement scripts/run_backtest.py following PHASE1.md section 12. Include walk-forward optimization and BacktestResults metrics.

# Step 14: 仪表盘
Implement dashboard/app.py as a Streamlit app with equity curve, signal monitor, risk gauge, and trade log pages.

# Step 15: 测试
Write and run all tests from PHASE1.md section 14 testing checklist. Fix any issues found.
```

### 3. Phase 2 — 实盘对接

```
# 先读取 Phase 2 规格
Read PHASE2.md and implement the exchange adapter abstract interface and adapter factory (section 2.1-2.2).

# 逐个实现交易所适配器
Implement the Alpaca adapter following PHASE2.md section 2.3.
Implement the Binance adapter following PHASE2.md section 2.4.
Implement the IB adapter following PHASE2.md section 2.5.
Implement the MT5 adapter following PHASE2.md section 2.6.

# 订单执行引擎
Implement the order execution engine following PHASE2.md section 3.

# 日志 + 归因
Implement the enhanced trade journal following PHASE2.md section 4.
Implement the attribution engine following PHASE2.md section 5.

# 通知系统
Implement the Telegram notifier following PHASE2.md section 6.

# 实盘交易循环
Implement run_live.py with pre-flight checks, position sync, kill switch, and gradual deployment following PHASE2.md section 7.
```

### 4. Phase 3 — 多Agent协同

```
# Agent 基础架构
Read PHASE3.md and implement the base agent class and agent registry (section 2).

# 专业Agent
Implement the Momentum agent following PHASE3.md section 3.1.
Implement the Mean Reversion agent following PHASE3.md section 3.2.
Implement the Event Driven agent following PHASE3.md section 3.3.
Implement the Statistical Arbitrage agent following PHASE3.md section 3.4.

# 编排器
Implement the Orchestrator with capital allocation and conflict resolution following PHASE3.md section 4.

# 全局风控 + 相关性监控
Implement the Global Risk Manager following PHASE3.md section 5.
Implement the Correlation Monitor following PHASE3.md section 6.

# 策略进化
Implement the Strategy Evolver following PHASE3.md section 7.

# 多Agent交易循环
Implement run_multi_agent.py following PHASE3.md section 8.
```

### 5. Dashboard — React Web App

```
# 后端
Read DASHBOARD.md section 2 and implement the FastAPI backend with all route files, schemas, and WebSocket manager.

# 前端脚手架
Read DASHBOARD.md section 1.3 and scaffold the React frontend with Vite + TypeScript + Tailwind + shadcn/ui.

# 逐页实现
Implement DashboardPage following DASHBOARD.md section 3.1 with equity curve, metric cards, P&L bars.
Implement SignalsPage following DASHBOARD.md section 3.2 with radar chart and factor panel.
Implement PositionsPage following DASHBOARD.md section 3.3 with risk gauges, heatmap, position table.
Implement JournalPage following DASHBOARD.md section 3.4 with trade table and Claude reasoning.
Implement BacktestPage following DASHBOARD.md section 3.5 with comparison charts.
Implement AgentsPage following DASHBOARD.md section 3.6 with allocation donut and correlation matrix.
Implement ExecutionPage following DASHBOARD.md section 3.7 with kill switch and order form.

# 移动端 + 暗黑模式
Apply mobile responsive adaptations from DASHBOARD.md section 4 to all pages.
Implement dark mode toggle and verify all components.
```

---

## 关键提醒

1. **始终从 paper trading 开始** — 配置文件中 `mode: "paper"` 和 `paper: True`
2. **API keys 用 .env 管理** — 永远不要硬编码密钥
3. **每完成一个 Step 就运行测试** — 不要等到最后
4. **风控模块是第一优先级** — 任何时候都不能绕过风控
5. **Claude API 成本控制** — 默认用 Sonnet，关键决策用 Opus
