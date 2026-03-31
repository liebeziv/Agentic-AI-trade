# Phase 3: Multi-Agent Orchestration

> **Duration**: Weeks 17+
> **Mode**: Live trading, multiple independent strategies
> **Prerequisite**: Phase 2 exit criteria met (3mo live profit, DD < 15%, > 100 trades)
> **Exit Criteria**: Portfolio Sharpe > 2.0, inter-strategy correlation < 0.3

---

## 1. Implementation Order

```
Step 1:  Agent base class + agent registry
Step 2:  Specialist agent — Momentum
Step 3:  Specialist agent — Mean Reversion
Step 4:  Specialist agent — Event Driven
Step 5:  Specialist agent — Statistical Arbitrage
Step 6:  Orchestrator — capital allocation engine
Step 7:  Orchestrator — conflict resolution
Step 8:  Global risk manager (portfolio-level)
Step 9:  Correlation monitor
Step 10: Strategy evolution loop (weekly)
Step 11: Multi-agent debate protocol (Claude-powered)
Step 12: Dashboard enhancements (per-agent views)
Step 13: Performance analytics + reporting
Step 14: Integration testing (all agents + orchestrator)
```

---

## 2. Agent Architecture

### 2.1 Base Agent Class (`src/strategy/strategies/base_agent.py`)

```python
from abc import ABC, abstractmethod

class BaseStrategyAgent(ABC):
    """Base class for all specialist trading agents."""
    
    def __init__(self, agent_id: str, config: dict, store: DataStore,
                 claude_agent: ClaudeAgent, technical: TechnicalEngine):
        self.agent_id = agent_id
        self.config = config
        self.store = store
        self.claude = claude_agent
        self.technical = technical
        self.is_active = True
        self.allocated_capital = 0.0
        self.positions: list[Position] = []
        self.performance = AgentPerformance()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
    
    @property
    @abstractmethod
    def description(self) -> str:
        """What this strategy does."""
    
    @property
    @abstractmethod
    def preferred_regimes(self) -> list[MarketRegime]:
        """Which market regimes this strategy performs best in."""
    
    @property
    @abstractmethod
    def target_instruments(self) -> list[str]:
        """Instruments this agent trades."""
    
    @abstractmethod
    async def generate_signals(
        self, market_data: dict[str, pl.DataFrame],
        technical: dict[str, TechnicalSnapshot],
        news: list[NewsItem],
        regime: MarketRegime,
    ) -> list[SignalScore]:
        """Generate trading signals. May return multiple signals."""
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """Return current strategy parameters (for logging/optimization)."""
    
    @abstractmethod
    def set_parameters(self, params: dict) -> None:
        """Update strategy parameters."""
    
    async def evaluate_fitness(self, regime: MarketRegime) -> float:
        """How well-suited is this agent for the current regime? 0.0-1.0"""
        if regime in self.preferred_regimes:
            return 0.8 + 0.2 * (self.performance.rolling_sharpe_30d / 3.0)
        return max(0.1, 0.4 * (self.performance.rolling_sharpe_30d / 3.0))
    
    def update_performance(self, trades: list[TradeRecord]) -> None:
        """Update rolling performance metrics."""
        self.performance.update(trades)


@dataclass
class AgentPerformance:
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    rolling_sharpe_30d: float = 0.0
    rolling_sharpe_90d: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    daily_returns: list[float] = field(default_factory=list)
    
    def update(self, trades: list[TradeRecord]) -> None:
        if not trades:
            return
        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.pnl > 0)
        self.total_pnl = sum(t.pnl for t in trades)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades else 0
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl <= 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 1
        self.profit_factor = gross_profit / gross_loss
        self.avg_trade_pnl = self.total_pnl / self.total_trades
        
        # Rolling Sharpe (annualized from daily returns)
        if len(self.daily_returns) >= 20:
            returns = np.array(self.daily_returns[-30:])
            self.rolling_sharpe_30d = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        if len(self.daily_returns) >= 60:
            returns = np.array(self.daily_returns[-90:])
            self.rolling_sharpe_90d = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        self.last_updated = datetime.utcnow()
```

### 2.2 Agent Registry (`src/orchestrator/agent_registry.py`)

```python
class AgentRegistry:
    """Manages the lifecycle of all strategy agents."""
    
    def __init__(self):
        self._agents: dict[str, BaseStrategyAgent] = {}
    
    def register(self, agent: BaseStrategyAgent) -> None:
        self._agents[agent.agent_id] = agent
        log.info("Agent registered", agent_id=agent.agent_id, name=agent.name)
    
    def deactivate(self, agent_id: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].is_active = False
    
    def activate(self, agent_id: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].is_active = True
    
    def get_active_agents(self) -> list[BaseStrategyAgent]:
        return [a for a in self._agents.values() if a.is_active]
    
    def get_agent(self, agent_id: str) -> BaseStrategyAgent | None:
        return self._agents.get(agent_id)
    
    def get_all_agents(self) -> list[BaseStrategyAgent]:
        return list(self._agents.values())
    
    def get_agent_summary(self) -> list[dict]:
        return [
            {
                "id": a.agent_id,
                "name": a.name,
                "active": a.is_active,
                "capital": a.allocated_capital,
                "positions": len(a.positions),
                "sharpe_30d": a.performance.rolling_sharpe_30d,
                "total_pnl": a.performance.total_pnl,
                "win_rate": a.performance.win_rate,
            }
            for a in self._agents.values()
        ]
```

---

## 3. Specialist Agents

### 3.1 Momentum Agent (`src/strategy/strategies/momentum.py`)

```python
class MomentumAgent(BaseStrategyAgent):
    """Trend-following agent. Buys strength, sells weakness."""
    
    name = "Momentum"
    description = "Follows established trends using MA alignment, ADX, and breakout patterns."
    preferred_regimes = [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN, MarketRegime.BREAKOUT]
    
    DEFAULT_PARAMS = {
        "fast_ma": 9,
        "medium_ma": 21,
        "slow_ma": 50,
        "adx_threshold": 25,
        "rsi_entry_long": (40, 70),     # Buy when RSI between 40-70 (not overbought)
        "rsi_entry_short": (30, 60),
        "atr_stop_multiple": 2.0,
        "atr_tp_multiple": 3.0,
        "volume_confirmation": True,
        "min_bars_in_trend": 5,
    }
    
    @property
    def target_instruments(self) -> list[str]:
        return self.config.get("instruments", [
            "EURUSD", "GBPUSD", "ES", "NQ", "BTC/USDT",
            "AAPL", "NVDA", "TSLA"
        ])
    
    async def generate_signals(self, market_data, technical, news, regime) -> list[SignalScore]:
        signals = []
        
        for instrument in self.target_instruments:
            snap = technical.get(instrument, {}).get("15m")
            if not snap:
                continue
            
            ind = snap.indicators
            p = self.params
            
            # Entry conditions: LONG
            ma_aligned_up = (
                ind.get("ema_9", 0) > ind.get("ema_21", 0) > ind.get("sma_50", 0)
            )
            adx_strong = ind.get("adx_14", 0) > p["adx_threshold"]
            rsi_ok_long = p["rsi_entry_long"][0] <= ind.get("rsi_14", 50) <= p["rsi_entry_long"][1]
            volume_ok = ind.get("obv_slope", 0) > 0 if p["volume_confirmation"] else True
            
            if ma_aligned_up and adx_strong and rsi_ok_long and volume_ok:
                atr = ind.get("atr_14", 0)
                close = ind.get("close", 0)
                signals.append(SignalScore(
                    instrument=instrument,
                    timestamp=datetime.utcnow(),
                    technical_score=0.7,
                    regime_score=0.8 if regime in self.preferred_regimes else 0.2,
                    composite_score=0.65,
                    action=TradeAction.BUY,
                    recommendation=TradeRecommendation(
                        action=TradeAction.BUY,
                        instrument=instrument,
                        confidence=70,
                        entry_price=close,
                        stop_loss=close - atr * p["atr_stop_multiple"],
                        take_profit_1=close + atr * p["atr_tp_multiple"],
                        risk_reward_ratio=p["atr_tp_multiple"] / p["atr_stop_multiple"],
                        reasoning=f"Momentum long: MA aligned up, ADX={ind.get('adx_14',0):.0f}",
                        key_factors=["ma_alignment", "adx_strong", "volume_confirm"],
                    ),
                ))
            
            # Entry conditions: SHORT (mirror logic)
            ma_aligned_down = (
                ind.get("ema_9", 0) < ind.get("ema_21", 0) < ind.get("sma_50", 0)
            )
            rsi_ok_short = p["rsi_entry_short"][0] <= ind.get("rsi_14", 50) <= p["rsi_entry_short"][1]
            volume_down = ind.get("obv_slope", 0) < 0 if p["volume_confirmation"] else True
            
            if ma_aligned_down and adx_strong and rsi_ok_short and volume_down:
                atr = ind.get("atr_14", 0)
                close = ind.get("close", 0)
                signals.append(SignalScore(
                    instrument=instrument,
                    timestamp=datetime.utcnow(),
                    technical_score=-0.7,
                    regime_score=0.8 if regime in self.preferred_regimes else 0.2,
                    composite_score=-0.65,
                    action=TradeAction.SELL,
                    recommendation=TradeRecommendation(
                        action=TradeAction.SELL,
                        instrument=instrument,
                        confidence=70,
                        entry_price=close,
                        stop_loss=close + atr * p["atr_stop_multiple"],
                        take_profit_1=close - atr * p["atr_tp_multiple"],
                        risk_reward_ratio=p["atr_tp_multiple"] / p["atr_stop_multiple"],
                        reasoning=f"Momentum short: MA aligned down, ADX={ind.get('adx_14',0):.0f}",
                        key_factors=["ma_alignment_bearish", "adx_strong", "volume_confirm"],
                    ),
                ))
        
        return signals
```

### 3.2 Mean Reversion Agent (`src/strategy/strategies/mean_reversion.py`)

```python
class MeanReversionAgent(BaseStrategyAgent):
    """Fades extremes. Buys oversold, sells overbought in ranging markets."""
    
    name = "Mean Reversion"
    description = "Trades reversals from Bollinger Band extremes and RSI divergences."
    preferred_regimes = [MarketRegime.RANGE_BOUND, MarketRegime.WEAK_TREND]
    
    DEFAULT_PARAMS = {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "adx_max": 20,                  # Only trade when ADX < 20 (ranging)
        "atr_stop_multiple": 1.5,
        "target_bb_mid": True,           # TP at middle Bollinger Band
        "require_rsi_divergence": False,  # Advanced: price new low but RSI higher
        "max_holding_bars": 20,          # Time exit after N bars
    }
    
    async def generate_signals(self, market_data, technical, news, regime) -> list[SignalScore]:
        signals = []
        
        for instrument in self.target_instruments:
            snap = technical.get(instrument, {}).get("15m")
            if not snap:
                continue
            
            ind = snap.indicators
            p = self.params
            
            # Only trade in ranging markets
            if ind.get("adx_14", 30) > p["adx_max"]:
                continue
            
            close = ind.get("close", 0)
            bb_lower = ind.get("bbands_lower", 0)
            bb_upper = ind.get("bbands_upper", 0)
            bb_mid = ind.get("bbands_mid", 0)
            rsi = ind.get("rsi_14", 50)
            atr = ind.get("atr_14", 0)
            
            # LONG: Price at lower BB + RSI oversold
            if close <= bb_lower and rsi <= p["rsi_oversold"]:
                tp = bb_mid if p["target_bb_mid"] else close + atr * 2
                signals.append(SignalScore(
                    instrument=instrument,
                    timestamp=datetime.utcnow(),
                    technical_score=0.6,
                    regime_score=0.7 if regime in self.preferred_regimes else 0.0,
                    composite_score=0.55,
                    action=TradeAction.BUY,
                    recommendation=TradeRecommendation(
                        action=TradeAction.BUY,
                        instrument=instrument,
                        confidence=65,
                        entry_price=close,
                        stop_loss=close - atr * p["atr_stop_multiple"],
                        take_profit_1=tp,
                        risk_reward_ratio=abs(tp - close) / (atr * p["atr_stop_multiple"]),
                        reasoning=f"Mean reversion long: Below BB lower, RSI={rsi:.0f}",
                        key_factors=["bb_oversold", "rsi_extreme", "range_bound"],
                    ),
                ))
            
            # SHORT: Price at upper BB + RSI overbought
            if close >= bb_upper and rsi >= p["rsi_overbought"]:
                tp = bb_mid if p["target_bb_mid"] else close - atr * 2
                signals.append(SignalScore(
                    instrument=instrument,
                    timestamp=datetime.utcnow(),
                    technical_score=-0.6,
                    regime_score=0.7 if regime in self.preferred_regimes else 0.0,
                    composite_score=-0.55,
                    action=TradeAction.SELL,
                    recommendation=TradeRecommendation(
                        action=TradeAction.SELL,
                        instrument=instrument,
                        confidence=65,
                        entry_price=close,
                        stop_loss=close + atr * p["atr_stop_multiple"],
                        take_profit_1=tp,
                        risk_reward_ratio=abs(close - tp) / (atr * p["atr_stop_multiple"]),
                        reasoning=f"Mean reversion short: Above BB upper, RSI={rsi:.0f}",
                        key_factors=["bb_overbought", "rsi_extreme", "range_bound"],
                    ),
                ))
        
        return signals
```

### 3.3 Event-Driven Agent (`src/strategy/strategies/event_driven.py`)

```python
class EventDrivenAgent(BaseStrategyAgent):
    """Trades around news events and sentiment shifts. Heavy Claude API usage."""
    
    name = "Event Driven"
    description = "Uses Claude API to analyze news impact and trade sentiment-driven moves."
    preferred_regimes = [MarketRegime.BREAKOUT, MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]
    
    DEFAULT_PARAMS = {
        "min_news_relevance": 0.7,
        "min_claude_confidence": 70,
        "sentiment_threshold": 0.6,       # Absolute sentiment score to trigger
        "max_news_age_hours": 2,
        "atr_stop_multiple": 2.5,         # Wider stops for event trades
        "atr_tp_multiple": 4.0,           # Larger targets (events move more)
        "require_technical_alignment": True,
    }
    
    async def generate_signals(self, market_data, technical, news, regime) -> list[SignalScore]:
        signals = []
        
        # Filter high-relevance recent news
        relevant_news = [
            n for n in news
            if n.relevance_score >= self.params["min_news_relevance"]
            and (datetime.utcnow() - n.timestamp).total_seconds() < self.params["max_news_age_hours"] * 3600
        ]
        
        if not relevant_news:
            return signals
        
        # Group news by instrument
        instrument_news: dict[str, list[NewsItem]] = {}
        for n in relevant_news:
            for inst in n.instruments:
                instrument_news.setdefault(inst, []).append(n)
        
        for instrument, inst_news in instrument_news.items():
            snap = technical.get(instrument, {}).get("15m")
            if not snap:
                continue
            
            # Ask Claude to analyze the news impact
            recommendation = await self.claude.analyze_market(
                instrument=instrument,
                technical={tf: technical.get(instrument, {}).get(tf) for tf in ["5m", "15m", "1h"]},
                news=inst_news,
                regime=regime,
                recent_trades=self.store.get_trades(limit=5),
                portfolio_state=self._get_portfolio_state(),
            )
            
            if not recommendation or recommendation.action == TradeAction.NO_TRADE:
                continue
            
            if recommendation.confidence < self.params["min_claude_confidence"]:
                continue
            
            # Optional: require technical alignment
            if self.params["require_technical_alignment"]:
                if recommendation.action in [TradeAction.BUY, TradeAction.STRONG_BUY]:
                    if snap.trend_bias == "bearish":
                        continue
                elif recommendation.action in [TradeAction.SELL, TradeAction.STRONG_SELL]:
                    if snap.trend_bias == "bullish":
                        continue
            
            signals.append(SignalScore(
                instrument=instrument,
                timestamp=datetime.utcnow(),
                sentiment_score=recommendation.confidence / 100,
                claude_confidence=recommendation.confidence / 100,
                composite_score=recommendation.confidence / 100 * (1 if "BUY" in recommendation.action.value else -1),
                action=recommendation.action,
                recommendation=recommendation,
            ))
        
        return signals
```

### 3.4 Statistical Arbitrage Agent (`src/strategy/strategies/stat_arb.py`)

```python
class StatArbAgent(BaseStrategyAgent):
    """Trades mean-reverting spreads between correlated instruments."""
    
    name = "Statistical Arbitrage"
    description = "Identifies and trades correlated pairs when spread deviates from equilibrium."
    preferred_regimes = [MarketRegime.RANGE_BOUND, MarketRegime.WEAK_TREND]
    
    DEFAULT_PARAMS = {
        "pairs": [
            ("AAPL", "MSFT"),
            ("GOOGL", "META"),
            ("GC", "SLV"),           # Gold vs Silver
            ("ES", "NQ"),            # S&P vs Nasdaq
            ("BTC/USDT", "ETH/USDT"),
        ],
        "lookback_period": 60,        # Days for cointegration test
        "z_score_entry": 2.0,
        "z_score_exit": 0.5,
        "z_score_stop": 3.5,
        "min_correlation": 0.7,
        "cointegration_pvalue": 0.05,
        "retest_interval_days": 7,
    }
    
    async def generate_signals(self, market_data, technical, news, regime) -> list[SignalScore]:
        signals = []
        
        for asset_a, asset_b in self.params["pairs"]:
            # Get price series
            bars_a = market_data.get(asset_a)
            bars_b = market_data.get(asset_b)
            if bars_a is None or bars_b is None:
                continue
            
            # Compute spread
            prices_a = bars_a["close"].to_numpy()
            prices_b = bars_b["close"].to_numpy()
            
            # Correlation check
            corr = np.corrcoef(prices_a[-self.params["lookback_period"]:],
                              prices_b[-self.params["lookback_period"]:])[0, 1]
            if abs(corr) < self.params["min_correlation"]:
                continue
            
            # Compute hedge ratio (OLS)
            from numpy.polynomial.polynomial import polyfit
            beta, alpha = polyfit(prices_b, prices_a, 1)
            spread = prices_a - beta * prices_b - alpha
            
            # Z-score
            spread_mean = np.mean(spread[-self.params["lookback_period"]:])
            spread_std = np.std(spread[-self.params["lookback_period"]:])
            if spread_std == 0:
                continue
            z_score = (spread[-1] - spread_mean) / spread_std
            
            # Entry signals
            p = self.params
            if z_score > p["z_score_entry"]:
                # Spread too wide: short A, long B (expect convergence)
                signals.append(self._create_pair_signal(
                    asset_a, asset_b, "SHORT_A_LONG_B", z_score, beta, spread_std
                ))
            elif z_score < -p["z_score_entry"]:
                # Spread too narrow: long A, short B
                signals.append(self._create_pair_signal(
                    asset_a, asset_b, "LONG_A_SHORT_B", z_score, beta, spread_std
                ))
        
        return signals
```

---

## 4. Orchestrator

### 4.1 Capital Allocation Engine (`src/orchestrator/orchestrator.py`)

```python
class Orchestrator:
    """Manages multi-agent capital allocation, conflict resolution, and portfolio coordination."""
    
    def __init__(self, registry: AgentRegistry, risk_manager: GlobalRiskManager,
                 store: DataStore, notifier: Notifier):
        self.registry = registry
        self.risk_mgr = risk_manager
        self.store = store
        self.notifier = notifier
        self.total_capital = 0.0
        self.allocation_weights: dict[str, float] = {}
    
    async def allocate_capital(self, total_capital: float, current_regime: MarketRegime) -> dict[str, float]:
        """Compute capital allocation per agent based on performance + regime fit."""
        agents = self.registry.get_active_agents()
        
        if not agents:
            return {}
        
        self.total_capital = total_capital
        
        # 1. Compute fitness scores
        fitness = {}
        for agent in agents:
            fitness[agent.agent_id] = await agent.evaluate_fitness(current_regime)
        
        # 2. Performance overlay (rolling Sharpe)
        sharpe_scores = {}
        for agent in agents:
            sharpe = max(0, agent.performance.rolling_sharpe_30d)
            sharpe_scores[agent.agent_id] = sharpe
        
        # 3. Correlation penalty
        correlation_penalty = self._compute_correlation_penalties(agents)
        
        # 4. Compute raw weights
        raw_weights = {}
        for agent in agents:
            aid = agent.agent_id
            raw = (
                fitness[aid] * 0.4 +
                min(sharpe_scores[aid] / 3.0, 1.0) * 0.4 +  # Cap at Sharpe 3
                (1.0 - correlation_penalty.get(aid, 0)) * 0.2
            )
            raw_weights[aid] = max(0.05, raw)  # Min 5% allocation
        
        # 5. Normalize to sum to 1.0
        total = sum(raw_weights.values())
        self.allocation_weights = {k: v / total for k, v in raw_weights.items()}
        
        # 6. Assign capital
        allocations = {}
        for agent in agents:
            allocated = total_capital * self.allocation_weights[agent.agent_id]
            agent.allocated_capital = allocated
            allocations[agent.agent_id] = allocated
        
        log.info("Capital allocated", allocations={
            aid: f"${amt:,.0f} ({self.allocation_weights[aid]:.0%})"
            for aid, amt in allocations.items()
        })
        
        return allocations
    
    async def resolve_conflicts(self, all_signals: dict[str, list[SignalScore]]) -> list[SignalScore]:
        """Resolve conflicting signals from different agents on the same instrument."""
        # Group by instrument
        instrument_signals: dict[str, list[tuple[str, SignalScore]]] = {}
        for agent_id, signals in all_signals.items():
            for signal in signals:
                instrument_signals.setdefault(signal.instrument, []).append((agent_id, signal))
        
        resolved = []
        for instrument, agent_signals in instrument_signals.items():
            if len(agent_signals) == 1:
                resolved.append(agent_signals[0][1])
                continue
            
            # Multiple agents on same instrument
            buy_signals = [(aid, s) for aid, s in agent_signals if "BUY" in s.action.value]
            sell_signals = [(aid, s) for aid, s in agent_signals if "SELL" in s.action.value]
            
            if buy_signals and sell_signals:
                # CONFLICT: agents disagree
                # Resolution: weighted vote by allocation weight * confidence
                buy_weight = sum(
                    self.allocation_weights.get(aid, 0) * s.claude_confidence
                    for aid, s in buy_signals
                )
                sell_weight = sum(
                    self.allocation_weights.get(aid, 0) * s.claude_confidence
                    for aid, s in sell_signals
                )
                
                if buy_weight > sell_weight * 1.3:  # Need 30% margin to override
                    # Take strongest buy signal
                    best = max(buy_signals, key=lambda x: x[1].composite_score)
                    resolved.append(best[1])
                    log.info("Conflict resolved: BUY wins", instrument=instrument)
                elif sell_weight > buy_weight * 1.3:
                    best = max(sell_signals, key=lambda x: abs(x[1].composite_score))
                    resolved.append(best[1])
                    log.info("Conflict resolved: SELL wins", instrument=instrument)
                else:
                    # Too close to call — skip
                    log.info("Conflict unresolved: NO_TRADE", instrument=instrument)
            else:
                # All agree on direction — take strongest signal
                best = max(agent_signals, key=lambda x: abs(x[1].composite_score))
                resolved.append(best[1])
        
        return resolved
    
    def _compute_correlation_penalties(self, agents: list[BaseStrategyAgent]) -> dict[str, float]:
        """Penalize agents whose returns are highly correlated with others."""
        penalties = {}
        daily_returns = {}
        
        for agent in agents:
            if len(agent.performance.daily_returns) >= 20:
                daily_returns[agent.agent_id] = np.array(agent.performance.daily_returns[-30:])
        
        for aid in daily_returns:
            correlations = []
            for other_aid, other_returns in daily_returns.items():
                if aid != other_aid and len(other_returns) == len(daily_returns[aid]):
                    corr = np.corrcoef(daily_returns[aid], other_returns)[0, 1]
                    correlations.append(abs(corr))
            
            penalties[aid] = max(correlations) if correlations else 0.0
        
        return penalties
```

### 4.2 Multi-Agent Debate Protocol

```python
class AgentDebate:
    """Claude-powered debate between agents for high-stakes decisions."""
    
    async def run_debate(
        self, instrument: str, signals: dict[str, SignalScore],
        market_context: dict
    ) -> TradeRecommendation:
        """
        When agents disagree on a high-value trade, run a structured debate.
        Each agent presents its case. Claude synthesizes a final decision.
        """
        debate_prompt = f"""
You are moderating a debate between trading agents about {instrument}.

MARKET CONTEXT:
{json.dumps(market_context, indent=2)}

AGENT POSITIONS:
"""
        for agent_id, signal in signals.items():
            rec = signal.recommendation
            debate_prompt += f"""
--- {agent_id} ({signal.action.value}, confidence={signal.claude_confidence:.0%}) ---
Reasoning: {rec.reasoning if rec else 'N/A'}
Key factors: {rec.key_factors if rec else []}
Entry: {rec.entry_price if rec else 'N/A'}
Stop: {rec.stop_loss if rec else 'N/A'}
"""
        
        debate_prompt += """
As the senior portfolio manager, synthesize these views.
Consider: Which agent has the strongest evidence? Do any risk warnings override the opportunity?
Provide your FINAL DECISION as JSON (same format as TradeRecommendation).
"""
        
        response = await self.claude.raw_query(debate_prompt)
        return parse_recommendation(response)
```

---

## 5. Global Risk Manager

### 5.1 Implementation (`src/orchestrator/global_risk_manager.py`)

```python
class GlobalRiskManager:
    """Portfolio-level risk management across all agents."""
    
    def __init__(self, config: dict):
        self.limits = config
        self.agent_risk_budgets: dict[str, float] = {}
    
    def check_portfolio_signal(
        self, signal: SignalScore, agent_id: str,
        all_positions: list[Position], portfolio: PortfolioState
    ) -> RiskCheckResult:
        """Check signal against portfolio-level risk limits."""
        
        checks = []
        
        # 1. Total portfolio drawdown
        if portfolio.max_drawdown_current >= self.limits["max_drawdown_pct"]:
            return RiskCheckResult(approved=False, reason="Portfolio drawdown limit")
        checks.append("portfolio_dd_ok")
        
        # 2. Cross-agent exposure to same instrument
        same_instrument = [p for p in all_positions if p.instrument == signal.instrument]
        if len(same_instrument) >= 2:
            return RiskCheckResult(approved=False, reason="Max exposure per instrument across agents")
        checks.append("instrument_exposure_ok")
        
        # 3. Directional concentration
        long_exposure = sum(p.quantity * p.current_price for p in all_positions if p.side == Side.BUY)
        short_exposure = sum(p.quantity * p.current_price for p in all_positions if p.side == Side.SELL)
        net_exposure_pct = abs(long_exposure - short_exposure) / portfolio.equity * 100
        if net_exposure_pct > self.limits.get("max_net_exposure_pct", 80):
            return RiskCheckResult(approved=False, reason=f"Net exposure {net_exposure_pct:.0f}% too high")
        checks.append("directional_ok")
        
        # 4. Agent's risk budget
        agent_used = sum(
            abs(p.entry_price - (p.stop_loss or p.entry_price)) * p.quantity
            for p in all_positions if p.instrument in self._get_agent_instruments(agent_id)
        )
        agent_budget = self.agent_risk_budgets.get(agent_id, portfolio.equity * 0.25)
        if agent_used >= agent_budget:
            return RiskCheckResult(approved=False, reason=f"Agent {agent_id} risk budget exhausted")
        checks.append("agent_budget_ok")
        
        # 5. Sector/correlation limit
        correlated = self._count_correlated_positions(signal.instrument, all_positions)
        if correlated >= self.limits.get("max_correlated_cross_agent", 4):
            return RiskCheckResult(approved=False, reason="Cross-agent correlated position limit")
        checks.append("correlation_ok")
        
        return RiskCheckResult(approved=True, checks_passed=checks)
```

---

## 6. Correlation Monitor

### 6.1 Implementation (`src/orchestrator/correlation_monitor.py`)

```python
class CorrelationMonitor:
    """Monitors inter-agent and inter-position correlations."""
    
    def __init__(self, registry: AgentRegistry, store: DataStore):
        self.registry = registry
        self.store = store
    
    def compute_agent_correlation_matrix(self, lookback_days: int = 30) -> pd.DataFrame:
        """Compute return correlation between all active agents."""
        agents = self.registry.get_active_agents()
        returns_data = {}
        
        for agent in agents:
            if len(agent.performance.daily_returns) >= lookback_days:
                returns_data[agent.agent_id] = agent.performance.daily_returns[-lookback_days:]
        
        if len(returns_data) < 2:
            return pd.DataFrame()
        
        df = pd.DataFrame(returns_data)
        return df.corr()
    
    def get_diversification_score(self) -> float:
        """
        0 = all agents perfectly correlated (no diversification)
        1 = all agents uncorrelated (maximum diversification)
        """
        corr_matrix = self.compute_agent_correlation_matrix()
        if corr_matrix.empty:
            return 0.5
        
        n = len(corr_matrix)
        off_diagonal = corr_matrix.values[np.triu_indices(n, k=1)]
        avg_correlation = np.mean(np.abs(off_diagonal))
        
        return 1.0 - avg_correlation
    
    def check_alerts(self) -> list[str]:
        """Return list of correlation alerts."""
        alerts = []
        corr_matrix = self.compute_agent_correlation_matrix()
        
        if corr_matrix.empty:
            return alerts
        
        n = len(corr_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.7:
                    a1 = corr_matrix.index[i]
                    a2 = corr_matrix.columns[j]
                    alerts.append(f"High correlation ({corr:.2f}) between {a1} and {a2}")
        
        return alerts
```

---

## 7. Strategy Evolution Loop

### 7.1 Implementation (`src/reflection/strategy_evolver.py`)

```python
class StrategyEvolver:
    """Weekly strategy review and parameter optimization."""
    
    def __init__(self, registry: AgentRegistry, claude: ClaudeAgent,
                 store: DataStore, notifier: Notifier):
        self.registry = registry
        self.claude = claude
        self.store = store
        self.notifier = notifier
    
    async def weekly_review(self) -> dict:
        """Run weekly strategy evolution cycle."""
        report = {"timestamp": datetime.utcnow().isoformat(), "agents": {}}
        
        for agent in self.registry.get_all_agents():
            trades = self.store.get_trades(
                agent_id=agent.agent_id,
                start=datetime.utcnow() - timedelta(days=7)
            )
            
            if len(trades) < 3:
                report["agents"][agent.agent_id] = {"status": "insufficient_trades"}
                continue
            
            # Ask Claude to analyze and suggest improvements
            analysis = await self._claude_strategy_review(agent, trades)
            
            # If Claude suggests parameter changes, backtest them
            if analysis.get("suggested_params"):
                bt_result = await self._backtest_params(
                    agent, analysis["suggested_params"], days=30
                )
                
                current_sharpe = agent.performance.rolling_sharpe_30d
                new_sharpe = bt_result.sharpe_ratio
                
                if new_sharpe > current_sharpe * 1.1:  # 10% improvement threshold
                    old_params = agent.get_parameters()
                    agent.set_parameters(analysis["suggested_params"])
                    
                    report["agents"][agent.agent_id] = {
                        "status": "parameters_updated",
                        "old_sharpe": current_sharpe,
                        "new_sharpe_backtest": new_sharpe,
                        "changes": analysis["suggested_params"],
                        "reasoning": analysis["reasoning"],
                    }
                    
                    log.info("Strategy evolved", agent=agent.agent_id,
                            old_sharpe=current_sharpe, new_sharpe=new_sharpe)
                else:
                    report["agents"][agent.agent_id] = {
                        "status": "no_improvement",
                        "current_sharpe": current_sharpe,
                        "tested_sharpe": new_sharpe,
                    }
            
            # Deactivate underperformers
            if agent.performance.rolling_sharpe_30d < -0.5 and agent.performance.total_trades > 20:
                self.registry.deactivate(agent.agent_id)
                report["agents"][agent.agent_id] = {"status": "DEACTIVATED", "reason": "Negative Sharpe"}
                await self.notifier.send_critical(f"Agent {agent.name} deactivated: Sharpe {agent.performance.rolling_sharpe_30d:.2f}")
        
        return report
    
    async def _claude_strategy_review(self, agent: BaseStrategyAgent, trades: list[TradeRecord]) -> dict:
        prompt = f"""
You are reviewing the performance of trading agent "{agent.name}".

STRATEGY: {agent.description}
CURRENT PARAMETERS: {json.dumps(agent.get_parameters(), indent=2)}

PERFORMANCE (last 7 days):
- Trades: {len(trades)}
- Wins: {sum(1 for t in trades if t.pnl > 0)}
- Total P&L: ${sum(t.pnl for t in trades):+,.2f}
- Win rate: {sum(1 for t in trades if t.pnl > 0) / len(trades) * 100:.0f}%
- Avg win: ${np.mean([t.pnl for t in trades if t.pnl > 0]) if any(t.pnl > 0 for t in trades) else 0:+,.2f}
- Avg loss: ${np.mean([t.pnl for t in trades if t.pnl <= 0]) if any(t.pnl <= 0 for t in trades) else 0:+,.2f}

TRADE DETAILS:
{json.dumps([{{"instrument": t.instrument, "side": t.side.value, "pnl_pct": t.pnl_pct, "exit_reason": t.exit_reason.value, "duration_min": t.duration_minutes}} for t in trades], indent=2)}

Analyze the performance and suggest parameter adjustments.
Output JSON:
{{
  "performance_grade": "A-F",
  "analysis": "2-3 sentence summary",
  "suggested_params": {{...}} or null if no changes needed,
  "reasoning": "Why these changes would help"
}}
"""
        return await self.claude.raw_query_json(prompt)
```

---

## 8. Multi-Agent Trading Loop

### 8.1 Implementation (`scripts/run_multi_agent.py`)

```python
async def multi_agent_loop(config):
    """Main loop for Phase 3 multi-agent trading."""
    
    # Initialize
    store = DataStore(config["db_path"])
    data_fetcher = MarketDataFetcher(config, store)
    technical = TechnicalEngine()
    claude = ClaudeAgent(config["claude"])
    regime_detector = RegimeDetector()
    
    # Initialize agents
    registry = AgentRegistry()
    registry.register(MomentumAgent("momentum_01", config, store, claude, technical))
    registry.register(MeanReversionAgent("meanrev_01", config, store, claude, technical))
    registry.register(EventDrivenAgent("event_01", config, store, claude, technical))
    registry.register(StatArbAgent("statarb_01", config, store, claude, technical))
    
    # Initialize orchestrator
    global_risk = GlobalRiskManager(config["risk_params"])
    orchestrator = Orchestrator(registry, global_risk, store, notifier)
    correlation_monitor = CorrelationMonitor(registry, store)
    evolver = StrategyEvolver(registry, claude, store, notifier)
    order_mgr = OrderManager(AdapterFactory, store, global_risk)
    
    # Schedule weekly evolution
    scheduler = AsyncScheduler()
    scheduler.add_job(evolver.weekly_review, trigger="cron", day_of_week="sun", hour=20)
    scheduler.start()
    
    interval = config["trading"]["analysis_interval_seconds"]
    
    while True:
        cycle_start = datetime.utcnow()
        
        # 1. Determine regime
        bars_1h = await data_fetcher.get_latest_bars("ES", "1h", count=100)
        regime = regime_detector.classify(bars_1h)
        
        # 2. Allocate capital
        account = await order_mgr.get_total_equity()
        await orchestrator.allocate_capital(account, regime)
        
        # 3. Gather signals from all active agents
        all_signals: dict[str, list[SignalScore]] = {}
        for agent in registry.get_active_agents():
            try:
                market_data = {}
                tech_data = {}
                for inst in agent.target_instruments:
                    market_data[inst] = await data_fetcher.get_latest_bars(inst, "15m", 250)
                    tech_data[inst] = await technical.multi_timeframe_analysis_async(inst, data_fetcher)
                
                news = await news_feed.get_recent_all(hours=4)
                signals = await agent.generate_signals(market_data, tech_data, news, regime)
                all_signals[agent.agent_id] = signals
                
            except Exception as e:
                log.error("Agent signal generation failed", agent=agent.agent_id, error=str(e))
        
        # 4. Resolve conflicts
        resolved_signals = await orchestrator.resolve_conflicts(all_signals)
        
        # 5. Risk check + execute each resolved signal
        all_positions = await order_mgr.sync_positions()
        flat_positions = [p for positions in all_positions.values() for p in positions]
        
        for signal in resolved_signals:
            risk_check = global_risk.check_portfolio_signal(
                signal, signal.recommendation.key_factors[0] if signal.recommendation else "",
                flat_positions, orchestrator.portfolio
            )
            if not risk_check.approved:
                continue
            
            # Execute...
            # (Same execution logic as Phase 2)
        
        # 6. Correlation check
        alerts = correlation_monitor.check_alerts()
        for alert in alerts:
            log.warning("Correlation alert", alert=alert)
        
        # Wait for next cycle
        elapsed = (datetime.utcnow() - cycle_start).total_seconds()
        await asyncio.sleep(max(0, interval - elapsed))
```

---

## 9. Testing Checklist

```
[ ] Base agent: Fitness score correctly weighted by regime + performance
[ ] Momentum agent: Generates BUY when MA aligned up + ADX > 25
[ ] Momentum agent: No signal when ADX < threshold
[ ] Mean reversion: Generates BUY at lower BB + RSI oversold
[ ] Mean reversion: No signal when ADX > max (trending market)
[ ] Event driven: Claude API called only for high-relevance news
[ ] Event driven: Technical alignment filter works
[ ] Stat arb: Z-score computation matches manual calculation
[ ] Stat arb: No signal when correlation below threshold
[ ] Orchestrator: Capital allocation sums to 100%
[ ] Orchestrator: Low-performing agent gets less capital
[ ] Orchestrator: Conflict resolution: same direction = strongest wins
[ ] Orchestrator: Conflict resolution: opposing signals need 30% margin
[ ] Global risk: Rejects when portfolio drawdown at limit
[ ] Global risk: Rejects when same instrument held by 2+ agents
[ ] Global risk: Net exposure check works
[ ] Correlation monitor: Detects high correlation between agents
[ ] Correlation monitor: Diversification score decreases with correlated agents
[ ] Strategy evolver: Claude suggests parameter changes
[ ] Strategy evolver: Only applies changes if backtest improves Sharpe by 10%+
[ ] Strategy evolver: Deactivates agent with negative Sharpe
[ ] Multi-agent loop: All agents generate signals in parallel
[ ] Multi-agent loop: Resolved signals pass through global risk
[ ] Debate protocol: Claude synthesizes conflicting views
[ ] Full integration: 4 agents + orchestrator + global risk + execution
```
