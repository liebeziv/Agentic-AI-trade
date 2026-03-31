# Phase 2: Live Execution + Trade Journal + Attribution

> **Duration**: Weeks 9-16
> **Mode**: Live trading with real capital
> **Prerequisite**: Phase 1 exit criteria met (3mo sim, Sharpe > 1.5, DD < 15%)
> **Exit Criteria**: 3-month live profit, Max DD < 15%, > 100 trades with full attribution

---

## 1. Implementation Order

```
Step 1:  Exchange adapter — abstract interface + adapter factory
Step 2:  Exchange adapter — Alpaca (US stocks, paper → live)
Step 3:  Exchange adapter — Binance (crypto)
Step 4:  Exchange adapter — Interactive Brokers (futures, HK stocks)
Step 5:  Exchange adapter — MT5 (forex)
Step 6:  Order execution engine (smart routing, retry, partial fill handling)
Step 7:  Position synchronization (adapter state ↔ local state)
Step 8:  Trade journal — full audit trail
Step 9:  Attribution engine — factor decomposition
Step 10: Claude reflection loop — post-trade analysis
Step 11: Alert system (Telegram)
Step 12: Live monitoring upgrades (dashboard enhancements)
Step 13: Gradual capital deployment (scaling rules)
Step 14: Integration testing on testnet/paper accounts
Step 15: Go-live checklist + kill switch
```

---

## 2. Exchange Adapter Layer

### 2.1 Abstract Interface (`src/execution/exchange_adapter.py`)

```python
from abc import ABC, abstractmethod

class ExchangeAdapter(ABC):
    """Unified interface for all exchange/broker connections."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection, authenticate."""
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Clean disconnect, cancel pending orders if configured."""
    
    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Account balance, margin, buying power."""
    
    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Submit order to exchange. Returns fill info or pending status."""
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled."""
    
    @abstractmethod
    async def modify_order(self, order_id: str, updates: dict) -> OrderResult:
        """Modify pending order (price, quantity). Not all exchanges support this."""
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Poll order status."""
    
    @abstractmethod
    async def get_open_orders(self, instrument: str | None = None) -> list[Order]:
        """Get all open/pending orders."""
    
    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions from the exchange."""
    
    @abstractmethod
    async def close_position(self, instrument: str, quantity: float | None = None) -> OrderResult:
        """Close a position. None quantity = close all."""
    
    @abstractmethod
    async def close_all_positions(self) -> list[OrderResult]:
        """Emergency: close everything."""
    
    @abstractmethod
    async def get_balance(self) -> Balance:
        """Cash, equity, margin available."""
    
    @abstractmethod
    async def subscribe_fills(self, callback: Callable) -> None:
        """Real-time fill notifications via WebSocket."""
    
    @abstractmethod
    async def subscribe_orders(self, callback: Callable) -> None:
        """Real-time order status updates."""
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Connection health check."""
    
    @abstractmethod
    def supports_instrument(self, instrument: str) -> bool:
        """Check if this adapter handles the given instrument."""


@dataclass
class AccountInfo:
    equity: float
    cash: float
    margin_used: float
    margin_available: float
    buying_power: float
    currency: str
    account_id: str


@dataclass
class Balance:
    total_equity: float
    available_cash: float
    unrealized_pnl: float
    margin_used: float


@dataclass
class OrderResult:
    success: bool
    order_id: str
    status: OrderStatus
    fill_price: float | None = None
    fill_quantity: float | None = None
    commission: float = 0.0
    message: str = ""
    exchange_order_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### 2.2 Adapter Factory

```python
class AdapterFactory:
    """Create the right adapter for each instrument."""
    
    _adapters: dict[str, ExchangeAdapter] = {}
    
    @classmethod
    def get_adapter(cls, instrument: str, config: dict) -> ExchangeAdapter:
        inst_type = classify_instrument(instrument)
        
        if inst_type == "us_stocks":
            key = "alpaca"
            if key not in cls._adapters:
                cls._adapters[key] = AlpacaAdapter(config["alpaca"])
            return cls._adapters[key]
        
        elif inst_type == "crypto":
            exchange = config["instruments"]["crypto"][instrument].get("exchange", "binance")
            key = f"crypto_{exchange}"
            if key not in cls._adapters:
                cls._adapters[key] = BinanceAdapter(config["binance"])
            return cls._adapters[key]
        
        elif inst_type == "futures" or inst_type == "hk_stocks":
            key = "ib"
            if key not in cls._adapters:
                cls._adapters[key] = IBAdapter(config["ib"])
            return cls._adapters[key]
        
        elif inst_type == "forex":
            key = "mt5"
            if key not in cls._adapters:
                cls._adapters[key] = MT5Adapter(config["mt5"])
            return cls._adapters[key]
        
        raise ValueError(f"No adapter for instrument: {instrument}")
    
    @classmethod
    async def connect_all(cls):
        for adapter in cls._adapters.values():
            await adapter.connect()
    
    @classmethod
    async def disconnect_all(cls):
        for adapter in cls._adapters.values():
            await adapter.disconnect()
```

### 2.3 Alpaca Adapter (`src/execution/adapters/alpaca_adapter.py`)

```python
"""Alpaca adapter for US stocks. Supports paper and live."""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus as AlpacaStatus

class AlpacaAdapter(ExchangeAdapter):
    def __init__(self, config: dict):
        self.client = TradingClient(
            api_key=config["api_key"],
            secret_key=config["secret_key"],
            paper=config.get("paper", True),  # DEFAULT TO PAPER
        )
        self._connected = False
    
    async def connect(self) -> None:
        account = self.client.get_account()
        if account.status != "ACTIVE":
            raise ConnectionError(f"Account not active: {account.status}")
        self._connected = True
        log.info("Alpaca connected", account_id=account.id, paper=self.client._use_raw_data)
    
    async def place_order(self, order: Order) -> OrderResult:
        side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL
        
        if order.order_type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=order.instrument,
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        elif order.order_type == OrderType.LIMIT:
            request = LimitOrderRequest(
                symbol=order.instrument,
                qty=order.quantity,
                side=side,
                limit_price=order.price,
                time_in_force=TimeInForce.GTC,
            )
        
        try:
            result = self.client.submit_order(request)
            return OrderResult(
                success=True,
                order_id=order.id,
                exchange_order_id=str(result.id),
                status=self._map_status(result.status),
                fill_price=float(result.filled_avg_price) if result.filled_avg_price else None,
                fill_quantity=float(result.filled_qty) if result.filled_qty else None,
            )
        except Exception as e:
            return OrderResult(success=False, order_id=order.id, status=OrderStatus.REJECTED, message=str(e))
    
    async def get_positions(self) -> list[Position]:
        positions = self.client.get_all_positions()
        return [
            Position(
                instrument=p.symbol,
                side=Side.BUY if float(p.qty) > 0 else Side.SELL,
                quantity=abs(float(p.qty)),
                entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pnl=float(p.unrealized_pl),
            )
            for p in positions
        ]
    
    async def close_all_positions(self) -> list[OrderResult]:
        self.client.close_all_positions(cancel_orders=True)
        return [OrderResult(success=True, order_id="CLOSE_ALL", status=OrderStatus.FILLED)]
```

### 2.4 Binance Adapter (`src/execution/adapters/binance_adapter.py`)

```python
"""Binance adapter for crypto spot/futures via ccxt."""

import ccxt.async_support as ccxt

class BinanceAdapter(ExchangeAdapter):
    def __init__(self, config: dict):
        self.exchange = ccxt.binance({
            "apiKey": config["api_key"],
            "secret": config["secret"],
            "sandbox": config.get("testnet", True),  # DEFAULT TO TESTNET
            "options": {"defaultType": config.get("market_type", "spot")},
        })
        self._connected = False
    
    async def connect(self) -> None:
        await self.exchange.load_markets()
        balance = await self.exchange.fetch_balance()
        self._connected = True
        log.info("Binance connected", testnet=self.exchange.sandbox)
    
    async def place_order(self, order: Order) -> OrderResult:
        side = "buy" if order.side == Side.BUY else "sell"
        
        try:
            if order.order_type == OrderType.MARKET:
                result = await self.exchange.create_market_order(
                    order.instrument, side, order.quantity
                )
            elif order.order_type == OrderType.LIMIT:
                result = await self.exchange.create_limit_order(
                    order.instrument, side, order.quantity, order.price
                )
            
            return OrderResult(
                success=True,
                order_id=order.id,
                exchange_order_id=result["id"],
                status=self._map_status(result["status"]),
                fill_price=result.get("average"),
                fill_quantity=result.get("filled"),
                commission=self._extract_fee(result),
            )
        except Exception as e:
            return OrderResult(success=False, order_id=order.id, status=OrderStatus.REJECTED, message=str(e))
    
    async def disconnect(self) -> None:
        await self.exchange.close()
        self._connected = False
```

### 2.5 Interactive Brokers Adapter (`src/execution/adapters/ib_adapter.py`)

```python
"""IB adapter for futures, HK stocks via ib_insync."""

from ib_insync import IB, Stock, Future, MarketOrder, LimitOrder

class IBAdapter(ExchangeAdapter):
    def __init__(self, config: dict):
        self.ib = IB()
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 7497)  # 7497=paper, 7496=live
        self.client_id = config.get("client_id", 1)
    
    async def connect(self) -> None:
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        log.info("IB connected", host=self.host, port=self.port)
    
    async def place_order(self, order: Order) -> OrderResult:
        contract = self._resolve_contract(order.instrument)
        
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(
                "BUY" if order.side == Side.BUY else "SELL",
                order.quantity
            )
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(
                "BUY" if order.side == Side.BUY else "SELL",
                order.quantity,
                order.price
            )
        
        trade = self.ib.placeOrder(contract, ib_order)
        # Wait for fill (with timeout)
        await asyncio.sleep(2)
        
        return OrderResult(
            success=trade.orderStatus.status in ["Filled", "Submitted"],
            order_id=order.id,
            exchange_order_id=str(trade.order.orderId),
            status=self._map_status(trade.orderStatus.status),
            fill_price=trade.orderStatus.avgFillPrice or None,
            fill_quantity=trade.orderStatus.filled or None,
            commission=sum(f.commission for f in trade.fills),
        )
    
    def _resolve_contract(self, instrument: str):
        """Map instrument string to IB contract."""
        FUTURES_MAP = {
            "ES": Future("ES", "CME", currency="USD"),
            "NQ": Future("NQ", "CME", currency="USD"),
            "GC": Future("GC", "COMEX", currency="USD"),
            "CL": Future("CL", "NYMEX", currency="USD"),
        }
        if instrument in FUTURES_MAP:
            return FUTURES_MAP[instrument]
        # HK stocks
        if instrument.endswith(".HK"):
            return Stock(instrument.replace(".HK", ""), "SEHK", "HKD")
        # Default US stock
        return Stock(instrument, "SMART", "USD")
```

### 2.6 MT5 Adapter (`src/execution/adapters/mt5_adapter.py`)

```python
"""MetaTrader 5 adapter for forex (Windows only, or via Wine/Docker)."""

import MetaTrader5 as mt5

class MT5Adapter(ExchangeAdapter):
    def __init__(self, config: dict):
        self.login = config["login"]
        self.password = config["password"]
        self.server = config["server"]
    
    async def connect(self) -> None:
        if not mt5.initialize():
            raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")
        authorized = mt5.login(self.login, password=self.password, server=self.server)
        if not authorized:
            raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")
        log.info("MT5 connected", server=self.server)
    
    async def place_order(self, order: Order) -> OrderResult:
        symbol = order.instrument
        action = mt5.TRADE_ACTION_DEAL
        order_type = mt5.ORDER_TYPE_BUY if order.side == Side.BUY else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": action,
            "symbol": symbol,
            "volume": order.quantity,
            "type": order_type,
            "sl": order.stop_loss or 0.0,
            "tp": order.take_profit or 0.0,
            "deviation": 20,  # max slippage in points
            "magic": 234000,
            "comment": f"atlas:{order.id}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if order.order_type == OrderType.LIMIT:
            request["action"] = mt5.TRADE_ACTION_PENDING
            request["type"] = mt5.ORDER_TYPE_BUY_LIMIT if order.side == Side.BUY else mt5.ORDER_TYPE_SELL_LIMIT
            request["price"] = order.price
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(success=False, order_id=order.id, status=OrderStatus.REJECTED, message=str(result.comment))
        
        return OrderResult(
            success=True,
            order_id=order.id,
            exchange_order_id=str(result.order),
            status=OrderStatus.FILLED,
            fill_price=result.price,
            fill_quantity=order.quantity,
            commission=result.comment_fee if hasattr(result, "comment_fee") else 0,
        )
```

---

## 3. Order Execution Engine

### 3.1 Implementation (`src/execution/order_manager.py`)

```python
class OrderManager:
    """Manages the full lifecycle of orders across exchanges."""
    
    def __init__(self, adapter_factory: AdapterFactory, store: DataStore, risk_manager: RiskManager):
        self.factory = adapter_factory
        self.store = store
        self.risk_mgr = risk_manager
        self.pending_orders: dict[str, Order] = {}
        self.retry_config = {"max_retries": 3, "backoff_seconds": 2}
    
    async def submit_order(self, order: Order) -> OrderResult:
        """Submit order with pre-flight checks and retry logic."""
        
        # Final risk check
        risk_check = self.risk_mgr.pre_order_check(order)
        if not risk_check.approved:
            log.warning("Order blocked by risk manager", order_id=order.id, reason=risk_check.reason)
            return OrderResult(success=False, order_id=order.id, status=OrderStatus.REJECTED, message=risk_check.reason)
        
        # Get adapter
        adapter = self.factory.get_adapter(order.instrument, self.config)
        
        # Smart order routing
        order = self._apply_smart_routing(order, adapter)
        
        # Submit with retry
        for attempt in range(self.retry_config["max_retries"]):
            try:
                result = await adapter.place_order(order)
                if result.success:
                    order.status = result.status
                    order.exchange_order_id = result.exchange_order_id
                    order.submitted_at = datetime.utcnow()
                    
                    if result.status == OrderStatus.FILLED:
                        order.fill_price = result.fill_price
                        order.fill_quantity = result.fill_quantity
                        order.filled_at = datetime.utcnow()
                        order.commission = result.commission
                        order.slippage = abs(result.fill_price - (order.price or 0))
                    
                    self.pending_orders[order.id] = order
                    self.store.save_order(order)
                    return result
                
                if "insufficient" in result.message.lower():
                    log.error("Insufficient funds", order_id=order.id)
                    return result  # Don't retry
                
            except Exception as e:
                log.warning("Order attempt failed", attempt=attempt, error=str(e))
                await asyncio.sleep(self.retry_config["backoff_seconds"] * (attempt + 1))
        
        return OrderResult(success=False, order_id=order.id, status=OrderStatus.REJECTED, message="Max retries exceeded")
    
    def _apply_smart_routing(self, order: Order, adapter: ExchangeAdapter) -> Order:
        """Apply smart order routing rules."""
        # Rule 1: Prefer limit orders for entries
        if order.order_type == OrderType.MARKET and order.stop_loss:
            # Convert market to aggressive limit (1 tick above ask for buy)
            # This reduces slippage
            spread_estimate = self._get_spread_estimate(order.instrument)
            if order.side == Side.BUY:
                order.order_type = OrderType.LIMIT
                order.price = order.entry_price_target + spread_estimate
            else:
                order.order_type = OrderType.LIMIT
                order.price = order.entry_price_target - spread_estimate
            order.time_in_force = "IOC"  # Immediate or cancel
        
        # Rule 2: Stop-loss orders are always MARKET (must fill)
        # This is handled at the stop-check level, not here
        
        return order
    
    async def sync_positions(self) -> dict[str, list[Position]]:
        """Synchronize local state with exchange state."""
        all_positions = {}
        for name, adapter in self.factory._adapters.items():
            try:
                positions = await adapter.get_positions()
                all_positions[name] = positions
            except Exception as e:
                log.error("Position sync failed", adapter=name, error=str(e))
        return all_positions
```

---

## 4. Trade Journal

### 4.1 Implementation (`src/reflection/trade_journal.py`)

```python
class TradeJournal:
    """Automatic trade journal with full context capture."""
    
    def __init__(self, store: DataStore, claude_agent: ClaudeAgent, notifier: Notifier):
        self.store = store
        self.claude = claude_agent
        self.notifier = notifier
    
    async def record_trade_open(self, order: Order, signal: SignalScore,
                                 technical: TechnicalSnapshot, regime: MarketRegime,
                                 news: list[NewsItem]) -> str:
        """Record everything at trade entry."""
        entry = {
            "trade_id": order.id,
            "timestamp": datetime.utcnow().isoformat(),
            "instrument": order.instrument,
            "side": order.side.value,
            "entry_price": order.fill_price,
            "quantity": order.fill_quantity,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            # Signal context
            "signal_composite": signal.composite_score,
            "signal_technical": signal.technical_score,
            "signal_sentiment": signal.sentiment_score,
            "signal_regime": signal.regime_score,
            "signal_claude_conf": signal.claude_confidence,
            # Market context
            "regime": regime.value,
            "technical_snapshot": asdict(technical),
            "news_headlines": [n.title for n in news[:5]],
            "claude_reasoning": order.claude_reasoning,
        }
        self.store.save_journal_entry("open", entry)
        
        # Notify
        await self.notifier.send(
            f"📈 OPEN {order.side.value} {order.instrument}\n"
            f"Price: {order.fill_price}\n"
            f"Size: {order.fill_quantity}\n"
            f"SL: {order.stop_loss} | TP: {order.take_profit}\n"
            f"Score: {signal.composite_score:.2f} | Conf: {signal.claude_confidence:.0%}"
        )
        return order.id
    
    async def record_trade_close(self, trade: TradeRecord,
                                   exit_technical: TechnicalSnapshot,
                                   exit_regime: MarketRegime) -> None:
        """Record everything at trade exit and trigger reflection."""
        self.store.save_trade(trade)
        
        # Claude reflection (async, non-blocking)
        try:
            reflection = await self.claude.reflect_on_trade(
                trade=trade,
                entry_context=self.store.get_journal_entry("open", trade.id),
                exit_context={"technical": asdict(exit_technical), "regime": exit_regime.value}
            )
            trade.factor_attribution = reflection.get("factor_attribution", {})
            trade.lessons_learned = reflection.get("lessons", "")
            self.store.update_trade(trade)
        except Exception as e:
            log.warning("Reflection failed", trade_id=trade.id, error=str(e))
        
        # Notify
        emoji = "✅" if trade.pnl > 0 else "❌"
        await self.notifier.send(
            f"{emoji} CLOSE {trade.side.value} {trade.instrument}\n"
            f"P&L: ${trade.pnl:+.2f} ({trade.pnl_pct:+.1f}%)\n"
            f"Exit: {trade.exit_reason.value}\n"
            f"Duration: {trade.duration_minutes:.0f}min\n"
            f"Lesson: {trade.lessons_learned[:100]}"
        )
```

---

## 5. Attribution Engine

### 5.1 Implementation (`src/reflection/attribution.py`)

```python
class AttributionEngine:
    """Decompose trade P&L into contributing factors."""
    
    def __init__(self, store: DataStore):
        self.store = store
    
    def attribute_trade(self, trade: TradeRecord) -> dict[str, float]:
        """
        Factor attribution for a single trade.
        Returns dict of factor → contribution score (-1 to +1).
        """
        attribution = {}
        
        # 1. Technical factor
        # Did the technical indicators correctly predict direction?
        tech = trade.technical_at_entry
        if trade.side == Side.BUY:
            tech_correct = tech.get("trend_bias") == "bullish"
        else:
            tech_correct = tech.get("trend_bias") == "bearish"
        attribution["technical"] = 1.0 if (tech_correct and trade.pnl > 0) else -0.5
        
        # 2. Sentiment factor
        # Was Claude's sentiment assessment aligned with outcome?
        claude_conf = trade.signal_score
        if trade.pnl > 0 and claude_conf > 0.6:
            attribution["sentiment"] = 0.8
        elif trade.pnl < 0 and claude_conf > 0.6:
            attribution["sentiment"] = -0.8  # High confidence wrong = bad signal
        else:
            attribution["sentiment"] = 0.0
        
        # 3. Regime factor
        regime = trade.regime_at_entry
        regime_favored = self._regime_favors_action(regime, trade.side.value)
        if regime_favored and trade.pnl > 0:
            attribution["regime"] = 0.7
        elif not regime_favored and trade.pnl < 0:
            attribution["regime"] = -0.7
        else:
            attribution["regime"] = 0.1
        
        # 4. Timing factor
        # How much of the potential move was captured?
        if trade.pnl > 0:
            entry_efficiency = self._compute_entry_efficiency(trade)
            exit_efficiency = self._compute_exit_efficiency(trade)
            attribution["timing"] = (entry_efficiency + exit_efficiency) / 2
        else:
            attribution["timing"] = -0.3
        
        # 5. Sizing factor
        # Was the position size appropriate for the volatility?
        vol_at_entry = tech.get("atr_14", 0)
        expected_move = vol_at_entry * 2  # 2 ATR expected range
        actual_move = abs(trade.exit_price - trade.entry_price)
        if actual_move > expected_move * 1.5:
            attribution["sizing"] = 0.5 if trade.pnl > 0 else -0.8
        else:
            attribution["sizing"] = 0.2
        
        # Normalize so values sum to trade outcome direction
        total = sum(abs(v) for v in attribution.values())
        if total > 0:
            direction = 1 if trade.pnl > 0 else -1
            attribution = {k: (abs(v) / total) * direction for k, v in attribution.items()}
        
        return attribution
    
    def generate_weekly_report(self, start: datetime, end: datetime) -> dict:
        """Aggregate attribution across all trades in the period."""
        trades = self.store.get_trades(start=start, end=end)
        
        if not trades:
            return {"error": "No trades in period"}
        
        report = {
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "total_trades": len(trades),
            "winning_trades": sum(1 for t in trades if t.pnl > 0),
            "total_pnl": sum(t.pnl for t in trades),
            "avg_attribution": {},
            "best_factor": "",
            "worst_factor": "",
            "top_3_lessons": [],
            "instrument_breakdown": {},
            "regime_breakdown": {},
        }
        
        # Average attribution per factor
        factor_sums = {}
        for trade in trades:
            for factor, value in trade.factor_attribution.items():
                factor_sums.setdefault(factor, []).append(value)
        
        for factor, values in factor_sums.items():
            report["avg_attribution"][factor] = sum(values) / len(values)
        
        # Best and worst factors
        sorted_factors = sorted(report["avg_attribution"].items(), key=lambda x: x[1])
        report["worst_factor"] = sorted_factors[0][0] if sorted_factors else ""
        report["best_factor"] = sorted_factors[-1][0] if sorted_factors else ""
        
        # Lessons (unique, from Claude reflections)
        lessons = [t.lessons_learned for t in trades if t.lessons_learned]
        report["top_3_lessons"] = lessons[:3]
        
        # Per-instrument P&L
        for trade in trades:
            inst = trade.instrument
            report["instrument_breakdown"].setdefault(inst, {"pnl": 0, "trades": 0})
            report["instrument_breakdown"][inst]["pnl"] += trade.pnl
            report["instrument_breakdown"][inst]["trades"] += 1
        
        # Per-regime P&L
        for trade in trades:
            regime = trade.regime_at_entry
            report["regime_breakdown"].setdefault(regime, {"pnl": 0, "trades": 0})
            report["regime_breakdown"][regime]["pnl"] += trade.pnl
            report["regime_breakdown"][regime]["trades"] += 1
        
        return report
```

---

## 6. Alert System

### 6.1 Notifier (`src/utils/notifier.py`)

```python
class Notifier:
    """Multi-channel notification system."""
    
    def __init__(self, config: dict):
        self.telegram_token = config.get("telegram_bot_token")
        self.telegram_chat_id = config.get("telegram_chat_id")
        self.channels = []
        if self.telegram_token:
            self.channels.append("telegram")
    
    async def send(self, message: str, severity: str = "INFO") -> None:
        for channel in self.channels:
            try:
                if channel == "telegram":
                    await self._send_telegram(message)
            except Exception as e:
                log.error("Notification failed", channel=channel, error=str(e))
    
    async def send_critical(self, message: str) -> None:
        """Send with urgency markers."""
        await self.send(f"🚨 CRITICAL 🚨\n{message}", severity="CRITICAL")
    
    async def send_daily_summary(self, portfolio: PortfolioState, trades: list[TradeRecord]) -> None:
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = len(trades) - wins
        total_pnl = sum(t.pnl for t in trades)
        
        msg = (
            f"📊 Daily Summary\n"
            f"{'─' * 20}\n"
            f"Trades: {len(trades)} ({wins}W / {losses}L)\n"
            f"Daily P&L: ${total_pnl:+,.2f}\n"
            f"Equity: ${portfolio.equity:,.2f}\n"
            f"Drawdown: {portfolio.max_drawdown_current:.1f}%\n"
            f"Heat: {portfolio.portfolio_heat_pct:.1f}%"
        )
        await self.send(msg)
    
    async def _send_telegram(self, message: str) -> None:
        from telegram import Bot
        bot = Bot(token=self.telegram_token)
        await bot.send_message(chat_id=self.telegram_chat_id, text=message, parse_mode="HTML")
```

---

## 7. Live Trading Loop Modifications

### 7.1 Key Differences from Paper Trading

```python
# In run_live.py — additions over run_paper.py:

# 1. Pre-flight checks
async def pre_flight_checks():
    """Must pass before live trading starts."""
    checks = {
        "phase_1_complete": verify_phase1_exit_criteria(),       # Sharpe > 1.5, DD < 15%
        "api_keys_valid": await verify_all_adapters(),
        "risk_params_loaded": verify_risk_config(),
        "kill_switch_ready": verify_kill_switch(),
        "initial_capital_set": verify_capital_deployment(),
        "telegram_working": await verify_notifications(),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"Pre-flight failed: {failed}")

# 2. Position sync every cycle
async def sync_cycle():
    """Reconcile local state with exchange state."""
    exchange_positions = await order_manager.sync_positions()
    local_positions = portfolio.positions
    discrepancies = find_discrepancies(exchange_positions, local_positions)
    if discrepancies:
        log.warning("Position discrepancy detected", details=discrepancies)
        await notifier.send_critical(f"Position mismatch: {discrepancies}")

# 3. Kill switch
class KillSwitch:
    def __init__(self, max_drawdown: float, max_daily_loss: float):
        self.active = False
    
    async def check(self, portfolio: PortfolioState) -> bool:
        if portfolio.max_drawdown_current >= self.max_drawdown:
            self.active = True
            await self._emergency_close_all()
            await notifier.send_critical(f"KILL SWITCH: DD {portfolio.max_drawdown_current:.1f}% >= {self.max_drawdown}%")
            return True
        return False
    
    async def _emergency_close_all(self):
        await AdapterFactory.close_all_positions()

# 4. Gradual capital deployment
DEPLOYMENT_SCHEDULE = {
    "week_1": 0.10,    # 10% of target capital
    "week_2": 0.20,
    "week_3": 0.30,
    "week_4": 0.50,
    "week_5": 0.70,
    "week_6": 0.85,
    "week_7": 1.00,    # Full deployment only after 6 weeks of live profit
}
```

---

## 8. Go-Live Checklist

```
PRE-LAUNCH:
[ ] Phase 1 sim results documented (Sharpe, DD, trade count)
[ ] All exchange API keys configured and tested
[ ] Paper trading on each exchange works correctly
[ ] Risk parameters reviewed and locked
[ ] Kill switch tested manually
[ ] Telegram alerts tested (open, close, critical)
[ ] Position sync tested against each exchange
[ ] Deployment schedule configured (start at 10%)
[ ] Emergency contact procedure documented
[ ] Manual override procedure documented

WEEK 1 (10% capital):
[ ] Monitor every trade manually
[ ] Compare live fills vs paper engine predictions
[ ] Track slippage per exchange
[ ] Verify commission calculations
[ ] Daily P&L matches exchange reports
[ ] Journal entries recording correctly

ONGOING:
[ ] Daily: Review journal, check alerts
[ ] Weekly: Attribution report, Claude strategy review
[ ] Monthly: Full performance review, parameter adjustment
[ ] Quarterly: Phase 3 readiness assessment
```

---

## 9. Testing Checklist

```
[ ] Alpaca adapter: Paper order placement and fill
[ ] Alpaca adapter: Position retrieval matches expected
[ ] Binance adapter: Testnet order placement (spot)
[ ] Binance adapter: Testnet order cancellation
[ ] IB adapter: Paper order for ES futures
[ ] IB adapter: Position sync across multiple instruments
[ ] MT5 adapter: Demo account order placement
[ ] Order manager: Retry logic on transient failure
[ ] Order manager: Risk manager blocks invalid order
[ ] Order manager: Smart routing converts market to limit
[ ] Position sync: Detects discrepancy between local and exchange
[ ] Kill switch: Triggers at max drawdown, closes all positions
[ ] Trade journal: Records full context at open and close
[ ] Attribution: Factor scores sum to 1.0 in magnitude
[ ] Telegram: Receives trade open/close/critical messages
[ ] Daily summary: Sent at configured time with correct stats
[ ] Full live pipeline: Signal → Risk → Size → Order → Fill → Journal → Notify
```
