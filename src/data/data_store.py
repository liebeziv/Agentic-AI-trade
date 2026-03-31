"""DuckDB-backed data store for atlas-trader."""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from src.types import (
    EconomicEvent, NewsItem, PortfolioState, SignalScore,
    TradeAction, TradeRecord, Side, ExitReason, OrderStatus,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bars (
    timestamp   TIMESTAMP NOT NULL,
    open        DOUBLE NOT NULL,
    high        DOUBLE NOT NULL,
    low         DOUBLE NOT NULL,
    close       DOUBLE NOT NULL,
    volume      DOUBLE NOT NULL,
    instrument  VARCHAR NOT NULL,
    timeframe   VARCHAR NOT NULL,
    source      VARCHAR,
    PRIMARY KEY (instrument, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS news (
    id               VARCHAR PRIMARY KEY,
    timestamp        TIMESTAMP NOT NULL,
    title            VARCHAR NOT NULL,
    summary          TEXT,
    source           VARCHAR,
    url              VARCHAR,
    relevance_score  DOUBLE,
    instruments      VARCHAR,   -- JSON array
    sentiment_score  DOUBLE,
    raw_text         TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id                  VARCHAR PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    instrument          VARCHAR NOT NULL,
    technical_score     DOUBLE,
    sentiment_score     DOUBLE,
    regime_score        DOUBLE,
    claude_confidence   DOUBLE,
    composite_score     DOUBLE,
    action              VARCHAR,
    recommendation_json TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id                      VARCHAR PRIMARY KEY,
    instrument              VARCHAR NOT NULL,
    side                    VARCHAR NOT NULL,
    entry_price             DOUBLE,
    exit_price              DOUBLE,
    quantity                DOUBLE,
    pnl                     DOUBLE,
    pnl_pct                 DOUBLE,
    commission_total        DOUBLE,
    entry_time              TIMESTAMP,
    exit_time               TIMESTAMP,
    duration_minutes        DOUBLE,
    exit_reason             VARCHAR,
    signal_score            DOUBLE,
    claude_reasoning_entry  TEXT,
    claude_reasoning_exit   TEXT,
    technical_at_entry      TEXT,
    regime_at_entry         VARCHAR,
    news_at_entry           TEXT,
    factor_attribution      TEXT,
    lessons_learned         TEXT
);

CREATE TABLE IF NOT EXISTS economic_events (
    id        VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    name      VARCHAR NOT NULL,
    country   VARCHAR,
    impact    VARCHAR,
    forecast  VARCHAR,
    previous  VARCHAR,
    actual    VARCHAR
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    timestamp      TIMESTAMP PRIMARY KEY,
    equity         DOUBLE,
    cash           DOUBLE,
    unrealized_pnl DOUBLE,
    realized_pnl   DOUBLE,
    daily_pnl      DOUBLE,
    max_drawdown   DOUBLE,
    positions_json TEXT
);

CREATE TABLE IF NOT EXISTS journal_entries (
    id             VARCHAR PRIMARY KEY,
    trade_id       VARCHAR NOT NULL,
    event          VARCHAR NOT NULL,   -- 'open' | 'close'
    timestamp      TIMESTAMP NOT NULL,
    instrument     VARCHAR NOT NULL,
    context_json   TEXT NOT NULL       -- full context snapshot
);
"""


class DataStore:
    def __init__(self, db_path: str = "./data/atlas.duckdb") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.conn.execute("SET threads=4")
        self.conn.execute(SCHEMA_SQL)
        log.info("DataStore ready", path=db_path)

    # ------------------------------------------------------------------ Bars

    def save_bars(self, df: pl.DataFrame) -> int:
        if df.is_empty():
            return 0
        self.conn.execute("""
            INSERT OR IGNORE INTO bars
            SELECT timestamp, open, high, low, close, volume, instrument, timeframe, source
            FROM df
        """)
        return len(df)

    def get_bars(
        self, instrument: str, timeframe: str,
        start: datetime, end: datetime
    ) -> pl.DataFrame:
        result = self.conn.execute("""
            SELECT * FROM bars
            WHERE instrument = ? AND timeframe = ?
              AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """, [instrument, timeframe, start, end]).pl()
        return result

    def get_latest_bars(
        self, instrument: str, timeframe: str, count: int = 200
    ) -> pl.DataFrame:
        result = self.conn.execute("""
            SELECT * FROM (
                SELECT * FROM bars
                WHERE instrument = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ) ORDER BY timestamp
        """, [instrument, timeframe, count]).pl()
        return result

    # ------------------------------------------------------------------ News

    def save_news(self, items: list[NewsItem]) -> int:
        if not items:
            return 0
        rows = [
            (
                f"{n.timestamp.isoformat()}_{hash(n.title) & 0xFFFFFF:06x}",
                n.timestamp, n.title, n.summary, n.source, n.url,
                n.relevance_score, json.dumps(n.instruments), 0.0, n.sentiment_raw,
            )
            for n in items
        ]
        self.conn.executemany("""
            INSERT OR IGNORE INTO news
            (id, timestamp, title, summary, source, url,
             relevance_score, instruments, sentiment_score, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        return len(rows)

    def get_recent_news(self, instrument: str, hours: int = 24) -> list[NewsItem]:
        since = datetime.utcnow() - timedelta(hours=hours)
        rows = self.conn.execute("""
            SELECT timestamp, title, summary, source, url, relevance_score, instruments
            FROM news
            WHERE timestamp >= ?
              AND instruments LIKE ?
            ORDER BY timestamp DESC
            LIMIT 50
        """, [since, f'%"{instrument}"%']).fetchall()

        return [
            NewsItem(
                timestamp=row[0], title=row[1], summary=row[2] or "",
                source=row[3] or "", url=row[4] or "",
                relevance_score=row[5] or 0.0,
                instruments=json.loads(row[6]) if row[6] else [],
            )
            for row in rows
        ]

    # ---------------------------------------------------------------- Signals

    def save_signal(self, signal: SignalScore) -> None:
        sig_id = f"SIG-{signal.instrument}-{signal.timestamp.strftime('%Y%m%d%H%M%S')}"
        rec_json = ""
        if signal.recommendation:
            import dataclasses
            rec_json = json.dumps(dataclasses.asdict(signal.recommendation), default=str)
        self.conn.execute("""
            INSERT OR REPLACE INTO signals
            (id, timestamp, instrument, technical_score, sentiment_score,
             regime_score, claude_confidence, composite_score, action, recommendation_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            sig_id, signal.timestamp, signal.instrument,
            signal.technical_score, signal.sentiment_score,
            signal.regime_score, signal.claude_confidence,
            signal.composite_score, signal.action.value, rec_json,
        ])

    def get_signals(
        self, instrument: str, start: datetime, end: datetime
    ) -> list[SignalScore]:
        rows = self.conn.execute("""
            SELECT timestamp, instrument, technical_score, sentiment_score,
                   regime_score, claude_confidence, composite_score, action
            FROM signals
            WHERE instrument = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """, [instrument, start, end]).fetchall()
        return [
            SignalScore(
                instrument=row[1], timestamp=row[0],
                technical_score=row[2], sentiment_score=row[3],
                regime_score=row[4], claude_confidence=row[5],
                composite_score=row[6], action=TradeAction(row[7]),
            )
            for row in rows
        ]

    # ---------------------------------------------------------------- Trades

    def save_trade(self, trade: TradeRecord) -> None:
        self.conn.execute("""
            INSERT OR REPLACE INTO trades VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            trade.id, trade.instrument, trade.side.value,
            trade.entry_price, trade.exit_price, trade.quantity,
            trade.pnl, trade.pnl_pct, trade.commission_total,
            trade.entry_time, trade.exit_time, trade.duration_minutes,
            trade.exit_reason.value, trade.signal_score,
            trade.claude_reasoning_entry, trade.claude_reasoning_exit,
            json.dumps(trade.technical_at_entry),
            trade.regime_at_entry,
            json.dumps(trade.news_at_entry),
            json.dumps(trade.factor_attribution),
            trade.lessons_learned,
        ])

    def get_trades(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        instrument: str | None = None,
        limit: int = 1000,
    ) -> list[TradeRecord]:
        conditions = []
        params: list = []
        if start:
            conditions.append("entry_time >= ?")
            params.append(start)
        if end:
            conditions.append("entry_time <= ?")
            params.append(end)
        if instrument:
            conditions.append("instrument = ?")
            params.append(instrument)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        rows = self.conn.execute(f"""
            SELECT id, instrument, side, entry_price, exit_price, quantity,
                   pnl, pnl_pct, commission_total, entry_time, exit_time,
                   duration_minutes, exit_reason, signal_score,
                   claude_reasoning_entry, claude_reasoning_exit,
                   technical_at_entry, regime_at_entry,
                   news_at_entry, factor_attribution, lessons_learned
            FROM trades {where}
            ORDER BY entry_time DESC
            LIMIT ?
        """, params).fetchall()

        return [
            TradeRecord(
                id=r[0], instrument=r[1], side=Side(r[2]),
                entry_price=r[3], exit_price=r[4], quantity=r[5],
                pnl=r[6], pnl_pct=r[7], commission_total=r[8],
                entry_time=r[9], exit_time=r[10],
                duration_minutes=r[11],
                exit_reason=ExitReason(r[12]),
                signal_score=r[13] or 0.0,
                claude_reasoning_entry=r[14] or "",
                claude_reasoning_exit=r[15] or "",
                technical_at_entry=json.loads(r[16]) if r[16] else {},
                regime_at_entry=r[17] or "",
                news_at_entry=json.loads(r[18]) if r[18] else [],
                factor_attribution=json.loads(r[19]) if r[19] else {},
                lessons_learned=r[20] or "",
            )
            for r in rows
        ]

    # ----------------------------------------------------------- Portfolio

    def save_portfolio_snapshot(self, state: PortfolioState) -> None:
        import dataclasses
        self.conn.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots
            (timestamp, equity, cash, unrealized_pnl, realized_pnl,
             daily_pnl, max_drawdown, positions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            state.timestamp, state.equity, state.cash,
            state.total_unrealized_pnl, state.total_realized_pnl,
            state.daily_pnl, state.max_drawdown_current,
            json.dumps([dataclasses.asdict(p) for p in state.positions], default=str),
        ])

    def get_portfolio_history(
        self, start: datetime, end: datetime
    ) -> pl.DataFrame:
        return self.conn.execute("""
            SELECT * FROM portfolio_snapshots
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """, [start, end]).pl()

    # -------------------------------------------------------- Economic Events

    def save_events(self, events: list[EconomicEvent]) -> None:
        rows = [
            (
                f"{e.timestamp.isoformat()}_{e.name[:20]}",
                e.timestamp, e.name, e.country, e.impact,
                e.forecast, e.previous, e.actual,
            )
            for e in events
        ]
        self.conn.executemany("""
            INSERT OR IGNORE INTO economic_events
            (id, timestamp, name, country, impact, forecast, previous, actual)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    def get_upcoming_events(self, hours: int = 4) -> list[EconomicEvent]:
        now = datetime.utcnow()
        end = now + timedelta(hours=hours)
        rows = self.conn.execute("""
            SELECT timestamp, name, country, impact, forecast, previous, actual
            FROM economic_events
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """, [now, end]).fetchall()
        return [
            EconomicEvent(
                timestamp=r[0], name=r[1], country=r[2] or "",
                impact=r[3] or "LOW", forecast=r[4] or "",
                previous=r[5] or "", actual=r[6] or "",
            )
            for r in rows
        ]

    # --------------------------------------------------------- Journal entries

    def save_journal_entry(self, event: str, trade_id: str,
                           instrument: str, context: dict) -> None:
        entry_id = f"JNL-{trade_id}-{event}"
        self.conn.execute("""
            INSERT OR REPLACE INTO journal_entries
            (id, trade_id, event, timestamp, instrument, context_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [entry_id, trade_id, event, datetime.utcnow(),
              instrument, json.dumps(context, default=str)])

    def get_journal_entry(self, event: str, trade_id: str) -> dict:
        row = self.conn.execute("""
            SELECT context_json FROM journal_entries
            WHERE trade_id = ? AND event = ?
        """, [trade_id, event]).fetchone()
        return json.loads(row[0]) if row else {}

    def update_trade(self, trade: TradeRecord) -> None:
        """Update attribution and lessons on an existing trade row."""
        self.conn.execute("""
            UPDATE trades SET
                factor_attribution   = ?,
                lessons_learned      = ?,
                claude_reasoning_exit = ?
            WHERE id = ?
        """, [
            json.dumps(trade.factor_attribution),
            trade.lessons_learned,
            trade.claude_reasoning_exit,
            trade.id,
        ])

    # -------------------------------------------------------------- Cleanup

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")

    def close(self) -> None:
        self.conn.close()
