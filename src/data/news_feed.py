"""News feed — NewsAPI + RSS sources."""
from __future__ import annotations
import asyncio
import hashlib
from datetime import datetime, timedelta

import aiohttp
import feedparser

from src.types import NewsItem
from src.utils.logger import get_logger

log = get_logger(__name__)

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # markets
    "https://www.ft.com/markets?format=rss",
]

INSTRUMENT_KEYWORDS: dict[str, list[str]] = {
    "EURUSD": ["euro", "ecb", "eurozone", "eur/usd", "eurusd", "european central bank"],
    "GBPUSD": ["pound", "sterling", "boe", "bank of england", "gbp", "brexit"],
    "USDJPY": ["yen", "boj", "bank of japan", "jpy", "japanese"],
    "AUDUSD": ["australian dollar", "rba", "reserve bank australia", "aud"],
    "ES":     ["s&p", "sp500", "s&p 500", "us stocks", "wall street", "spx"],
    "NQ":     ["nasdaq", "tech stocks", "qqq", "ndx"],
    "GC":     ["gold", "xau", "precious metals", "gold price", "bullion"],
    "CL":     ["crude", "oil", "wti", "brent", "opec", "petroleum"],
    "BTC/USDT": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "ETH/USDT": ["ethereum", "eth", "defi", "smart contracts"],
    "SOL/USDT": ["solana", "sol"],
    "AAPL":   ["apple", "aapl", "iphone", "tim cook"],
    "NVDA":   ["nvidia", "nvda", "gpu", "jensen huang", "ai chips"],
    "MSFT":   ["microsoft", "msft", "azure", "satya nadella"],
    "TSLA":   ["tesla", "tsla", "elon musk", "electric vehicle", "ev"],
    "AMZN":   ["amazon", "amzn", "aws", "jeff bezos"],
    "META":   ["meta", "facebook", "instagram", "mark zuckerberg"],
    "GOOGL":  ["google", "alphabet", "googl", "sundar pichai"],
    "SPY":    ["s&p 500", "market rally", "stock market", "equities"],
}


def _compute_relevance(text: str, instrument: str) -> float:
    keywords = INSTRUMENT_KEYWORDS.get(instrument, [])
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(1.0, hits / max(1, len(keywords)) * 3)


def _dedup_key(title: str) -> str:
    return hashlib.md5(title.lower().strip().encode()).hexdigest()


class NewsFeed:
    def __init__(self, config: dict) -> None:
        self.api_key: str = config.get("newsapi_key", "")
        self.fetch_interval: int = config.get("fetch_interval_seconds", 300)
        self._cache: dict[str, NewsItem] = {}
        self._last_fetch: datetime | None = None

    async def fetch_all(self, instruments: list[str]) -> list[NewsItem]:
        """Fetch from all sources and return deduplicated, relevance-scored items."""
        items: list[NewsItem] = []

        rss_items = await self._fetch_rss()
        items.extend(rss_items)

        if self.api_key:
            api_items = await self._fetch_newsapi(instruments)
            items.extend(api_items)

        # Deduplicate
        seen: set[str] = set()
        unique: list[NewsItem] = []
        for item in items:
            key = _dedup_key(item.title)
            if key not in seen:
                seen.add(key)
                unique.append(item)

        # Score relevance per instrument
        for item in unique:
            combined = f"{item.title} {item.summary}"
            item.instruments = [
                instr for instr in instruments
                if _compute_relevance(combined, instr) > 0.1
            ]
            if item.instruments:
                item.relevance_score = max(
                    _compute_relevance(combined, i) for i in item.instruments
                )

        self._last_fetch = datetime.utcnow()
        return [i for i in unique if i.instruments]

    async def get_recent(self, instrument: str, hours: int = 4) -> list[NewsItem]:
        """Return cached items relevant to instrument from last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        return [
            item for item in self._cache.values()
            if instrument in item.instruments and item.timestamp >= since
        ]

    async def _fetch_rss(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        loop = asyncio.get_event_loop()

        async def parse_feed(url: str) -> list[NewsItem]:
            try:
                feed = await loop.run_in_executor(None, lambda: feedparser.parse(url))
                result = []
                for entry in feed.entries[:20]:
                    ts_struct = getattr(entry, "published_parsed", None)
                    if ts_struct:
                        ts = datetime(*ts_struct[:6])
                    else:
                        ts = datetime.utcnow()

                    summary = getattr(entry, "summary", "") or ""
                    result.append(NewsItem(
                        timestamp=ts,
                        title=entry.get("title", ""),
                        summary=summary[:500],
                        source=feed.feed.get("title", url),
                        url=entry.get("link", ""),
                        sentiment_raw=summary,
                    ))
                return result
            except Exception as exc:
                log.warning("RSS fetch failed", url=url, error=str(exc))
                return []

        tasks = [parse_feed(url) for url in RSS_FEEDS]
        results = await asyncio.gather(*tasks)
        for batch in results:
            items.extend(batch)
        return items

    async def _fetch_newsapi(self, instruments: list[str]) -> list[NewsItem]:
        # Gather keywords for all instruments
        all_keywords: list[str] = []
        for instr in instruments:
            all_keywords.extend(INSTRUMENT_KEYWORDS.get(instr, [])[:2])
        query = " OR ".join(set(all_keywords[:10]))

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": self.api_key,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
        except Exception as exc:
            log.warning("NewsAPI fetch failed", error=str(exc))
            return []

        items = []
        for article in data.get("articles", []):
            try:
                ts = datetime.fromisoformat(
                    article["publishedAt"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except Exception:
                ts = datetime.utcnow()
            items.append(NewsItem(
                timestamp=ts,
                title=article.get("title", ""),
                summary=article.get("description", "") or "",
                source=article.get("source", {}).get("name", "NewsAPI"),
                url=article.get("url", ""),
                sentiment_raw=article.get("content", "") or "",
            ))
        return items
