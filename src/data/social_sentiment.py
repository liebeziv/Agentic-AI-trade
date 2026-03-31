"""Social sentiment fetcher — Reddit (PRAW) + StockTwits REST API."""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import aiohttp

from src.utils.logger import get_logger

log = get_logger(__name__)

INSTRUMENT_KEYWORDS: dict[str, list[str]] = {
    "AAPL":     ["apple", "aapl", "iphone", "tim cook"],
    "NVDA":     ["nvidia", "nvda", "gpu", "jensen huang", "ai chips"],
    "MSFT":     ["microsoft", "msft", "azure", "satya nadella"],
    "TSLA":     ["tesla", "tsla", "elon musk", "electric vehicle", "ev"],
    "AMZN":     ["amazon", "amzn", "aws", "jeff bezos"],
    "META":     ["meta", "facebook", "instagram", "mark zuckerberg"],
    "GOOGL":    ["google", "alphabet", "googl", "sundar pichai"],
    "SPY":      ["s&p 500", "market rally", "stock market", "equities", "spy"],
    "EURUSD":   ["euro", "ecb", "eurozone", "eur/usd", "eurusd", "european central bank"],
    "BTC/USDT": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "ETH/USDT": ["ethereum", "eth", "defi", "smart contracts"],
    "GC":       ["gold", "xau", "precious metals", "gold price", "bullion"],
    "CL":       ["crude", "oil", "wti", "brent", "opec", "petroleum"],
    "ES":       ["s&p", "sp500", "s&p 500", "us stocks", "wall street", "spx"],
}

_BULLISH_WORDS = {"bull", "buy", "moon", "rally", "long", "breakout", "surge"}
_BEARISH_WORDS = {"bear", "sell", "crash", "short", "dump", "drop", "panic"}

_REDDIT_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "cryptocurrency"]

# StockTwits uses ticker symbols directly; map our instrument IDs to their format
_STOCKTWITS_SYMBOL_MAP: dict[str, str] = {
    "AAPL":     "AAPL",
    "NVDA":     "NVDA",
    "MSFT":     "MSFT",
    "TSLA":     "TSLA",
    "AMZN":     "AMZN",
    "META":     "META",
    "GOOGL":    "GOOGL",
    "SPY":      "SPY",
    "BTC/USDT": "BTC.X",
    "ETH/USDT": "ETH.X",
    "GC":       "GC_F",
    "CL":       "CL_F",
    "ES":       "ES_F",
    "EURUSD":   "EUR.USD",
}


def _simple_sentiment(text: str) -> float:
    """Rule-based sentiment scorer. Returns a value in [-1.0, 1.0]."""
    words = set(text.lower().split())
    bullish_hits = len(words & _BULLISH_WORDS)
    bearish_hits = len(words & _BEARISH_WORDS)
    total = bullish_hits + bearish_hits
    if total == 0:
        return 0.0
    raw = (bullish_hits - bearish_hits) / total
    # Scale to [-1, 1] with a mild dampening — full score only when heavily one-sided
    return max(-1.0, min(1.0, raw))


def _url_hash(url: str) -> str:
    return hashlib.md5(url.strip().encode()).hexdigest()


def _match_instruments(text: str) -> list[str]:
    text_lower = text.lower()
    matched = []
    for instrument, keywords in INSTRUMENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(instrument)
    return matched


@dataclass
class SocialPost:
    timestamp: datetime
    text: str
    source: str          # "reddit" or "stocktwits"
    instrument: str
    sentiment_score: float  # -1..+1
    score: int              # upvotes / likes
    url: str = ""


class SocialSentimentFeed:
    """Fetches social sentiment from Reddit (via asyncpraw) and StockTwits REST API."""

    def __init__(self, config: dict) -> None:
        self._reddit_client_id: str = config.get("reddit_client_id", "")
        self._reddit_client_secret: str = config.get("reddit_client_secret", "")
        self._reddit_user_agent: str = config.get(
            "reddit_user_agent", "atlas-trader/1.0"
        )
        self._stocktwits_token: str = config.get("stocktwits_token", "")

    async def fetch_all(self, instruments: list[str]) -> list[SocialPost]:
        """Fetch from Reddit and StockTwits, deduplicate by URL hash, return newest-first."""
        reddit_posts, stocktwits_posts = await asyncio.gather(
            self._fetch_reddit(instruments),
            self._fetch_stocktwits(instruments),
            return_exceptions=False,
        )

        all_posts: list[SocialPost] = []
        all_posts.extend(reddit_posts)
        all_posts.extend(stocktwits_posts)

        seen_hashes: set[str] = set()
        unique: list[SocialPost] = []
        for post in all_posts:
            key = _url_hash(post.url) if post.url else _url_hash(post.text[:120])
            if key not in seen_hashes:
                seen_hashes.add(key)
                unique.append(post)

        unique.sort(key=lambda p: p.timestamp, reverse=True)
        return unique

    async def _fetch_reddit(self, instruments: list[str]) -> list[SocialPost]:
        """Fetch recent posts from financial subreddits using asyncpraw."""
        try:
            import asyncpraw  # type: ignore
        except ImportError:
            log.warning("asyncpraw not installed; Reddit fetch skipped. pip install asyncpraw")
            return []

        if not self._reddit_client_id or not self._reddit_client_secret:
            log.warning("Reddit credentials not configured; skipping Reddit fetch")
            return []

        posts: list[SocialPost] = []
        try:
            reddit = asyncpraw.Reddit(
                client_id=self._reddit_client_id,
                client_secret=self._reddit_client_secret,
                user_agent=self._reddit_user_agent,
            )

            # Build keyword → instrument reverse map for fast lookup
            keyword_to_instruments: dict[str, list[str]] = {}
            target_instruments = set(instruments)
            for instr in target_instruments:
                for kw in INSTRUMENT_KEYWORDS.get(instr, []):
                    keyword_to_instruments.setdefault(kw, []).append(instr)

            async with reddit:
                for sub_name in _REDDIT_SUBREDDITS:
                    try:
                        subreddit = await reddit.subreddit(sub_name)
                        async for submission in subreddit.hot(limit=50):
                            combined_text = f"{submission.title} {submission.selftext}"
                            matched = _match_instruments(combined_text)
                            matched = [m for m in matched if m in target_instruments]
                            if not matched:
                                continue

                            ts = datetime.utcfromtimestamp(submission.created_utc)
                            sentiment = _simple_sentiment(combined_text)
                            upvotes = int(submission.score) if submission.score else 0
                            url = f"https://www.reddit.com{submission.permalink}"

                            for instrument in matched:
                                posts.append(SocialPost(
                                    timestamp=ts,
                                    text=combined_text[:500],
                                    source="reddit",
                                    instrument=instrument,
                                    sentiment_score=sentiment,
                                    score=upvotes,
                                    url=url,
                                ))
                    except Exception as exc:
                        log.warning("Reddit subreddit fetch failed", subreddit=sub_name, error=str(exc))
                        continue

        except Exception as exc:
            log.warning("Reddit fetch failed", error=str(exc))
            return []

        return posts

    async def _fetch_stocktwits(self, instruments: list[str]) -> list[SocialPost]:
        """Fetch sentiment from StockTwits streams for each instrument."""
        posts: list[SocialPost] = []

        async def fetch_symbol(session: aiohttp.ClientSession, instrument: str) -> list[SocialPost]:
            symbol = _STOCKTWITS_SYMBOL_MAP.get(instrument)
            if not symbol:
                return []

            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            params: dict = {}
            if self._stocktwits_token:
                params["access_token"] = self._stocktwits_token

            try:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 429:
                        log.warning("StockTwits rate limited", symbol=symbol)
                        return []
                    if resp.status != 200:
                        log.warning("StockTwits non-200 response", symbol=symbol, status=resp.status)
                        return []
                    data = await resp.json(content_type=None)
            except Exception as exc:
                log.warning("StockTwits HTTP error", symbol=symbol, error=str(exc))
                return []

            result: list[SocialPost] = []
            for msg in data.get("messages", []):
                try:
                    body: str = msg.get("body", "")
                    created_at: str = msg.get("created_at", "")
                    try:
                        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        ts = datetime.utcnow()

                    # Extract official sentiment label if present
                    entities = msg.get("entities", {})
                    sentiment_entity = entities.get("sentiment")
                    if sentiment_entity and isinstance(sentiment_entity, dict):
                        basic = sentiment_entity.get("basic", "")
                        if basic == "Bullish":
                            sentiment = 0.6
                        elif basic == "Bearish":
                            sentiment = -0.6
                        else:
                            sentiment = _simple_sentiment(body)
                    else:
                        sentiment = _simple_sentiment(body)

                    likes: int = msg.get("likes", {}).get("total", 0) if isinstance(msg.get("likes"), dict) else 0
                    msg_id = str(msg.get("id", ""))
                    post_url = f"https://stocktwits.com/message/{msg_id}" if msg_id else ""

                    result.append(SocialPost(
                        timestamp=ts,
                        text=body[:500],
                        source="stocktwits",
                        instrument=instrument,
                        sentiment_score=sentiment,
                        score=likes,
                        url=post_url,
                    ))
                except Exception as exc:
                    log.warning("StockTwits message parse error", error=str(exc))
                    continue
            return result

        try:
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_symbol(session, instr) for instr in instruments]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, list):
                        posts.extend(res)
                    elif isinstance(res, Exception):
                        log.warning("StockTwits task error", error=str(res))
        except Exception as exc:
            log.warning("StockTwits fetch session error", error=str(exc))
            return []

        return posts

    def get_aggregate_sentiment(
        self,
        posts: list[SocialPost],
        instrument: str,
        hours: int = 4,
    ) -> float:
        """
        Weighted average sentiment for an instrument over the last N hours.

        Weight per post = max(1, post.score) so high-upvote posts dominate.
        Returns 0.0 if no matching posts exist.
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        relevant = [
            p for p in posts
            if p.instrument == instrument and p.timestamp >= since
        ]
        if not relevant:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for post in relevant:
            weight = max(1, post.score)
            weighted_sum += post.sentiment_score * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return max(-1.0, min(1.0, weighted_sum / total_weight))
