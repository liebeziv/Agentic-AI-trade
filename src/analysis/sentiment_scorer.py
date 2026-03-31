"""Unified sentiment scorer — merges news and social signals per instrument."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from src.types import NewsItem
from src.data.social_sentiment import SocialPost


@dataclass
class SentimentScore:
    instrument: str
    timestamp: datetime
    news_score: float       # -1..+1
    social_score: float     # -1..+1
    combined_score: float   # weighted combination
    news_count: int
    social_count: int
    confidence: float       # 0..1, based on sample size


class SentimentScorer:
    """
    Combines news-derived sentiment and social sentiment into a single
    per-instrument score, suitable for use as a signal input.
    """

    def __init__(
        self,
        news_weight: float = 0.6,
        social_weight: float = 0.4,
    ) -> None:
        if abs(news_weight + social_weight - 1.0) > 1e-6:
            raise ValueError(
                f"news_weight + social_weight must equal 1.0, "
                f"got {news_weight} + {social_weight} = {news_weight + social_weight}"
            )
        self.news_weight = news_weight
        self.social_weight = social_weight

    def score(
        self,
        instrument: str,
        news_items: list,
        social_posts: list,
        hours: int = 4,
    ) -> SentimentScore:
        """
        Compute a SentimentScore for one instrument.

        news_items  — list of NewsItem; filtered to those mentioning `instrument`
                      within the last `hours` hours.
        social_posts — list of SocialPost; filtered similarly.
        """
        since = datetime.utcnow() - timedelta(hours=hours)

        # --- news ---
        relevant_news: list[NewsItem] = [
            n for n in news_items
            if instrument in n.instruments and n.timestamp >= since
        ]
        news_count = len(relevant_news)
        if news_count > 0:
            news_score = sum(n.sentiment_score for n in relevant_news) / news_count
        else:
            news_score = 0.0

        # --- social ---
        relevant_social: list[SocialPost] = [
            p for p in social_posts
            if p.instrument == instrument and p.timestamp >= since
        ]
        social_count = len(relevant_social)
        if social_count > 0:
            social_score = sum(p.sentiment_score for p in relevant_social) / social_count
        else:
            social_score = 0.0

        # --- combined ---
        combined_score = (
            self.news_weight * news_score
            + self.social_weight * social_score
        )
        combined_score = max(-1.0, min(1.0, combined_score))

        # --- confidence — grows with sample size, saturates at 10 items ---
        confidence = min(1.0, (news_count + social_count) / 10)

        return SentimentScore(
            instrument=instrument,
            timestamp=datetime.utcnow(),
            news_score=max(-1.0, min(1.0, news_score)),
            social_score=max(-1.0, min(1.0, social_score)),
            combined_score=combined_score,
            news_count=news_count,
            social_count=social_count,
            confidence=confidence,
        )

    def score_all(
        self,
        instruments: list[str],
        news_items: list,
        social_posts: list,
    ) -> dict[str, SentimentScore]:
        """Return a mapping of instrument → SentimentScore for every instrument."""
        return {
            instrument: self.score(instrument, news_items, social_posts)
            for instrument in instruments
        }

    def to_signal_input(self, score: SentimentScore) -> float:
        """
        Convert a SentimentScore to a single float suitable for use as
        `sentiment_score` in a SignalScore.

        Returns combined_score * confidence, clamped to [-1, 1].
        A low-confidence score is damped toward zero regardless of direction.
        """
        raw = score.combined_score * score.confidence
        return max(-1.0, min(1.0, raw))
