from datetime import datetime, timezone

from tradingai.news.analysis.social_noise_filter import filter_social_posts, is_low_effort
from tradingai.news.analysis.sentiment_scorer import score_social_posts
from tradingai.news.models import SocialPost


def _post(title: str, summary: str | None = None, score: float = 10.0) -> SocialPost:
    return SocialPost(
        title=title,
        summary=summary,
        platform="reddit",
        source="r/stocks",
        published_at=datetime.now(timezone.utc),
        engagement_score=score,
        provider="reddit",
    )


def test_is_low_effort_rejects_meme_posts():
    assert is_low_effort("to the moon")
    assert is_low_effort("YOLO")


def test_filter_social_posts_requires_ticker_mention():
    posts = [
        _post("NVDA earnings look strong with data center growth"),
        _post("Market is crazy today"),
    ]
    kept, removed = filter_social_posts(posts, "NVDA")
    assert len(kept) == 1
    assert removed == 1
    assert "NVDA" in kept[0].title


def test_filter_social_posts_removes_low_engagement_reddit():
    posts = [
        _post("NVDA looks bullish on AI demand", score=1.0),
        _post("NVDA beat estimates and raised guidance", score=50.0),
    ]
    kept, removed = filter_social_posts(posts, "NVDA")
    assert len(kept) == 1
    assert kept[0].engagement_score == 50.0


def test_score_social_posts_returns_bounded_value():
    posts = [
        _post("NVDA beat estimates with strong growth", score=20.0),
        _post("NVDA demand remains weak and bearish", score=15.0),
    ]
    kept, _ = filter_social_posts(posts, "NVDA")
    score = score_social_posts(kept)
    assert score is not None
    assert 0.0 <= score <= 1.0
