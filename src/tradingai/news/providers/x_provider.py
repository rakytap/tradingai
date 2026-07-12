from __future__ import annotations

from datetime import datetime, timezone

import httpx

from tradingai.news.config import get_x_bearer_token
from tradingai.news.models import SocialPost
from tradingai.news.providers.base import SocialProvider

X_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"


class XProvider(SocialProvider):
    name = "x"

    def _require_token(self) -> str:
        token = get_x_bearer_token()
        if not token:
            raise ValueError(
                "X_BEARER_TOKEN is not configured. Add it to .env to enable X sentiment."
            )
        return token

    def fetch_posts(self, ticker: str, count: int = 15) -> list[SocialPost]:
        token = self._require_token()
        query = f"${ticker.upper()} lang:en -is:retweet -is:reply"
        params = {
            "query": query,
            "max_results": min(count, 100),
            "tweet.fields": "created_at,public_metrics,lang",
        }
        headers = {"Authorization": f"Bearer {token}"}

        with httpx.Client(timeout=20.0) as client:
            response = client.get(X_SEARCH_URL, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()

        posts: list[SocialPost] = []
        for tweet in payload.get("data", []):
            text = tweet.get("text")
            if not text:
                continue
            metrics = tweet.get("public_metrics", {})
            engagement = float(
                metrics.get("like_count", 0)
                + metrics.get("retweet_count", 0)
                + metrics.get("reply_count", 0)
            )
            published_at = None
            if tweet.get("created_at"):
                published_at = datetime.fromisoformat(
                    tweet["created_at"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            posts.append(
                SocialPost(
                    title=text[:280],
                    summary=text if len(text) > 280 else None,
                    url=f"https://x.com/i/web/status/{tweet.get('id')}",
                    platform="x",
                    source="X",
                    published_at=published_at,
                    engagement_score=engagement,
                    provider=self.name,
                )
            )
        return posts
