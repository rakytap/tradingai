from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx

from tradingai.news.config import (
    get_reddit_client_id,
    get_reddit_client_secret,
    get_reddit_user_agent,
)
from tradingai.news.models import SocialPost
from tradingai.news.providers.base import SocialProvider

DEFAULT_SUBREDDITS = ("stocks", "investing", "wallstreetbets", "StockMarket")
REDDIT_BASE = "https://www.reddit.com"


class RedditProvider(SocialProvider):
    name = "reddit"

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": get_reddit_user_agent()}

    def _fetch_with_praw(self, ticker: str, count: int) -> list[SocialPost]:
        import praw

        client_id = get_reddit_client_id()
        client_secret = get_reddit_client_secret()
        if not client_id or not client_secret:
            raise ValueError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are required when "
                "the public Reddit API is unavailable."
            )

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=get_reddit_user_agent(),
        )

        posts: list[SocialPost] = []
        per_sub = max(3, count // len(DEFAULT_SUBREDDITS))
        for subreddit_name in DEFAULT_SUBREDDITS:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(
                ticker.upper(), sort="top", time_filter="week", limit=per_sub
            ):
                posts.append(
                    SocialPost(
                        title=submission.title,
                        summary=submission.selftext or None,
                        url=f"{REDDIT_BASE}{submission.permalink}",
                        platform="reddit",
                        source=f"r/{subreddit_name}",
                        published_at=datetime.fromtimestamp(
                            submission.created_utc, tz=timezone.utc
                        ),
                        engagement_score=float(submission.score),
                        provider=self.name,
                    )
                )
        return posts

    def _search_subreddit(
        self,
        client: httpx.Client,
        subreddit: str,
        ticker: str,
        count: int,
    ) -> list[SocialPost]:
        url = f"{REDDIT_BASE}/r/{subreddit}/search.json"
        params = {
            "q": ticker.upper(),
            "restrict_sr": "on",
            "sort": "top",
            "t": "week",
            "limit": count,
        }
        response = client.get(url, params=params, headers=self._headers(), timeout=20.0)
        response.raise_for_status()
        payload = response.json()

        posts: list[SocialPost] = []
        for child in payload.get("data", {}).get("children", []):
            data = child.get("data", {})
            title = data.get("title")
            if not title:
                continue
            created = data.get("created_utc")
            published_at = None
            if created is not None:
                published_at = datetime.fromtimestamp(created, tz=timezone.utc)
            posts.append(
                SocialPost(
                    title=title,
                    summary=data.get("selftext") or None,
                    url=f"{REDDIT_BASE}{data.get('permalink', '')}"
                    if data.get("permalink")
                    else None,
                    platform="reddit",
                    source=f"r/{data.get('subreddit', subreddit)}",
                    published_at=published_at,
                    engagement_score=float(data.get("score", 0)),
                    provider=self.name,
                )
            )
        return posts

    def _fetch_with_public_api(self, ticker: str, count: int) -> list[SocialPost]:
        per_sub = max(3, count // len(DEFAULT_SUBREDDITS))
        collected: list[SocialPost] = []
        with httpx.Client() as client:
            for subreddit in DEFAULT_SUBREDDITS:
                try:
                    collected.extend(
                        self._search_subreddit(client, subreddit, ticker, per_sub)
                    )
                    time.sleep(0.2)
                except Exception:
                    continue
        return collected

    def fetch_posts(self, ticker: str, count: int = 15) -> list[SocialPost]:
        try:
            posts = self._fetch_with_public_api(ticker, count)
            if posts:
                return posts[: count * 2]
        except Exception:
            pass

        if get_reddit_client_id() and get_reddit_client_secret():
            return self._fetch_with_praw(ticker, count)[: count * 2]

        return []
