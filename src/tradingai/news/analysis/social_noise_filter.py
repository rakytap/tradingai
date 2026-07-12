from __future__ import annotations

import re
from datetime import datetime, timezone

MIN_TEXT_LENGTH = 20
MIN_REDDIT_SCORE = 3
LOW_EFFORT_MAX_LENGTH = 50

MEME_ONLY_PATTERNS = [
    r"^(to the moon|moon|lambo|wen\s|yolo|hodl|diamond hands)[\s!.]*$",
    r"^[\U0001F680\U0001F4C8\U0001F4B0\s!.]+$",
]

SPAM_PATTERNS = [
    r"free\s+signals?",
    r"join\s+my\s+(discord|telegram)",
    r"dm\s+me",
    r"guaranteed\s+returns?",
]


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def mentions_ticker(text: str, ticker: str) -> bool:
    pattern = rf"\$?{re.escape(ticker.upper())}\b"
    return bool(re.search(pattern, text.upper()))


def is_low_effort(text: str) -> bool:
    cleaned = _normalize(text)
    if len(cleaned) < MIN_TEXT_LENGTH:
        return True
    if len(cleaned) < LOW_EFFORT_MAX_LENGTH and cleaned.isupper():
        return True
    lowered = cleaned.lower()
    for pattern in MEME_ONLY_PATTERNS:
        if re.fullmatch(pattern, lowered, flags=re.IGNORECASE):
            return True
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


def filter_social_posts(posts: list, ticker: str) -> tuple[list, int]:
    """Return filtered posts and count of removed noisy items."""
    kept = []
    removed = 0
    seen_titles: set[str] = set()

    for post in posts:
        text = _normalize(f"{post.title} {post.summary or ''}")
        if not mentions_ticker(text, ticker):
            removed += 1
            continue
        if is_low_effort(text):
            removed += 1
            continue
        if post.engagement_score is not None and post.engagement_score < MIN_REDDIT_SCORE:
            if post.platform == "reddit":
                removed += 1
                continue
        title_key = post.title.strip().lower()
        if title_key in seen_titles:
            removed += 1
            continue
        seen_titles.add(title_key)
        kept.append(post)

    return kept, removed
