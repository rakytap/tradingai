from __future__ import annotations

import time
from datetime import datetime, timezone

from tradingai.news.config import get_sec_identity
from tradingai.news.models import EarningsMaterial, FilingItem
from tradingai.news.providers.base import FilingsProvider

_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_TTL_SECONDS = 900


def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value) -> None:
    _CACHE[key] = (time.time() + _CACHE_TTL_SECONDS, value)


def _ensure_sec_identity() -> None:
    identity = get_sec_identity()
    if not identity:
        raise ValueError(
            "SEC_IDENTITY is required for SEC EDGAR access. "
            'Set SEC_IDENTITY="Your Name your@email.com" in .env.'
        )
    from edgar import set_identity

    set_identity(identity)


def _parse_filing_date(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _excerpt(text: str | None, limit: int = 500) -> str | None:
    if not text:
        return None
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


class SecEdgarProvider(FilingsProvider):
    name = "sec_edgar"

    def _get_company(self, ticker: str):
        cache_key = f"company:{ticker.upper()}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        _ensure_sec_identity()
        from edgar import Company

        company = Company(ticker.upper())
        _cache_set(cache_key, company)
        return company

    def fetch_filings(
        self, ticker: str, forms: list[str] | None = None, count: int = 5
    ) -> list[FilingItem]:
        company = self._get_company(ticker)
        requested_forms = forms or ["8-K", "10-Q", "10-K"]
        filings = company.get_filings(form=requested_forms).latest(count)

        items: list[FilingItem] = []
        for filing in filings:
            excerpt = None
            try:
                text = filing.text()
                excerpt = _excerpt(text)
            except Exception:
                excerpt = filing.description

            items.append(
                FilingItem(
                    form=str(filing.form),
                    filed_at=_parse_filing_date(filing.filing_date),
                    accession_number=getattr(filing, "accession_number", None),
                    title=filing.description,
                    excerpt=excerpt,
                    url=getattr(filing, "filing_url", None)
                    or getattr(filing, "homepage_url", None),
                    provider=self.name,
                )
            )
        return items

    def fetch_earnings_materials(self, ticker: str) -> EarningsMaterial | None:
        company = self._get_company(ticker)
        filings = company.get_filings(form="8-K").latest(10)
        if not filings:
            return None

        for filing in filings:
            press_release = None
            transcript = None
            try:
                for attachment in filing.attachments:
                    name = str(getattr(attachment, "document", "") or "").lower()
                    description = str(getattr(attachment, "description", "") or "").lower()
                    text = None
                    try:
                        text = attachment.text()
                    except Exception:
                        text = None

                    if not text:
                        continue

                    if "99.1" in name or "press release" in description:
                        press_release = text
                    elif "99.2" in name or "transcript" in description:
                        transcript = text
            except Exception:
                continue

            if press_release or transcript:
                return EarningsMaterial(
                    filed_at=_parse_filing_date(filing.filing_date),
                    press_release=press_release,
                    transcript=transcript,
                    provider=self.name,
                )

        return None
