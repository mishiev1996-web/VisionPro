"""
scrapers/utils.py — Shared utilities: retry, rate-limit, validation, timezone.
"""
from __future__ import annotations

import datetime as dt
import time
import functools
from typing import Callable, Optional, TypeVar

import requests


T = TypeVar("T")

MSK = dt.timezone(dt.timedelta(hours=3))


def utc_to_msk(utc_dt: dt.datetime) -> dt.datetime:
    """Convert a naive or aware UTC datetime to MSK (UTC+3)."""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=dt.timezone.utc)
    return utc_dt.astimezone(MSK)


def format_msk(utc_str: str) -> str:
    """Convert ISO UTC string to 'YYYY-MM-DD HH:MM MSK' format."""
    try:
        s = utc_str.replace("Z", "+00:00")
        if "+" not in s and "-" not in s[10:]:
            s += "+00:00"
        parsed = dt.datetime.fromisoformat(s)
        msk = utc_to_msk(parsed)
        return msk.strftime("%Y-%m-%d %H:%M") + " МСК"
    except Exception:
        return utc_str


def format_msk_short(utc_str: str) -> str:
    """Convert ISO UTC string to 'DD.MM HH:MM МСК' format."""
    try:
        s = utc_str.replace("Z", "+00:00")
        if "+" not in s and "-" not in s[10:]:
            s += "+00:00"
        parsed = dt.datetime.fromisoformat(s)
        msk = utc_to_msk(parsed)
        return msk.strftime("%d.%m %H:%M") + " МСК"
    except Exception:
        return utc_str


def today_msk() -> dt.date:
    """Get today's date in MSK timezone."""
    return dt.datetime.now(MSK).date()


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """Decorator: retry on failure with exponential backoff."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts:
                        wait = delay * (backoff ** (attempt - 1))
                        time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


def safe_request(url: str, timeout: int = 10, retries: int = 3,
                 headers: Optional[dict] = None) -> Optional[requests.Response]:
    """HTTP GET with retry, timeout, and error handling. Returns None on failure."""
    default_headers = {"User-Agent": "FootballAI/1.0"}
    if headers:
        default_headers.update(headers)

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers=default_headers)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return None
        except (requests.ConnectionError, requests.Timeout):
            if attempt < retries:
                time.sleep(2 ** attempt)
            continue
        except Exception:
            return None
    return None


def validate_match(m: dict) -> bool:
    """Check that a match dict has required fields with valid types."""
    required = ["home", "away"]
    for field in required:
        val = m.get(field)
        if not val or not isinstance(val, str) or len(val.strip()) == 0:
            return False
    score_fields = ["score_home", "score_away"]
    for field in score_fields:
        val = m.get(field)
        if val is not None and val != "":
            try:
                int(val)
            except (ValueError, TypeError):
                pass
    return True


def normalize_team_name(name: str) -> str:
    """Normalize team name for consistent matching."""
    name = name.strip().lower()
    name = name.replace(" fc", "").replace(" f.c.", "").replace("afc ", "")
    name = name.replace("united", "utd").replace("city", "cty")
    return name


def dedup_matches(matches: list) -> list:
    """Remove duplicate matches by (home, away, date) key."""
    seen = set()
    out = []
    for m in matches:
        key = (
            normalize_team_name(m.get("home", "")),
            normalize_team_name(m.get("away", "")),
            m.get("date", "")[:10],
        )
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out
