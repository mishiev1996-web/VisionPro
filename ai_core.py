"""
ai_core.py — Shared LLM infrastructure for football and tennis AI analysis.

Provides: TLS session, API key loading, chat wrapper with retries,
and unified PROB-line parser.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

import config

logger = logging.getLogger("ai_core")


# ── TLS Session ──────────────────────────────────────────────────────────────

class _TLSAdapter(HTTPAdapter):
    """Custom TLS adapter for Polza.ai compatibility."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


_session = requests.Session()
_session.mount('https://', _TLSAdapter())
_session.trust_env = True


# ── API Key ──────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Load Polza.ai API key from env or file."""
    key = os.environ.get("POLZA_API_KEY", "")
    if not key:
        key_path = os.path.join("Апи", "key.txt")
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                key = f.read().strip()
    return key


# ── Chat ─────────────────────────────────────────────────────────────────────

def chat(messages: List[Dict[str, str]], model: str = None,
         temperature: float = 0.7, max_tokens: int = 1500,
         max_retries: int = 3, timeout: int = 120) -> Optional[str]:
    """Call Polza.ai chat completions with retries and model fallback.

    Returns the assistant content string, or None on failure.
    """
    api_key = get_api_key()
    if not api_key:
        logger.warning("No Polza API key found")
        return None
    if not model:
        model = config.DEFAULT_AI_MODEL

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.debug(f"LLM request: model={model} key={api_key[:12]}... url={config.POLZA_BASE_URL}/chat/completions")

    for attempt in range(max_retries):
        try:
            try:
                resp = _session.post(
                    f"{config.POLZA_BASE_URL}/chat/completions",
                    headers=headers, json=payload, timeout=timeout,
                )
            except Exception:
                resp = requests.post(
                    f"{config.POLZA_BASE_URL}/chat/completions",
                    headers=headers, json=payload, timeout=timeout,
                )

            if resp.status_code != 200:
                logger.error(f"Polza {resp.status_code}: {resp.text[:500]}")
                logger.error(f"  Key: {api_key[:12]}... | Model: {model} | URL: {config.POLZA_BASE_URL}/chat/completions")
                resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content and content.strip():
                return content

            # Empty response — retry with fallback model
            if attempt < max_retries - 1:
                fallback = "deepseek/deepseek-v4-flash"
                logger.info(f"Empty response from {model}, retrying with {fallback}")
                model = fallback
                time.sleep(2)
                continue
            return content or None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            logger.error(f"LLM error after {max_retries} attempts: {e}")
            return None


# ── PROB Parser ──────────────────────────────────────────────────────────────

def parse_prob_line(text: str) -> Optional[Dict[str, Any]]:
    """Extract structured predictions from a PROB: line in LLM output.

    Handles football format:
        PROB:home=X.XX:draw=X.XX:away=X.XX[:total_over=X.XX:total_under=X.XX]
             [:btts_yes=X.XX:btts_no=X.XX][:bet=...][:confidence=...]
    And tennis format:
        PROB:p1=X.XX:p2=X.XX[:bet=...][:confidence=...]

    Returns dict with percentages (0-100) and metadata, or None if no PROB found.
    """
    # Try football format first (home/draw/away)
    prob_match = re.search(
        r'PROB:home=([\d.]+):draw=([\d.]+):away=([\d.]+)'
        r'(?::total_over=([\d.]+):total_under=([\d.]+))?'
        r'(?::btts_yes=([\d.]+):btts_no=([\d.]+))?'
        r'(?::bet=([^:]+))?(?::confidence=([^\s]+))?',
        text
    )
    if prob_match:
        result = {
            "home_win": round(float(prob_match.group(1)) * 100, 1),
            "draw": round(float(prob_match.group(2)) * 100, 1),
            "away_win": round(float(prob_match.group(3)) * 100, 1),
        }
        if prob_match.group(4):
            result["total_over_2_5"] = round(float(prob_match.group(4)) * 100, 1)
        if prob_match.group(5):
            result["total_under_2_5"] = round(float(prob_match.group(5)) * 100, 1)
        if prob_match.group(6):
            result["btts_yes"] = round(float(prob_match.group(6)) * 100, 1)
        if prob_match.group(7):
            result["btts_no"] = round(float(prob_match.group(7)) * 100, 1)
        if prob_match.group(8):
            result["main_bet"] = prob_match.group(8).strip()
        if prob_match.group(9):
            result["confidence"] = prob_match.group(9).strip()
        return result

    # Try tennis format (p1/p2)
    prob_match = re.search(
        r'PROB:p1=([\d.]+):p2=([\d.]+)'
        r'(?::bet=([^:]+))?(?::confidence=([^\s]+))?',
        text
    )
    if prob_match:
        p1_val = float(prob_match.group(1))
        p2_val = float(prob_match.group(2))

        # If values > 1, treat as odds and convert to probabilities
        if p1_val > 1 or p2_val > 1:
            total = 1 / p1_val + 1 / p2_val
            p1_prob = round((1 / p1_val) / total, 2)
            p2_prob = round((1 / p2_val) / total, 2)
        else:
            p1_prob = p1_val
            p2_prob = p2_val

        result = {
            "player1_win": round(p1_prob * 100, 1),
            "player2_win": round(p2_prob * 100, 1),
        }
        if prob_match.group(3):
            result["main_bet"] = prob_match.group(3).strip()
        if prob_match.group(4):
            result["confidence"] = prob_match.group(4).strip()
        return result

    return None
