"""
telegram_auth.py — Telegram Web App initData validation.

Validates the initData string from Telegram Web App using HMAC-SHA256
to ensure requests genuinely originate from Telegram.

Reference: https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Optional, Dict, Any
from urllib.parse import parse_qs, unquote


def _get_bot_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        token_path = os.path.join(os.path.dirname(__file__), "Апи", "telegram_token.txt")
        if os.path.exists(token_path):
            with open(token_path, "r") as f:
                token = f.read().strip()
    return token


def validate_init_data(init_data: str, bot_token: Optional[str] = None,
                       max_age_seconds: int = 86400) -> Optional[Dict[str, Any]]:
    """Validate Telegram Web App initData.

    Returns parsed user dict on success, None on failure.
    max_age_seconds: reject initData older than this (default 24h).
    """
    if not init_data:
        return None

    token = bot_token or _get_bot_token()
    if not token:
        return None

    try:
        parsed = parse_qs(init_data, keep_blank_values=True)
    except Exception:
        return None

    received_hash = parsed.get("hash", [None])[0]
    if not received_hash:
        return None

    # Build data_check_string: all fields except "hash", sorted alphabetically
    data_check_pairs = []
    for key in sorted(parsed.keys()):
        if key == "hash":
            continue
        data_check_pairs.append(f"{key}={parsed[key][0]}")
    data_check_string = "\n".join(data_check_pairs)

    # Compute HMAC-SHA256
    secret_key = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    computed_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed_hash, received_hash):
        return None

    # Check auth_date freshness
    auth_date_str = parsed.get("auth_date", [None])[0]
    if auth_date_str:
        try:
            auth_ts = int(auth_date_str)
            if time.time() - auth_ts > max_age_seconds:
                return None
        except (ValueError, TypeError):
            pass

    # Parse user JSON
    user_json = parsed.get("user", [None])[0]
    if user_json:
        try:
            return json.loads(unquote(user_json))
        except (json.JSONDecodeError, TypeError):
            pass

    return {"validated": True}


def extract_user_id(init_data: str, bot_token: Optional[str] = None) -> Optional[int]:
    """Extract and validate user_id from initData. Returns None if invalid."""
    user = validate_init_data(init_data, bot_token)
    if user and isinstance(user, dict):
        return user.get("id")
    return None
