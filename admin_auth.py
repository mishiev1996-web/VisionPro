"""Add admin API key authentication"""
import os
import secrets

# Load or generate admin key
ADMIN_KEY_FILE = os.path.join(os.path.dirname(__file__), "..", "admin_key.txt")

def _get_admin_key():
    """Load admin key from env or file, generate if not exists."""
    key = os.environ.get("ADMIN_API_KEY", "")
    if key:
        return key
    if os.path.exists(ADMIN_KEY_FILE):
        with open(ADMIN_KEY_FILE, "r") as f:
            return f.read().strip()
    # Generate new key
    key = secrets.token_urlsafe(32)
    os.makedirs(os.path.dirname(ADMIN_KEY_FILE), exist_ok=True)
    with open(ADMIN_KEY_FILE, "w") as f:
        f.write(key)
    return key

ADMIN_KEY = _get_admin_key()

def verify_admin_key(request: Request):
    """FastAPI dependency: check X-Admin-Key header."""
    from fastapi import HTTPException
    key = request.headers.get("X-Admin-Key", "")
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key header")
