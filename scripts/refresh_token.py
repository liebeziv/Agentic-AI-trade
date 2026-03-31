"""Auto-refresh Claude Code OAuth token into .env — run as a cron or before trading."""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path

CREDENTIALS = Path.home() / ".claude" / ".credentials.json"
ENV_FILE = Path(__file__).parent.parent / ".env"
REFRESH_URL = "https://claude.ai/api/auth/oauth/token"


def load_credentials() -> dict:
    with open(CREDENTIALS, encoding="utf-8") as f:
        return json.load(f).get("claudeAiOauth", {})


def save_credentials(oauth: dict) -> None:
    creds_path = CREDENTIALS
    with open(creds_path, encoding="utf-8") as f:
        data = json.load(f)
    data["claudeAiOauth"] = oauth
    with open(creds_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def token_expires_in(oauth: dict) -> float:
    """Return seconds until token expires."""
    exp_ms = oauth.get("expiresAt", 0)
    return exp_ms / 1000 - time.time()


def refresh_token(oauth: dict) -> dict:
    """Call Anthropic OAuth refresh endpoint and return updated oauth dict."""
    import urllib.request
    import urllib.error

    refresh_tok = oauth.get("refreshToken", "")
    if not refresh_tok:
        raise ValueError("No refresh token available")

    payload = json.dumps({
        "grant_type": "refresh_token",
        "refresh_token": refresh_tok,
    }).encode()

    req = urllib.request.Request(
        REFRESH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Token refresh failed: {exc.code} {exc.reason}") from exc

    # Update oauth dict with new tokens
    if "access_token" in body:
        oauth["accessToken"] = body["access_token"]
    if "refresh_token" in body:
        oauth["refreshToken"] = body["refresh_token"]
    if "expires_in" in body:
        oauth["expiresAt"] = int((time.time() + body["expires_in"]) * 1000)

    return oauth


def write_to_env(token: str) -> None:
    """Update ANTHROPIC_API_KEY in .env file."""
    try:
        content = ENV_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = ""

    if "ANTHROPIC_API_KEY=" in content:
        content = re.sub(r"ANTHROPIC_API_KEY=.*", f"ANTHROPIC_API_KEY={token}", content)
    else:
        content = f"ANTHROPIC_API_KEY={token}\n" + content

    ENV_FILE.write_text(content, encoding="utf-8")


def main(force: bool = False) -> None:
    oauth = load_credentials()
    expires_in = token_expires_in(oauth)

    print(f"Token expires in: {expires_in/3600:.1f}h")

    # Refresh if less than 1 hour remaining or forced
    if expires_in < 3600 or force:
        print("Refreshing token...")
        try:
            oauth = refresh_token(oauth)
            save_credentials(oauth)
            print("Token refreshed successfully")
        except Exception as exc:
            print(f"Refresh failed: {exc}")
            print("Using existing token (may expire soon)")

    token = oauth.get("accessToken", "")
    write_to_env(token)
    print(f"Token written to .env: {token[:20]}...")
    print(f"Now expires in: {token_expires_in(oauth)/3600:.1f}h")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    main(force=force)
