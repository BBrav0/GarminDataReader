#!/usr/bin/env python3
"""One-time Withings OAuth setup for Ben's smart scale data.

Requires in /Users/bensopenclaw/garmin/.env:
  WITHINGS_CLIENT_ID=...
  WITHINGS_CLIENT_SECRET=...
  WITHINGS_REDIRECT_URI=http://localhost:5000/callback

Opens/prints an auth URL, catches the localhost callback, exchanges the short-lived
code for tokens, and stores them in ~/.hermes/secrets/withings_tokens.json.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent.absolute()
HERMES_SECRETS = Path.home() / ".hermes" / "secrets"
TOKEN_PATH = HERMES_SECRETS / "withings_tokens.json"
API_ENDPOINT = "https://wbsapi.withings.net"
AUTH_ENDPOINT = "https://account.withings.com/oauth2_user/authorize2"
SCOPES = "user.info,user.metrics,user.activity"


def ensure_venv() -> None:
    """Re-execute script with venv Python if not already using it."""
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        return

    for env_name in (".venv", "venv"):
        venv_python = SCRIPT_DIR / env_name / "bin" / "python3"
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)


def load_config() -> tuple[str, str, str]:
    load_dotenv(SCRIPT_DIR / ".env")
    client_id = os.getenv("WITHINGS_CLIENT_ID")
    client_secret = os.getenv("WITHINGS_CLIENT_SECRET")
    redirect_uri = os.getenv("WITHINGS_REDIRECT_URI", "http://localhost:5000/callback")
    missing = [name for name, value in {
        "WITHINGS_CLIENT_ID": client_id,
        "WITHINGS_CLIENT_SECRET": client_secret,
    }.items() if not value]
    if missing:
        print("ERROR: missing " + ", ".join(missing) + " in /Users/bensopenclaw/garmin/.env")
        sys.exit(1)
    assert client_id is not None
    assert client_secret is not None
    return client_id, client_secret, redirect_uri


def sign(params: dict[str, object], client_secret: str) -> str:
    # Withings signing docs: sign action + client_id plus timestamp OR nonce,
    # ordered alphabetically by key name and joined as values with commas.
    params_to_sign = {
        "action": params["action"],
        "client_id": params["client_id"],
    }
    if "nonce" in params and params["nonce"]:
        params_to_sign["nonce"] = params["nonce"]
    if "timestamp" in params and params["timestamp"]:
        params_to_sign["timestamp"] = params["timestamp"]
    data = ",".join(str(params_to_sign[key]) for key in sorted(params_to_sign))
    return hmac.new(client_secret.encode(), data.encode(), hashlib.sha256).hexdigest()


def get_nonce(client_id: str, client_secret: str) -> str:
    params: dict[str, object] = {
        "action": "getnonce",
        "client_id": client_id,
        "timestamp": int(time.time()),
    }
    params["signature"] = sign(params, client_secret)
    resp = requests.post(f"{API_ENDPOINT}/v2/signature", data=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 0:
        raise RuntimeError(f"Withings getnonce failed: {payload}")
    return payload["body"]["nonce"]


def request_token(client_id: str, client_secret: str, redirect_uri: str, code: str) -> dict[str, object]:
    params: dict[str, object] = {
        "action": "requesttoken",
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "code": code,
        "grant_type": "authorization_code",
    }
    resp = requests.post(f"{API_ENDPOINT}/v2/oauth2", data=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 0:
        raise RuntimeError(f"Withings token exchange failed: {payload}")
    body = payload["body"]
    body["obtained_at"] = int(time.time())
    return body


class CallbackHandler(BaseHTTPRequestHandler):
    server_version = "WithingsAuth/1.0"

    def log_message(self, format: str, *args: object) -> None:  # quiet default HTTP logs
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        self.server.auth_code = qs.get("code", [None])[0]  # type: ignore[attr-defined]
        self.server.auth_state = qs.get("state", [None])[0]  # type: ignore[attr-defined]
        self.server.auth_error = qs.get("error", [None])[0]  # type: ignore[attr-defined]
        ok = self.server.auth_code and not self.server.auth_error  # type: ignore[attr-defined]
        message = "Withings authorization captured. You can close this tab." if ok else "Withings authorization failed. Check the terminal."
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(message.encode())


def main() -> None:
    ensure_venv()
    client_id, client_secret, redirect_uri = load_config()
    parsed_redirect = urllib.parse.urlparse(redirect_uri)
    host = parsed_redirect.hostname or "localhost"
    port = parsed_redirect.port or 5000
    state = secrets.token_urlsafe(24)
    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": SCOPES,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    auth_url = AUTH_ENDPOINT + "?" + urllib.parse.urlencode(params)

    suppress_auth_url = os.getenv("WITHINGS_SUPPRESS_AUTH_URL") == "1"
    if suppress_auth_url:
        print("Opening Withings authorization URL in the default browser.")
    else:
        print("Open this Withings authorization URL:")
        print(auth_url)
    print("\nWaiting for callback on", redirect_uri)

    server = HTTPServer((host, port), CallbackHandler)
    timeout_seconds = int(os.getenv("WITHINGS_OAUTH_TIMEOUT_SECONDS", "180"))
    server.timeout = timeout_seconds
    server.auth_code = None  # type: ignore[attr-defined]
    server.auth_state = None  # type: ignore[attr-defined]
    server.auth_error = None  # type: ignore[attr-defined]
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    deadline = time.time() + timeout_seconds
    while time.time() < deadline and not server.auth_code and not server.auth_error:  # type: ignore[attr-defined]
        server.handle_request()

    if server.auth_error:  # type: ignore[attr-defined]
        raise SystemExit(f"ERROR: Withings authorization returned error: {server.auth_error}")  # type: ignore[attr-defined]
    if not server.auth_code:  # type: ignore[attr-defined]
        raise SystemExit("ERROR: timed out waiting for Withings callback")
    if server.auth_state != state:  # type: ignore[attr-defined]
        raise SystemExit("ERROR: OAuth state mismatch; refusing token exchange")

    tokens = request_token(client_id, client_secret, redirect_uri, server.auth_code)  # type: ignore[arg-type, attr-defined]
    HERMES_SECRETS.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(tokens, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TOKEN_PATH.chmod(0o600)
    print(f"Saved Withings tokens to {TOKEN_PATH}")
    print("Withings OAuth setup complete.")


if __name__ == "__main__":
    main()
