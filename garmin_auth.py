#!/usr/bin/env python3
"""Helpers for Garmin authentication with graceful rate-limit retries.

Compatible with garminconnect >= 0.3.0 (native DI OAuth, no garth).
"""

import os
import time
from pathlib import Path

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectTooManyRequestsError,
)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_INITIAL_DELAY_SECONDS = 30
DEFAULT_MAX_DELAY_SECONDS = 120
DEFAULT_TOKENSTORE_DIR = Path.home() / ".garminconnect"
MOBILE_SIGNIN_URL = (
    "https://sso.garmin.com/mobile/sso/en-US/sign-in"
    "?clientId=GCM_ANDROID_DARK"
    "&service=https%3A%2F%2Fmobile.integration.garmin.com%2Fgcm%2Fandroid"
)
MOBILE_SERVICE_URL = "https://mobile.integration.garmin.com/gcm/android"

RATE_LIMIT_HINTS = ("429", "too many requests", "rate limit")
BROWSER_BOOTSTRAP_HINTS = ("cloudflare", "just a moment", "portal login failed (non-json): http 403")


def is_rate_limit_error(error):
    """Return True when an exception indicates Garmin rate limiting."""
    if isinstance(error, GarminConnectTooManyRequestsError):
        return True

    # v0.3.0 wraps 429 inside GarminConnectAuthenticationError — check the chain.
    current = error
    while current is not None:
        if isinstance(current, GarminConnectTooManyRequestsError):
            return True
        msg = str(current).lower()
        if any(hint in msg for hint in RATE_LIMIT_HINTS):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    status = getattr(getattr(error, "response", None), "status_code", None)
    if status == 429:
        return True

    return False


def should_try_browser_bootstrap(error):
    """Return True when a stealth browser bootstrap is worth trying."""
    if is_rate_limit_error(error):
        return True

    message = str(error).lower()
    return any(hint in message for hint in BROWSER_BOOTSTRAP_HINTS)


def retry_on_rate_limit(
    action,
    description,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    initial_delay_seconds=DEFAULT_INITIAL_DELAY_SECONDS,
    max_delay_seconds=DEFAULT_MAX_DELAY_SECONDS,
):
    """Run *action* with exponential backoff on 429 errors.

    Fails loudly (raises) after *max_attempts* so the caller never hangs.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except Exception as error:
            if is_rate_limit_error(error) and attempt < max_attempts:
                delay = min(max_delay_seconds, initial_delay_seconds * (2 ** (attempt - 1)))
                print(
                    f"⚠ {description} rate-limited "
                    f"(attempt {attempt}/{max_attempts}): {error}"
                )
                print(f"  Waiting {delay}s before retrying...")
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(f"{description} failed after {max_attempts} attempts")


def get_tokenstore_path(tokenstore=None):
    """Return the configured Garmin tokenstore directory."""
    configured = tokenstore or os.getenv("GARMINTOKENS")
    return Path(configured).expanduser() if configured else DEFAULT_TOKENSTORE_DIR


def tokenstore_file_exists(tokenstore_path):
    """Return True when a Garmin tokenstore file already exists."""
    tokenstore_path = Path(tokenstore_path).expanduser()
    if tokenstore_path.is_file():
        return True
    return (tokenstore_path / "garmin_tokens.json").exists()


def persist_tokens(client, tokenstore_path):
    """Persist refreshed Garmin tokens, warning instead of failing."""
    try:
        client.client.dump(str(tokenstore_path))
    except Exception as exc:
        print(f"⚠ Could not persist Garmin tokens: {exc}")


def login_to_garmin(email, password, tokenstore=None):
    """Authenticate to Garmin Connect and persist tokens for reuse.

    On first run, performs a full SSO login and writes tokens to
    *tokenstore_path* so subsequent runs can skip SSO entirely.
    """
    tokenstore_path = get_tokenstore_path(tokenstore)

    def login_with_tokenstore():
        client = Garmin(email, password)
        client.login(tokenstore=str(tokenstore_path))
        persist_tokens(client, tokenstore_path)
        return client

    def login_with_credentials():
        client = Garmin(email, password)
        client.login()
        persist_tokens(client, tokenstore_path)
        return client

    def login_with_camoufox():
        try:
            from camoufox.sync_api import Camoufox
        except Exception as import_error:
            raise RuntimeError(
                "Garmin browser bootstrap requested, but Camoufox is unavailable"
            ) from import_error

        print("⚠ Falling back to stealth browser login to bootstrap Garmin tokens...")
        ticket_holder = {}
        login_error_holder = {}

        with Camoufox(headless=False) as browser:
            page = browser.new_page()

            def handle_response(response):
                if "mobile/api/login" not in response.url:
                    return
                try:
                    payload = response.json()
                except Exception:
                    payload = {
                        "responseStatus": {
                            "type": "UNKNOWN",
                            "message": response.text()[:500],
                        }
                    }

                response_type = payload.get("responseStatus", {}).get("type")
                if response_type == "SUCCESSFUL":
                    ticket_holder["ticket"] = payload.get("serviceTicketId")
                else:
                    login_error_holder["payload"] = payload

            page.on("response", handle_response)
            page.goto(MOBILE_SIGNIN_URL, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(4000)
            page.locator("input#email").fill(email)
            page.locator("input#password").fill(password)
            page.locator("button[type='submit']").click()
            page.wait_for_timeout(15000)

        if "payload" in login_error_holder:
            payload = login_error_holder["payload"]
            response_type = payload.get("responseStatus", {}).get("type")
            if response_type == "MFA_REQUIRED":
                raise RuntimeError(
                    "Garmin browser bootstrap hit MFA and cannot continue non-interactively"
                )
            raise RuntimeError(f"Garmin browser bootstrap failed: {payload}")

        ticket = ticket_holder.get("ticket")
        if not ticket:
            raise RuntimeError("Garmin browser bootstrap did not capture a service ticket")

        seeded_client = Garmin(email, password)
        seeded_client.client._establish_session(ticket, service_url=MOBILE_SERVICE_URL)
        persist_tokens(seeded_client, tokenstore_path)

        verified_client = Garmin(email, password)
        verified_client.login(tokenstore=str(tokenstore_path))
        persist_tokens(verified_client, tokenstore_path)
        return verified_client

    if tokenstore_file_exists(tokenstore_path):
        try:
            return login_with_tokenstore()
        except Exception as error:
            print(f"⚠ Cached Garmin token reuse failed; falling back to credential login: {error}")

    try:
        return retry_on_rate_limit(
            login_with_credentials,
            "Garmin authentication",
            max_attempts=DEFAULT_MAX_ATTEMPTS,
            initial_delay_seconds=DEFAULT_INITIAL_DELAY_SECONDS,
            max_delay_seconds=DEFAULT_MAX_DELAY_SECONDS,
        )
    except Exception as error:
        if should_try_browser_bootstrap(error):
            return login_with_camoufox()
        raise
