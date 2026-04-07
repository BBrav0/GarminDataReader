#!/usr/bin/env python3
"""Helpers for Garmin authentication with graceful rate-limit retries."""

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from garminconnect import Garmin

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_INITIAL_DELAY_SECONDS = 30
DEFAULT_MAX_DELAY_SECONDS = 300
RATE_LIMIT_HINTS = (
    "429",
    "too many requests",
    "rate limit",
    "ratelimit",
)


def _walk_exception_chain(error):
    """Yield an exception and any nested exceptions attached to it."""
    seen = set()
    pending = [error]

    while pending:
        current = pending.pop()
        if current is None:
            continue

        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current

        nested = getattr(current, "error", None)
        if nested is not None:
            pending.append(nested)

        cause = getattr(current, "__cause__", None)
        if cause is not None:
            pending.append(cause)

        context = getattr(current, "__context__", None)
        if context is not None:
            pending.append(context)

        for arg in getattr(current, "args", ()):
            if isinstance(arg, BaseException):
                pending.append(arg)


def _extract_response(error):
    """Return the first HTTP response object found in an exception chain."""
    for current in _walk_exception_chain(error):
        response = getattr(current, "response", None)
        if response is not None:
            return response
    return None


def _extract_status_code(error):
    """Return the first HTTP status code found in an exception chain."""
    response = _extract_response(error)
    if response is None:
        return None
    return getattr(response, "status_code", None)


def _retry_after_seconds(error):
    """Parse a Retry-After header if present."""
    response = _extract_response(error)
    if response is None:
        return None

    header_value = response.headers.get("Retry-After")
    if not header_value:
        return None

    try:
        return max(0, int(header_value))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(header_value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(delay))
    except (TypeError, ValueError, OverflowError):
        return None


def is_rate_limit_error(error):
    """Return True when an exception indicates Garmin rate limiting."""
    if _extract_status_code(error) == 429:
        return True

    for current in _walk_exception_chain(error):
        message = str(current).lower()
        if any(hint in message for hint in RATE_LIMIT_HINTS):
            return True

    return False


def retry_on_rate_limit(
    action,
    description,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    initial_delay_seconds=DEFAULT_INITIAL_DELAY_SECONDS,
    max_delay_seconds=DEFAULT_MAX_DELAY_SECONDS,
):
    """Run an action and retry with exponential backoff when Garmin returns 429."""
    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except Exception as error:
            should_retry = is_rate_limit_error(error) and attempt < max_attempts
            if not should_retry:
                raise

            delay_seconds = min(
                max_delay_seconds,
                initial_delay_seconds * (2 ** (attempt - 1)),
            )
            retry_after = _retry_after_seconds(error)
            if retry_after is not None:
                delay_seconds = min(
                    max_delay_seconds,
                    max(delay_seconds, retry_after),
                )

            print(
                f"⚠ {description} hit Garmin rate limits "
                f"(attempt {attempt}/{max_attempts}): {error}"
            )
            print(f"  Waiting {delay_seconds} seconds before retrying...")
            time.sleep(delay_seconds)

    raise RuntimeError(f"{description} failed after {max_attempts} attempts")


def login_to_garmin(email, password):
    """Create a Garmin client and log in with 429-aware retries."""

    def do_login():
        client = Garmin(email, password)
        client.login()
        return client

    return retry_on_rate_limit(do_login, "Garmin authentication")
