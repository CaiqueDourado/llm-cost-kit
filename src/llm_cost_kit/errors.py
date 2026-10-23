"""Is an LLM error worth retrying, or will it fail the same way every time?

HTTP 429 means two different things:

- ``rate_limit_exceeded``: retry with backoff, it will pass;
- ``insufficient_quota`` / ``billing_not_active``: the account has no credit. Retrying three
  times with exponential backoff only fails slower.

The account check runs BEFORE the transient one, because quota messages contain "429" and
plan messages contain "403". Testing "429" first is the classic bug: it retries an empty
account. Classification is by the error text, so it works with any SDK.
"""

from __future__ import annotations

ACCOUNT = (
    "insufficient_quota", "insufficient quota", "insufficient balance", "insufficient_balance",
    "billing_not_active", "billing", "account is not active", "exceeded your current quota",
    "quota exceeded", "payment required", "credit balance is too low", "credit",
    "not available in your subscription", "upgrade your plan",
)
SERVICE = (
    "connection error", "connection refused", "max retries exceeded", "name or service not known",
    "nodename nor servname", "failed to establish a new connection", "no route to host",
)
TRANSIENT = (
    "429", "500", "502", "503", "504", "529", "timeout", "timed out", "overloaded", "connection reset",
    "rate limit", "rate_limit", "resource_exhausted", "unavailable", "temporarily",
)


def _t(err) -> str:
    return str(err or "").lower()


def is_account(err) -> bool:
    """No credit, billing off, quota used up, or the plan does not include the model."""
    return any(x in _t(err) for x in ACCOUNT)


def is_service_down(err) -> bool:
    """Nobody on the other side: wrong host, service stopped (a local model not running)."""
    return any(x in _t(err) for x in SERVICE)


def is_transient(err) -> bool:
    return any(x in _t(err) for x in TRANSIENT)


def should_retry(err) -> bool:
    """True when another try can give another result. Account first, then service, then transient."""
    if is_account(err) or is_service_down(err):
        return False
    return is_transient(err)


def kind(err) -> str:
    """``account``, ``service``, ``transient`` or ``other``: one word for the log."""
    if is_account(err):
        return "account"
    if is_service_down(err):
        return "service"
    if is_transient(err):
        return "transient"
    return "other"


def is_environment(err) -> bool:
    """Not a code bug: account, service or limit. No code change fixes it."""
    return kind(err) != "other"
