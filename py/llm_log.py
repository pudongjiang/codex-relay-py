"""LLM request/response logging with a configurable enable/disable switch.

Controlled by the env var ``CODEX_RELAY_LOG_LLM`` (default: ``"1"`` = enabled).
Set to ``"0"`` to disable.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("CODEX_RELAY_LOG_LLM", "0") != "0"


def is_enabled() -> bool:
    return _ENABLED


def log_request(model: str, url: str, body: dict) -> None:
    if not _ENABLED:
        return
    try:
        payload = json.dumps(body, ensure_ascii=False, indent=2)
    except Exception:
        payload = str(body)
    logger.info(
        "LLM request -> %s [model=%s]\n%s",
        url,
        model,
        payload,
    )


def log_response(model: str, status: int, body: str) -> None:
    if not _ENABLED:
        return
    try:
        parsed = json.loads(body)
        formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        formatted = body
    logger.info(
        "LLM response <- [model=%s, status=%s]\n%s",
        model,
        status,
        formatted,
    )


def log_stream_request(model: str, url: str, body: dict) -> None:
    if not _ENABLED:
        return
    try:
        payload = json.dumps(body, ensure_ascii=False, indent=2)
    except Exception:
        payload = str(body)
    logger.info(
        "LLM stream request -> %s [model=%s]\n%s",
        url,
        model,
        payload,
    )


def log_stream_response(model: str, status: int, text: str) -> None:
    if not _ENABLED:
        return
    logger.info(
        "LLM stream result <- [model=%s, status=%s, len=%s]\n%s",
        model,
        status,
        len(text),
        text,
    )
