"""Thin wrapper around the Anthropic SDK with retry on transient errors."""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import tenacity
from dotenv import load_dotenv

# Walk up from this file to find .env — works however the CLI is invoked.
load_dotenv(Path(__file__).parents[2] / ".env", override=True)
load_dotenv(override=True)  # fallback: search from CWD


class AnthropicClient:
    """Wrapper that exposes a single ``complete`` call and handles retries."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the underlying ``anthropic.Anthropic`` SDK client.

        Reads the ``ANTHROPIC_API_KEY`` env var when ``api_key`` is not given.
        """
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        # The SDK accepts ``api_key=None`` and will read the env itself, but
        # being explicit makes test substitution easier.
        self._client = anthropic.Anthropic(api_key=key)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=8),
        retry=tenacity.retry_if_exception_type(
            (
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
            )
        ),
        reraise=True,
    )
    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> str:
        """Send one non-streaming message and return the assistant text.

        Retries up to three times with exponential backoff on transient
        connection / rate-limit / 5xx errors. Other errors propagate.
        """
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
