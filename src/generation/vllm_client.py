"""LLM generation via a local vLLM OpenAI-compatible server.

Replaces TTT-Discover's tinker sampling path. One prompt -> ``n`` completions (the group of
candidate solutions to grade). We return only the final assistant ``content`` (reasoning tokens,
if the server exposes them separately as ``reasoning_content``, are dropped).
"""
from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        reasoning_effort: str | None = None,
        thinking_token_budget: int | None = None,
        enable_thinking: bool | None = None,
        request_timeout: float = 3600.0,
        max_retries: int = 4,
        max_concurrency: int = 8,
    ):
        """
        Args:
            base_url: vLLM OpenAI endpoint, e.g. "http://localhost:8000/v1".
            model: served model name, e.g. "openai/gpt-oss-120b".
            api_key: ignored by vLLM but required by the client (any non-empty string).
            reasoning_effort: for gpt-oss ("low"/"medium"/"high"); sent via extra_body. None to omit.
            thinking_token_budget: for Qwen3 — cap reasoning tokens; vLLM forces </think> once hit.
                Requires the server launched with ``--reasoning-parser qwen3``. None to omit.
            enable_thinking: for Qwen3 — set False to disable thinking entirely (via
                chat_template_kwargs). None to omit (leave the model default).
            request_timeout: per-request timeout (generation with reasoning can be slow).
            max_retries: client-side retries on transient errors.
            max_concurrency: cap on in-flight requests to the server (across all callers).
        """
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.thinking_token_budget = thinking_token_budget
        self.enable_thinking = enable_thinking
        self._sem = asyncio.Semaphore(max_concurrency)

    async def generate(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """Return ``n`` completion strings for ``prompt`` (empty strings for empty content)."""
        extra_body: dict = {}
        if self.reasoning_effort is not None:              # gpt-oss
            extra_body["reasoning_effort"] = self.reasoning_effort
        if self.thinking_token_budget is not None:         # Qwen3 (needs --reasoning-parser qwen3)
            extra_body["thinking_token_budget"] = self.thinking_token_budget
        if self.enable_thinking is not None:               # Qwen3
            extra_body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        async with self._sem:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        return [(choice.message.content or "") for choice in resp.choices]
