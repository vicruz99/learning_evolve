"""LLM generation via a local vLLM OpenAI-compatible server.

Replaces TTT-Discover's tinker sampling path. One prompt -> ``n`` completions (the group of
candidate solutions to grade), returned as a ``GenResult`` that carries, alongside the answer text:

  * ``reasonings``     — per-choice ``reasoning_content`` (the hidden chain of thought). Kept both
                         because it is the bulk of the decode cost and because SFT/RL variants need it.
  * ``finish_reasons`` — per-choice stop reason; ``"length"`` means the completion was truncated by
                         ``max_tokens`` (it burned the whole budget and likely emitted no code block).
  * ``usage``          — exact token accounting for the request.

Token-granularity caveat: the OpenAI schema reports ``usage`` **per request**, i.e. summed over the
``n`` choices, so exact per-candidate token counts are not available when ``n > 1``. Per candidate we
therefore record the exact ``finish_reason`` plus content/reasoning character counts, and rely on the
request-level totals for throughput and prefix-cache accounting.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _int_attr(obj, *path) -> int:
    """Walk a chain of optional attributes, returning 0 unless the whole chain resolves to a number.

    Every field we read is optional in the OpenAI schema and vLLM may omit any of them (e.g.
    ``cached_tokens`` only appears with prefix caching enabled), so nothing here may raise.
    """
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return 0
    return int(obj) if isinstance(obj, (int, float)) else 0


@dataclass
class GenResult:
    """One vLLM request's output: ``n`` completions plus the request's token accounting."""
    texts: list[str] = field(default_factory=list)            # assistant content (what we parse code from)
    reasonings: list[str] = field(default_factory=list)       # reasoning_content, "" if not exposed
    finish_reasons: list[str] = field(default_factory=list)   # "stop" | "length" | ...
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0    # prefix-cache hits within prompt_tokens (0 if server omits it)
    completion_tokens: int = 0       # summed over all n choices
    reasoning_tokens: int = 0        # subset of completion_tokens, 0 if server omits it
    latency: float = 0.0             # wall-clock seconds for the request

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def truncated(self) -> int:
        """Choices cut off by ``max_tokens`` — these gate the request's return and usually yield no code."""
        return sum(1 for r in self.finish_reasons if r == "length")


# Field name vLLM uses for the hidden chain of thought. It varies by version and reasoning parser
# (`openai_gptoss` on vLLM 0.11 emits `reasoning`; the qwen/deepseek parsers emit `reasoning_content`),
# and the OpenAI SDK passes unknown fields straight through, so we accept either.
REASONING_FIELDS = ("reasoning_content", "reasoning")


def _reasoning_of(message) -> str:
    for field in REASONING_FIELDS:
        value = getattr(message, field, None)
        if value:
            return value
    return ""


def _to_result(resp, latency: float) -> GenResult:
    """Convert a chat-completion response into a GenResult, tolerating omitted usage fields."""
    return GenResult(
        texts=[(c.message.content or "") for c in resp.choices],
        reasonings=[_reasoning_of(c.message) for c in resp.choices],
        finish_reasons=[(c.finish_reason or "") for c in resp.choices],
        prompt_tokens=_int_attr(resp, "usage", "prompt_tokens"),
        cached_prompt_tokens=_int_attr(resp, "usage", "prompt_tokens_details", "cached_tokens"),
        completion_tokens=_int_attr(resp, "usage", "completion_tokens"),
        reasoning_tokens=_int_attr(resp, "usage", "completion_tokens_details", "reasoning_tokens"),
        latency=latency,
    )


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
        self._base_url = base_url
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
        self._tokenize_unavailable = False   # so the /tokenize fallback warning fires at most once

    async def cache_counters(self) -> dict[str, int]:
        """Scrape the server's cumulative prefix-cache counters from ``/metrics``.

        Needed because this vLLM build reports ``usage.prompt_tokens_details.cached_tokens = 0`` even
        when the cache is in fact serving most of the prefill — ``/metrics`` is the reliable source.
        Take a snapshot before and after a generation and diff to get that generation's hit rate.

        CAVEATS: these counters are **server-global** (a delta mixes the traffic of every run sharing
        the server) and counted **per sequence**, so with ``n > 1`` queries ~= n * prompt_tokens. Use
        the ratio hits/queries; the absolute counts are not comparable to ``usage.prompt_tokens``.

        Returns {} on any failure (metrics are diagnostics, never load-bearing).
        """
        wanted = {"vllm:prefix_cache_queries_total": "cache_queries",
                  "vllm:prefix_cache_hits_total": "cache_hits"}
        try:
            url = self._base_url.rstrip("/").removesuffix("/v1") + "/metrics"
            async with httpx.AsyncClient(timeout=10.0) as http:
                body = (await http.get(url)).text
            out: dict[str, int] = {}
            for line in body.splitlines():
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if name in wanted:
                    out[wanted[name]] = int(float(line.rsplit(" ", 1)[1]))
            return out
        except Exception as e:                             # noqa: BLE001 - diagnostics only
            logger.debug(f"could not scrape {self._base_url} /metrics: {e}")
            return {}

    async def health(self) -> bool:
        """Is the server up and serving? Used to wait one out rather than lose a generation to it.

        ``/health`` is the cheap liveness endpoint and needs no model name, so a vLLM that is up but
        still loading weights answers it only once it is ready to serve — which is exactly the moment
        a waiting run should retry.
        """
        try:
            url = self._base_url.rstrip("/").removesuffix("/v1") + "/health"
            async with httpx.AsyncClient(timeout=10.0) as http:
                return (await http.get(url)).status_code == 200
        except Exception:                                  # noqa: BLE001 - a probe, never load-bearing
            return False

    async def count_tokens(self, texts: list[str]) -> list[int]:
        """Exact token counts for ``texts``, via the server's ``/tokenize`` (the served model's own
        tokenizer — no local transformers dependency).

        Needed because this vLLM build omits ``usage.completion_tokens_details`` entirely, so the
        server never reports how much of ``completion_tokens`` was reasoning. Counting the captured
        reasoning text here turns that into a real measurement instead of a chars/4 estimate.

        ``/tokenize`` takes one string per request, so these are fired concurrently (they are
        CPU-only and fast). Empty strings cost no request. Returns ``[]`` if the endpoint is
        unavailable, so callers fall back to the character-based estimate.
        """
        if not texts:
            return []
        url = self._base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
        sem = asyncio.Semaphore(16)          # separate from _sem: must not compete for generation slots

        async def one(http: httpx.AsyncClient, text: str) -> int:
            if not text:
                return 0
            async with sem:
                resp = await http.post(url, json={"model": self.model, "prompt": text,
                                                  "add_special_tokens": False})
            resp.raise_for_status()
            return int(resp.json()["count"])

        try:
            async with httpx.AsyncClient(timeout=120.0) as http:
                return list(await asyncio.gather(*[one(http, t) for t in texts]))
        except Exception as e:                             # noqa: BLE001 - measurement only
            if not self._tokenize_unavailable:
                self._tokenize_unavailable = True
                logger.warning(f"/tokenize unavailable ({e}); reasoning token counts will be "
                               f"estimated from character counts instead")
            return []

    async def generate(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> GenResult:
        """Return a ``GenResult`` with ``n`` completions for ``prompt`` (empty strings for empty content)."""
        extra_body: dict = {}
        if self.reasoning_effort is not None:              # gpt-oss
            extra_body["reasoning_effort"] = self.reasoning_effort
        if self.thinking_token_budget is not None:         # Qwen3 (needs --reasoning-parser qwen3)
            extra_body["thinking_token_budget"] = self.thinking_token_budget
        if self.enable_thinking is not None:               # Qwen3
            extra_body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        t0 = time.perf_counter()
        async with self._sem:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
        return _to_result(resp, time.perf_counter() - t0)
