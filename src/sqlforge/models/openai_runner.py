"""Frontier OpenAI API inference runner with rate limiting, retries, and cost guards.

Features:
1. Pinned reference model: gpt-4o-mini-2024-07-18.
2. Hard dollar spending budget ceiling ($50.00 USD maximum).
3. Explicit opt-in safety guard (allow_live_api=True) preventing unintended charges.
4. Bounded exponential backoff distinguishing transient from permanent API errors.
5. Zero external dependencies: implements standard HTTP communication via urllib.
6. Secret redaction: never serializes or logs raw API keys.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, cast

from sqlforge.schemas.evaluation import GenerationResult

logger = logging.getLogger(__name__)

# Official OpenAI gpt-4o-mini pricing per 1M tokens (as of 2024-2026)
DEFAULT_INPUT_COST_PER_1M = 0.150  # $0.150 per 1M prompt tokens
DEFAULT_OUTPUT_COST_PER_1M = 0.600  # $0.600 per 1M completion tokens


class OpenAIRunnerError(RuntimeError):
    """Base exception for OpenAI model runner failures."""


class APIOptInRequiredError(OpenAIRunnerError):
    """Raised when a live API call is attempted without explicit authorization."""


class APIBudgetExceededError(OpenAIRunnerError):
    """Raised when cumulative API expenditure exceeds hard dollar limit."""


class OpenAIPermanentError(OpenAIRunnerError):
    """Raised when OpenAI API returns non-retryable 4xx errors (e.g. 400, 401, 403, 404)."""


class OpenAITransientError(OpenAIRunnerError):
    """Raised when retries are exhausted for transient failures (e.g. 429 rate limit, 5xx server errors)."""


class RateLimiter:
    """Thread-safe request rate throttle."""

    def __init__(self, max_qps: float = 5.0) -> None:
        self.min_interval = 1.0 / max_qps if max_qps > 0 else 0.0
        self.last_call_time = 0.0

    def throttle(self) -> None:
        """Enforce minimum spacing between outbound network requests."""
        if self.min_interval <= 0.0:
            return
        elapsed = time.monotonic() - self.last_call_time
        remaining = self.min_interval - elapsed
        if remaining > 0.0:
            time.sleep(remaining)
        self.last_call_time = time.monotonic()


class OpenAIRunner:
    """ModelRunner for OpenAI frontier models with budget, safety, and retry guards."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini-2024-07-18",
        api_key: str | None = None,
        allow_live_api: bool = False,
        spending_limit_usd: float = 50.0,
        max_cumulative_budget_usd: float | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        max_qps: float = 5.0,
        max_retries: int = 5,
        base_delay_sec: float = 1.0,
        max_delay_sec: float = 16.0,
        api_base_url: str = "https://api.openai.com/v1",
        input_cost_per_1m: float = DEFAULT_INPUT_COST_PER_1M,
        output_cost_per_1m: float = DEFAULT_OUTPUT_COST_PER_1M,
        http_transport: Callable[[urllib.request.Request, float], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize OpenAI API runner.

        Args:
            model_id: Pinned OpenAI model identifier.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable).
            allow_live_api: Explicit opt-in flag required to make live remote requests.
            spending_limit_usd: Hard maximum dollar budget ceiling (cannot exceed $50.00).
            max_qps: Maximum queries per second (default 5.0).
            max_retries: Maximum exponential backoff retries on transient errors.
            base_delay_sec: Initial retry backoff delay in seconds.
            max_delay_sec: Maximum retry backoff delay ceiling in seconds.
            api_base_url: OpenAI API endpoint URL.
            input_cost_per_1m: Prompt token cost per 1M tokens in USD.
            output_cost_per_1m: Completion token cost per 1M tokens in USD.
            http_transport: Optional custom transport callable for dependency-injection/mock testing.
        """
        self._model_id = model_id
        # Securely resolve API key without exposing in state logs
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.allow_live_api = allow_live_api

        # Hard guard: spending limit cannot exceed $50.00 USD
        resolved_limit = (
            max_cumulative_budget_usd
            if max_cumulative_budget_usd is not None
            else spending_limit_usd
        )
        self.spending_limit_usd = min(max(0.0, float(resolved_limit)), 50.0)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max(0, int(max_retries))
        self.base_delay_sec = max(0.1, float(base_delay_sec))
        self.max_delay_sec = max(self.base_delay_sec, float(max_delay_sec))
        self.api_base_url = api_base_url.rstrip("/")
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self._http_transport = http_transport

        self._rate_limiter = RateLimiter(max_qps=max_qps)
        self._cumulative_cost_usd: float = 0.0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_requests: int = 0

    @property
    def model_id(self) -> str:
        """Pinned OpenAI model identifier."""
        return self._model_id

    @property
    def cumulative_cost_usd(self) -> float:
        """Cumulative dollar expenditure incurred during this runner's lifecycle."""
        return round(self._cumulative_cost_usd, 6)

    @property
    def total_tokens(self) -> int:
        """Total tokens processed across all requests."""
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def masked_key(self) -> str:
        """Masked representation of API key."""
        return self.masked_api_key()

    def masked_api_key(self) -> str:
        """Return masked key representation for safe logging."""
        if not self._api_key:
            return "(not configured)"
        cleaned = self._api_key.strip()
        if len(cleaned) <= 8:
            return "sk-...[REDACTED]"
        return f"{cleaned[:4]}...[REDACTED]...{cleaned[-4:]}"

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate dollar cost from prompt and completion token counts."""
        cost = (prompt_tokens * (self.input_cost_per_1m / 1_000_000.0)) + (
            completion_tokens * (self.output_cost_per_1m / 1_000_000.0)
        )
        return round(cost, 8)

    def generate(
        self,
        prompt: str,
        example_id: str = "example",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        stop_tokens: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate SQL completion for a prompt via OpenAI Chat Completions API.

        Args:
            prompt: Text prompt string.
            example_id: Evaluation example identifier.
            temperature: Sampling temperature (strictly 0.0 for greedy decoding).
            max_new_tokens: Maximum completion token budget (default 512).
            stop_tokens: Optional token sequences to halt generation.
            **kwargs: Extra parameters.

        Returns:
            GenerationResult with predicted SQL, token counts, latency, and cost.

        Raises:
            APIOptInRequiredError: If live network calls are attempted without allow_live_api=True.
            APIBudgetExceededError: If spending limit ($50.00) is reached.
            OpenAIPermanentError: On non-retryable 4xx client errors (e.g. 401 Unauthorized).
            OpenAITransientError: When retry budget is exhausted for transient errors.
        """
        # 1. Budget ceiling assertion
        if self._cumulative_cost_usd >= self.spending_limit_usd:
            raise APIBudgetExceededError(
                f"Cumulative API spending ceiling (${self.spending_limit_usd:.2f} USD) reached. "
                f"Incurred: ${self._cumulative_cost_usd:.4f} USD across {self._total_requests} requests. "
                f"Aborting execution to prevent unauthorized charges."
            )

        # 2. Live API opt-in assertion
        if not self.allow_live_api and self._http_transport is None:
            raise APIOptInRequiredError(
                "Live commercial API execution is disabled by default. "
                "To authorize live requests to OpenAI, set allow_live_api=True or pass the --live-api CLI flag."
            )

        # 3. API key validation
        if not self._api_key and self._http_transport is None:
            raise OpenAIPermanentError(
                "OPENAI_API_KEY is not set. Export OPENAI_API_KEY in your environment or pass api_key."
            )

        # 4. Construct payload
        effective_stop = stop_tokens if stop_tokens is not None else [";", "```", "<|im_end|>"]
        payload = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
            "stop": effective_stop,
        }

        # 5. Execute with rate limiting and bounded retries
        start_time = time.monotonic()
        response_data = self._execute_with_retries(payload)
        latency_ms = (time.monotonic() - start_time) * 1000.0

        # 6. Parse response tokens and content
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "") or ""
        finish_reason = choice.get("finish_reason", "stop")

        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(prompt.split()))
        completion_tokens = usage.get("completion_tokens", len(raw_content.split()))

        # 7. Compute cost and update cumulative budget tracker
        call_cost = self.calculate_cost(prompt_tokens, completion_tokens)
        self._cumulative_cost_usd += call_cost
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_requests += 1

        # Clean SQL completion
        clean_sql = self._clean_generated_sql(raw_content)

        return GenerationResult(
            example_id=example_id,
            generated_sql=clean_sql,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2),
            finish_reason=str(finish_reason),
            cost_usd=round(call_cost, 6),
            model_id=self._model_id,
        )

    def _execute_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute HTTP request with exponential backoff on transient errors."""
        url = f"{self.api_base_url}/chat/completions"
        encoded_data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=encoded_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key or 'mock'}",
            },
            method="POST",
        )

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._rate_limiter.throttle()

            try:
                if self._http_transport is not None:
                    # Test injection / custom transport
                    return self._http_transport(req, 30.0)

                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    resp_bytes = resp.read()
                    return cast(dict[str, Any], json.loads(resp_bytes.decode("utf-8")))

            except urllib.error.HTTPError as exc:
                last_error = exc
                status_code = exc.code
                err_body = exc.read().decode("utf-8", errors="replace")

                # Permanent client errors: abort immediately
                if status_code in {400, 401, 403, 404}:
                    raise OpenAIPermanentError(
                        f"OpenAI API permanent error (HTTP {status_code}): {err_body}"
                    ) from exc

                # Transient errors (429 Rate Limit, 5xx Server Error)
                if (status_code == 429 or status_code >= 500) and attempt < self.max_retries:
                    delay = min(self.max_delay_sec, self.base_delay_sec * (2**attempt))
                    logger.warning(
                        f"OpenAI transient HTTP {status_code}. Backing off for {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})."
                    )
                    time.sleep(delay)
                    continue

                raise OpenAITransientError(
                    f"OpenAI transient error HTTP {status_code} after {self.max_retries} retries: {err_body}"
                ) from exc

            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    delay = min(self.max_delay_sec, self.base_delay_sec * (2**attempt))
                    logger.warning(
                        f"OpenAI network/timeout error: {exc}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})."
                    )
                    time.sleep(delay)
                    continue

                raise OpenAITransientError(
                    f"OpenAI network timeout after {self.max_retries} retries: {exc}"
                ) from exc

        raise OpenAITransientError(f"OpenAI retries exhausted: {last_error}")

    @staticmethod
    def _clean_generated_sql(text: str) -> str:
        """Strip markdown fences and leading/trailing noise from model output."""
        cleaned = text.strip()
        # Handle markdown SQL blocks: ```sql ... ```
        if "```" in cleaned:
            parts = cleaned.split("```")
            for p in parts:
                p_strip = p.strip()
                if p_strip.startswith("sql"):
                    cleaned = p_strip[3:].strip()
                    break
                elif p_strip.upper().startswith("SELECT") or p_strip.upper().startswith("WITH"):
                    cleaned = p_strip
                    break

        # Ensure single statement ending with semicolon
        if not cleaned.endswith(";"):
            cleaned = f"{cleaned};"
        return cleaned
