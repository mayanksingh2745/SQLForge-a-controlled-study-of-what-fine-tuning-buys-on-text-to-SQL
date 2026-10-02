"""Unit tests for model runners: OpenAIRunner and LocalHFModelRunner."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from sqlforge.models.local_hf import LocalHFModelRunner, LocalModelDependencyError
from sqlforge.models.openai_runner import (
    APIBudgetExceededError,
    APIOptInRequiredError,
    OpenAIPermanentError,
    OpenAIRunner,
)


class TestOpenAIRunner:
    """Tests for OpenAI API model runner safety, retries, budgeting, and secret redaction."""

    def test_opt_in_guard_raises_when_disabled(self) -> None:
        runner = OpenAIRunner(api_key="sk-test-secret-key-12345", allow_live_api=False)
        with pytest.raises(APIOptInRequiredError, match="allow_live_api=True"):
            runner.generate("SELECT * FROM users;", example_id="ex1")

    def test_missing_api_key_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        runner = OpenAIRunner(api_key=None, allow_live_api=True)
        with pytest.raises(OpenAIPermanentError, match="OPENAI_API_KEY is not set"):
            runner.generate("SELECT 1;", example_id="ex1")

    def test_secret_redaction(self) -> None:
        runner = OpenAIRunner(api_key="sk-proj-abc123xyz789secretkey", allow_live_api=False)
        masked = runner.masked_key
        assert "abc123xyz789secretkey" not in masked
        assert masked.startswith("sk-p")
        assert masked.endswith("tkey")
        assert "[REDACTED]" in masked

    def test_spending_limit_ceiling_clamped_at_fifty(self) -> None:
        # User requested $100 budget -> capped to $50.00
        runner = OpenAIRunner(spending_limit_usd=100.0)
        assert runner.spending_limit_usd == 50.0

    def test_budget_exceeded_guard(self) -> None:
        # Create runner with microscopic budget ($0.000001)
        runner = OpenAIRunner(
            api_key="sk-test-key",
            allow_live_api=True,
            spending_limit_usd=0.000001,
            http_transport=lambda req, timeout: {
                "choices": [{"message": {"content": "SELECT 1;"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )
        # First call succeeds and consumes ~$0.000045 > $0.000001
        res1 = runner.generate("SELECT 1;", example_id="ex1")
        assert res1.generated_sql == "SELECT 1;"

        # Second call must raise APIBudgetExceededError
        with pytest.raises(APIBudgetExceededError, match="spending ceiling"):
            runner.generate("SELECT 2;", example_id="ex2")

    def test_successful_mock_transport_call(self) -> None:
        def mock_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            assert req.get_header("Authorization") == "Bearer sk-test-key"
            assert isinstance(req.data, bytes)
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["model"] == "gpt-4o-mini-2024-07-18"
            assert payload["temperature"] == 0.0
            return {
                "choices": [
                    {"message": {"content": "```sql\nSELECT name FROM users WHERE id = 1;\n```"}}
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10},
            }

        runner = OpenAIRunner(
            api_key="sk-test-key",
            allow_live_api=True,
            http_transport=mock_transport,
        )

        res = runner.generate("Question: find name", example_id="ex_mock")
        assert res.generated_sql == "SELECT name FROM users WHERE id = 1;"
        assert res.prompt_tokens == 40
        assert res.completion_tokens == 10
        assert res.cost_usd is not None and res.cost_usd > 0.0
        assert res.example_id == "ex_mock"

    def test_retry_on_transient_error(self) -> None:
        attempts = 0

        def flaky_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                # First attempt: rate limited 429
                raise urllib.error.HTTPError(
                    url="https://api.openai.com/v1/chat/completions",
                    code=429,
                    msg="Too Many Requests",
                    hdrs=None,  # type: ignore
                    fp=None,
                )
            return {
                "choices": [{"message": {"content": "SELECT 42;"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

        runner = OpenAIRunner(
            api_key="sk-test-key",
            allow_live_api=True,
            base_delay_sec=0.01,
            max_delay_sec=0.05,
            http_transport=flaky_transport,
        )

        res = runner.generate("Test query", example_id="ex_flaky")
        assert attempts == 2
        assert res.generated_sql == "SELECT 42;"

    def test_permanent_error_fails_immediately_without_retrying(self) -> None:
        attempts = 0

        def unauthorized_transport(req: urllib.request.Request, timeout: float) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            raise urllib.error.HTTPError(
                url="https://api.openai.com/v1/chat/completions",
                code=401,
                msg="Unauthorized",
                hdrs=None,  # type: ignore
                fp=None,
            )

        runner = OpenAIRunner(
            api_key="sk-invalid-key",
            allow_live_api=True,
            max_retries=5,
            http_transport=unauthorized_transport,
        )

        with pytest.raises(OpenAIPermanentError, match="HTTP 401"):
            runner.generate("Test query", example_id="ex_unauth")

        assert attempts == 1  # No retries on 401!

    def test_cost_calculation(self) -> None:
        runner = OpenAIRunner(api_key="sk-test", allow_live_api=False)
        # 1M prompt tokens ($0.150) + 1M completion tokens ($0.600) = $0.750
        cost = runner.calculate_cost(1_000_000, 1_000_000)
        assert pytest.approx(cost, 0.0001) == 0.750


class TestLocalHFModelRunner:
    """Tests for local causal language model runner and dependency boundaries."""

    def test_mock_inference_callable_offline_execution(self) -> None:
        def mock_infer(prompt: str, max_tokens: int) -> tuple[str, int, int]:
            assert "Question:" in prompt
            return "SELECT department, count(*) FROM staff GROUP BY department;", 35, 12

        runner = LocalHFModelRunner(
            model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            temperature=0.0,
            max_new_tokens=512,
            inference_fn=mock_infer,
        )

        assert runner.model_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        res = runner.generate("Question: count staff by department", example_id="ex_qwen")
        assert res.generated_sql == "SELECT department, count(*) FROM staff GROUP BY department;"
        assert res.prompt_tokens == 35
        assert res.completion_tokens == 12
        assert res.cost_usd is None  # Local open-weight models have $0 API cost

    def test_missing_torch_raises_dependency_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        runner = LocalHFModelRunner(model_id="Qwen/Qwen2.5-Coder-7B-Instruct")
        # Mask torch and transformers to simulate minimal environment
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "transformers", None)

        with pytest.raises(LocalModelDependencyError, match="torch.*transformers"):
            runner.generate("Question: select all", example_id="ex_err")
