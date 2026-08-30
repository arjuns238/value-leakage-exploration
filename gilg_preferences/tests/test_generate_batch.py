import pytest

pytestmark = pytest.mark.models

from dotenv import load_dotenv

load_dotenv()

from src.models import ConfigurableMockModel, GenerateRequest, BatchResult
from src.types import Message


class TestGenerateBatchMock:
    """Unit tests for generate_batch using mock model."""

    def test_returns_list_of_batch_results(self):
        """generate_batch should return a list of BatchResult matching input length."""
        model = ConfigurableMockModel(response="test response")

        requests = [
            GenerateRequest(messages=[{"role": "user", "content": "msg1"}]),
            GenerateRequest(messages=[{"role": "user", "content": "msg2"}]),
            GenerateRequest(messages=[{"role": "user", "content": "msg3"}]),
        ]

        results = model.generate_batch(requests)

        assert len(results) == 3
        assert all(isinstance(r, BatchResult) for r in results)
        assert all(r.ok for r in results)
        assert all(r.unwrap() == "test response" for r in results)

    def test_empty_requests_returns_empty_list(self):
        """generate_batch with empty list should return empty list."""
        model = ConfigurableMockModel(response="test")

        results = model.generate_batch([])

        assert results == []

    def test_single_request(self):
        """generate_batch with single request should work."""
        model = ConfigurableMockModel(response="single")

        requests = [GenerateRequest(messages=[{"role": "user", "content": "hi"}])]
        results = model.generate_batch(requests)

        assert len(results) == 1
        assert results[0].ok
        assert results[0].unwrap() == "single"

    def test_max_concurrent_parameter_accepted(self):
        """generate_batch should accept max_concurrent parameter."""
        model = ConfigurableMockModel(response="ok")

        requests = [
            GenerateRequest(messages=[{"role": "user", "content": "msg"}])
            for _ in range(5)
        ]

        # Should not raise
        results = model.generate_batch(requests, max_concurrent=2)

        assert len(results) == 5
        assert all(r.ok for r in results)


class TestBatchResult:
    """Tests for BatchResult dataclass and its methods."""

    def test_ok_returns_true_when_no_error(self):
        """BatchResult.ok should return True when error is None."""
        result = BatchResult(response="success", error=None)

        assert result.ok is True

    def test_ok_returns_false_when_error_present(self):
        """BatchResult.ok should return False when error is set."""
        result = BatchResult(response=None, error=ValueError("test error"))

        assert result.ok is False

    def test_unwrap_returns_response_when_no_error(self):
        """BatchResult.unwrap() should return response when successful."""
        result = BatchResult(response="success", error=None)

        assert result.unwrap() == "success"

    def test_unwrap_raises_when_error_present(self):
        """BatchResult.unwrap() should raise the stored error."""
        error = ValueError("test error")
        result = BatchResult(response=None, error=error)

        with pytest.raises(ValueError, match="test error"):
            result.unwrap()

    def test_unwrap_raises_original_error_type(self):
        """BatchResult.unwrap() should raise the original error type."""
        error = RuntimeError("runtime issue")
        result = BatchResult(response=None, error=error)

        with pytest.raises(RuntimeError):
            result.unwrap()

    def test_can_iterate_and_filter_failures(self):
        """Should be able to easily filter successful vs failed results."""
        results = [
            BatchResult(response="a", error=None),
            BatchResult(response=None, error=ValueError("fail")),
            BatchResult(response="b", error=None),
        ]

        successes = [r for r in results if r.ok]
        failures = [r for r in results if not r.ok]

        assert len(successes) == 2
        assert len(failures) == 1
        assert [r.unwrap() for r in successes] == ["a", "b"]

    def test_error_details_returns_none_on_success(self):
        """error_details() should return None when no error."""
        result = BatchResult(response="success", error=None)

        assert result.error_details() is None

    def test_error_details_includes_error_type_and_message(self):
        """error_details() should include error type and message."""
        result = BatchResult(response=None, error=ValueError("something went wrong"))

        details = result.error_details()

        assert "ValueError" in details
        assert "something went wrong" in details

    def test_error_details_includes_api_attributes(self):
        """error_details() should extract API-specific error attributes."""
        # Create a mock API error with extra attributes
        error = Exception("API error")
        error.status_code = 429
        error.body = {"error": "rate limited"}

        result = BatchResult(response=None, error=error)
        details = result.error_details()

        assert "429" in details
        assert "rate limited" in details


class TestGenerateRequest:
    """Tests for GenerateRequest dataclass."""

    def test_create_with_all_params(self):
        """Should create request with all parameters."""
        tools = [{"type": "function", "function": {"name": "test"}}]
        req = GenerateRequest(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            tools=tools,
        )

        assert req.temperature == 0.5
        assert req.tools == tools


# =============================================================================
# Integration Tests (real API calls)
# =============================================================================

pytestmark_api = pytest.mark.api


@pytest.mark.api
class TestGenerateBatchIntegration:
    """Integration tests for generate_batch with real API calls."""

    @pytest.fixture(scope="class")
    def client(self):
        from src.models import get_client
        return get_client(
            model_name="llama-3.1-8b",
            max_new_tokens=16,
        )

    def test_batch_returns_valid_responses(self, client):
        """Batch should return BatchResult objects with valid responses."""
        requests = [
            GenerateRequest(
                messages=[{"role": "user", "content": "Say 'one'"}],
                temperature=0.0,
            ),
            GenerateRequest(
                messages=[{"role": "user", "content": "Say 'two'"}],
                temperature=0.0,
            ),
            GenerateRequest(
                messages=[{"role": "user", "content": "Say 'three'"}],
                temperature=0.0,
            ),
        ]

        results = client.generate_batch(requests, max_concurrent=3)

        assert len(results) == 3
        assert all(isinstance(r, BatchResult) for r in results)
        for r in results:
            assert r.ok, f"Request failed: {r.error_details()}"
            assert len(r.unwrap()) > 0

    def test_batch_completes_faster_than_serial_estimate(self, client):
        """Batch of N requests should complete in less than N * single_request_time."""
        import time

        # First, time a single request to get baseline
        single_req = GenerateRequest(
            messages=[{"role": "user", "content": "Say 'test'"}],
            temperature=0.0,
        )
        start = time.time()
        client.generate(single_req.messages, single_req.temperature)
        single_time = time.time() - start

        # Now run a batch of 10 requests
        requests = [
            GenerateRequest(
                messages=[{"role": "user", "content": f"Say '{i}'"}],
                temperature=0.0,
            )
            for i in range(10)
        ]

        start = time.time()
        batch_results = client.generate_batch(requests, max_concurrent=10)
        batch_time = time.time() - start

        # Batch should be faster than 10 * single_time
        serial_estimate = single_time * 10
        speedup = serial_estimate / batch_time

        successful = sum(1 for r in batch_results if r.ok)
        print(f"\nSingle request time: {single_time:.2f}s")
        print(f"Batch time (10 requests): {batch_time:.2f}s")
        print(f"Serial estimate: {serial_estimate:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Successful requests: {successful}/10")

        assert len(batch_results) == 10
        # At least half should succeed
        assert successful >= 5, f"Too many failures: only {successful}/10 succeeded"
        # Batch should be faster than serial (allowing for variance)
        assert speedup > 1.0, f"Expected speedup > 1.0, got {speedup:.2f}x"

    def test_batch_preserves_order(self, client):
        """Results should be in the same order as requests."""
        requests = [
            GenerateRequest(
                messages=[{"role": "user", "content": "What is 1+1? Reply with just the number."}],
                temperature=0.0,
            ),
            GenerateRequest(
                messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                temperature=0.0,
            ),
            GenerateRequest(
                messages=[{"role": "user", "content": "What is 3+3? Reply with just the number."}],
                temperature=0.0,
            ),
        ]

        results = client.generate_batch(requests, max_concurrent=3)

        for r in results:
            assert r.ok, f"Request failed: {r.error_details()}"

        # Results should correspond to 2, 4, 6
        assert "2" in results[0].unwrap()
        assert "4" in results[1].unwrap()
        assert "6" in results[2].unwrap()

    def test_batch_with_tools(self, client):
        """Batch should work with tool use requests."""
        import json

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_choice",
                    "description": "Submit your choice",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "type": "string",
                                "enum": ["A", "B"],
                            }
                        },
                        "required": ["choice"],
                    },
                },
            }
        ]

        requests = [
            GenerateRequest(
                messages=[{"role": "user", "content": "Pick A."}],
                temperature=0.0,
                tools=tools,
            ),
            GenerateRequest(
                messages=[{"role": "user", "content": "Pick B."}],
                temperature=0.0,
                tools=tools,
            ),
        ]

        results = client.generate_batch(requests, max_concurrent=2)

        assert len(results) == 2
        for r in results:
            assert r.ok, f"Tool call failed: {r.error_details()}"
            parsed = json.loads(r.unwrap())
            assert "choice" in parsed
            assert parsed["choice"].upper() in ("A", "B")

