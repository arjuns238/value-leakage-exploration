"""Real API tests. Skip with: pytest -m 'not api'"""

import pytest

pytestmark = pytest.mark.models

from dotenv import load_dotenv

load_dotenv()

from src.models import get_client
from src.types import Message


pytestmark = pytest.mark.api


@pytest.fixture(scope="module")
def llama_client():
    return get_client(
        model_name="llama-3.1-8b",
        max_new_tokens=64,
    )


class TestInferenceClientGenerate:
    """Tests for inference client generate()."""

    def test_generate_returns_string(self, llama_client):
        """generate() should return a non-empty string."""
        messages: list[Message] = [
            {"role": "user", "content": "Say 'hello' and nothing else."}
        ]

        response = llama_client.generate(messages, temperature=0.0)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_with_system_message(self, llama_client):
        """generate() should handle system messages."""
        messages: list[Message] = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        response = llama_client.generate(messages, temperature=0.0)

        assert isinstance(response, str)
        assert "4" in response

    def test_generate_with_conversation(self, llama_client):
        """generate() should handle multi-turn conversations."""
        messages: list[Message] = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]

        response = llama_client.generate(messages, temperature=0.0)

        assert "Alice" in response

