"""Tests for ranking response format parsing and refusal handling."""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.task_data import Task, OriginDataset
from src.types import RankingRefusal, PreferenceType
from src.measurement.elicitation.response_format import (
    RegexRankingFormat,
    XMLRankingFormat,
    ToolUseRankingFormat,
)
from src.measurement.elicitation.measurer import RankingMeasurer
from src.measurement.elicitation.refusal_judge import PreferenceRefusalResult


pytestmark = [pytest.mark.measurement, pytest.mark.ranking]


class TestRankingRefusalHandling:
    """Test that ranking formats handle refusals like other formats."""

    @pytest.mark.asyncio
    async def test_ranking_refusal_returns_refusal_string(self):
        """Ranking format should return 'refusal' for refusal responses, not raise."""
        task_labels = ("A", "B", "C", "D", "E")
        fmt = RegexRankingFormat(task_labels)

        with patch("src.measurement.elicitation.response_format.refusal_judge") as mock_judge:
            mock_refusal_result = PreferenceRefusalResult(is_refusal=True, refusal_type="no_preferences")
            mock_judge.judge_preference_refusal_async = AsyncMock(return_value=mock_refusal_result)

            result = await fmt.parse("I cannot rank these tasks as I have no preferences.")
            assert result == "refusal"

    @pytest.mark.asyncio
    async def test_ranking_measurer_creates_refusal_result(self):
        """RankingMeasurer should create RankingRefusal for refusal responses."""
        tasks = [
            Task(prompt=f"Task {i}", origin=OriginDataset.WILDCHAT, id=f"task_{i}", metadata={})
            for i in range(5)
        ]

        mock_format = MagicMock()
        mock_format.parse = AsyncMock(return_value="refusal")

        mock_prompt = MagicMock()
        mock_prompt.tasks = tasks
        mock_prompt.kind = PreferenceType.PRE_TASK_RANKING
        mock_prompt.response_format = mock_format

        measurer = RankingMeasurer()
        response = await measurer.parse("I have no preferences", mock_prompt)

        assert isinstance(response.result, RankingRefusal)
        assert response.result.tasks == tasks
        assert response.result.preference_type == PreferenceType.PRE_TASK_RANKING


class TestResponseFormatParsing:
    """Test different response formats parse correctly in realistic scenarios."""

    @pytest.mark.asyncio
    async def test_regex_format_realistic_responses(self):
        """Test regex format handles realistic model outputs (no API fallback)."""
        task_labels = ("A", "B", "C", "D", "E")
        fmt = RegexRankingFormat(task_labels)

        test_cases = [
            ("A > C > B > E > D", [0, 2, 1, 4, 3]),
            ("A, C, B, E, D", [0, 2, 1, 4, 3]),
            ("A C B E D", [0, 2, 1, 4, 3]),
            ("a > c > b > e > d", [0, 2, 1, 4, 3]),
            ("A > B > C > D > E", [0, 1, 2, 3, 4]),
        ]

        for response, expected in test_cases:
            result = await fmt.parse(response)
            assert result == expected, f"Failed for: {response}"

    @pytest.mark.asyncio
    async def test_xml_format_realistic_responses(self):
        """Test XML format handles realistic model outputs."""
        task_labels = ("A", "B", "C", "D", "E")
        fmt = XMLRankingFormat(task_labels)

        test_cases = [
            ("<ranking>A > C > B > E > D</ranking>", [0, 2, 1, 4, 3]),
            ("Here's my ranking:\n<ranking>A, C, B, E, D</ranking>", [0, 2, 1, 4, 3]),
            ("<ranking>\nA > B > C > D > E\n</ranking>", [0, 1, 2, 3, 4]),
        ]

        for response, expected in test_cases:
            result = await fmt.parse(response)
            assert result == expected, f"Failed for: {response}"

    @pytest.mark.asyncio
    async def test_tool_use_format(self):
        """Test tool_use format parses JSON correctly."""
        task_labels = ("A", "B", "C", "D", "E")
        fmt = ToolUseRankingFormat(task_labels)

        test_cases = [
            ('{"ranking": ["A", "C", "B", "E", "D"]}', [0, 2, 1, 4, 3]),
            ('{"ranking": ["a", "b", "c", "d", "e"]}', [0, 1, 2, 3, 4]),
        ]

        for response, expected in test_cases:
            result = await fmt.parse(response)
            assert result == expected

        tools = fmt.tools
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "submit_ranking"
