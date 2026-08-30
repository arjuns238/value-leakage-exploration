"""Tests for semantic parser that require API calls."""

import pytest

from src.measurement.elicitation.semantic_parser import (
    parse_choice_async,
    parse_rating_async,
    parse_ranking_async,
    parse_qualitative_async,
    ParseError,
)


pytestmark = [pytest.mark.api, pytest.mark.measurement]


class TestSemanticParserChoices:
    """Tests for parsing binary choices."""

    @pytest.mark.asyncio
    async def test_parse_choice_clear_a(self):
        """Parse clear choice for option A."""
        result = await parse_choice_async(
            "I would definitely prefer to do Task A. It seems more interesting.",
            "Task A", "Task B"
        )
        assert result == "a"

    @pytest.mark.asyncio
    async def test_parse_choice_clear_b(self):
        """Parse clear choice for option B."""
        result = await parse_choice_async(
            "Task B is the one I'd choose without hesitation.",
            "Task A", "Task B"
        )
        assert result == "b"

    @pytest.mark.asyncio
    async def test_parse_choice_refusal(self):
        """Parse refusal to choose."""
        result = await parse_choice_async(
            "I don't have personal preferences and cannot choose between tasks.",
            "Task A", "Task B"
        )
        assert result == "refusal"

    @pytest.mark.asyncio
    async def test_parse_choice_with_negation(self):
        """Parse choice expressed through negation."""
        result = await parse_choice_async(
            "I wouldn't want to do Task A, so I'll go with the other one.",
            "Task A", "Task B"
        )
        assert result == "b"


class TestSemanticParserRatings:
    """Tests for parsing numerical ratings."""

    @pytest.mark.asyncio
    async def test_parse_rating_embedded_in_text(self):
        """Parse rating embedded in conversational text."""
        result = await parse_rating_async(
            "After thinking about it, I'd give this task a solid 7. It's pretty interesting.",
            scale_min=1, scale_max=10
        )
        assert result == 7.0

    @pytest.mark.asyncio
    async def test_parse_rating_worded(self):
        """Parse rating expressed in words."""
        result = await parse_rating_async(
            "I'd rate this eight out of ten.",
            scale_min=1, scale_max=10
        )
        assert result == 8.0


class TestSemanticParserRankings:
    """Tests for parsing rankings."""

    @pytest.mark.asyncio
    async def test_parse_ranking_natural_language(self):
        """Parse ranking expressed in natural language."""
        result = await parse_ranking_async(
            "I'd most prefer Task C, followed by A, then E, B, and finally D.",
            ("A", "B", "C", "D", "E")
        )
        # C=2, A=0, E=4, B=1, D=3
        assert result == [2, 0, 4, 1, 3]

    @pytest.mark.asyncio
    async def test_parse_ranking_numbered_list(self):
        """Parse ranking as numbered list."""
        result = await parse_ranking_async(
            "My ranking:\n1. Task B\n2. Task D\n3. Task A\n4. Task C\n5. Task E",
            ("A", "B", "C", "D", "E")
        )
        # B=1, D=3, A=0, C=2, E=4
        assert result == [1, 3, 0, 2, 4]

    @pytest.mark.asyncio
    async def test_parse_ranking_unclear_raises(self):
        """Unclear ranking should raise ParseError."""
        with pytest.raises(ParseError):
            await parse_ranking_async(
                "I'm not sure how to rank these. They all seem equally interesting.",
                ("A", "B", "C", "D", "E")
            )


class TestSemanticParserQualitative:
    """Tests for parsing qualitative values."""

    @pytest.mark.asyncio
    async def test_parse_qualitative_synonym(self):
        """Parse qualitative value expressed as synonym."""
        result = await parse_qualitative_async(
            "I strongly concur with this approach.",
            ("strongly agree", "agree", "neutral", "disagree", "strongly disagree")
        )
        assert result == "strongly agree"

    @pytest.mark.asyncio
    async def test_parse_qualitative_negation(self):
        """Parse qualitative value expressed through negation."""
        result = await parse_qualitative_async(
            "I don't agree with this at all.",
            ("strongly agree", "agree", "neutral", "disagree", "strongly disagree")
        )
        assert result in ("disagree", "strongly disagree")
