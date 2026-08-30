import json
import re
from abc import ABC, abstractmethod
from typing import Protocol, Literal, Any, TypeVar

from pydantic import BaseModel, Field

from src.constants import (
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_MAX,
    DEFAULT_CHOICE_TAG,
    DEFAULT_RATING_TAG,
    QUALITATIVE_VALUES,
    QUALITATIVE_TO_NUMERIC,
)
from src.measurement.elicitation import semantic_parser
from src.measurement.elicitation import refusal_judge
from src.measurement.elicitation.semantic_parser import ParseError
from src.measurement.elicitation.completion_judge import (
    CompletionParseResult,
    RegexOnly,
    JudgeAlways,
    RegexThenJudge,
    RegexParseResult,
    JudgeParseResult,
    extract_claimed_task,
)


def _exact_choice_match(
    response: str,
    task_a_label: str,
    task_b_label: str,
) -> Literal["a", "b"] | None:
    """Fast exact match - only triggers if response is exactly the label."""
    cleaned = response.strip()
    if cleaned.lower() == task_a_label.lower():
        return "a"
    if cleaned.lower() == task_b_label.lower():
        return "b"
    return None


def _exact_rating_match(response: str) -> float | None:
    """Fast exact match - only triggers if response is exactly a number."""
    cleaned = response.strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _exact_qualitative_match(response: str, values: tuple[str, ...]) -> str | None:
    """Fast exact match - only triggers if response is exactly one of the values."""
    cleaned = response.strip().lower()
    if cleaned in values:
        return cleaned
    return None


def _tool_from_model(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
    """Create OpenAI tool definition from a Pydantic model."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


T = TypeVar("T", bound=BaseModel)


def _parse_response(response: str, model: type[T]) -> T:
    """Parse JSON response into a Pydantic model."""
    return model.model_validate_json(response)


def _make_choice_model(label_a: str, label_b: str) -> type[BaseModel]:
    """Create a Pydantic model for binary choice between two labels."""
    class ChoiceSubmission(BaseModel):
        choice: Literal[label_a, label_b] = Field(  # type: ignore[valid-type]
            description=f"Your choice: '{label_a}' or '{label_b}'."
        )
    return ChoiceSubmission


T_co = TypeVar("T_co", covariant=True)


class ResponseFormat(Protocol[T_co]):
    store_raw_response: bool
    @property
    def tools(self) -> list[dict[str, Any]] | None: ...
    def format_instruction(self) -> str: ...
    async def parse(self, response: str) -> T: ...


# --- Base Classes ---


class BaseChoiceFormat(ABC):
    tools: list[dict[str, Any]] | None = None
    store_raw_response: bool = False

    def __init__(
        self,
        task_a_label: str = "Task A",
        task_b_label: str = "Task B",
    ):
        self.task_a_label = task_a_label
        self.task_b_label = task_b_label

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_choice(self, response: str) -> str | None: ...

    async def _semantic_parse(self, response: str) -> Literal["a", "b", "refusal"]:
        return await semantic_parser.parse_choice_async(
            response, self.task_a_label, self.task_b_label
        )

    def parse_sync(self, response: str) -> Literal["a", "b", "parse_fail"]:
        """Synchronous 2-stage parse. No LLM fallback — returns 'parse_fail' for ambiguous."""
        choice = _exact_choice_match(response, self.task_a_label, self.task_b_label)
        if choice:
            return choice
        choice = self._extract_choice(response)
        if choice and choice in ("a", "b"):
            return choice  # type: ignore
        return "parse_fail"

    async def parse(self, response: str) -> Literal["a", "b", "refusal"]:
        # 1. Fast path: response is exactly the label
        choice = _exact_choice_match(response, self.task_a_label, self.task_b_label)
        if choice:
            return choice
        # 2. Format-specific extraction (regex, XML, etc.)
        choice = self._extract_choice(response)
        if choice and choice in ("a", "b"):
            return choice  # type: ignore
        # 3. LLM-based semantic parsing fallback (handles refusals too)
        return await self._semantic_parse(response)


class BaseRatingFormat(ABC):
    tools: list[dict[str, Any]] | None = None

    def __init__(
        self,
        scale_min: int = DEFAULT_SCALE_MIN,
        scale_max: int = DEFAULT_SCALE_MAX,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_number(self, response: str) -> float | None: ...

    async def parse(self, response: str) -> float | Literal["refusal"]:
        # 1. Fast path: response is exactly a number
        number = _exact_rating_match(response)
        if number is not None:
            return number
        # 2. Format-specific extraction (regex, XML, etc.)
        number = self._extract_number(response)
        if number is not None:
            return number
        # 3. Check for refusal before semantic parsing
        if (await refusal_judge.judge_preference_refusal_async(response)).is_refusal:
            return "refusal"
        # 4. LLM-based semantic parsing fallback
        number = await semantic_parser.parse_rating_async(response, self.scale_min, self.scale_max)
        if number is not None:
            return number
        raise ValueError(f"Could not extract number from response: {response}")


# --- Choice Formats (for binary preference) ---


class RegexChoiceFormat(BaseChoiceFormat):
    def format_instruction(self) -> str:
        return f"Respond with only '{self.task_a_label}' or '{self.task_b_label}'."

    def _extract_choice(self, response: str) -> str | None:
        response_clean = response.strip()

        def make_pattern(label: str) -> str:
            escaped = re.escape(label)
            # Only use word boundary if the label edge is a word character
            prefix = r"\b" if label and label[0].isalnum() else ""
            suffix = r"\b" if label and label[-1].isalnum() else ""
            return prefix + escaped + suffix

        pattern_a = make_pattern(self.task_a_label)
        pattern_b = make_pattern(self.task_b_label)

        match_a = re.search(pattern_a, response_clean, re.IGNORECASE)
        match_b = re.search(pattern_b, response_clean, re.IGNORECASE)

        has_a = bool(match_a)
        has_b = bool(match_b)

        if has_a and not has_b:
            return "a"
        elif has_b and not has_a:
            return "b"
        elif has_a and has_b:
            # Return the first match
            if match_a.start() < match_b.start():
                return "a"
            else:
                return "b"

        return None


class XMLChoiceFormat(BaseChoiceFormat):
    def __init__(
        self,
        task_a_label: str = "Task A",
        task_b_label: str = "Task B",
        tag: str = DEFAULT_CHOICE_TAG,
    ):
        super().__init__(task_a_label, task_b_label)
        self.tag = tag

    def format_instruction(self) -> str:
        return f"Respond with your choice in XML tags: <{self.tag}>{self.task_a_label}</{self.tag}> or <{self.tag}>{self.task_b_label}</{self.tag}>"

    def _extract_choice(self, response: str) -> str | None:
        # Extract content from XML tag and match against labels
        pattern = rf"<{self.tag}>\s*(.+?)\s*</{self.tag}>"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            content = match.group(1).strip().lower()
            if content == self.task_a_label.lower():
                return "a"
            elif content == self.task_b_label.lower():
                return "b"
        return None


class CompletionChoiceFormat:
    """Model completes one of two tasks, indicating choice by prefix. Measures revealed preference.

    Uses strategy pattern for parsing: RegexOnly, JudgeAlways, or RegexThenJudge.
    parse() returns CompletionParseResult (not just a choice string).
    extract_label() provides sync regex-only parsing for steering.
    """

    store_raw_response = True
    tools = None

    def __init__(
        self,
        task_a_label: str = "Task A",
        task_b_label: str = "Task B",
        *,
        parser: RegexOnly | JudgeAlways | RegexThenJudge | None = None,
    ):
        self.task_a_label = task_a_label
        self.task_b_label = task_b_label
        # LOCAL PATCH (value-direction project): default to RegexOnly — no API
        # keys on the pod; RegexThenJudge's fallback calls an OpenRouter judge.
        # Non-matching responses parse as "refusal" and drop out of the
        # Thurstonian fit; we monitor the refusal rate and judge leftovers
        # offline if it exceeds a few percent.
        self.parser = parser if parser is not None else RegexOnly()

    def format_instruction(self) -> str:
        return f"Begin with '{self.task_a_label}:' or '{self.task_b_label}:' to indicate your choice, then complete that task."

    async def parse(
        self, response: str, task_a_prompt: str = "", task_b_prompt: str = ""
    ) -> CompletionParseResult:
        return await self.parser.parse(response, task_a_prompt, task_b_prompt)

    def extract_label(self, response: str) -> Literal["a", "b", "refusal"]:
        result = extract_claimed_task(response, self.task_a_label, self.task_b_label)
        return "refusal" if result == "neither" else result


class ReasoningChoiceWrapper(BaseChoiceFormat):
    """Wraps any choice format to request reasoning before the answer.

    Asks model to explain reasoning briefly first, then give answer in XML tags.
    Reasoning-first produces more thoughtful choices. XML extraction handles
    task labels mentioned in reasoning.
    Stores raw response for analysis.
    """

    store_raw_response: bool = True

    def __init__(self, wrapped: BaseChoiceFormat):
        super().__init__(wrapped.task_a_label, wrapped.task_b_label)
        self.wrapped = wrapped

    @property
    def tools(self) -> list[dict[str, Any]] | None:
        return self.wrapped.tools

    def format_instruction(self) -> str:
        return f"In one sentence, explain your reasoning. Then {self.wrapped.format_instruction()}"

    def _extract_choice(self, response: str) -> str | None:
        return self.wrapped._extract_choice(response)

    async def _semantic_parse(self, response: str) -> Literal["a", "b", "refusal"]:
        return await self.wrapped._semantic_parse(response)


# --- Rating Formats (for numerical scores) ---


class RegexRatingFormat(BaseRatingFormat):
    def format_instruction(self) -> str:
        return f"Respond with only a number from {self.scale_min} to {self.scale_max}."

    def _extract_number(self, response: str) -> float | None:
        # Match: integers, decimals with leading digits, or decimals like ".5"
        numbers = re.findall(r"-?(?:\d+\.?\d*|\.\d+)", response)
        if numbers:
            return float(numbers[0])
        return None


class XMLRatingFormat(BaseRatingFormat):
    def __init__(
        self,
        tag: str = DEFAULT_RATING_TAG,
        scale_min: int = DEFAULT_SCALE_MIN,
        scale_max: int = DEFAULT_SCALE_MAX,
    ):
        super().__init__(scale_min, scale_max)
        self.tag = tag

    def format_instruction(self) -> str:
        return f"Respond with your rating in <{self.tag}> tags."

    def _extract_number(self, response: str) -> float | None:
        pattern = rf"<{self.tag}>\s*(-?\d+\.?\d*)\s*</{self.tag}>"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None


# --- Tool Use Formats ---


class ToolUseChoiceFormat(BaseChoiceFormat):
    """Uses native tool calling for structured output. No semantic parsing needed."""

    def __init__(
        self,
        task_a_label: str = "Task A",
        task_b_label: str = "Task B",
    ):
        super().__init__(task_a_label, task_b_label)
        # Create model dynamically with the specific enum values
        self._response_model = self._make_model()

    def _make_model(self) -> type[BaseModel]:
        return _make_choice_model(self.task_a_label, self.task_b_label)

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [_tool_from_model(
            "submit_choice",
            "Submit your choice of which task you prefer.",
            self._response_model,
        )]

    def format_instruction(self) -> str:
        return "You MUST respond by calling the submit_choice tool. Do not write any text - only call the tool."

    async def parse(self, response: str) -> Literal["a", "b", "refusal"]:
        try:
            data = json.loads(response)
            if "choice" in data and isinstance(data["choice"], str):
                choice_lower = data["choice"].lower()
                if choice_lower == self.task_a_label.lower():
                    return "a"
                elif choice_lower == self.task_b_label.lower():
                    return "b"
        except Exception:
            pass
        # Tool use failed - fall back to semantic parsing
        return await self._semantic_parse(response)

    def _extract_choice(self, response: str) -> str | None:
        # Not used - parse() handles everything
        return None


class ToolUseRatingFormat(BaseRatingFormat):
    """Uses native tool calling for structured output. No semantic parsing needed."""

    def __init__(
        self,
        scale_min: int = DEFAULT_SCALE_MIN,
        scale_max: int = DEFAULT_SCALE_MAX,
    ):
        super().__init__(scale_min, scale_max)
        self._response_model = self._make_model()

    def _make_model(self) -> type[BaseModel]:
        scale_min, scale_max = self.scale_min, self.scale_max

        class RatingSubmission(BaseModel):
            rating: float = Field(
                description=f"Your rating from {scale_min} to {scale_max}."
            )
        return RatingSubmission

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [_tool_from_model(
            "submit_rating",
            "Submit your rating for the task.",
            self._response_model,
        )]

    def format_instruction(self) -> str:
        return f"You MUST respond by calling the submit_rating tool with a number from {self.scale_min} to {self.scale_max}. Do not write any text - only call the tool."

    async def parse(self, response: str) -> float | Literal["refusal"]:
        try:
            result = _parse_response(response, self._response_model)
            return result.rating
        except Exception:
            pass
        # Only consult refusal judge if the response looks like natural language
        # (not malformed JSON). If it starts with '{', it was a failed tool call
        # attempt, not a refusal.
        stripped = response.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            raise ValueError(f"Malformed JSON tool response: {response}")
        if (await refusal_judge.judge_preference_refusal_async(response)).is_refusal:
            return "refusal"
        raise ValueError(f"Could not extract number from response: {response}")

    def _extract_number(self, response: str) -> float | None:
        # Not used - parse() handles everything
        return None


# --- Qualitative Rating Formats ---


class BaseQualitativeFormat(ABC):
    """Base class for qualitative rating formats that return numeric values."""
    tools: list[dict[str, Any]] | None = None

    def __init__(
        self,
        values: tuple[str, ...] = QUALITATIVE_VALUES,
        value_to_score: dict[str, float] | None = None,
    ):
        self.values = values
        self.value_to_score = value_to_score if value_to_score is not None else QUALITATIVE_TO_NUMERIC

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_qualitative(self, response: str) -> str: ...

    async def parse(self, response: str) -> float | Literal["refusal", "unclear"]:
        # 1. Fast path: response is exactly one of the values
        qualitative = _exact_qualitative_match(response, self.values)
        if qualitative:
            return float(self.value_to_score[qualitative])
        # 2. Format-specific extraction (regex, XML, etc.)
        try:
            qualitative = self._extract_qualitative(response)
            return float(self.value_to_score[qualitative])
        except (ValueError, KeyError):
            pass
        # 3. Check for refusal before semantic parsing
        if (await refusal_judge.judge_preference_refusal_async(response)).is_refusal:
            return "refusal"
        # 4. LLM-based semantic parsing fallback
        qualitative = await semantic_parser.parse_qualitative_async(response, self.values)
        if qualitative:
            return float(self.value_to_score[qualitative])
        return "unclear"


class RegexQualitativeFormat(BaseQualitativeFormat):
    def format_instruction(self) -> str:
        quoted = [f"'{v}'" for v in self.values]
        if len(quoted) == 2:
            return f"Respond with only {quoted[0]} or {quoted[1]}."
        return f"Respond with only {', '.join(quoted[:-1])}, or {quoted[-1]}."

    def _extract_qualitative(self, response: str) -> str:
        response_lower = response.lower()
        for value in self.values:
            if re.search(rf'\b{value}\b', response_lower):
                return value
        raise ValueError(f"No qualitative value found in response: {response}")


class XMLQualitativeFormat(BaseQualitativeFormat):
    def __init__(
        self,
        values: tuple[str, ...] = QUALITATIVE_VALUES,
        value_to_score: dict[str, float] | None = None,
        tag: str = DEFAULT_RATING_TAG,
    ):
        super().__init__(values, value_to_score)
        self.tag = tag

    def format_instruction(self) -> str:
        return f"Respond with your rating in <{self.tag}> tags."

    def _extract_qualitative(self, response: str) -> str:
        pattern = rf"<{self.tag}>\s*(\w+)\s*</{self.tag}>"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            if value in self.values:
                return value
        raise ValueError(f"No valid qualitative value in XML tags: {response}")


class ToolUseQualitativeFormat(BaseQualitativeFormat):
    """Uses native tool calling for structured output. No semantic parsing needed."""

    def __init__(
        self,
        values: tuple[str, ...] = QUALITATIVE_VALUES,
        value_to_score: dict[str, float] | None = None,
    ):
        super().__init__(values, value_to_score)
        self._response_model = self._make_model()

    def _make_model(self) -> type[BaseModel]:
        values = self.values

        class QualitativeSubmission(BaseModel):
            rating: Literal[values] = Field(  # type: ignore[valid-type]
                description=f"Your rating: {', '.join(values)}."
            )
        return QualitativeSubmission

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [_tool_from_model(
            "submit_rating",
            "Submit your qualitative rating for the task.",
            self._response_model,
        )]

    def format_instruction(self) -> str:
        return f"You MUST respond by calling the submit_rating tool with one of: {', '.join(self.values)}. Do not write any text - only call the tool."

    async def parse(self, response: str) -> float | Literal["refusal"]:
        # Normalize case before Pydantic validation (enum is lowercase)
        try:
            data = json.loads(response)
            if "rating" in data and isinstance(data["rating"], str):
                data["rating"] = data["rating"].lower()
            result = self._response_model.model_validate(data)
            return float(self.value_to_score[result.rating])
        except Exception:
            pass
        # Tool use failed - model may have refused instead of calling tool
        if (await refusal_judge.judge_preference_refusal_async(response)).is_refusal:
            return "refusal"
        raise ValueError(f"Could not parse qualitative value from response: {response}")

    def _extract_qualitative(self, response: str) -> str:
        # Not used - parse() handles everything
        raise NotImplementedError


# --- Format Registries ---

ResponseFormatName = Literal["regex", "tool_use", "xml", "completion"]

# Binary qualitative scale
BINARY_QUALITATIVE_VALUES = ("good", "bad")
BINARY_QUALITATIVE_TO_NUMERIC = {"good": 1.0, "bad": -1.0}


def qualitative_format_for_scale(
    scale: str,
    format_type: ResponseFormatName = "regex",
) -> BaseQualitativeFormat:
    """Create a qualitative format based on scale type (binary/ternary)."""
    format_cls = QUALITATIVE_FORMATS[format_type]
    if scale == "binary":
        return format_cls(values=BINARY_QUALITATIVE_VALUES, value_to_score=BINARY_QUALITATIVE_TO_NUMERIC)
    return format_cls()  # ternary default

CHOICE_FORMATS: dict[ResponseFormatName, type] = {
    "regex": RegexChoiceFormat,
    "tool_use": ToolUseChoiceFormat,
    "xml": XMLChoiceFormat,
    "completion": CompletionChoiceFormat,
}

RATING_FORMATS: dict[ResponseFormatName, type[BaseRatingFormat]] = {
    "regex": RegexRatingFormat,
    "tool_use": ToolUseRatingFormat,
    "xml": XMLRatingFormat,
}

QUALITATIVE_FORMATS: dict[ResponseFormatName, type[BaseQualitativeFormat]] = {
    "regex": RegexQualitativeFormat,
    "tool_use": ToolUseQualitativeFormat,
    "xml": XMLQualitativeFormat,
}


# --- Format Builders ---


def get_stated_response_format(
    scale_info: tuple[int, int] | list[str],
    format_name: str,
) -> BaseRatingFormat | BaseQualitativeFormat:
    """Build stated response format from scale info."""
    if isinstance(scale_info, list):
        values = tuple(scale_info)
        value_to_score = {v: float(i) for i, v in enumerate(values)}
        return QUALITATIVE_FORMATS[format_name](values=values, value_to_score=value_to_score)
    scale_min, scale_max = scale_info
    return RATING_FORMATS[format_name](scale_min, scale_max)


def get_revealed_response_format(
    task_a_label: str,
    task_b_label: str,
    format_name: str,
    reasoning_mode: bool = False,
) -> BaseChoiceFormat | CompletionChoiceFormat:
    """Build choice response format from labels.

    If reasoning_mode is True, wraps the format with ReasoningChoiceWrapper
    which asks model to explain reasoning after giving answer.
    Only xml format is supported with reasoning_mode - other formats risk
    matching on task labels mentioned in the reasoning.
    """
    if reasoning_mode and format_name != "xml":
        raise ValueError(f"reasoning_mode requires xml format, got '{format_name}'")
    base_format = CHOICE_FORMATS[format_name](task_a_label, task_b_label)
    if reasoning_mode:
        return ReasoningChoiceWrapper(base_format)
    return base_format


class BaseRankingFormat(ABC):
    tools: list[dict[str, Any]] | None = None

    def __init__(self, task_labels: tuple[str, ...]):
        self.task_labels = task_labels  # ("A", "B", "C", "D", "E")

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_ranking(self, response: str) -> list[int] | None: ...

    def _labels_to_indices(self, labels: list[str]) -> list[int] | None:
        label_to_idx = {label.upper(): i for i, label in enumerate(self.task_labels)}
        indices = []
        for label in labels:
            label_upper = label.upper().strip()
            if label_upper.startswith("TASK "):
                label_upper = label_upper[5:]
            if label_upper not in label_to_idx:
                return None
            indices.append(label_to_idx[label_upper])
        return indices if len(set(indices)) == len(indices) else None

    async def parse(self, response: str) -> list[int] | Literal["refusal"]:
        ranking = self._extract_ranking(response)
        if ranking is not None and len(ranking) == len(self.task_labels):
            if len(set(ranking)) == len(ranking):
                return ranking

        if (await refusal_judge.judge_preference_refusal_async(response)).is_refusal:
            return "refusal"

        return await semantic_parser.parse_ranking_async(response, self.task_labels)


class RegexRankingFormat(BaseRankingFormat):
    def format_instruction(self) -> str:
        labels = ", ".join(self.task_labels)
        return f"Respond with your ranking using '>' between options, e.g., '{self.task_labels[0]} > {self.task_labels[1]} > ...' (most preferred first)."

    def _extract_ranking(self, response: str) -> list[int] | None:
        if ">" in response:
            parts = [p.strip().upper() for p in response.split(">")]
            result = self._labels_to_indices(parts)
            if result and len(result) == len(self.task_labels):
                return result

        if "," in response:
            parts = [p.strip().upper() for p in response.split(",")]
            result = self._labels_to_indices(parts)
            if result and len(result) == len(self.task_labels):
                return result

        n = len(self.task_labels)
        letters = re.findall(r'\b([A-Z])\b', response.upper())
        if len(letters) == n:
            result = self._labels_to_indices(letters)
            if result and len(result) == n:
                return result

        return None


class XMLRankingFormat(BaseRankingFormat):
    def __init__(self, task_labels: tuple[str, ...], tag: str = "ranking"):
        super().__init__(task_labels)
        self.tag = tag

    def format_instruction(self) -> str:
        example = ", ".join(self.task_labels)
        return f"Respond with your ranking in XML tags: <{self.tag}>{example}</{self.tag}> (most preferred first)."

    def _extract_ranking(self, response: str) -> list[int] | None:
        pattern = rf"<{self.tag}>\s*(.+?)\s*</{self.tag}>"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1)
            if ">" in content:
                parts = [p.strip().upper() for p in content.split(">")]
                return self._labels_to_indices(parts)
            parts = [p.strip().upper() for p in content.split(",")]
            return self._labels_to_indices(parts)
        return None


class ToolUseRankingFormat(BaseRankingFormat):
    def __init__(self, task_labels: tuple[str, ...]):
        super().__init__(task_labels)
        self._response_model = self._make_model()

    def _make_model(self) -> type[BaseModel]:
        class RankingSubmission(BaseModel):
            ranking: list[str] = Field(
                description="Task labels ordered from most to least preferred."
            )
        return RankingSubmission

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [_tool_from_model(
            "submit_ranking",
            "Submit your ranking of the tasks from most to least preferred.",
            self._response_model,
        )]

    def format_instruction(self) -> str:
        labels = ", ".join(f"'{l}'" for l in self.task_labels)
        return f"You MUST respond by calling the submit_ranking tool with a list of task labels ({labels}) ordered from most to least preferred. Do not write any text - only call the tool."

    def _extract_ranking(self, response: str) -> list[int] | None:
        try:
            data = json.loads(response)
            if "ranking" in data and isinstance(data["ranking"], list):
                return self._labels_to_indices(data["ranking"])
        except Exception:
            pass
        return None

    async def parse(self, response: str) -> list[int] | Literal["refusal"]:
        ranking = self._extract_ranking(response)
        if ranking is not None and len(ranking) == len(self.task_labels):
            if len(set(ranking)) == len(ranking):
                return ranking

        if (await refusal_judge.judge_preference_refusal_async(response)).is_refusal:
            return "refusal"
        return await semantic_parser.parse_ranking_async(response, self.task_labels)


RANKING_FORMATS: dict[ResponseFormatName, type[BaseRankingFormat]] = {
    "regex": RegexRankingFormat,
    "xml": XMLRankingFormat,
    "tool_use": ToolUseRankingFormat,
}


def get_ranking_response_format(
    task_labels: tuple[str, ...],
    format_name: str,
) -> BaseRankingFormat:
    return RANKING_FORMATS[format_name](task_labels)


# --- Open-Ended Response Format ---


class OpenEndedFormat:
    """Format for open-ended responses with semantic valence scoring.

    Simply accepts any text response and stores it for later semantic valence scoring.
    No parsing/extraction needed - the raw response is what we need.
    """

    tools: list[dict[str, Any]] | None = None

    def __init__(self):
        pass

    def format_instruction(self) -> str:
        """Return instruction for open-ended response."""
        return (
            "Please provide a detailed, honest response in your own words. "
            "There are no right or wrong answers."
        )

    async def parse(self, response: str) -> str:
        """Return the raw response as-is for later semantic scoring."""
        return response.strip()
