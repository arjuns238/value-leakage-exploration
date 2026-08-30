from abc import ABC, abstractmethod
from os.path import commonprefix
from typing import Any, Literal

from src.task_data import Task
from src.types import (
    Message,
    PreferenceType,
    PreferencePrompt,
)
from src.measurement.elicitation.measurer import (
    Measurer,
    RevealedPreferenceMeasurer,
    StatedScoreMeasurer,
    RankingMeasurer,
)
from src.measurement.elicitation.response_format import (
    ResponseFormat,
    CompletionChoiceFormat,
)
from src.measurement.elicitation.prompt_templates.template import PromptTemplate


# =============================================================================
# Prompt Builders
# =============================================================================


class PromptBuilder(ABC):
    preference_type: PreferenceType

    def __init__(
        self,
        measurer: Measurer,
        response_format: ResponseFormat[Any],
        template: PromptTemplate,
        system_prompt: str | None = None,
        context_messages: list[Message] | None = None,
    ):
        self.measurer = measurer
        self.response_format = response_format
        self.template = template
        self.system_prompt = system_prompt
        self.context_messages = context_messages

    def _prepend_context(self, messages: list[Message]) -> list[Message]:
        prefix: list[Message] = []
        if self.system_prompt:
            prefix.append({"role": "system", "content": self.system_prompt})
        if self.context_messages:
            prefix.extend(self.context_messages)
        return [*prefix, *messages] if prefix else messages

    @abstractmethod
    def build(self, task: Task, *args: Any) -> PreferencePrompt: ...


class PreTaskRevealedPromptBuilder(PromptBuilder):
    preference_type = PreferenceType.PRE_TASK_REVEALED

    def build(self, task_a: Task, task_b: Task) -> PreferencePrompt:
        content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
            task_a=task_a.prompt,
            task_b=task_b.prompt,
        )
        messages = self._prepend_context([{"role": "user", "content": content}])
        return PreferencePrompt(
            messages=messages,
            tasks=[task_a, task_b],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class BaseModelRevealedPromptBuilder(PreTaskRevealedPromptBuilder):
    """For base model logprob cloze measurement.

    Inherits build() unchanged — produces identical prompt content to the instruct builder.
    Adds cloze_prefix/cloze_suffixes for logprob discrimination.
    """

    @property
    def cloze_prefix(self) -> str:
        """Common prefix of choice labels, to append for logprob discrimination.
        "Task A" / "Task B" -> "Task"
        """
        a = self.response_format.task_a_label
        b = self.response_format.task_b_label
        return commonprefix([a, b]).rstrip()

    @property
    def cloze_suffixes(self) -> tuple[str, str]:
        """Discriminative suffixes after cloze_prefix: (" A", " B")."""
        p = self.cloze_prefix
        a = self.response_format.task_a_label
        b = self.response_format.task_b_label
        return (a[len(p):], b[len(p):])


class PreTaskStatedPromptBuilder(PromptBuilder):
    preference_type = PreferenceType.PRE_TASK_STATED

    def build(self, task: Task) -> PreferencePrompt:
        format_args = {
            "format_instruction": self.response_format.format_instruction(),
            "task": task.prompt,
        }

        if "scale_min" in self.template.required_placeholders:
            format_args["scale_min"] = str(self.response_format.scale_min)
            format_args["scale_max"] = str(self.response_format.scale_max)

        content = self.template.format(**format_args)
        messages = self._prepend_context([{"role": "user", "content": content}])
        return PreferencePrompt(
            messages=messages,
            tasks=[task],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PostTaskStatedPromptBuilder(PromptBuilder):
    """Creates multi-turn: (1) task prompt, (2) completion, (3) stated preference request."""
    preference_type = PreferenceType.POST_TASK_STATED

    def build(self, task: Task, completion_text: str) -> PreferencePrompt:
        format_args: dict[str, str] = {
            "format_instruction": self.response_format.format_instruction(),
        }
        if "scale_min" in self.template.required_placeholders:
            format_args["scale_min"] = str(self.response_format.scale_min)
            format_args["scale_max"] = str(self.response_format.scale_max)

        stated_content = self.template.format(**format_args)
        messages = self._prepend_context([
            {"role": "user", "content": task.prompt},
            {"role": "assistant", "content": completion_text},
            {"role": "user", "content": stated_content},
        ])
        return PreferencePrompt(
            messages=messages,
            tasks=[task],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PostTaskRevealedPromptBuilder(PromptBuilder):
    """Creates multi-turn prompt for binary preference after completing both tasks.

    Messages: [user: task_a] → [asst: completion_a] → [user: task_b] → [asst: completion_b] → [user: which preferred?]
    """
    preference_type = PreferenceType.POST_TASK_REVEALED

    def build(
        self, task_a: Task, task_b: Task, completion_a: str, completion_b: str
    ) -> PreferencePrompt:
        preference_content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
        )
        messages = self._prepend_context([
            {"role": "user", "content": task_a.prompt},
            {"role": "assistant", "content": completion_a},
            {"role": "user", "content": task_b.prompt},
            {"role": "assistant", "content": completion_b},
            {"role": "user", "content": preference_content},
        ])
        return PreferencePrompt(
            messages=messages,
            tasks=[task_a, task_b],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PreTaskRankingPromptBuilder(PromptBuilder):
    """Creates prompt for ranking multiple tasks by preference."""
    preference_type = PreferenceType.PRE_TASK_RANKING

    def build(self, tasks: list[Task]) -> PreferencePrompt:
        # Format tasks as task_a, task_b, task_c, task_d, task_e
        task_texts = {f"task_{chr(97+i)}": t.prompt for i, t in enumerate(tasks)}
        content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
            **task_texts,
        )
        messages = self._prepend_context([{"role": "user", "content": content}])
        return PreferencePrompt(
            messages=messages,
            tasks=tasks,
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PostTaskRankingPromptBuilder(PromptBuilder):
    """Creates multi-turn prompt for ranking tasks after completing them.

    Messages: [user: task_a] → [asst: completion_a] → ... → [user: rank them]
    """
    preference_type = PreferenceType.POST_TASK_RANKING

    def build(self, tasks: list[Task], completions: list[str]) -> PreferencePrompt:
        if len(tasks) != len(completions):
            raise ValueError("tasks and completions must have same length")

        messages: list[Message] = []
        for task, completion in zip(tasks, completions):
            messages.append({"role": "user", "content": task.prompt})
            messages.append({"role": "assistant", "content": completion})

        ranking_content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
        )
        messages.append({"role": "user", "content": ranking_content})
        messages = self._prepend_context(messages)

        return PreferencePrompt(
            messages=messages,
            tasks=tasks,
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class MultiTurnRevealedPromptBuilder(PromptBuilder):
    """Split tasks across turns with assistant prefill between them.

    Messages: [user: intro + task_a + format_instruction] -> [asst: prefill] -> [user: task_b]
    Model generates completion of preferred task.
    """
    preference_type = PreferenceType.PRE_TASK_REVEALED

    def __init__(self, prefill: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.prefill = prefill

    def build(self, task_a: Task, task_b: Task) -> PreferencePrompt:
        first_turn = self.template.format(
            format_instruction=self.response_format.format_instruction(),
            task_a=task_a.prompt,
        )
        task_b_label = self.response_format.task_b_label if isinstance(self.response_format, CompletionChoiceFormat) else "Task B"
        messages = self._prepend_context([
            {"role": "user", "content": first_turn},
            {"role": "assistant", "content": self.prefill},
            {"role": "user", "content": f"{task_b_label}:\n{task_b.prompt}"},
        ])
        return PreferencePrompt(
            messages=messages,
            tasks=[task_a, task_b],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class OpenEndedPromptBuilder(PromptBuilder):
    """Creates multi-turn prompt for open-ended response after task completion.

    Messages: [user: task] → [asst: completion] → [user: open-ended question]
    Returns raw response text for semantic valence scoring.
    """
    preference_type = PreferenceType.OPEN_ENDED

    def build(self, task: Task, completion_text: str) -> PreferencePrompt:
        open_ended_content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
        )
        messages = self._prepend_context([
            {"role": "user", "content": task.prompt},
            {"role": "assistant", "content": completion_text},
            {"role": "user", "content": open_ended_content},
        ])
        return PreferencePrompt(
            messages=messages,
            tasks=[task],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )
