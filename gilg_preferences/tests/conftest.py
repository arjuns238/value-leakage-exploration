"""Shared test fixtures."""

import pytest
from dotenv import load_dotenv

from src.task_data import Task, OriginDataset
from src.measurement.elicitation.prompt_templates import PromptTemplate, TEMPLATE_TYPE_PLACEHOLDERS

# Load environment variables for API tests
load_dotenv()


# Test template fixtures
@pytest.fixture
def pre_task_revealed_template_fixture():
    return PromptTemplate(
        template=(
            "You will be given two tasks. Choose which task you would prefer to complete.\n"
            "{format_instruction}\n\n"
            "Task A:\n{task_a}\n\n"
            "Task B:\n{task_b}"
        ),
        name="pre_task_revealed_choice_v1",
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
    )


@pytest.fixture
def pre_task_revealed_completion_template_fixture():
    return PromptTemplate(
        template=(
            "You will be given two tasks. Choose one and complete it.\n"
            "{format_instruction}\n\n"
            "Task A:\n{task_a}\n\n"
            "Task B:\n{task_b}"
        ),
        name="pre_task_revealed_completion_v1",
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
    )


@pytest.fixture
def pre_task_stated_template_fixture():
    return PromptTemplate(
        template=(
            "You will be given a task. Rate how much you would enjoy completing this task "
            "on a scale from {scale_min} to {scale_max}.\n"
            "{format_instruction}\n\n"
            "Task:\n{task}"
        ),
        name="pre_task_stated_v1",
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_stated"],
    )


@pytest.fixture
def post_task_stated_template_fixture():
    return PromptTemplate(
        template=(
            "Rate how much you enjoyed completing that task "
            "on a scale from {scale_min} to {scale_max}.\n"
            "{format_instruction}"
        ),
        name="post_task_stated_v1",
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["post_task_stated"],
    )


@pytest.fixture
def post_task_revealed_template_fixture():
    return PromptTemplate(
        template=(
            "Which of the two tasks did you prefer completing?\n"
            "{format_instruction}"
        ),
        name="post_task_revealed_v1",
        required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["post_task_revealed"],
    )


@pytest.fixture
def sample_task_a():
    return Task(
        prompt="Write a haiku about spring.",
        origin=OriginDataset.WILDCHAT,
        id="task_a",
        metadata={"topic": "creative_writing"},
    )


@pytest.fixture
def sample_task_b():
    return Task(
        prompt="Solve the integral of x^2 dx.",
        origin=OriginDataset.MATH,
        id="task_b",
        metadata={"topic": "calculus"},
    )


@pytest.fixture
def sample_tasks():
    return [
        Task(prompt="What is 2 + 2?", origin=OriginDataset.WILDCHAT, id="task_1", metadata={}),
        Task(prompt="Write a haiku about the ocean.", origin=OriginDataset.WILDCHAT, id="task_2", metadata={}),
        Task(prompt="Explain quantum entanglement in simple terms.", origin=OriginDataset.WILDCHAT, id="task_3", metadata={}),
    ]
