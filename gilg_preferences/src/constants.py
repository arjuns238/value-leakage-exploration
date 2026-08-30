"""Centralized constants for the preferences measurement system.

This module contains all magic numbers and strings used throughout the codebase
to ensure consistency and make configuration changes easier.
"""

# Rating scale defaults
# These define the default numerical range for rating-based measurements.
# A 1-10 scale is commonly used in preference elicitation as it provides
# sufficient granularity while remaining intuitive for evaluators.
DEFAULT_SCALE_MIN = 1
DEFAULT_SCALE_MAX = 10

# XML tag defaults for structured response parsing
# Using descriptive tag names that clearly indicate their purpose
DEFAULT_CHOICE_TAG = "choice"
DEFAULT_RATING_TAG = "rating"

# Qualitative rating scale
QUALITATIVE_VALUES = ("good", "neutral", "bad")
QUALITATIVE_TO_NUMERIC = {
    "good": 1,
    "neutral": 0,
    "bad": -1,
}
NUMERIC_TO_QUALITATIVE = {v: k for k, v in QUALITATIVE_TO_NUMERIC.items()}

# Adjective pick-list scale (10-point ordinal)
ADJECTIVE_VALUES = (
    "dreading", "reluctant", "unenthusiastic", "indifferent", "curious",
    "interested", "engaged", "eager", "excited", "thrilled",
)
ADJECTIVE_TO_NUMERIC = {adj: float(i + 1) for i, adj in enumerate(ADJECTIVE_VALUES)}

# Fruit rating scale
FRUIT_VALUES = ("lemon", "grape", "orange", "banana", "apple")
FRUIT_TO_NUMERIC = {fruit: float(i) for i, fruit in enumerate(FRUIT_VALUES)}
