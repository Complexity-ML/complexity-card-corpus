"""Sense-aware two-dimensional variable decks for authored text templates."""

from .catalog import (
    brainstorming_variable_by,
    critique_variable_by,
    empathy_variable_by,
    reasoning_variable_by,
    safety_variable_by,
)
from .audit import (
    StaticTextProgress,
    TemplateDensityProgress,
    aggregate_static_text_progress,
    analyze_static_text,
    analyze_static_text_progress,
    analyze_template_density,
)
from .matrix import VariableBy2D

__all__ = (
    "StaticTextProgress",
    "TemplateDensityProgress",
    "VariableBy2D",
    "aggregate_static_text_progress",
    "analyze_static_text",
    "analyze_static_text_progress",
    "analyze_template_density",
    "brainstorming_variable_by",
    "critique_variable_by",
    "empathy_variable_by",
    "reasoning_variable_by",
    "safety_variable_by",
)
