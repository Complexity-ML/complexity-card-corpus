from __future__ import annotations


CORE_TASKS = (
    "brainstorming_creativity",
    "context_clarification",
    "conversation_empathy",
    "critique_revision",
    "explanation_learning",
    "extraction_classification",
    "grounded_qa",
    "planning_comparison",
    "practical_action",
    "reasoning_verification",
    "safety_uncertainty",
    "summarization_synthesis",
    "troubleshooting",
    "writing_transformation",
)
CASUAL_TASK = "casual_conversation"
ALL_TASKS = (CASUAL_TASK, *CORE_TASKS)


def validate_v2_plan() -> None:
    if len(ALL_TASKS) != len(set(ALL_TASKS)):
        raise ValueError("V2 task names must be unique")
    if not CORE_TASKS or CASUAL_TASK in CORE_TASKS:
        raise ValueError("V2 requires one separate casual family")


validate_v2_plan()


__all__ = (
    "ALL_TASKS",
    "CASUAL_TASK",
    "CORE_TASKS",
    "validate_v2_plan",
)
