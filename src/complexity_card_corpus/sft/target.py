from __future__ import annotations

import re
from typing import Any

from ..training_cards import TrainingCards
from .language import (
    _final_assistant_target,
    _inline_sentence,
    _labelled_fields,
    _sentence,
    _stable_index,
)


def _naturalize_assistant_target(
    messages: list[dict[str, str]],
    *,
    task: str,
    cards: TrainingCards,
    example_id: str,
) -> str:
    """Project card contracts into varied, direct assistant prose for SFT.

    The authored corpus keeps explicit completion labels because they make the
    source auditable. The model-facing projection deliberately removes those
    labels so the model learns the answer rather than a single house format.
    """

    response = _final_assistant_target(messages)
    variant = _stable_index(
        f"assistant-target:{example_id}:{cards.surface}:{cards.style}", 8
    )
    if task == "explanation_learning":
        fields = _labelled_fields(response, ("Core idea", "Example", "Check"))
        if set(fields) == {"Core idea", "Example", "Check"}:
            idea = re.sub(
                r"^(?:in plain terms,\s*|the key distinction is that\s+)",
                "",
                fields["Core idea"],
                flags=re.IGNORECASE,
            )
            idea = _sentence(idea)
            example = _sentence(fields["Example"])
            check = _sentence(fields["Check"])
            templates = (
                "{idea} For example, {example} {check}",
                "In simple terms, {inline_idea} For example, {inline_example} {check}",
                "{idea} You can see this in practice: {example} To check your understanding, {inline_check}",
                "The key point is that {inline_idea} For instance, {inline_example} {check}",
            )
            return templates[variant % len(templates)].format(
                idea=idea,
                inline_idea=_inline_sentence(idea),
                example=_inline_sentence(example),
                inline_example=_inline_sentence(example),
                check=check,
                inline_check=_inline_sentence(check),
            )
    elif task == "reasoning_verification":
        fields = _labelled_fields(response, ("Equation", "Total", "Check"))
        if set(fields) == {"Equation", "Total", "Check"}:
            check = re.sub(
                r"^(?:independently,\s*|inspect the supplied values, then note that\s*|use a second view of the values;\s*)",
                "",
                fields["Check"],
                flags=re.IGNORECASE,
            )
            templates = (
                "{equation}, so the result is {total}. As an independent check, {check}.",
                "The result is {total}: {equation}. This is consistent because {check}.",
                "Using the supplied values gives {equation}. Therefore, {total}. To verify it, {check}.",
                "{equation}. That gives {total}; checking from the other direction, {check}.",
            )
            return templates[variant % len(templates)].format(
                equation=fields["Equation"],
                total=fields["Total"],
                check=check,
            )
    elif task == "summarization_synthesis":
        fields = _labelled_fields(response, ("Decision", "Action", "Open point"))
        if set(fields) == {"Decision", "Action", "Open point"}:
            open_point = _sentence(fields["Open point"])
            templates = (
                "The decision is to {decision}. {action}. {open_point}",
                "They decided to {decision}. Next, {action}. {open_point}",
                "In summary, the decision is to {decision}; {action}. {open_point}",
                "The recorded decision is to {decision}, and {inline_action} {open_point}",
            )
            return templates[variant % len(templates)].format(
                decision=fields["Decision"],
                action=fields["Action"],
                inline_action=_inline_sentence(fields["Action"]),
                open_point=open_point,
            )
    elif task == "grounded_qa":
        direct = re.sub(
            r"^(?:Based on Source [A-Za-z0-9]+:|Source [A-Za-z0-9]+ supports this answer:|According to Source [A-Za-z0-9]+:)\s*",
            "",
            response,
        )
        direct = re.sub(
            r"^The documented answer is:\s*",
            "",
            direct,
        )
        direct = re.sub(
            r"\s+This is limited to Source [A-Za-z0-9]+\.?$",
            "",
            direct,
        )
        return direct
    elif task == "critique_revision":
        fields = _labelled_fields(response, ("Weakness", "Revision"))
        if set(fields) == {"Weakness", "Revision"}:
            weakness_text = re.sub(
                r",?\s*which makes the original difficult to verify\.?$",
                "",
                fields["Weakness"],
                flags=re.IGNORECASE,
            )
            weakness_text = re.sub(
                r"\s+The revision must stay within the recorded facts\.?$",
                "",
                weakness_text,
                flags=re.IGNORECASE,
            )
            weakness = _sentence(weakness_text)
            revision = _sentence(fields["Revision"])
            templates = (
                "{revision} This fixes the main problem because {inline_weakness}",
                "{revision} The draft previously failed because {inline_weakness}",
                "{revision}",
                "{revision} This avoids the unsupported part of the original because {inline_weakness}",
            )
            return templates[variant % len(templates)].format(
                weakness=weakness,
                inline_weakness=_inline_sentence(weakness),
                revision=revision,
            )
    elif task == "safety_uncertainty":
        match = re.fullmatch(
            r"Immediate action:\s*(.*?)\s+Boundary:\s*(.*?)\s+(Escalate\b.*)",
            response,
        )
        if match is not None:
            action = _sentence(match.group(1))
            boundary = _sentence(match.group(2))
            escalation = _sentence(match.group(3))
            templates = (
                "{action} {boundary} {escalation}",
                "First, {inline_action} {boundary} Next, {inline_escalation}",
                "The safest immediate step is clear. {action} {boundary} Then {inline_escalation}",
                "{action} {boundary} {escalation}",
            )
            return templates[variant % len(templates)].format(
                action=action,
                inline_action=_inline_sentence(action),
                boundary=boundary,
                inline_boundary=_inline_sentence(boundary),
                escalation=escalation,
                inline_escalation=_inline_sentence(escalation),
            )
    elif task == "practical_action":
        fields = _labelled_fields(response, ("Next step", "Owner", "Timing"))
        if set(fields) == {"Next step", "Owner", "Timing"}:
            step = _sentence(fields["Next step"])
            owner = _sentence(fields["Owner"])
            timing = _sentence(fields["Timing"])
            if timing.lower().startswith("before "):
                timing = "Complete this " + _inline_sentence(timing)
            templates = (
                "{step} {owner} {timing}",
                "First, {inline_step} {timing} {owner}",
                "{timing} Before committing, {inline_step} {owner}",
                "The safest workable move is clear. {step} {owner} {timing}",
            )
            return templates[variant % len(templates)].format(
                step=step,
                inline_step=_inline_sentence(step),
                owner=owner,
                timing=timing,
            )
    elif task == "context_clarification":
        fields = _labelled_fields(response, ("Understood",))
        if fields:
            return _sentence(fields["Understood"])
    elif task == "brainstorming_creativity":
        direct = re.sub(
            r"\s+(?:Each description states.*|The three retained ideas remain feasible.*|The alternatives emphasize.*)$",
            "",
            response,
            flags=re.IGNORECASE,
        )
        return direct.strip()
    elif task == "writing_transformation":
        direct = re.sub(
            r"^(?:Support reply|Project update|Internal note|Public notice|Short brief)\s+[A-Z0-9]+:\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"^(?:Meeting|Status|Update)\s+[A-Z0-9]+\s*[—:-]\s*",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        fields = _labelled_fields(direct, ("Decision", "Action", "Open item"))
        if set(fields) == {"Decision", "Action", "Open item"}:
            return " ".join(_sentence(fields[name]) for name in fields)
        direct = re.sub(
            r"\bRemaining work:\s*([A-Za-z])",
            lambda match: match.group(1).upper(),
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\bBlocker:\s*([A-Za-z])",
            lambda match: match.group(1).upper(),
            direct,
            flags=re.IGNORECASE,
        )
        return direct
    elif task in {
        "conversation_empathy",
        "extraction_classification",
    }:
        return response
    elif task == "planning_comparison":
        match = re.fullmatch(
            r"(.*?)\s+Sequence:\s*(.*?)\s+Fallback trigger:\s*(.*)",
            response,
            flags=re.DOTALL,
        )
        if match is not None:
            choice = _sentence(match.group(1))
            sequence = _sentence(match.group(2))
            fallback = _sentence(match.group(3))
            templates = (
                "{choice} Then {inline_sequence} If that path fails, {inline_fallback}",
                "{choice} {sequence} {fallback}",
                "{choice} {sequence} {fallback}",
                "{sequence} On those constraints, {inline_choice} If needed, {inline_fallback}",
            )
            return templates[variant % len(templates)].format(
                choice=choice,
                inline_choice=_inline_sentence(choice),
                sequence=sequence,
                inline_sequence=_inline_sentence(sequence),
                fallback=fallback,
                inline_fallback=_inline_sentence(fallback),
            )
        return response
    elif task == "troubleshooting":
        direct = re.sub(
            r"\bDirect check:\s*(?:confirm that\s*)?",
            "Confirm that ",
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\bRegression check:\s*(?:repeat\s*)?",
            "Afterward, repeat ",
            direct,
            flags=re.IGNORECASE,
        )
        steps = [
            match.group(1).strip()
            for match in re.finditer(
                r"(?:^|\s)\d+\.\s*(.*?)(?=\s+\d+\.\s|$)",
                direct,
                flags=re.DOTALL,
            )
        ]
        if len(steps) >= 3:
            if variant % 4 == 1:
                return "First, " + " Next, ".join(
                    _inline_sentence(step) for step in steps
                )
            if variant % 4 == 2:
                return "\n".join(f"- {_sentence(step)}" for step in steps)
            if variant % 4 == 3:
                return " ".join(_sentence(step) for step in steps)
        return direct
    return re.sub(
        r"^(?:Next step|Owner|Timing|Core idea|Example|Check|Decision|Action|Open point|Weakness|Revision|Immediate action|Boundary):\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )


def _apply_semantic_resolution(
    target: str,
    *,
    task: str,
    metadata: dict[str, Any],
    example_id: str,
) -> str:
    """Ground a generated answer in the scenario's authored semantic cards.

    Scenario Forge deliberately varies intent, state, boundary, fallback and
    success condition. Earlier SFT projection discarded those distinctions and
    retained only the domain renderer's answer, which made many otherwise
    different scenarios collapse to the same target. This projection turns the
    authored distinctions into ordinary prose. It never uses scenario IDs,
    hashes or lexical trace labels to manufacture uniqueness.
    """

    strict_output_tasks = {"extraction_classification"}
    if task in strict_output_tasks or not metadata.get("scenario_id"):
        return target
    required = (
        "subject",
        "surface_intent",
        "source_state",
        "source_constraint",
        "fallback_surface",
        "desired_outcome",
    )
    if any(not str(metadata.get(name, "")).strip() for name in required):
        return target

    source_variant = int(metadata.get("variant", 0)) % 4
    if source_variant == 0:
        return target

    state = _sentence(str(metadata["source_state"]))
    constraint = _sentence(str(metadata["source_constraint"]))
    fallback = _sentence(str(metadata["fallback_surface"]))
    outcome = _sentence(str(metadata["desired_outcome"]))
    # These clauses come from authored scenario fields. Keep them as complete
    # sentences instead of joining noun phrases with generic scaffolding such
    # as "the result should leave ...". The latter was grammatically fragile
    # and taught a visible house style rather than natural answers.
    full_templates = (
        (
            "{state} {constraint} {outcome} If the main path remains blocked, "
            "{inline_fallback}"
        ),
        (
            "{constraint} {state} If the decisive condition is still missing, "
            "{inline_fallback} {outcome}"
        ),
        ("{state} {outcome} {constraint} Otherwise, {inline_fallback}"),
        (
            "{constraint} {outcome} If that cannot be justified, "
            "{inline_fallback} {state}"
        ),
    )
    if source_variant == 1:
        short_templates = (
            "{state} If that condition still blocks progress, {inline_fallback}",
            "{state} If it remains unresolved, {inline_fallback}",
        )
        template = short_templates[
            _stable_index(f"short-resolution:{example_id}", len(short_templates))
        ]
    elif source_variant == 2:
        standard_templates = (
            "{constraint} {outcome} If that cannot be established, {inline_fallback}",
            "{outcome} {constraint} If the decisive condition remains unresolved, "
            "{inline_fallback}",
        )
        template = standard_templates[
            _stable_index(f"standard-resolution:{example_id}", len(standard_templates))
        ]
    else:
        template = full_templates[
            _stable_index(f"full-resolution:{example_id}", len(full_templates))
        ]
    resolution = template.format(
        state=state,
        constraint=constraint,
        fallback=fallback,
        inline_fallback=_inline_sentence(fallback).rstrip(".!?"),
        outcome=outcome,
    )
    return f"{target.rstrip()}\n\n{_sentence(resolution)}"
