from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from ..variable_by import reasoning_envelope_variable_by
from ..variable_by.reservoirs import planning_constraint_surfaces
from .language import _labelled_fields


REASONING_ENVELOPE_VERSION = "v18"
REASONING_ENVELOPE_TASKS = frozenset(
    {
        "reasoning_verification",
        "planning_comparison",
        "explanation_learning",
        "critique_revision",
        "troubleshooting",
    }
)
REASONING_ENVELOPE_ACTIVE_TASKS = frozenset(
    {
        "reasoning_verification",
        "planning_comparison",
        "troubleshooting",
    }
)
REASONING_ENVELOPE_FINAL_COUNTS = {
    "planning_comparison": 24,
}
_ENVELOPE = re.compile(
    r"\A<think>\n(?P<think>.+?)\n</think>\n<final>\n(?P<final>.+?)\n</final>\Z",
    flags=re.DOTALL,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"\d+(?:[.,/]\d+)*")


def _number_values(text: str) -> set[str]:
    """Return comparable numeric values, including fraction components.

    A fraction such as ``204/1008`` establishes both 204 and 1008. Treating
    the complete fraction as one opaque token made a final phrasing such as
    ``204 outcomes out of 1008`` look unsupported even though both values
    were explicitly present in the reasoning.
    """

    return {
        component
        for matched in _NUMBER.findall(text)
        for component in matched.split("/")
    }


@dataclass(frozen=True)
class ReasoningEnvelope:
    think: str
    final: str
    card_hand: str

    @property
    def text(self) -> str:
        return f"<think>\n{self.think}\n</think>\n<final>\n{self.final}\n</final>"


def parse_reasoning_envelope(text: str) -> ReasoningEnvelope | None:
    match = _ENVELOPE.fullmatch(text.strip())
    if match is None:
        return None
    think = match.group("think").strip()
    final = match.group("final").strip()
    if any(tag in think or tag in final for tag in ("<think>", "</think>", "<final>", "</final>")):
        return None
    return ReasoningEnvelope(think=think, final=final, card_hand="")


def _without_hand(text: str) -> str:
    return re.sub(
        r"^(?:For hand\s+[A-Z0-9]+\s*:\s*|Hand\s+[A-Z0-9]+\s*[—:-]\s*)",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def _inline(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip()).rstrip(" .")
    if re.match(r"A\b", value):
        return value
    return value[:1].lower() + value[1:] if value else value


def _sentence(text: str) -> str:
    value = re.sub(r"\s+", " ", text.strip())
    if not value:
        return ""
    value = value[:1].upper() + value[1:]
    return value if value.endswith((".", "?", "!")) else value + "."


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text.strip()) if part.strip()]


def _clean_total(text: str) -> str:
    value = re.sub(
        r"^(?:this gives|the result is|the computed value is|therefore,|"
        r"the supplied values produce|completing those operations produces|"
        r"after applying every stated quantity, the answer is|"
        r"the resulting quantity is|the final evaluated amount is)\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).rstrip(" .")
    # ``Total`` may append a second explanatory sentence containing fresh
    # notation (for example, a 100:1 unit identity).  The equation and check
    # belong in ``think``; ``final`` should carry only the direct result.
    sentences = _sentences(value)
    return (sentences[0] if sentences else value).rstrip(" .")


def _layout_variants(parts: list[str]) -> tuple[str, ...]:
    clean = [_sentence(part) for part in parts if part.strip()]
    if not clean:
        raise ValueError("reasoning envelope cannot render an empty final")
    if len(clean) == 1:
        # A bullet or numbered list with one item adds meaningless syntax and,
        # for calculations, can look like a new unsupported numeric value.
        return (clean[0],) * 6
    paragraph = " ".join(clean)
    lines = "\n".join(clean)
    spaced = "\n\n".join(clean)
    bullets = "\n".join(f"- {part}" for part in clean)
    numbered = "\n".join(f"{index}. {part}" for index, part in enumerate(clean, 1))
    compact = " ".join(part.rstrip(".") + ";" for part in clean[:-1])
    if clean[:-1]:
        compact += " " + clean[-1]
    else:
        compact = clean[0]
    return paragraph, lines, spaced, bullets, numbered, compact


def _fallback_analysis(metadata: dict[str, Any]) -> tuple[str, str]:
    state = str(metadata.get("source_state") or metadata.get("state") or "the stated condition")
    constraint = str(
        metadata.get("source_constraint")
        or metadata.get("constraint")
        or "the supplied evidence limits the conclusion"
    )
    desired = str(
        metadata.get("desired_outcome")
        or metadata.get("fallback")
        or "the result remains independently checkable"
    )
    return (
        _sentence(
            f"{state.rstrip('.')} while "
            f"{constraint[:1].lower() + constraint[1:].rstrip('.')}"
        ),
        _sentence(desired),
    )


def _reasoning_parts(
    source: str, final: str, metadata: dict[str, Any]
) -> tuple[str, str, tuple[str, ...]]:
    fields = _labelled_fields(_without_hand(source), ("Equation", "Total", "Check"))
    if set(fields) == {"Equation", "Total", "Check"}:
        equation = fields["Equation"]
        check = re.sub(
            r"^(?:independently,\s*|inspect the supplied values, then note that\s*|"
            r"use a second view of the values;\s*|a second view confirms that\s*|"
            r"verify the result by noting that\s*|"
            r"reversing or decomposing the operation shows that\s*|"
            r"an independent reconstruction confirms that\s*|"
            r"the quantities remain consistent because\s*|"
            r"a separate numerical route establishes that\s*)",
            "",
            fields["Check"],
            flags=re.IGNORECASE,
        )
        equation = _sentence(equation)
        check = _sentence(check)
        total = _clean_total(fields["Total"])
        final_patterns = (
            "The result is {total}", "{total} answers the calculation directly",
            "Therefore, the result is {total}", "This gives {total}",
            "{total} is the computed value", "The supplied values produce {total}",
            "The required quantity is {total}", "Calculation yields {total}",
            "The evaluated total comes to {total}", "So the requested value is {total}",
            "The resulting amount equals {total}", "{total} is the numerical answer",
            "Completing the operation gives {total}", "{total} is the final quantity",
            "From these values, {total} is the outcome", "The calculation resolves to {total}",
            "{total} is the supported total", "Altogether, this produces {total}",
            "The requested computation returns {total}", "{total} is the resulting value",
            "After evaluation, {total} answers the question", "The figures establish {total}",
            "The operation produces a total of {total}", "{total} is the final evaluated result",
            "Direct evaluation returns {total}", "{total} follows from the supplied operation",
            "The requested computation gives {total}", "{total} completes the calculation",
            "Evaluating the relation produces {total}", "{total} is established by the figures",
            "The quantities combine to give {total}", "{total} is the supported outcome",
        )
        finals = tuple(_sentence(pattern.format(total=total)) for pattern in final_patterns)
        return equation, check, finals
    parts = _sentences(final)
    first = parts[0] if parts else final
    direct = re.split(r"\s+because\s+", first, maxsplit=1, flags=re.IGNORECASE)[0]
    analysis = _sentence(first)
    verification = _sentence(
        parts[-1]
        if len(parts) > 1
        else metadata.get("desired_outcome", first)
    )
    return analysis, verification, _layout_variants([direct])


def _planning_parts(
    source: str, final: str, metadata: dict[str, Any]
) -> tuple[str, str, tuple[str, ...]]:
    response = _without_hand(source)
    match = re.fullmatch(
        r"(.*?)\s+Sequence:\s*(.*?)\s+Fallback trigger:\s*(.*)",
        response,
        flags=re.DOTALL,
    )
    if match is None:
        analysis, verification = _fallback_analysis(metadata)
        return analysis, verification, _layout_variants(_sentences(final))
    head, sequence, fallback = (part.strip() for part in match.groups())
    choice_prefix = (
        r"Choose\b|Select\b|Proceed with\b|Use Option\b|Pick\b|"
        r"A is\b|Option A\b|The viable\b|The comparison\b|Only A\b|"
        r"The hard gates\b|A survives\b|The compliant\b|The decision\b|"
        r"On the stated\b|After screening\b|The feasible\b|All binding\b|"
        r"A alone\b|The shortlist\b|Under the hard limits\b"
    )
    # A criteria sentence may itself begin with "A is ...".  The actual
    # choice is the last matching sentence before ``Sequence``, not the first.
    choice_starts = list(
        re.finditer(
            rf"(?:^|(?<=\.)\s+)(?=(?:{choice_prefix}))",
            head,
        )
    )
    choice_start = choice_starts[-1].end() if choice_starts else None
    choice = head[choice_start:].strip() if choice_start is not None else head
    criteria = head[:choice_start].strip() if choice_start is not None else head
    constraint = str(metadata.get("constraint") or metadata.get("source_constraint") or "the hard limits")
    constraint_surfaces = planning_constraint_surfaces(constraint)
    constraint = constraint_surfaces[
        int.from_bytes(
            hashlib.sha256(f"planning-boundary:{source}".encode()).digest()[:8],
            "big",
        )
        % len(constraint_surfaces)
    ]
    option_match = re.fullmatch(
        r"(?:Choose A:\s*(.*?)|Choose Option A,\s*(.*?),\s*as the compliant candidate|"
        r"Choose the viable option, A:\s*(.*?))\.?",
        choice,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if option_match is not None:
        option = next(group for group in option_match.groups() if group).rstrip(" .")
        choices = (
            f"Choose A: {option}",
            f"Select A: {option}",
            f"Proceed with A: {option}",
            f"Use Option A, {option}",
            f"Pick A, {option}, because it meets every hard constraint",
            f"A is the viable choice: {option}",
            f"A is the only compliant option: {option}",
            f"Option A meets the binding criteria: {option}",
            f"The viable option is A: {option}",
            f"The comparison leaves A as the choice: {option}",
            f"Only A remains eligible: {option}",
            f"The hard gates leave Option A: {option}",
            f"A survives every required check, so choose {option}",
            f"The compliant candidate is A, {option}",
            f"The decision is A: {option}",
            f"On the stated constraints, select A: {option}",
            f"After screening the options, use A: {option}",
            f"The feasible selection is A: {option}",
            f"All binding tests point to A: {option}",
            f"A alone satisfies the fixed requirements: {option}",
            f"The shortlist resolves to A: {option}",
            f"Under the hard limits, proceed with A: {option}",
            f"Screening identifies A as viable: {option}",
            f"The constraints support A: {option}",
        )
        final_variants = tuple(
            _layout_variants([variant, sequence, fallback])[index % 6]
            for index, variant in enumerate(choices)
        )
    else:
        final_variants = _layout_variants([choice, sequence, fallback])
    fallback_analysis, _fallback_verification = _fallback_analysis(metadata)
    return (
        _sentence(criteria) or fallback_analysis,
        _sentence(constraint),
        final_variants,
    )


def _explanation_parts(
    source: str, final: str, metadata: dict[str, Any]
) -> tuple[str, str, tuple[str, ...]]:
    fields = _labelled_fields(_without_hand(source), ("Core idea", "Example", "Check"))
    if set(fields) == {"Core idea", "Example", "Check"}:
        idea = _sentence(fields["Core idea"])
        example = _sentence(fields["Example"])
        check = _sentence(fields["Check"])
        variants = (
            f"{idea} {example} {check}",
            f"{idea}\n{example}\n{check}",
            f"{idea}\n\n{example}\n\n{check}",
            f"{example} {idea} {check}",
            f"{idea} {check} {example}",
            f"{check} {idea} {example}",
        )
        constraint = str(metadata.get("constraint") or metadata.get("source_constraint") or "the explanation must remain accurate")
        return idea, _sentence(constraint), variants
    analysis, verification = _fallback_analysis(metadata)
    return analysis, verification, _layout_variants(_sentences(final))


def _critique_parts(
    source: str, final: str, metadata: dict[str, Any]
) -> tuple[str, str, tuple[str, ...]]:
    fields = _labelled_fields(_without_hand(source), ("Weakness", "Revision"))
    if set(fields) == {"Weakness", "Revision"}:
        weakness = _sentence(fields["Weakness"])
        revision = fields["Revision"].strip().rstrip(" .")
        variants = (
            _sentence(revision),
            f"Use this narrower wording: {_sentence(revision)}",
            f"A grounded revision is: {_sentence(revision)}",
            f"The corrected version reads: {_sentence(revision)}",
            f"A narrower supported version is: {_sentence(revision)}",
            f"Revise the passage as follows: {_sentence(revision)}",
        )
        constraint = str(metadata.get("constraint") or metadata.get("source_constraint") or "the revision must preserve supported facts")
        return weakness, _sentence(constraint), variants
    marker = re.search(r"(?:A clearer version is|Use this revision):\s*(.*)$", final, re.DOTALL | re.IGNORECASE)
    if marker:
        weakness = final[: marker.start()].strip()
        revision = marker.group(1).strip().strip("'\"")
        return _sentence(weakness), _sentence(metadata.get("constraint", "Keep the revision grounded")), _layout_variants([revision])
    analysis, verification = _fallback_analysis(metadata)
    return analysis, verification, _layout_variants(_sentences(final))


def _troubleshooting_parts(
    source: str, final: str, metadata: dict[str, Any]
) -> tuple[str, str, tuple[str, ...]]:
    steps = [
        match.group(1).strip()
        for match in re.finditer(
            r"(?:^|\s)(?:\d+\.|[-*])\s*(.*?)(?=\s+(?:\d+\.|[-*])\s|$)",
            final,
            flags=re.DOTALL,
        )
    ]
    if not steps:
        steps = _sentences(final)
    fallback_analysis, fallback_verification = _fallback_analysis(metadata)
    analysis = _sentence(
        steps[1] if len(steps) > 1 else (steps[0] if steps else fallback_analysis)
    )
    verification_candidates = [
        sentence
        for sentence in _sentences(final)
        if re.search(
            r"\b(?:direct|regression|verify|checks?|baseline|control)\b",
            sentence,
            re.IGNORECASE,
        )
    ]
    verification = _sentence(
        verification_candidates[-1]
        if verification_candidates
        else fallback_verification
    )
    return analysis, verification, _layout_variants(steps)


_PARTS = {
    "reasoning_verification": _reasoning_parts,
    "planning_comparison": _planning_parts,
    "explanation_learning": _explanation_parts,
    "critique_revision": _critique_parts,
    "troubleshooting": _troubleshooting_parts,
}


def reasoning_envelope_card_hand(
    task: str,
    seed: str,
    final_count: int | None = None,
) -> str:
    if final_count is None:
        final_count = REASONING_ENVELOPE_FINAL_COUNTS.get(task, 6)
    material = reasoning_envelope_variable_by(
        task,
        analysis="Analysis.",
        analysis_inline="analysis",
        verification="Verification.",
        verification_inline="verification",
        final_variants=tuple(f"final-{index}" for index in range(final_count)),
    )
    indices = material.deal_indices(seed)
    return "|".join(
        (
            f"opening={indices['opening'][task]}",
            f"think={indices['think'][task]}",
            f"final={indices['final'][task]}",
        )
    )


def render_reasoning_envelope(
    *,
    task: str,
    source_response: str,
    natural_final: str,
    metadata: dict[str, Any],
    seed: str,
) -> ReasoningEnvelope | None:
    if task not in REASONING_ENVELOPE_TASKS:
        return None
    analysis, verification, finals = _PARTS[task](
        source_response,
        natural_final,
        metadata,
    )
    analysis = _sentence(analysis)
    verification = _sentence(verification)
    variable_by = reasoning_envelope_variable_by(
        task,
        analysis=analysis,
        analysis_inline=_inline(analysis),
        verification=verification,
        verification_inline=_inline(verification),
        final_variants=finals,
    )
    dealt = variable_by.deal(seed)
    think = re.sub(r"[ \t]+", " ", dealt["think"][task]).strip()
    final = dealt["final"][task].strip()
    hand = reasoning_envelope_card_hand(task, seed, len(finals))
    return ReasoningEnvelope(think=think, final=final, card_hand=hand)


def envelope_digest(envelope: ReasoningEnvelope) -> str:
    return hashlib.sha256(envelope.text.encode()).hexdigest()


def audit_reasoning_envelopes(
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
    target_key: str = "_projected_target",
    maximum_exact_think_share: float = 0.05,
) -> dict[str, Any]:
    """Validate V18 syntax, scope, length and per-family think diversity."""

    by_task: dict[str, list[ReasoningEnvelope]] = defaultdict(list)
    envelope_example_ids: dict[str, list[str]] = defaultdict(list)
    invalid: list[str] = []
    unexpected: list[str] = []
    missing: list[str] = []
    tags = ("<think>", "</think>", "<final>", "</final>")
    for row in rows:
        task = str(row["task"])
        target = str(row[target_key])
        envelope = parse_reasoning_envelope(target)
        if any(tag in target for tag in tags) and envelope is None:
            invalid.append(str(row["example_id"]))
            continue
        if envelope is None:
            if enabled and task in REASONING_ENVELOPE_ACTIVE_TASKS:
                missing.append(str(row["example_id"]))
            continue
        if task not in REASONING_ENVELOPE_ACTIVE_TASKS:
            unexpected.append(str(row["example_id"]))
            continue
        by_task[task].append(envelope)
        envelope_example_ids[task].append(str(row["example_id"]))

    task_audits: dict[str, Any] = {}
    diversity_failures: list[str] = []
    length_failures: list[str] = []
    generic_plan_failures: list[str] = []
    card_hand_failures: list[str] = []
    calculation_failures: list[str] = []
    calculation_failure_examples: list[dict[str, Any]] = []
    for task, envelopes in sorted(by_task.items()):
        think_counts = Counter(envelope.think.casefold() for envelope in envelopes)
        maximum_count = max(think_counts.values(), default=0)
        maximum_share = maximum_count / len(envelopes) if envelopes else 0.0
        think_words = [len(envelope.think.split()) for envelope in envelopes]
        final_words = [len(envelope.final.split()) for envelope in envelopes]
        length_ok = all(8 <= words <= 120 for words in think_words) and all(
            1 <= words <= 220 for words in final_words
        )
        diversity_ok = (
            len(envelopes) < 100 or maximum_share <= maximum_exact_think_share
        )
        generic_ok = all(
            "i should" not in envelope.think.casefold()
            and "chain of thought" not in envelope.think.casefold()
            for envelope in envelopes
        )
        task_rows = [row for row in rows if str(row["task"]) == task]
        hand_counts = Counter(
            reasoning_envelope_card_hand(task, str(row["example_id"]))
            for row in task_rows
            if parse_reasoning_envelope(str(row[target_key])) is not None
        )
        maximum_hand_count = max(hand_counts.values(), default=0)
        maximum_hand_share = (
            maximum_hand_count / len(envelopes) if envelopes else 0.0
        )
        card_hands_ok = (
            len(envelopes) < 100
            or maximum_hand_share <= maximum_exact_think_share
        )
        calculation_ok = True
        if task == "reasoning_verification":
            for example_id, envelope in zip(
                envelope_example_ids[task], envelopes, strict=True
            ):
                missing_numbers = sorted(
                    _number_values(envelope.final)
                    - _number_values(envelope.think)
                )
                if not missing_numbers:
                    continue
                calculation_ok = False
                if len(calculation_failure_examples) < 20:
                    calculation_failure_examples.append(
                        {
                            "example_id": example_id,
                            "missing_numbers": missing_numbers,
                            "think": envelope.think,
                            "final": envelope.final,
                        }
                    )
        if not length_ok:
            length_failures.append(task)
        if not diversity_ok:
            diversity_failures.append(task)
        if not generic_ok:
            generic_plan_failures.append(task)
        if not card_hands_ok:
            card_hand_failures.append(task)
        if not calculation_ok:
            calculation_failures.append(task)
        task_audits[task] = {
            "examples": len(envelopes),
            "distinct_think": len(think_counts),
            "maximum_exact_think_share": round(maximum_share, 6),
            "think_words_min": min(think_words, default=0),
            "think_words_max": max(think_words, default=0),
            "final_words_min": min(final_words, default=0),
            "final_words_max": max(final_words, default=0),
            "diversity_passed": diversity_ok,
            "length_passed": length_ok,
            "generic_plan_passed": generic_ok,
            "distinct_card_hands": len(hand_counts),
            "maximum_card_hand_share": round(maximum_hand_share, 6),
            "card_hand_diversity_passed": card_hands_ok,
            "calculation_consistency_passed": calculation_ok,
        }

    checks = {
        "all_tags_are_balanced": not invalid,
        "only_reasoning_tasks_use_envelopes": not unexpected,
        "every_reasoning_task_row_is_enveloped": not missing if enabled else True,
        "all_active_reasoning_tasks_are_present": (
            set(by_task) == set(REASONING_ENVELOPE_ACTIVE_TASKS)
            if enabled
            else True
        ),
        "think_exact_share_at_most_5_percent_per_family": not diversity_failures,
        "think_and_final_lengths_are_bounded": not length_failures,
        "no_generic_i_should_plans": not generic_plan_failures,
        "reasoning_card_hand_share_at_most_5_percent_per_family": (
            not card_hand_failures
        ),
        "reasoning_final_numbers_are_established_in_think": (
            not calculation_failures
        ),
    }
    return {
        "enabled": enabled,
        "version": REASONING_ENVELOPE_VERSION if enabled else None,
        "tasks": task_audits,
        "invalid_examples": invalid[:20],
        "unexpected_examples": unexpected[:20],
        "missing_examples": missing[:20],
        "calculation_failure_examples": calculation_failure_examples,
        "checks": checks,
        "passed": all(checks.values()),
    }
