from __future__ import annotations

import json
import re
from typing import Any

from ..training_cards import TrainingCards
from ..variable_by.reservoirs import (
    summary_decision_surfaces,
    troubleshooting_first_step_surfaces,
)
from .answer_development import develop_answer
from .language import (
    _final_assistant_target,
    _inline_sentence,
    _labelled_fields,
    _sentence,
)
from .response_cards import card_variant, render_response_card_hand


def _response_phrase(
    cards: TrainingCards,
    choices: tuple[str, ...],
    *,
    offset: int,
) -> str:
    return choices[card_variant(cards, len(choices), offset=offset)]


def _clean_prefix(value: str, pattern: str) -> str:
    return re.sub(pattern, "", value.strip(), flags=re.IGNORECASE).strip()


def _canonicalize_json_keys(value: Any) -> Any:
    """Normalize model-facing JSON keys without changing source values."""

    if isinstance(value, list):
        return [_canonicalize_json_keys(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        canonical_key = str(key).casefold()
        if canonical_key in normalized:
            raise ValueError(
                f"JSON target contains colliding keys after normalization: {key!r}"
            )
        normalized[canonical_key] = _canonicalize_json_keys(item)
    return normalized


def _canonicalize_json_target(response: str) -> str:
    """Return one compact, consistently cased JSON training contract."""

    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return response
    if not isinstance(payload, dict):
        return response
    return json.dumps(
        _canonicalize_json_keys(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _split_clarification_response(
    response: str,
) -> tuple[str, str, str] | None:
    """Recover the three authored clauses without exposing their labels."""

    question_end = response.find("?")
    if question_end < 0:
        return None
    default = response[question_end + 1 :].strip()
    before_default = response[: question_end + 1].strip()
    marker_match = re.search(
        r"(?:One point to resolve|Before proceeding):\s*",
        before_default,
        flags=re.IGNORECASE,
    )
    if marker_match is not None:
        question_start = marker_match.start()
    else:
        sentence_break = before_default.rfind(". ", 0, question_end)
        if sentence_break < 0:
            return None
        question_start = sentence_break + 2
    restatement = before_default[:question_start].strip()
    question = before_default[question_start:].strip()
    restatement = _clean_prefix(
        restatement,
        r"^(?:Understood|My current reading|What is clear|The supported interpretation is limited):\s*",
    )
    question = _clean_prefix(
        question,
        r"^(?:One point to resolve|Before proceeding):\s*",
    )
    default = _clean_prefix(
        default,
        r"^(?:Until confirmed|For now|Pending that answer|As a reversible default),\s*",
    )
    if not all((restatement, question, default)):
        return None
    return restatement, question, default


def _split_empathy_response(response: str) -> dict[str, str] | None:
    """Recover authored empathy roles while preserving exactly one question."""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", response.strip())
        if sentence.strip()
    ]
    if len(sentences) < 3 or not sentences[-1].endswith("?"):
        return None
    body = sentences[:-1]
    clauses = {
        "acknowledgment": _sentence(body[0]),
    }
    if len(body) > 2:
        clauses["state_reflection"] = " ".join(
            _sentence(sentence) for sentence in body[1:-1]
        )
    clauses["agency"] = _sentence(body[-1])
    clauses["question"] = _sentence(sentences[-1])
    return clauses


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
    if task == "explanation_learning":
        fields = _labelled_fields(response, ("Core idea", "Example", "Check"))
        if set(fields) == {"Core idea", "Example", "Check"}:
            idea = re.sub(
                r"^(?:in plain terms,\s*|the key distinction is that\s+)",
                "",
                fields["Core idea"],
                flags=re.IGNORECASE,
            )
            idea = idea.strip().rstrip(" .")
            example = fields["Example"].strip().rstrip(" .")
            example = re.sub(
                r"\s+(?:This applies the distinction directly|This turns the definition into a checkable case|The example makes the mechanism visible)\.?",
                "",
                example,
                flags=re.IGNORECASE,
            )
            check = _clean_prefix(
                fields["Check"],
                r"^(?:as a transfer test,\s*|to verify the idea,\s*)",
            )
            clauses = {
                "idea": _response_phrase(
                    cards,
                    (_sentence(idea),) * 6,
                    offset=1,
                ),
                "example": _response_phrase(
                    cards,
                    (_sentence(example),) * 6,
                    offset=2,
                ),
                "check": _response_phrase(
                    cards,
                    (_sentence(check),) * 5,
                    offset=3,
                ),
            }
            return render_response_card_hand(
                clauses,
                cards=cards,
            )
    elif task == "reasoning_verification":
        fields = _labelled_fields(response, ("Equation", "Total", "Check"))
        if set(fields) == {"Equation", "Total", "Check"}:
            equation = _clean_prefix(
                fields["Equation"],
                r"^(?:using the supplied values,\s*|the direct calculation is\s+|represent the required operation as\s+|evaluating the quantities gives\s+|the numerical relation is\s+|mapping each supplied quantity to its role gives\s+|following the stated order of operations yields\s+|the quantities combine in this form:\s*|a direct numerical model of the prompt is\s+)",
            )
            total = _clean_prefix(
                fields["Total"],
                r"^(?:this gives\s+|the result is\s+|the computed value is\s+|therefore,\s*|the supplied values produce\s+|completing those operations produces\s+|after applying every stated quantity, the answer is\s+|the resulting quantity is\s+|the final evaluated amount is\s+)",
            )
            check = _clean_prefix(
                fields["Check"],
                r"^(?:independently,\s*|inspect the supplied values, then note that\s*|use a second view of the values;\s*|a second view confirms that\s*|verify the result by noting that\s*|reversing or decomposing the operation shows that\s*|an independent reconstruction confirms that\s*|the quantities remain consistent because\s*|a separate numerical route establishes that\s*)",
            )
            check = re.sub(
                r"\bA occupies slot (\d+), immediately before B at slot (\d+)",
                r"slot \1 is occupied by A, immediately before B at slot \2",
                check,
            )
            if cards.response_bridge in {"compact", "plain"}:
                # These two response cards intentionally provide the direct
                # length band. Keep the three required semantic roles while
                # omitting optional explanatory tails after the first sentence
                # or authored em-dash expansion.
                concise_equation = re.split(
                    r"(?<=[.!?])\s+", equation, maxsplit=1
                )[0]
                concise_total = re.split(
                    r"(?<=[.!?])\s+", total, maxsplit=1
                )[0]
                concise_check = re.split(r"\s+—\s+", check, maxsplit=1)[0]
                concise_check = re.split(
                    r"(?<=[.!?])\s+", concise_check, maxsplit=1
                )[0]
                concise_clauses = {
                    "equation": _sentence(concise_equation),
                    "total": _sentence(concise_total),
                    "check": _sentence(concise_check),
                }
                return render_response_card_hand(
                    concise_clauses,
                    cards=cards,
                )
            clauses = {
                "equation": _response_phrase(
                    cards,
                    (
                        _sentence(equation),
                        f"Start with {_inline_sentence(equation)}",
                        f"The calculation is {_inline_sentence(equation)}",
                        f"Evaluating the quantities gives {_inline_sentence(equation)}",
                        f"The numerical relation is {_inline_sentence(equation)}",
                    ),
                    offset=11,
                ),
                "total": _response_phrase(
                    cards,
                    (
                        f"The result is {_inline_sentence(total)}",
                        f"That gives {_inline_sentence(total)}",
                        f"Therefore, {_inline_sentence(total)}",
                        f"So the answer is {_inline_sentence(total)}",
                        _sentence(total),
                    ),
                    offset=12,
                ),
                "check": _response_phrase(
                    cards,
                    (
                        f"As an independent check, {_inline_sentence(check)}",
                        f"A second calculation confirms that {_inline_sentence(check)}",
                        f"This is consistent because {_inline_sentence(check)}",
                        f"To verify the result, note that {_inline_sentence(check)}",
                        _sentence(check),
                    ),
                    offset=13,
                ),
            }
            return render_response_card_hand(
                clauses,
                cards=cards,
            )
    elif task == "summarization_synthesis":
        fields = _labelled_fields(response, ("Decision", "Action", "Open point"))
        if set(fields) == {"Decision", "Action", "Open point"}:
            decision = _clean_prefix(
                fields["Decision"],
                r"^(?:the record is to\s+|proceed by choosing to\s+|the agreed direction is to\s+)",
            ).rstrip(" .")
            action = fields["Action"].strip().rstrip(" .")
            open_point = fields["Open point"].strip().rstrip(" .")
            owned_action = re.fullmatch(
                r"(.+), owned by ([A-Z][A-Za-z'-]+), is due by day (\d+)",
                action,
            )
            if owned_action is not None:
                work, owner, day = owned_action.groups()
                action = f"{owner} will {work[:1].lower() + work[1:]} by day {day}"
            owner_owns = re.fullmatch(
                r"([A-Z][A-Za-z'-]+) owns (.+) for day (\d+)",
                action,
            )
            if owner_owns is not None:
                owner, work, day = owner_owns.groups()
                action = f"{owner} will {work[:1].lower() + work[1:]} by day {day}"
            clauses = {
                "decision": _response_phrase(
                    cards,
                    summary_decision_surfaces(_inline_sentence(decision)),
                    offset=21,
                ),
                "action": _response_phrase(
                    cards,
                    (
                        _sentence(action),
                        f"Next, {_inline_sentence(action)}",
                        _sentence(action),
                        f"For execution, {_inline_sentence(action)}",
                    ),
                    offset=22,
                ),
                "open_point": _response_phrase(
                    cards,
                    (
                        _sentence(open_point),
                        _sentence(open_point),
                        _sentence(open_point),
                        _sentence(open_point),
                    ),
                    offset=23,
                ),
            }
            return render_response_card_hand(
                clauses,
                cards=cards,
            )
    elif task == "grounded_qa":
        direct = response
        # Evidence-scope labels and source bookkeeping are generator controls,
        # not useful answer content.  Remove that leading clause completely so
        # the model learns the documented result and its evidence boundary.
        direct = re.sub(
            r"^(?:Evidence scope|Grounding|Source basis)\s*[:—-]\s*"
            r"[^.!?]*(?:\.|$)\s*",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"^(?:Documented answer|Supported finding|Known result)\s*"
            r"[:—-]\s*",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\s*[-—]\s*(?:Evidence scope|Grounding|Source basis|"
            r"Documented answer|Supported finding|Known result|Corroborating fact|"
            r"Additional evidence|Related fact|Unknown boundary|Evidence limit|"
            r"Unsupported field)\.(?=\s|$)",
            " ",
            direct,
            flags=re.IGNORECASE,
        ).strip()
        wrapper = re.compile(
            r"^(?:Based on Source [A-Za-z0-9]+:|Source [A-Za-z0-9]+ supports this answer:|According to Source [A-Za-z0-9]+:|The supplied record establishes this:|Supported facts:|The documented answer is:)\s*",
            flags=re.IGNORECASE,
        )
        while True:
            unwrapped = wrapper.sub("", direct)
            if unwrapped == direct:
                break
            direct = unwrapped
        direct = re.sub(
            r"^(?=[^.!?]*\bSource [A-Za-z0-9]+\b)"
            r"(?=[^.!?]*(?:support|evidence|ground|document|fact|record|"
            r"answer|result|finding|response|content|portion|conclusion))"
            r"[^.!?]*(?:\.|$)\s*",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"^(?:Documented answer|Supported finding|Known result)\s*"
            r"[:—-]\s*",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\s+(?:This is|The answer remains) limited to Source [A-Za-z0-9]+\.?$",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", direct.strip())
            if sentence.strip()
        ]
        if len(sentences) >= 2:
            clauses = {
                "documented": sentences[0],
                "boundary": sentences[1],
                "verification": " ".join(sentences[2:]),
            }
            return render_response_card_hand(clauses, cards=cards)
        return direct
    elif task == "critique_revision":
        fields = _labelled_fields(response, ("Weakness", "Revision"))
        if set(fields) == {"Weakness", "Revision"}:
            weakness_text = re.sub(
                r"\s+(?:Faithful|Bounded)\s*$",
                "",
                fields["Weakness"],
                flags=re.IGNORECASE,
            )
            weakness_text = re.sub(
                r",?\s*which makes the original difficult to verify\.?$",
                "",
                weakness_text,
                flags=re.IGNORECASE,
            )
            weakness_text = re.sub(
                r"\s+The revision must stay within the recorded facts\.?$",
                "",
                weakness_text,
                flags=re.IGNORECASE,
            )
            weakness = weakness_text.strip().rstrip(" .")
            revision = fields["Revision"].strip().rstrip(" .")
            clauses = {
                "revision": _response_phrase(
                    cards,
                    (_sentence(revision),) * 4,
                    offset=31,
                ),
                "weakness": _response_phrase(
                    cards,
                    (_sentence(weakness),) * 4,
                    offset=32,
                ),
            }
            return render_response_card_hand(
                clauses,
                cards=cards,
            )
    elif task == "safety_uncertainty":
        match = re.fullmatch(
            r"(?:Immediate action|Protective step|First safeguard)\s*(?::|—)\s*(.*?)\s+"
            r"(?:Boundary|Safety boundary|Verification limit)\s*(?::|—)\s*(.*?)\s+"
            r"(?:(?:Escalation|Trusted channel|Next contact)\s*(?::|—)\s*(.*)"
            r"|(Escalate\b.*))",
            response,
        )
        if match is not None:
            action = match.group(1).strip().rstrip(" .")
            boundary = match.group(2).strip().rstrip(" .")
            escalation = (match.group(3) or match.group(4)).strip().rstrip(" .")
            clauses = {
                "action": _response_phrase(
                    cards,
                    (_sentence(action),) * 6,
                    offset=41,
                ),
                "boundary": _response_phrase(
                    cards,
                    (_sentence(boundary),) * 4,
                    offset=42,
                ),
                "escalation": _response_phrase(
                    cards,
                    (_sentence(escalation),) * 4,
                    offset=43,
                ),
            }
            return render_response_card_hand(
                clauses,
                cards=cards,
            )
    elif task == "practical_action":
        fields = _labelled_fields(response, ("Next step", "Owner", "Timing"))
        if set(fields) == {"Next step", "Owner", "Timing"}:
            step = _clean_prefix(
                fields["Next step"].strip().rstrip(" ."),
                r"^(?:the next workable move is to\s+|start by choosing to\s+|start by\s+|first,\s*)",
            )
            owner = fields["Owner"].strip().rstrip(" .")
            timing = fields["Timing"].strip().rstrip(" .")
            if timing.lower().startswith("before "):
                timing = "complete this " + timing
            timing = re.sub(
                r"(until\s+the\s+[^.]*[,;][^.]*\band\b[^.]*?)\s+is recorded\b",
                r"\1 are recorded",
                timing,
                flags=re.IGNORECASE,
            )
            verified_prefix = timing.partition(" is verified")[0]
            if "," in verified_prefix and " and " in verified_prefix:
                timing = timing.replace(" is verified", " are verified", 1)
            clauses = {
                "step": _response_phrase(
                    cards,
                    (_sentence(step),) * 8,
                    offset=51,
                ),
                "owner": _response_phrase(
                    cards,
                    (_sentence(owner),) * 3,
                    offset=52,
                ),
                "timing": _response_phrase(
                    cards,
                    (_sentence(timing),) * 3,
                    offset=53,
                ),
            }
            return render_response_card_hand(
                clauses,
                cards=cards,
            )
        direct = re.sub(
            r"\bKeep the rationale attached to the action:\s*([a-z])",
            lambda match: (
                "Keep the rationale attached to this choice. "
                + match.group(1).upper()
            ),
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\bUse this choice:\s*([a-z])",
            lambda match: "Use this choice. " + match.group(1).upper(),
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\bThen complete the concrete next step:\s*",
            "Then ",
            direct,
            flags=re.IGNORECASE,
        )
        return direct
    elif task == "context_clarification":
        clauses = _split_clarification_response(response)
        if clauses is not None:
            restatement, question, default = clauses
            rendered = {
                "restatement": _response_phrase(
                    cards,
                    (_sentence(restatement),) * 6,
                    offset=81,
                ),
                "question": _response_phrase(
                    cards,
                    (_sentence(question),) * 5,
                    offset=82,
                ),
                "default": _response_phrase(
                    cards,
                    (_sentence(default),) * 6,
                    offset=83,
                ),
            }
            return render_response_card_hand(rendered, cards=cards)
        fields = _labelled_fields(response, ("Understood",))
        if fields:
            return _sentence(fields["Understood"])
        direct = re.sub(
            r"^(?:My current reading|What is clear|The supported interpretation is limited):\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\b(?:One point to resolve|Before proceeding):\s*",
            _response_phrase(
                cards,
                (
                    "One detail would resolve this: ",
                    "Before proceeding, ",
                    "The remaining question is: ",
                    "Please clarify one point: ",
                ),
                offset=82,
            ),
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\b(?:Until confirmed|For now|Pending that answer|As a reversible default),\s*",
            _response_phrase(
                cards,
                (
                    "Until that is confirmed, ",
                    "For now, ",
                    "Pending the answer, ",
                    "The reversible default is to ",
                ),
                offset=83,
            ),
            direct,
            flags=re.IGNORECASE,
        )
        return direct
    elif task == "brainstorming_creativity":
        # Remove only the generic audit sentences themselves.  The previous
        # tail-anchored expression removed everything after one of these
        # sentences, including the actual comparison and selection whenever
        # response-card ordering placed them later in the answer.
        direct = re.sub(
            r"(?:^|\s+)(?:Each description states|"
            r"The three retained(?: [A-Za-z][A-Za-z0-9_-]*){0,4} "
            r"ideas remain feasible|"
            r"(?:The )?[^.!?]*options remain bounded by the explicit brief|"
            r"Each retained option satisfies the stated criteria|"
            r"The selected [^.!?]*option is concrete enough to test first)"
            r"[^.!?]*(?:[.!?]|$)",
            " ",
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"^(?:Candidate set|Options|Possible directions):\s*",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\b(?:Criteria review|Constraint review|Fit with the brief|"
            r"Outcome review|Comparison result|Practical result):\s*",
            "",
            direct,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\bSelect (?:this option|the strongest fit):\s*",
            "Select ",
            direct,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", direct).strip()
    elif task == "writing_transformation":
        direct = re.sub(
            r"^(?:Here is the revised text|The concise version is):\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"^(?:Support reply|Project update|Internal note|Public notice|Short brief)\s+[A-Z0-9]+:\s*",
            "",
            direct,
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
    elif task == "conversation_empathy":
        clauses = _split_empathy_response(response)
        if clauses is not None:
            return render_response_card_hand(clauses, cards=cards)
        return response
    elif task == "extraction_classification":
        return _canonicalize_json_target(response)
    elif task == "planning_comparison":
        match = re.fullmatch(
            r"(.*?)\s+Sequence:\s*(.*?)\s+Fallback trigger:\s*(.*)",
            response,
            flags=re.DOTALL,
        )
        if match is not None:
            head = match.group(1).strip()
            criteria_match = re.match(r"(.*?)(Choose\b.*)$", head, flags=re.DOTALL)
            criteria = criteria_match.group(1).strip() if criteria_match else ""
            choice = criteria_match.group(2).strip() if criteria_match else head
            sequence = match.group(2).strip().rstrip(" .")
            fallback = match.group(3).strip().rstrip(" .")
            reject_variants = (
                "Rule out {option} because",
                "{option} is not viable because",
                "{option} fails the constraints because",
                "The criteria eliminate {option} because",
                "Exclude {option} because",
                "{option} does not qualify because",
            )
            reject_pattern = reject_variants[
                card_variant(cards, len(reject_variants), offset=60)
            ]
            criteria = re.sub(
                r"\bReject ([A-Z]) because\b",
                lambda match: reject_pattern.format(option=match.group(1)),
                criteria,
            )
            budget_variants = (
                ("B fails the budget test", "C fails both"),
                ("B is over budget", "C misses both"),
                ("The budget rules out B", "The remaining constraints rule out C on both"),
                ("Option B exceeds the budget", "Option C violates both"),
                ("B does not meet the budget", "C does not satisfy both"),
                ("The cost test eliminates B", "Two constraints eliminate C:"),
            )
            budget_b, budget_c = budget_variants[
                card_variant(cards, len(budget_variants), offset=65)
            ]
            criteria = re.sub(
                r"\bB fails the budget test\b",
                budget_b,
                criteria,
            )
            criteria = re.sub(
                r"\bC fails both\b",
                budget_c,
                criteria,
            )
            hard_constraint_variants = (
                "The hard constraints remove",
                "The requirements rule out",
                "Constraint checks eliminate",
                "The comparison excludes",
                "The non-negotiable limits remove",
                "Applying every hard limit removes",
            )
            criteria = re.sub(
                r"\bThe hard constraints remove\b",
                hard_constraint_variants[
                    card_variant(cards, len(hard_constraint_variants), offset=66)
                ],
                criteria,
            )
            clauses = {
                "criteria": _response_phrase(
                    cards,
                    (
                        _sentence(criteria),
                        _sentence(criteria),
                        _sentence(criteria),
                    ),
                    offset=61,
                ) if criteria else "",
                "choice": _response_phrase(
                    cards,
                    (
                        _sentence(choice),
                        f"On that basis, {_inline_sentence(choice)}",
                        f"The viable choice is clear. {_sentence(choice)}",
                    ),
                    offset=62,
                ),
                "sequence": _response_phrase(
                    cards,
                    (
                        _sentence(sequence),
                        f"Then {_inline_sentence(sequence)}",
                        f"Use this order: {_sentence(sequence)}",
                    ),
                    offset=63,
                ),
                "fallback": _response_phrase(
                    cards,
                    (
                        _sentence(fallback),
                        f"If the plan fails, {_inline_sentence(fallback)}",
                        f"The fallback condition is simple. {_sentence(fallback)}",
                    ),
                    offset=64,
                ),
            }
            return render_response_card_hand(
                clauses,
                cards=cards,
            )
        return response
    elif task == "troubleshooting":
        # ``check:`` is an authoring label elsewhere in the corpus, but this
        # family also used it inside the otherwise natural phrase "perform
        # this check:". Keep the meaning while ensuring the model-facing
        # target cannot teach the same visible rubric token.
        direct = re.sub(
            r"\bperform this check:\s*",
            "perform this test: ",
            response,
            flags=re.IGNORECASE,
        )
        direct = re.sub(
            r"\bDirect check:\s*(?:confirm that\s*)?",
            "Confirm that ",
            direct,
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
            rendered_steps = []
            for index, step in enumerate(steps):
                repeated_environment = re.match(
                    r"^In an isolated test environment,\s*perform this test:\s*(.*)$",
                    step,
                    flags=re.IGNORECASE,
                )
                if (
                    repeated_environment is not None
                    and "in an isolated test environment"
                    in repeated_environment.group(1).lower()
                ):
                    step = repeated_environment.group(1)
                if index == 0:
                    rendered_steps.append(
                        _response_phrase(
                            cards,
                            tuple(
                                _sentence(surface)
                                for surface in troubleshooting_first_step_surfaces(
                                    _inline_sentence(step)
                                )
                            ),
                            offset=71,
                        )
                    )
                elif index and cards.response_bridge == "stepwise":
                    rendered_steps.append(f"Next, {_inline_sentence(step)}")
                else:
                    rendered_steps.append(_sentence(step))
            if cards.response_layout == "paragraph":
                return " ".join(rendered_steps)
            if cards.response_layout == "line_breaks":
                return "\n".join(rendered_steps)
            if cards.response_layout == "spaced_lines":
                return "\n\n".join(rendered_steps)
            return "\n".join(
                (
                    f"{index}. {text}"
                    if cards.response_layout == "numbered"
                    else f"- {text}"
                )
                for index, text in enumerate(rendered_steps, start=1)
            )
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
    """Preserve an already-authored answer without generic padding.

    Earlier releases appended one generic resolution paragraph to every task
    from a shared cross-family reservoir. That taught repetition: the same
    closing phrases recurred across otherwise-diverse prompts because only
    ``subject`` varied between draws. Short answers are now developed inside
    the task-specific generators in ``tasks/knowledge.py``, ``tasks/reasoning.py``,
    and ``tasks/communication.py``, where real per-scenario facts already live,
    instead of here.
    """

    return develop_answer(
        target,
        task=task,
        metadata=metadata,
        example_id=example_id,
    )
