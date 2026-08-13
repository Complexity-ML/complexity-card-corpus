from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool, prompt_variant_plans
from ..reservoirs.writing import WRITING_AUDIENCES, WRITING_CASES, WRITING_CHANNELS
from ._axes import PEOPLE
from ._common import render_v2_row, validate_complete_rows


TASK = "writing_transformation"

_TRANSFORMS = {
    "formal": "use professional language and make the request explicit",
    "concise": "retain the action, timing, recipient context, and reason",
    "friendly": "use a warm but unambiguous request",
    "plain_language": "use common words and short direct sentences",
}

_PROMPTS = (
    "Transform the source into {scenario[transform]} writing; {scenario[guidance]}. Source: {scenario[source]}",
    "Rewrite this message in a {scenario[transform]} style. Specifically, {scenario[guidance]}. Text: {scenario[source]}",
    "Preserve the facts while changing the expression to {scenario[transform]}; {scenario[guidance]}. Original: {scenario[source]}",
    "Produce a clean {scenario[transform]} version of the following. The transformation must {scenario[guidance]}. {scenario[source]}",
    "Edit this into {scenario[transform]} prose while keeping its operational meaning; {scenario[guidance]}. Draft: {scenario[source]}",
    "Give me the finished {scenario[transform]} message, not editing advice. Ensure you {scenario[guidance]}. Draft: {scenario[source]}",
    "Restate the message for direct use in a {scenario[transform]} register; {scenario[guidance]}. Message: {scenario[source]}",
    "Make this ready to send in a {scenario[transform]} form. Preserve every required fact and {scenario[guidance]}. {scenario[source]}",
)

_PROMPT_FUNCTIONS = (
    ("request_transformation", "specify_style", "supply_source"),
    ("request_rewrite", "specify_style", "supply_source"),
    ("preserve_facts", "request_expression_change", "supply_original"),
    ("request_clean_version", "specify_transformation", "supply_source"),
    ("request_edit", "preserve_meaning", "supply_draft"),
    ("request_finished_message", "forbid_meta_advice", "supply_draft"),
    ("request_restatement", "specify_register", "supply_message"),
    ("request_send_ready_message", "preserve_facts", "supply_source"),
)

_SOURCE_TEMPLATES = (
    "Hey {person}, quick thing about the {artifact}: could you maybe {action} sometime before {deadline}? Otherwise, {reason}. {audience_source} {channel_source} Thanks.",
    "Hi {person} — the {artifact} still needs a change. Please {action} by {deadline}, since {reason}. {channel_source} {audience_source}",
    "Message for {person}: I think we should {action} in the {artifact} before {deadline}. The reason is that {reason}. {audience_source} {channel_source}",
    "Could {person} take another look at the {artifact} and {action}? It needs to happen by {deadline} so {reason}. {channel_source} {audience_source}",
    "Just flagging the {artifact} for {person}. We need to {action} before {deadline}; that way, {reason}. {audience_source} {channel_source}",
    "{person}, there is one remaining item in the {artifact}: {action}. Please handle it by {deadline}, because {reason}. {channel_source} {audience_source}",
    "About the {artifact}, can you ask {person} to {action} no later than {deadline}? This matters because {reason}. {audience_source} {channel_source}",
    "Draft note to {person}: maybe {action} in the {artifact} before {deadline}. We need this so {reason}. {channel_source} {audience_source}",
)

_ANSWER_TEMPLATES = {
    "formal": (
        "{person}, please {action_answer} in the {artifact_answer} by {deadline_answer} {channel_answer}. This is necessary so {reason_answer}, {audience_answer}.",
        "Please {action_answer} in the {artifact_answer} by {deadline_answer}, {person}. The update will ensure {reason_answer} {audience_answer} when it appears {channel_answer}.",
        "{person}, the {artifact_answer} requires an update {channel_answer}: please {action_answer} by {deadline_answer}. This will ensure {reason_answer} {audience_answer}.",
        "By {deadline_answer}, please {action_answer} in the {artifact_answer}, {person}. That wording is needed {audience_answer} and will ensure {reason_answer} {channel_answer}.",
    ),
    "concise": (
        "{person}: {action_cap} in the {artifact_answer} by {deadline_answer} {channel_answer}. This ensures {reason_answer} {audience_answer}.",
        "By {deadline_answer}, {person} must {action_answer} in the {artifact_answer}. It keeps the message useful {audience_answer} {channel_answer} and ensures {reason_answer}.",
        "{person}, {action_answer} in the {artifact_answer} by {deadline_answer}. Send it {channel_answer} {audience_answer} so {reason_answer}.",
        "Required by {deadline_answer}: {action_answer} in the {artifact_answer}. {person} owns the change; it ensures {reason_answer} {audience_answer} {channel_answer}.",
    ),
    "friendly": (
        "Hi {person}, could you {action_answer} in the {artifact_answer} by {deadline_answer}? That will ensure {reason_answer} {audience_answer} when we share it {channel_answer}.",
        "Hello {person} — would you please {action_answer} in the {artifact_answer} by {deadline_answer}? It will make the message work {audience_answer} {channel_answer} and ensure {reason_answer}.",
        "Thanks for taking a look, {person}. Could you {action_answer} in the {artifact_answer} by {deadline_answer} so {reason_answer} {audience_answer} {channel_answer}?",
        "Hi {person}, one update remains: please {action_answer} in the {artifact_answer} by {deadline_answer}. This will help {audience_answer} and ensure {reason_answer} when it is posted {channel_answer}.",
    ),
    "plain_language": (
        "{person}, {action_answer} in the {artifact_answer} by {deadline_answer}. Put the result {channel_answer} {audience_answer} so {reason_answer}.",
        "The {artifact_answer} needs one change. {person} must {action_answer} by {deadline_answer}; then {reason_answer} {audience_answer} {channel_answer}.",
        "By {deadline_answer}, {person} should {action_answer} in the {artifact_answer}. This makes sure {reason_answer} {audience_answer} when sent {channel_answer}.",
        "{person} needs to {action_answer} in the {artifact_answer}. Finish by {deadline_answer} so {reason_answer} {audience_answer} {channel_answer}.",
    ),
}


def writing_transformation_capacity() -> int:
    return len(WRITING_CASES) * len(WRITING_AUDIENCES) * len(WRITING_CHANNELS) * len(_TRANSFORMS) * 2


def render_writing_transformation_rows() -> list[dict[str, object]]:
    rows = []
    for case_index, case in enumerate(WRITING_CASES):
        domain, artifact, artifact_answer, action, action_answer, deadline, deadline_answer, reason, reason_answer = case
        for audience_index, (audience_source, audience_answer) in enumerate(WRITING_AUDIENCES):
            for channel_index, (channel_source, channel_answer) in enumerate(WRITING_CHANNELS):
                person = PEOPLE[(case_index + audience_index + channel_index) % len(PEOPLE)]
                source_variant = (case_index + audience_index + channel_index) % len(_SOURCE_TEMPLATES)
                facts = {
                    "person": person,
                    "artifact": artifact,
                    "artifact_answer": artifact_answer,
                    "action": action,
                    "action_answer": action_answer,
                    "action_cap": action_answer[0].upper() + action_answer[1:],
                    "deadline": deadline,
                    "deadline_answer": deadline_answer,
                    "reason": reason,
                    "reason_answer": reason_answer,
                    "audience_source": audience_source,
                    "audience_answer": audience_answer,
                    "channel_source": channel_source,
                    "channel_answer": channel_answer,
                }
                source = _SOURCE_TEMPLATES[source_variant].format(**facts)
                for transform_index, (transform, base_guidance) in enumerate(_TRANSFORMS.items()):
                    for style_offset in range(2):
                        guidance = f"{base_guidance}; " + (
                            "return only the rewritten message"
                            if style_offset == 0
                            else "make the requested action easy to identify"
                        )
                        answer_variant = (audience_index + channel_index + transform_index + style_offset) % len(_ANSWER_TEMPLATES[transform])
                        target = _ANSWER_TEMPLATES[transform][answer_variant].format(**facts)
                        variables = RoleSeparatedVariableBy(
                            VariableBy2D(
                                {
                                    "scenario": {"transform": (transform,), "guidance": (guidance,), "source": (source,)},
                                    "prompt": {"transform_request": _PROMPTS},
                                    "answer": {"transformed_text": (target,)},
                                }
                            )
                        )
                        deck = V2RoleSeparatedDeck(
                            name=f"{TASK}:{domain}:{transform}:{answer_variant}",
                            variables=variables,
                            prompt_pools=(V2SubcardPool("transform_request", SurfaceRole.PROMPT, ("{prompt[transform_request]}",)),),
                            answer_pools=(V2SubcardPool("transformed_text", SurfaceRole.ANSWER, ("{answer[transformed_text]}",)),),
                            prompt_plans=prompt_variant_plans(
                                sense="transform_request",
                                pool_name="transform_request",
                                functions=_PROMPT_FUNCTIONS,
                            ),
                        )
                        case_id = f"{domain}:{case_index}:{audience_index}:{channel_index}:{transform}:{style_offset}"
                        rows.append(
                            render_v2_row(
                                task=TASK,
                                case_id=case_id,
                                domain=domain,
                                difficulty="easy",
                                deck=deck,
                                facts={**facts, "transform": transform, "source": source, "target": target, "source_variant": source_variant, "answer_variant": answer_variant},
                                validator={"kind": "exact", "expected": target},
                            )
                        )
    return validate_complete_rows(TASK, rows, writing_transformation_capacity())


__all__ = ("render_writing_transformation_rows", "writing_transformation_capacity")
