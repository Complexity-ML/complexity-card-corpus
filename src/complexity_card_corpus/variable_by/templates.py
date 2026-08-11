from __future__ import annotations


SAFETY_ANSWER_TEMPLATES = {
    "protective_action": (
        "{label[protective_action]}: {action[protective]} {grounding[action]}",
        "{label[protective_action]}: {action[protective]} — {grounding[action]}",
        "{label[protective_action]}: {grounding[action]} {action[protective]}",
    ),
    "boundary": (
        "{label[boundary]}: {boundary[verification]} {grounding[boundary]}",
        "{label[boundary]}: {boundary[verification]} — {grounding[boundary]}",
        "{label[boundary]}: {grounding[boundary]} {boundary[verification]}",
    ),
    "escalation_channel": (
        "{label[escalation]}: {channel[trusted]} {grounding[channel]}",
        "{label[escalation]}: {channel[trusted]} — {grounding[channel]}",
        "{label[escalation]}: {grounding[channel]} {channel[trusted]}",
    ),
}

EMPATHY_TEMPLATES = {
    "data": ("{label[source]}: \"{quote[person]}\"",),
    "goal": (
        "{goal[acknowledge]} {constraint[agency]} {constraint[question]}",
    ),
    "answer_grounding": (
        "{acknowledgment[domain]} {reflection[state]}",
        "{acknowledgment[domain]} — {reflection[state]}",
        "{reflection[state]} {acknowledgment[domain]}",
    ),
    "answer_agency": (
        "{agency[choice]} {question[optional]}",
        "{agency[choice]} — {question[optional]}",
        "{question[optional]} {agency[choice]}",
    ),
}

REASONING_TEMPLATES = {
    "data": (
        "{label[problem]}: {problem[statement]}",
        "{label[problem]} — {problem[statement]}",
        "{problem[statement]} — {label[problem]}.",
    ),
    "goal": (
        "{label[goal]}: {goal[instruction]} {constraint[verification]}",
        "{label[goal]} — {goal[instruction]} {constraint[verification]}",
        "{goal[instruction]} {constraint[verification]} — {label[goal]}.",
    ),
    "situation": (
        "{label[situation]}: {situation[calculation]}",
        "{label[situation]} — {situation[calculation]}",
        "{situation[calculation]} — {label[situation]}.",
    ),
    "calculation": (
        "{calculation[equation]} {calculation[total]}",
        "{calculation[equation]} — {calculation[total]}",
        "{calculation[total]} {calculation[equation]}",
    ),
    "verification": (
        "{verification[check]} {explanation[quantity_role]}",
        "{verification[check]} — {explanation[quantity_role]}",
        "{explanation[quantity_role]} {verification[check]}",
    ),
}

CRITIQUE_TEMPLATES = {
    "goal": (
        "{critique[diagnosis]} {critique[revision]} {constraint[evidence]}",
        "{critique[revision]} {constraint[evidence]} {critique[diagnosis]}",
        "{constraint[evidence]} {critique[diagnosis]} {critique[revision]}",
    ),
    "answer": (
        "{critique[weakness]} {critique[revised_text]}",
        "{critique[weakness]}\n{critique[revised_text]}",
        "{critique[weakness]}\n\n{critique[revised_text]}",
    ),
}

GROUNDED_QA_TEMPLATES = {
    "data": (
        "{label[source]} {source[code]}: {source[passage]}",
        "{label[source]} {source[code]} — {source[passage]}",
        "{source[passage]} — {label[source]} {source[code]}.",
    ),
    "goal": (
        "{label[request]}: {request[instruction]} {constraint[grounding]}",
        "{label[request]} — {constraint[grounding]} {request[instruction]}",
        "{request[instruction]} {constraint[grounding]} — {label[request]}.",
    ),
    "answer_scope": (
        "{label[evidence]}: {scope[answer]}",
        "{label[evidence]} — {scope[answer]}",
        "{scope[answer]} — {label[evidence]}.",
    ),
    "answer_complete": (
        "{label[documented]}: {answer[complete]}",
        "{label[documented]} — {answer[complete]}",
        "{answer[complete]} — {label[documented]}.",
    ),
    "situation": (
        "{label[situation]}: {situation[grounded]}",
        "{label[situation]} — {situation[grounded]}",
        "{situation[grounded]} — {label[situation]}.",
    ),
    "rule": (
        "{label[rule]}: {constraint[grounding]}",
        "{label[rule]} — {constraint[grounding]}",
        "{constraint[grounding]} — {label[rule]}.",
    ),
}
