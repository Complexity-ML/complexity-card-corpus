from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

MAXIMUM_ROLE_PHRASE_SHARE = 0.05
MAXIMUM_ROLE_SENTENCE_SHARE = 0.05
MAXIMUM_FUNCTION_PHRASE_SHARE = 0.15
MAXIMUM_FINAL_PHRASE_SHARE = 0.05
MAXIMUM_FINAL_SENTENCE_SHARE = 0.05
MAXIMUM_FINAL_SENTENCES = 3


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _SENTENCE.split(text.strip()) if part.strip())


def _normalized_sentence(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _phrases(text: str, words: int = 4) -> set[str]:
    tokens = _WORD.findall(text.lower())
    return {
        " ".join(tokens[index : index + words])
        for index in range(max(0, len(tokens) - words + 1))
    }


def _function_names(messages: list[dict[str, str]]) -> tuple[str, ...]:
    if len(messages) == 4:
        return (
            "user_opening",
            "assistant_entry",
            "user_follow_up",
            "assistant_closing",
        )
    if len(messages) == 6:
        return (
            "user_opening",
            "assistant_entry",
            "user_follow_up",
            "assistant_follow_up",
            "user_shift",
            "assistant_closing",
        )
    return tuple(f"turn_{index}_{message['role']}" for index, message in enumerate(messages))


def _largest(counter: Counter[str], denominator: int) -> dict[str, Any]:
    if not counter or denominator < 1:
        return {"value": "", "count": 0, "share": 0.0, "distinct": 0}
    value, count = min(counter.items(), key=lambda item: (-item[1], item[0]))
    return {
        "value": value,
        "count": count,
        "share": round(count / denominator, 6),
        "distinct": len(counter),
    }


def audit_casual_conversation_quality(
    rows: list[dict[str, Any]],
    *,
    task: str = "casual_conversation",
) -> dict[str, Any]:
    """Audit repetition and composition across every conversational role.

    Phrase and sentence counters use per-message support: repeating a phrase twice
    inside one message does not make that single training example count twice.
    Function-level statistics distinguish openings, follow-ups, and closings so a
    clean aggregate cannot hide a repetitive conversational stage.
    """

    selected = [row for row in rows if row.get("task") == task]
    if not selected:
        return {
            "task": task,
            "present": False,
            "examples": 0,
            "passed": False,
            "violations": ["casual_conversation is absent"],
            "thresholds": {
                "maximum_role_phrase_share": MAXIMUM_ROLE_PHRASE_SHARE,
                "maximum_role_sentence_share": MAXIMUM_ROLE_SENTENCE_SHARE,
                "maximum_function_phrase_share": MAXIMUM_FUNCTION_PHRASE_SHARE,
                "maximum_final_phrase_share": MAXIMUM_FINAL_PHRASE_SHARE,
                "maximum_final_sentence_share": MAXIMUM_FINAL_SENTENCE_SHARE,
                "maximum_final_sentences": MAXIMUM_FINAL_SENTENCES,
            },
        }

    role_messages: Counter[str] = Counter()
    role_phrases: dict[str, Counter[str]] = defaultdict(Counter)
    role_sentences: dict[str, Counter[str]] = defaultdict(Counter)
    function_messages: Counter[str] = Counter()
    function_phrases: dict[str, Counter[str]] = defaultdict(Counter)
    function_exact: dict[str, Counter[str]] = defaultdict(Counter)
    final_phrases: Counter[str] = Counter()
    final_sentences: Counter[str] = Counter()
    final_sentence_counts: Counter[int] = Counter()
    malformed_role_sequences = 0

    for row in selected:
        messages = row["messages"]
        expected_roles = ["user" if index % 2 == 0 else "assistant" for index in range(len(messages))]
        if len(messages) not in {4, 6} or [message["role"] for message in messages] != expected_roles:
            malformed_role_sequences += 1
        functions = _function_names(messages)
        for function, message in zip(functions, messages, strict=True):
            role = message["role"]
            text = message["content"]
            sentence_signatures = {
                signature
                for sentence in _sentences(text)
                if (signature := _normalized_sentence(sentence))
            }
            phrase_signatures = _phrases(text)
            role_messages[role] += 1
            role_phrases[role].update(phrase_signatures)
            role_sentences[role].update(sentence_signatures)
            function_messages[function] += 1
            function_phrases[function].update(phrase_signatures)
            function_exact[function][" ".join(text.lower().split())] += 1

        final = messages[-1]["content"]
        sentences = _sentences(final)
        final_sentence_counts[len(sentences)] += 1
        # Six-word spans catch memorized closing formulas without rejecting
        # ordinary English connective fragments such as "a useful way to".
        final_phrases.update(_phrases(final, words=6))
        final_sentences.update(
            signature
            for sentence in sentences
            if (signature := _normalized_sentence(sentence))
        )

    role_stats = {
        role: {
            "messages": count,
            "largest_four_word_phrase": _largest(role_phrases[role], count),
            "largest_exact_sentence": _largest(role_sentences[role], count),
        }
        for role, count in sorted(role_messages.items())
    }
    function_stats = {
        function: {
            "messages": count,
            "largest_four_word_phrase": _largest(function_phrases[function], count),
            "largest_exact_message": _largest(function_exact[function], count),
        }
        for function, count in sorted(function_messages.items())
    }
    final_count = len(selected)
    largest_final_phrase = _largest(final_phrases, final_count)
    largest_final_sentence = _largest(final_sentences, final_count)
    violations: list[str] = []
    if malformed_role_sequences:
        violations.append(f"{malformed_role_sequences} malformed role sequences")
    for role, stats in role_stats.items():
        if stats["largest_four_word_phrase"]["share"] > MAXIMUM_ROLE_PHRASE_SHARE:
            violations.append(f"{role} four-word phrase repetition exceeds 5%")
        if stats["largest_exact_sentence"]["share"] > MAXIMUM_ROLE_SENTENCE_SHARE:
            violations.append(f"{role} exact sentence repetition exceeds 5%")
    for function, stats in function_stats.items():
        if stats["largest_four_word_phrase"]["share"] > MAXIMUM_FUNCTION_PHRASE_SHARE:
            violations.append(f"{function} phrase repetition exceeds 15%")
    if largest_final_phrase["share"] > MAXIMUM_FINAL_PHRASE_SHARE:
        violations.append("final phrase repetition exceeds 5%")
    if largest_final_sentence["share"] > MAXIMUM_FINAL_SENTENCE_SHARE:
        violations.append("final sentence repetition exceeds 5%")
    if max(final_sentence_counts, default=0) > MAXIMUM_FINAL_SENTENCES:
        violations.append("a final response exceeds three sentences")

    return {
        "task": task,
        "present": True,
        "examples": len(selected),
        "passed": not violations,
        "violations": violations,
        "thresholds": {
            "maximum_role_phrase_share": MAXIMUM_ROLE_PHRASE_SHARE,
            "maximum_role_sentence_share": MAXIMUM_ROLE_SENTENCE_SHARE,
            "maximum_function_phrase_share": MAXIMUM_FUNCTION_PHRASE_SHARE,
            "maximum_final_phrase_share": MAXIMUM_FINAL_PHRASE_SHARE,
            "maximum_final_sentence_share": MAXIMUM_FINAL_SENTENCE_SHARE,
            "maximum_final_sentences": MAXIMUM_FINAL_SENTENCES,
        },
        "malformed_role_sequences": malformed_role_sequences,
        "final_sentence_counts": dict(sorted(final_sentence_counts.items())),
        "largest_final_six_word_phrase": largest_final_phrase,
        "largest_final_exact_sentence": largest_final_sentence,
        "roles": role_stats,
        "functions": function_stats,
    }
