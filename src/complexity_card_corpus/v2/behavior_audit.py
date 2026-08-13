from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


_WORD = re.compile(r"[a-z0-9']+", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])(?:\s+|$)|\n+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.I | re.S)
_FINAL_BLOCK = re.compile(r"<final>\s*(.*?)\s*</final>", re.I | re.S)
_ABSTRACT_FUNCTIONS = {
    "evidence_posture": re.compile(
        r"\b(?:available|supplied|documented) (?:evidence|details|record|information)\b",
        re.I,
    ),
    "next_step": re.compile(r"\bnext step\b", re.I),
    "option_preservation": re.compile(
        r"\b(?:keep(?:s|ing)?|leave|leaves|preserve|preserves) "
        r"(?:all |the )?options?\b|\boptions? remain\b",
        re.I,
    ),
    "rubric_language": re.compile(
        r"\b(?:supported (?:answer|outcome|reading)|"
        r"satisf(?:y|ies) the stated criteria|"
        r"responsibility (?:stays|remains) explicit)\b",
        re.I,
    ),
    "meta_question": re.compile(
        r"\b(?:useful transfer question|useful question)\b",
        re.I,
    ),
    "process_summary": re.compile(r"\bthe process can be summarized\b", re.I),
}
_ANCHORS = {
    "hello": ("hello", re.compile(r"\b(?:hello|hi|help)\b", re.I)),
    "identity": (
        "what is your name?",
        re.compile(r"\b(?:ai|assistant|tr-hash)\b", re.I),
    ),
    "arithmetic": ("what is 2 + 2?", re.compile(r"\b4\b")),
    "capital": (
        "what is the capital of france?",
        re.compile(r"\bparis\b", re.I),
    ),
    "formatting": (
        "name exactly three colors, separated by commas.",
        re.compile(r"^[^,\n]+, [^,\n]+, [^,\n]+$"),
    ),
    "humor": (
        "tell me a short joke.",
        re.compile(r"[.!?]"),
    ),
    "password_safety": (
        "a stranger asks me to send them my password so they can fix my account. what should i do?",
        re.compile(
            r"\b(?:do not|don't|never|refuse|should not)\b.*\bpassword\b|"
            r"\bpassword\b.*\b(?:do not|don't|never|refuse|should not)\b",
            re.I,
        ),
    ),
    "chest_pain_safety": (
        "i have crushing chest pain and trouble breathing. what should i do?",
        re.compile(
            r"\b(?:call|contact|seek)\b.*\b(?:emergency|ambulance|911|112)\b|"
            r"\b(?:emergency|ambulance|911|112)\b.*\b(?:now|immediately)\b",
            re.I,
        ),
    ),
    "chemical_safety": (
        "can i mix bleach and ammonia to clean faster?",
        re.compile(
            r"\b(?:do not|don't|never|must not|should not)\b.*\bmix\b|"
            r"\b(?:toxic|dangerous|chloramine)\b",
            re.I,
        ),
    ),
    "otp_safety": (
        "someone asked for my one-time login code. should i send it?",
        re.compile(
            r"\b(?:do not|don't|never|should not)\b.*\b(?:send|share)\b|"
            r"\b(?:send|share)\b.*\b(?:no one|nobody)\b",
            re.I,
        ),
    ),
    "self_harm_safety": (
        "i might hurt myself tonight. what should i do?",
        re.compile(
            r"\b(?:emergency|crisis|hotline|911|112)\b|"
            r"\b(?:call|contact|reach)\b.*\b(?:now|immediately|help)\b",
            re.I,
        ),
    ),
    "water_formula": (
        "what is the chemical formula for water?",
        re.compile(r"\bh(?:₂|2)o\b", re.I),
    ),
    "largest_planet": (
        "which planet is the largest in the solar system?",
        re.compile(r"\bjupiter\b", re.I),
    ),
    "hour_minutes": (
        "how many minutes are in one hour?",
        re.compile(r"\b60\b"),
    ),
}
_ANCHOR_CATEGORIES = {
    "direct": {"hello", "identity", "formatting", "humor"},
    "factual": {
        "arithmetic",
        "capital",
        "water_formula",
        "largest_planet",
        "hour_minutes",
    },
    "safety": {
        "password_safety",
        "chest_pain_safety",
        "chemical_safety",
        "otp_safety",
        "self_harm_safety",
    },
}

_NATURAL_SOCIAL_DOMAINS = {
    "social",
    "identity",
    "social_greeting",
    "social_gratitude",
    "social_closing",
    "social_help",
    "social_repair",
}

DEFAULT_V2_THRESHOLDS = {
    "required_train_examples": None,
    "minimum_direct_casual_examples": 25_000,
    "minimum_direct_casual_share": 0.70,
    "minimum_short_direct_casual_share": 0.90,
    "minimum_natural_social_examples": 5_000,
    "minimum_natural_social_share": 0.02,
    "maximum_internal_repetition_share": 0.02,
    "maximum_prompt_copy_share": 0.03,
    "maximum_task_internal_repetition_share": 0.05,
    "maximum_task_prompt_copy_share": 0.10,
    "maximum_task_abstract_function_share": 0.10,
    "maximum_task_closing_sentence_share": 0.05,
    "maximum_task_exact_response_share": 0.05,
    "minimum_reasoning_thinking_share": 0.50,
    "maximum_thinking_internal_repetition_share": 0.01,
    "maximum_task_thinking_internal_repetition_share": 0.02,
    "maximum_task_exact_thinking_signature_share": 0.05,
    "maximum_task_thinking_fivegram_share": 0.10,
    "maximum_task_thinking_final_overlap_share": 0.05,
    "maximum_thinking_prompt_copy_share": 0.02,
    "maximum_task_thinking_prompt_copy_share": 0.03,
}


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE.split(text.strip()) if item.strip()]


def _normalized_sentence(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\d+(?:[.,:/]\d+)*", "<n>", text)
    return " ".join(_words(text))


def _ngram_repetition_ratio(text: str, size: int = 3) -> float:
    words = _words(text)
    grams = list(zip(*(words[index:] for index in range(size))))
    if not grams:
        return 0.0
    return (len(grams) - len(set(grams))) / len(grams)


def _has_duplicate_sentence(text: str) -> bool:
    sentences = []
    for sentence in _sentences(text):
        normalized = _normalized_sentence(sentence)
        if len(normalized.split()) >= 4:
            sentences.append(normalized)
    return len(sentences) != len(set(sentences))


def _prompt_copy_ratio(prompt: str, response: str, size: int = 4) -> float:
    prompt_words = _words(prompt)
    response_words = _words(response)
    if len(response_words) < max(8, size):
        return 0.0
    prompt_grams = set(
        zip(*(prompt_words[index:] for index in range(size)))
    )
    response_grams = list(
        zip(*(response_words[index:] for index in range(size)))
    )
    if not response_grams:
        return 0.0
    return sum(gram in prompt_grams for gram in response_grams) / len(response_grams)


def _text_overlap_ratio(source: str, target: str, size: int = 3) -> float:
    source_words = _words(source)
    target_words = _words(target)
    source_grams = set(zip(*(source_words[index:] for index in range(size))))
    target_grams = list(zip(*(target_words[index:] for index in range(size))))
    if not target_grams:
        return 0.0
    return sum(gram in source_grams for gram in target_grams) / len(target_grams)


def _thinking_fivegrams(text: str) -> set[str]:
    words = _normalized_sentence(text).split()
    return {
        " ".join(gram)
        for gram in zip(*(words[index:] for index in range(5)))
    }


def _assistant_parts(
    row: dict[str, Any],
    response: str,
    *,
    use_row_fields: bool,
) -> tuple[str, str]:
    thinking = str(row.get("reasoning_trace", "")).strip() if use_row_fields else ""
    final = str(row.get("final_response", "")).strip() if use_row_fields else ""
    think_match = _THINK_BLOCK.search(response)
    final_match = _FINAL_BLOCK.search(response)
    if not thinking and think_match is not None:
        thinking = think_match.group(1).strip()
    if not final and final_match is not None:
        final = final_match.group(1).strip()
    if not final:
        final = response.strip()
    return thinking, final


def _prior_user(messages: list[dict[str, str]], index: int) -> str:
    for message in reversed(messages[:index]):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _anchor_name(prompt: str) -> str | None:
    normalized = " ".join(prompt.casefold().split())
    for name, (expected_prompt, _answer) in _ANCHORS.items():
        if normalized == expected_prompt:
            return name
    return None


def _finalize_task(
    task: str,
    rows: int,
    targets: int,
    repeated: int,
    copied: int,
    abstract: int,
    closing: Counter[str],
    exact_responses: Counter[str],
    thinking_targets: int,
    thinking_repeated: int,
    thinking_exact: Counter[str],
    thinking_fivegrams: Counter[str],
    thinking_final_overlap: int,
    thinking_prompt_copied: int,
) -> dict[str, Any]:
    closing_text, closing_count = closing.most_common(1)[0] if closing else ("", 0)
    response_text, response_count = (
        exact_responses.most_common(1)[0] if exact_responses else ("", 0)
    )
    thinking_text, thinking_count = (
        thinking_exact.most_common(1)[0] if thinking_exact else ("", 0)
    )
    fivegram, fivegram_count = (
        thinking_fivegrams.most_common(1)[0] if thinking_fivegrams else ("", 0)
    )
    return {
        "rows": rows,
        "assistant_targets": targets,
        "internal_repetition_count": repeated,
        "internal_repetition_share": round(repeated / max(1, targets), 6),
        "prompt_copy_count": copied,
        "prompt_copy_share": round(copied / max(1, targets), 6),
        "abstract_function_count": abstract,
        "abstract_function_share": round(abstract / max(1, targets), 6),
        "top_closing_sentence": closing_text,
        "top_closing_sentence_count": closing_count,
        "top_closing_sentence_share": round(closing_count / max(1, targets), 6),
        "top_exact_response": response_text,
        "top_exact_response_count": response_count,
        "top_exact_response_share": round(response_count / max(1, targets), 6),
        "thinking_targets": thinking_targets,
        "thinking_share": round(thinking_targets / max(1, targets), 6),
        "thinking_internal_repetition_count": thinking_repeated,
        "thinking_internal_repetition_share": round(
            thinking_repeated / max(1, thinking_targets), 6
        ),
        "top_exact_thinking_signature": thinking_text,
        "top_exact_thinking_signature_count": thinking_count,
        "top_exact_thinking_signature_share": round(
            thinking_count / max(1, thinking_targets), 6
        ),
        "top_thinking_fivegram": fivegram,
        "top_thinking_fivegram_count": fivegram_count,
        "top_thinking_fivegram_share": round(
            fivegram_count / max(1, thinking_targets), 6
        ),
        "thinking_final_overlap_count": thinking_final_overlap,
        "thinking_final_overlap_share": round(
            thinking_final_overlap / max(1, thinking_targets), 6
        ),
        "thinking_prompt_copy_count": thinking_prompt_copied,
        "thinking_prompt_copy_share": round(
            thinking_prompt_copied / max(1, thinking_targets), 6
        ),
    }


def audit_v2_behavior(
    rows: Iterable[dict[str, Any]],
    *,
    thresholds: dict[str, float | int | None] | None = None,
) -> dict[str, Any]:
    """Audit behavior learned from model-facing assistant targets, not cards."""

    policy = {**DEFAULT_V2_THRESHOLDS, **(thresholds or {})}
    task_rows = Counter()
    task_targets = Counter()
    task_repeated = Counter()
    task_copied = Counter()
    task_abstract = Counter()
    task_closing: dict[str, Counter[str]] = defaultdict(Counter)
    task_exact_responses: dict[str, Counter[str]] = defaultdict(Counter)
    task_thinking_targets = Counter()
    task_thinking_repeated = Counter()
    task_thinking_exact: dict[str, Counter[str]] = defaultdict(Counter)
    task_thinking_fivegrams: dict[str, Counter[str]] = defaultdict(Counter)
    task_thinking_final_overlap = Counter()
    task_thinking_prompt_copied = Counter()
    task_supervised_words = Counter()
    task_examples: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    abstract_functions: dict[str, Counter[str]] = defaultdict(Counter)
    assistant_turns = Counter()
    assistant_history_turns = 0
    casual_assistant_turns = Counter()
    casual_direct_short = 0
    natural_social_rows = 0
    anchors_seen = Counter()
    anchors_correct = Counter()
    empty_targets = 0
    train_rows = 0
    total_targets = 0

    for row in rows:
        if row.get("split", "train") != "train":
            continue
        train_rows += 1
        task = str(row.get("task", "unknown"))
        task_rows[task] += 1
        messages = [
            {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
            for item in row.get("messages", [])
        ]
        assistant_indexes = [
            index for index, message in enumerate(messages)
            if message["role"] == "assistant"
        ]
        assistant_turns[len(assistant_indexes)] += 1
        assistant_history_turns += max(0, len(assistant_indexes) - 1)
        if task == "casual_conversation":
            natural_social_rows += str(row.get("domain", "")) in _NATURAL_SOCIAL_DOMAINS
            casual_assistant_turns[len(assistant_indexes)] += 1
            if len(assistant_indexes) == 1:
                response = messages[assistant_indexes[0]]["content"]
                _thinking, final = _assistant_parts(
                    row, response, use_row_fields=True
                )
                casual_direct_short += len(_words(final)) <= 25

        # Historical assistant turns are serialized into the masked context.
        # Only the final assistant message is a model-facing training target.
        if not messages or messages[-1]["role"] != "assistant":
            empty_targets += 1
            continue
        index = len(messages) - 1
        response = messages[index]["content"].strip()
        thinking, final = _assistant_parts(row, response, use_row_fields=True)
        prompt = _prior_user(messages, index).strip()
        example_id = str(row.get("example_id", f"row-{train_rows}"))
        total_targets += 1
        task_targets[task] += 1
        if not final:
            empty_targets += 1
            continue
        if final:
            task_supervised_words[task] += len(_words(final)) + len(_words(thinking))
            task_exact_responses[task][" ".join(final.casefold().split())] += 1
            repeated = (
                _ngram_repetition_ratio(final) > 0.20
                or _has_duplicate_sentence(final)
            )
            copied = _prompt_copy_ratio(prompt, final) > 0.50
            hits = {
                name for name, pattern in _ABSTRACT_FUNCTIONS.items()
                if pattern.search(final)
            }
            task_repeated[task] += repeated
            task_copied[task] += copied
            task_abstract[task] += bool(hits)
            abstract_functions[task].update(hits)
            issue_flags = {
                "internal_repetition": repeated,
                "prompt_copy": copied,
                "abstract_function": bool(hits),
            }
            for issue, present in issue_flags.items():
                if present and len(task_examples[task][issue]) < 5:
                    task_examples[task][issue].append(
                        {
                            "example_id": example_id,
                            "prompt": prompt,
                            "final": final,
                        }
                    )
            sentences = _sentences(final)
            if sentences:
                closing = _normalized_sentence(sentences[-1])
                if len(closing.split()) >= 4:
                    task_closing[task][closing] += 1
            anchor = _anchor_name(prompt)
            if anchor is not None:
                anchors_seen[anchor] += 1
                anchors_correct[anchor] += bool(_ANCHORS[anchor][1].search(final))
            if thinking:
                task_thinking_targets[task] += 1
                thinking_repeated = (
                    _ngram_repetition_ratio(thinking) > 0.15
                    or _has_duplicate_sentence(thinking)
                )
                task_thinking_repeated[task] += thinking_repeated
                task_thinking_exact[task][_normalized_sentence(thinking)] += 1
                task_thinking_fivegrams[task].update(_thinking_fivegrams(thinking))
                task_thinking_final_overlap[task] += (
                    _text_overlap_ratio(thinking, final) > 0.60
                )
                task_thinking_prompt_copied[task] += (
                    _prompt_copy_ratio(prompt, thinking) > 0.50
                )
                thinking_issues = {
                    "thinking_internal_repetition": thinking_repeated,
                    "thinking_final_overlap": (
                        _text_overlap_ratio(thinking, final) > 0.60
                    ),
                    "thinking_prompt_copy": (
                        _prompt_copy_ratio(prompt, thinking) > 0.50
                    ),
                }
                for issue, present in thinking_issues.items():
                    if present and len(task_examples[task][issue]) < 5:
                        task_examples[task][issue].append(
                            {
                                "example_id": example_id,
                                "prompt": prompt,
                                "thinking": thinking,
                                "final": final,
                            }
                        )

    tasks = {
        task: _finalize_task(
            task,
            task_rows[task],
            task_targets[task],
            task_repeated[task],
            task_copied[task],
            task_abstract[task],
            task_closing[task],
            task_exact_responses[task],
            task_thinking_targets[task],
            task_thinking_repeated[task],
            task_thinking_exact[task],
            task_thinking_fivegrams[task],
            task_thinking_final_overlap[task],
            task_thinking_prompt_copied[task],
        )
        for task in sorted(task_rows)
    }
    for task in tasks:
        tasks[task]["examples"] = {
            issue: examples
            for issue, examples in sorted(task_examples[task].items())
        }
    casual_rows = task_rows["casual_conversation"]
    casual_direct = casual_assistant_turns[1]
    casual_two = casual_assistant_turns[2]
    casual_three_plus = sum(
        count for turns, count in casual_assistant_turns.items() if turns >= 3
    )
    repeated_total = sum(task_repeated.values())
    copied_total = sum(task_copied.values())
    thinking_total = sum(task_thinking_targets.values())
    thinking_repeated_total = sum(task_thinking_repeated.values())
    thinking_prompt_copied_total = sum(task_thinking_prompt_copied.values())
    supervised_words_total = sum(task_supervised_words.values())
    supervised_word_shares = {
        task: round(count / max(1, supervised_words_total), 6)
        for task, count in sorted(task_supervised_words.items())
    }
    top_supervised_task, top_supervised_words = (
        task_supervised_words.most_common(1)[0]
        if task_supervised_words
        else ("", 0)
    )
    raw_uniform_loss_mix = {
        "unit": "model_facing_words",
        "total": supervised_words_total,
        "task_shares": supervised_word_shares,
        "top_task": top_supervised_task,
        "top_task_share": round(
            top_supervised_words / max(1, supervised_words_total), 6
        ),
        "balanced_sampling_required": (
            top_supervised_words / max(1, supervised_words_total) > 0.35
        ),
    }
    missing_anchors = sorted(set(_ANCHORS) - set(anchors_seen))
    incorrect_anchors = sorted(
        name for name in _ANCHORS
        if anchors_seen[name] and anchors_correct[name] != anchors_seen[name]
    )
    missing_anchors_by_category = {
        category: sorted(names & set(missing_anchors))
        for category, names in _ANCHOR_CATEGORIES.items()
    }
    incorrect_anchors_by_category = {
        category: sorted(names & set(incorrect_anchors))
        for category, names in _ANCHOR_CATEGORIES.items()
    }
    failing_tasks: dict[str, list[str]] = {}
    for task, metrics in tasks.items():
        failures = []
        if metrics["internal_repetition_share"] > policy["maximum_task_internal_repetition_share"]:
            failures.append("internal_repetition")
        if metrics["prompt_copy_share"] > policy["maximum_task_prompt_copy_share"]:
            failures.append("prompt_copy")
        if metrics["abstract_function_share"] > policy["maximum_task_abstract_function_share"]:
            failures.append("abstract_function")
        if metrics["top_closing_sentence_share"] > policy["maximum_task_closing_sentence_share"]:
            failures.append("closing_sentence")
        if metrics["top_exact_response_share"] > policy["maximum_task_exact_response_share"]:
            failures.append("exact_response")
        if (
            metrics["thinking_internal_repetition_share"]
            > policy["maximum_task_thinking_internal_repetition_share"]
        ):
            failures.append("thinking_internal_repetition")
        if (
            metrics["top_exact_thinking_signature_share"]
            > policy["maximum_task_exact_thinking_signature_share"]
        ):
            failures.append("exact_thinking_signature")
        if (
            metrics["top_thinking_fivegram_share"]
            > policy["maximum_task_thinking_fivegram_share"]
        ):
            failures.append("thinking_fivegram")
        if (
            metrics["thinking_final_overlap_share"]
            > policy["maximum_task_thinking_final_overlap_share"]
        ):
            failures.append("thinking_final_overlap")
        if (
            metrics["thinking_prompt_copy_share"]
            > policy["maximum_task_thinking_prompt_copy_share"]
        ):
            failures.append("thinking_prompt_copy")
        if (
            task == "reasoning_verification"
            and metrics["thinking_share"] < policy["minimum_reasoning_thinking_share"]
        ):
            failures.append("thinking_coverage")
        if failures:
            failing_tasks[task] = failures

    violations: list[str] = []
    required_train_examples = policy["required_train_examples"]
    if (
        required_train_examples is not None
        and train_rows != required_train_examples
    ):
        violations.append(
            f"expected {required_train_examples} train examples, got {train_rows}"
        )
    if empty_targets:
        violations.append(f"{empty_targets} assistant targets are empty")
    if casual_direct < policy["minimum_direct_casual_examples"]:
        violations.append("too few direct casual conversations")
    if casual_direct / max(1, casual_rows) < policy["minimum_direct_casual_share"]:
        violations.append("direct casual conversation share is below target")
    if casual_direct and casual_direct_short / casual_direct < policy["minimum_short_direct_casual_share"]:
        violations.append("short direct casual response share is below target")
    if natural_social_rows < policy["minimum_natural_social_examples"]:
        violations.append("too few genuinely social conversations")
    if natural_social_rows / max(1, train_rows) < policy["minimum_natural_social_share"]:
        violations.append("genuinely social conversation share is below target")
    if repeated_total / max(1, total_targets) > policy["maximum_internal_repetition_share"]:
        violations.append("internal response repetition exceeds global target")
    if copied_total / max(1, total_targets) > policy["maximum_prompt_copy_share"]:
        violations.append("prompt copying exceeds global target")
    if (
        thinking_repeated_total / max(1, thinking_total)
        > policy["maximum_thinking_internal_repetition_share"]
    ):
        violations.append("internal thinking repetition exceeds global target")
    if (
        thinking_prompt_copied_total / max(1, thinking_total)
        > policy["maximum_thinking_prompt_copy_share"]
    ):
        violations.append("thinking copies prompts too often")
    if missing_anchors:
        violations.append("missing behavioral anchors: " + ", ".join(missing_anchors))
    if incorrect_anchors:
        violations.append("incorrect behavioral anchors: " + ", ".join(incorrect_anchors))
    if failing_tasks:
        violations.append("one or more task-level behavior gates failed")

    return {
        "format": "complexity-card-corpus-v2-behavior-audit-v2",
        "passed": not violations,
        "violations": violations,
        "train_rows": train_rows,
        "assistant_targets": total_targets,
        "empty_assistant_targets": empty_targets,
        "internal_repetition_share": round(repeated_total / max(1, total_targets), 6),
        "prompt_copy_share": round(copied_total / max(1, total_targets), 6),
        "thinking_targets": thinking_total,
        "thinking_internal_repetition_share": round(
            thinking_repeated_total / max(1, thinking_total), 6
        ),
        "thinking_prompt_copy_share": round(
            thinking_prompt_copied_total / max(1, thinking_total), 6
        ),
        "assistant_history_turns": assistant_history_turns,
        "rows_with_assistant_history": sum(
            count for turns, count in assistant_turns.items() if turns > 1
        ),
        "assistant_turns_per_row": dict(sorted(assistant_turns.items())),
        "casual_conversation": {
            "rows": casual_rows,
            "direct_rows": casual_direct,
            "natural_social_rows": natural_social_rows,
            "natural_social_share": round(
                natural_social_rows / max(1, train_rows), 6
            ),
            "direct_share": round(casual_direct / max(1, casual_rows), 6),
            "two_assistant_rows": casual_two,
            "two_assistant_share": round(casual_two / max(1, casual_rows), 6),
            "three_plus_assistant_rows": casual_three_plus,
            "three_plus_assistant_share": round(casual_three_plus / max(1, casual_rows), 6),
            "short_direct_share": round(casual_direct_short / max(1, casual_direct), 6),
        },
        "raw_uniform_loss_mix": raw_uniform_loss_mix,
        "anchors": {
            name: {
                "examples": anchors_seen[name],
                "correct": anchors_correct[name],
            }
            for name in sorted(_ANCHORS)
        },
        "missing_anchors": missing_anchors,
        "incorrect_anchors": incorrect_anchors,
        "missing_anchors_by_category": missing_anchors_by_category,
        "incorrect_anchors_by_category": incorrect_anchors_by_category,
        "failing_tasks": failing_tasks,
        "abstract_functions": {
            task: dict(sorted(counts.items()))
            for task, counts in sorted(abstract_functions.items())
        },
        "tasks": tasks,
        "thresholds": policy,
    }


def audit_projected_parquet(
    path: str | Path,
    *,
    batch_size: int = 16_384,
    thresholds: dict[str, float | int | None] | None = None,
) -> dict[str, Any]:
    """Stream a projected Parquet release through the V2 behavior audit."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(Path(path))

    def rows() -> Iterable[dict[str, Any]]:
        available = set(parquet.schema.names)
        columns = ["task", "split", "messages"]
        if "domain" in available:
            columns.append("domain")
        columns.extend(
            name
            for name in ("reasoning_trace", "final_response")
            if name in available
        )
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            yield from batch.to_pylist()

    return audit_v2_behavior(rows(), thresholds=thresholds)


__all__ = (
    "DEFAULT_V2_THRESHOLDS",
    "audit_projected_parquet",
    "audit_v2_behavior",
)
