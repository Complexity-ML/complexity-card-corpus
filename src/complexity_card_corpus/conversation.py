from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .build import file_sha256
from .instruct import INSTRUCTION_SCHEMA


DATASET_ID = "complexity-conversation-v1"
SOURCE = "Complexity original deterministic conversation scenarios"
LICENSE = "CC BY-NC 4.0"
VERSION = "1.0.0"

RAW_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("family", pa.string()),
        ("topic", pa.string()),
        ("language", pa.string()),
        ("split", pa.string()),
        ("fields_json", pa.string()),
        ("source", pa.string()),
        ("source_urls", pa.list_(pa.string())),
        ("license", pa.string()),
        ("version", pa.string()),
    ]
)


NAMES = (
    "Alex", "Amira", "Ben", "Chloe", "Diego", "Elena", "Farah", "Gabriel",
    "Hana", "Isaac", "Julia", "Kai", "Lina", "Maya", "Nora", "Omar",
)
TIMES = ("this morning", "this afternoon", "after lunch", "this evening")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
ACTIVITIES = (
    "reading", "walking", "cooking", "drawing", "gardening", "listening to music",
    "cycling", "taking photos", "learning languages", "doing puzzles",
)
DRINKS = ("tea", "coffee", "sparkling water", "hot chocolate", "lemon water")
RECIPIENTS = ("a colleague", "a friend", "my manager", "the project team")
MINUTES = ("10", "15", "20", "30")

EXPLANATIONS = (
    (
        "a backup",
        "A backup is a separate copy of important files. It helps you recover them if the original files are lost, damaged, or deleted.",
    ),
    (
        "a budget",
        "A budget is a simple plan for income and spending. It shows what money is available and helps prevent spending more than intended.",
    ),
    (
        "two-factor authentication",
        "Two-factor authentication asks for a second proof of identity after a password, such as a code from your phone. It makes an account harder to access with a stolen password alone.",
    ),
    (
        "cloud storage",
        "Cloud storage keeps files on remote servers that you access through the internet. It can make files available across devices, but access still depends on your account and connection.",
    ),
    (
        "a calendar reminder",
        "A calendar reminder is an alert linked to an event or task. It helps you remember what needs to happen and when.",
    ),
    (
        "a strong password",
        "A strong password is long, unique, and difficult to guess. Using a different password for each account limits the damage if one password is exposed.",
    ),
    (
        "sleep mode",
        "Sleep mode keeps a computer session in memory while using less power. It resumes quickly, but it is not the same as shutting the computer down.",
    ),
    (
        "Wi-Fi",
        "Wi-Fi connects devices to a local network using radio signals instead of a cable. The network can then provide access to the internet through its router.",
    ),
)

PRACTICAL = (
    ("organize a cluttered desk", ("Remove anything that does not belong on the desk.", "Group the remaining items by purpose.", "Keep only frequently used items within reach.")),
    ("prepare for a short meeting", ("Write down the desired outcome.", "List the two or three points that matter most.", "Bring any document needed for a decision.")),
    ("remember a morning appointment", ("Add the appointment to your calendar.", "Set one reminder the evening before.", "Set a second reminder with enough travel time.")),
    ("start a simple weekly budget", ("Write down expected income.", "List fixed expenses first.", "Choose a realistic limit for flexible spending.")),
    ("pack for a day trip", ("Check the weather and schedule.", "Pack water, identification, and any required tickets.", "Add only the clothing or equipment the plan requires.")),
    ("reduce phone distractions", ("Silence nonessential notifications.", "Move distracting apps away from the home screen.", "Choose a specific time to check messages.")),
    ("tidy an email inbox", ("Delete obvious junk.", "Archive messages that need no action.", "Turn the remaining messages into a short action list.")),
    ("plan a simple dinner", ("Check what ingredients are already available.", "Choose one main dish with few steps.", "Make a short list of only the missing ingredients.")),
)

REWRITES = (
    ("send me the file now", "Could you please send me the file when you have a moment?"),
    ("meeting moved to three", "The meeting has been moved to 3:00 p.m."),
    ("i cant come because im sick", "I’m sorry, but I can’t attend because I’m unwell."),
    ("need answer today", "Could you please let me know by the end of today?"),
    ("thanks for fixing it", "Thank you for taking the time to fix this."),
    ("we should talk about the delay", "Could we discuss the delay and agree on the next steps?"),
    ("your instructions are confusing", "Could you clarify the instructions, particularly the expected final result?"),
    ("ill send details later", "I’ll send the remaining details later today."),
)

AMBIGUOUS = (
    ("Can you prepare it for tomorrow?", "What would you like me to prepare, and what time tomorrow is it needed?"),
    ("Please make it shorter.", "Which text should I shorten, and is there a target length?"),
    ("Help me choose the best one.", "What options are you comparing, and which criteria matter most to you?"),
    ("Can you remind me later?", "What should I remind you about, and at what time?"),
    ("I need a plan for next week.", "What is the goal of the plan, and how much time is available next week?"),
    ("Can you explain that again?", "Which part should I explain again, and would an example help?"),
    ("Please improve this message.", "Please share the message and tell me the tone you want."),
    ("Book something for Friday.", "What should be booked, for how many people, and at what time on Friday?"),
)

NOTES = (
    ("The appointment is on Tuesday at 10:00. Bring identification. Arrive ten minutes early.", "Appointment: Tuesday at 10:00. Bring identification and arrive ten minutes early."),
    ("The team agreed to test the new page on Thursday. Maya will collect feedback. The final decision is Friday.", "The page will be tested Thursday, Maya will collect feedback, and the final decision is due Friday."),
    ("The train leaves at 08:20. The platform will appear on the board. Boarding closes five minutes before departure.", "The train leaves at 08:20; check the board for the platform and board at least five minutes early."),
    ("There is enough rice for dinner. Vegetables are missing. The shop closes at 19:00.", "Rice is available, vegetables still need to be bought, and the shop closes at 19:00."),
    ("The document needs a title and one final proofread. The figures are already complete. It must be sent tomorrow.", "Add a title and do one final proofread before sending the document tomorrow; the figures are complete."),
    ("The package arrived at reception. Reception closes at 18:00. A photo ID is required for collection.", "Collect the package from reception before 18:00 and bring photo identification."),
)

FAMILIES = (
    "social",
    "preference_chat",
    "practical_help",
    "clarification",
    "concise_explanation",
    "rewrite",
    "short_summary",
    "conversation_memory",
)


def _hash_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _split(record_id: str, validation_percent: int) -> str:
    return "validation" if _hash_int(record_id) % 100 < validation_percent else "train"


def _choose(items: tuple[Any, ...], key: str) -> Any:
    return items[_hash_int(key) % len(items)]


def _render(messages: list[dict[str, str]]) -> str:
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{labels[m['role']]}: {m['content']}" for m in messages)


def _scenario(family: str, index: int, rng: random.Random) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    key = f"{family}:{index}:{rng.randrange(1 << 31)}"
    name = _choose(NAMES, key + ":name")
    activity = _choose(ACTIVITIES, key + ":activity")
    time = _choose(TIMES, key + ":time")
    day = _choose(DAYS, key + ":day")

    if family == "social":
        variant = _hash_int(key) % 6
        if variant == 0:
            opening = f"Hi, I’m {name}. I have a little time to chat {time}."
            response = f"Hi {name}! It’s nice to meet you. What would you like to talk about?"
        elif variant == 1:
            opening = f"I tried {activity} {time}, and it went better than expected."
            response = f"That’s good to hear. What part of {activity} went especially well?"
        elif variant == 2:
            opening = f"Thanks for helping me organize my plans for {day}."
            response = "You’re welcome. I’m glad the plan feels more manageable now."
        elif variant == 3:
            opening = f"I had a difficult {day} and I’m feeling tired."
            response = "I’m sorry it was difficult. Would you like to talk about what made the day tiring?"
        elif variant == 4:
            opening = f"I finally finished my {activity} task {time}."
            response = "That’s good to hear. Finishing a demanding task can be a real relief."
        else:
            opening = f"I’m a little nervous about my plans for {day}."
            response = f"That makes sense. Which part of {day} feels most uncertain right now?"
        fields = {"opening": opening, "intent": "brief supportive response"}
        messages = [{"role": "user", "content": opening}, {"role": "assistant", "content": response}]
        return "everyday social exchange", fields, messages

    if family == "preference_chat":
        drink = _choose(DRINKS, key + ":drink")
        fields = {"name": name, "preference": activity, "drink": drink, "day": day}
        messages = [
            {"role": "user", "content": f"I’m {name}. On {day}, I enjoy {activity} and usually have {drink}."},
            {"role": "assistant", "content": f"That sounds like a pleasant {day} routine, {name}. What do you enjoy most about {activity}?"},
            {"role": "user", "content": "It helps me slow down and focus."},
            {"role": "assistant", "content": f"That makes sense. {activity.capitalize()} can be a useful way to create a calmer, more focused part of the day."},
        ]
        return "personal preference", fields, messages

    if family == "practical_help":
        goal, steps = _choose(PRACTICAL, key)
        minutes = _choose(MINUTES, key + ":minutes")
        fields = {"goal": goal, "available_time": f"{minutes} minutes", "day": day, "time": time, "steps": list(steps)}
        messages = [
            {"role": "user", "content": f"I have about {minutes} minutes {time} on {day} to {goal}. What should I do first?"},
            {
                "role": "assistant",
                "content": "Here is a simple approach:\n" + "\n".join(
                    f"{number}. {step}" for number, step in enumerate(steps, start=1)
                ),
            },
        ]
        return goal, fields, messages

    if family == "clarification":
        request, response = _choose(AMBIGUOUS, key)
        variant = _hash_int(key + ":variant") % 4
        if variant == 0:
            request = request.replace("tomorrow", day).replace("Friday", day)
            response = response.replace("tomorrow", day).replace("Friday", day)
        elif variant == 1:
            request = f"{request} It is for {name}."
            response = f"{response} Is there anything specific {name} needs?"
        elif variant == 2:
            request = f"{request} I would like to handle it {time}."
            response = f"{response} I’ll also take {time} into account."
        else:
            request = f"{request} This is related to my {activity} plan."
            response = f"{response} What part of the {activity} plan does it concern?"
        fields = {"ambiguous_request": request, "missing_information": response}
        messages = [{"role": "user", "content": request}, {"role": "assistant", "content": response}]
        return "request clarification", fields, messages

    if family == "concise_explanation":
        concept, explanation = _choose(EXPLANATIONS, key)
        context = _choose(
            (
                "I saw the term in my settings.",
                "I am explaining it to a friend.",
                "I only need the basic idea.",
                "I have not used it before.",
            ),
            key + ":context",
        )
        fields = {"name": name, "concept": concept, "context": context, "canonical_explanation": explanation}
        messages = [
            {"role": "user", "content": f"I’m {name}. Can you explain {concept} in simple terms? {context}"},
            {"role": "assistant", "content": explanation},
        ]
        return concept, fields, messages

    if family == "rewrite":
        original, canonical = _choose(REWRITES, key)
        recipient = _choose(RECIPIENTS, key + ":recipient")
        context = f"I want to send it to {recipient} {time} on {day}."
        fields = {"original_message": original, "requested_style": "clear and polite", "recipient": recipient, "day": day, "time": time, "meaning": canonical}
        messages = [
            {"role": "user", "content": f"Rewrite this clearly and politely: \"{original}\" {context}"},
            {"role": "assistant", "content": canonical},
        ]
        return "message rewrite", fields, messages

    if family == "short_summary":
        variant = _hash_int(key + ":note") % 3
        if variant == 0:
            note = f"The appointment is on {day} {time}. {name} should bring identification. Arriving ten minutes early is recommended."
            summary = f"{name} has an appointment on {day} {time} and should bring identification and arrive ten minutes early."
        elif variant == 1:
            note = f"The team will review the document on {day}. {name} will collect comments. The final decision is due the following morning."
            summary = f"The document review is on {day}, {name} will collect comments, and the final decision is due the next morning."
        else:
            note = f"There is enough rice for dinner on {day}. Vegetables are missing. The shop closes {time}."
            summary = f"Rice is available for {day}, but vegetables must be bought before the shop closes {time}."
        fields = {"note": note, "required_detail_count": 3, "summary": summary}
        messages = [
            {"role": "user", "content": f"Summarize this note in one sentence: {note}"},
            {"role": "assistant", "content": summary},
        ]
        return "short note", fields, messages

    if family == "conversation_memory":
        fields = {"name": name, "activity": activity, "schedule": f"{day} {time}"}
        messages = [
            {"role": "user", "content": f"My name is {name}. I plan to do some {activity} on {day}."},
            {"role": "assistant", "content": f"Nice to meet you, {name}. I’ll keep your {day} plan in mind for this conversation."},
            {"role": "user", "content": "What did I say I planned to do?"},
            {"role": "assistant", "content": f"You said you planned to do some {activity} on {day}."},
        ]
        return "short conversation memory", fields, messages

    raise ValueError(f"unknown conversation family: {family}")


def _validate_messages(messages: list[dict[str, str]]) -> None:
    if len(messages) not in {2, 4}:
        raise ValueError("conversation examples must contain two or four messages")
    for position, message in enumerate(messages):
        expected = "user" if position % 2 == 0 else "assistant"
        if message["role"] != expected or not message["content"].strip():
            raise ValueError("conversation roles must alternate and contain text")
    if len(messages[-1]["content"]) > 700:
        raise ValueError("assistant response is too long for the compact corpus")


def build_conversation_dataset(
    output_root: Path,
    *,
    examples: int = 2_000,
    seed: int = 42,
    validation_percent: int = 5,
) -> dict[str, Any]:
    if examples < len(FAMILIES):
        raise ValueError(f"examples must be at least {len(FAMILIES)}")
    if not 1 <= validation_percent <= 25:
        raise ValueError("validation_percent must be between 1 and 25")

    rng = random.Random(seed)
    raw_rows: list[dict[str, Any]] = []
    instruction_rows: list[dict[str, Any]] = []
    rendered_seen: set[str] = set()
    base_target, remainder = divmod(examples, len(FAMILIES))
    for family_position, family in enumerate(FAMILIES):
        target = base_target + int(family_position < remainder)
        accepted = 0
        candidate = 0
        while accepted < target:
            if candidate > max(10_000, target * 100):
                raise RuntimeError(f"not enough unique scenarios for {family}")
            topic, fields, messages = _scenario(family, candidate, rng)
            candidate += 1
            _validate_messages(messages)
            rendered_text = _render(messages)
            if rendered_text in rendered_seen:
                continue
            rendered_seen.add(rendered_text)
            record_id = f"{DATASET_ID}:{family}:{accepted:06d}"
            split = _split(record_id, validation_percent)
            fields_json = json.dumps(fields, ensure_ascii=False, sort_keys=True)
            raw_rows.append(
                {
                    "record_id": record_id,
                    "family": family,
                    "topic": topic,
                    "language": "en",
                    "split": split,
                    "fields_json": fields_json,
                    "source": SOURCE,
                    "source_urls": [],
                    "license": LICENSE,
                    "version": VERSION,
                }
            )
            instruction_rows.append(
                {
                    "example_id": record_id.replace(DATASET_ID, "conversation-instruct", 1),
                    "task": family,
                    "mode": "instruct" if len(messages) == 2 else "chat",
                    "difficulty": "easy",
                    "dataset_id": DATASET_ID,
                    "domain": "general_conversation",
                    "language": "en",
                    "split": split,
                    "messages": messages,
                    "prompt": messages[0]["content"],
                    "response": messages[-1]["content"],
                    "rendered_text": rendered_text,
                    "source_keys": [record_id],
                    "evidence": [fields_json],
                    "answer_json": fields_json,
                    "source": SOURCE,
                    "source_urls": [],
                    "license": LICENSE,
                    "version": VERSION,
                }
            )
            accepted += 1

    temporary = output_root.with_name(f"{output_root.name}.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    raw_path = temporary / "raw_records.parquet"
    instructions_path = temporary / "instructions.parquet"
    pq.write_table(pa.Table.from_pylist(raw_rows, schema=RAW_SCHEMA), raw_path, compression="zstd")
    pq.write_table(
        pa.Table.from_pylist(instruction_rows, schema=INSTRUCTION_SCHEMA),
        instructions_path,
        compression="zstd",
        use_dictionary=True,
    )
    counts = {
        "examples": len(instruction_rows),
        "examples_by_split": dict(sorted(Counter(row["split"] for row in instruction_rows).items())),
        "examples_by_task": dict(sorted(Counter(row["task"] for row in instruction_rows).items())),
        "examples_by_mode": dict(sorted(Counter(row["mode"] for row in instruction_rows).items())),
    }
    manifest = {
        "format": "complexity-conversation-instruct-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license": LICENSE,
        "counts": counts,
        "generation": {
            "method": "deterministic original conversational scenarios",
            "seed": seed,
            "validation_percent": validation_percent,
            "model_generated": False,
            "external_instruction_datasets": [],
            "target_model_scale": "approximately 100M parameters",
            "response_policy": "brief everyday assistant responses; no expert workflows",
        },
        "files": {
            "raw_records.parquet": {"bytes": raw_path.stat().st_size, "sha256": file_sha256(raw_path)},
            "instructions.parquet": {"bytes": instructions_path.stat().st_size, "sha256": file_sha256(instructions_path)},
        },
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest
