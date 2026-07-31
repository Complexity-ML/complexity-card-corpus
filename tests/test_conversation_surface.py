from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from complexity_card_corpus.conversation_blueprint import BLUEPRINT_SCHEMA
from complexity_card_corpus.conversation_surface import (
    _task_messages,
    build_conversation_surface_pilot,
)


TASK_DOMAINS = (
    "auto_repair",
    "coffee_ordering",
    "movie_ticket",
    "pizza_ordering",
    "restaurant_reservation",
    "ride_booking",
)
EMOTIONS = (
    "afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive",
    "ashamed", "caring", "confident", "content", "devastated", "disappointed",
    "disgusted", "embarrassed", "excited", "faithful", "furious", "grateful",
    "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely",
    "nostalgic", "prepared", "proud", "sad", "sentimental", "surprised",
    "terrified", "trusting",
)


def _blueprint(kind: str, category: str, index: int) -> dict:
    turns = (2, 4, 6, 8)[index % 4]
    task = kind == "task_oriented"
    task_stages = (
        "state_goal", "acknowledge_goal", "provide_detail",
        "ask_for_missing_detail", "choose_option", "present_bounded_options",
        "confirm_choice", "confirm_next_step",
    )
    empathy_stages = (
        "share_situation", "acknowledge_emotion", "expand_feeling",
        "invite_detail_without_assumption", "reflect_on_need",
        "offer_grounded_support", "follow_up", "close_supportively",
    )
    stages = list((task_stages if task else empathy_stages)[:turns])
    if task and index % 5 == 0:
        stages[-1] = "offer_safe_alternative"
    blueprint_id = f"blueprint:{kind}:{category}:{index:03d}"
    return {
        "blueprint_id": blueprint_id,
        "source_structure_id": f"structure:{kind}:{category}:{index:03d}",
        "source_record_id": f"record:{kind}:{category}:{index:03d}",
        "source_record_sha256": f"{index + 1:064x}",
        "source_dataset": f"source-{kind}",
        "source_revision": "a" * 40,
        "source_license": "CC BY 4.0" if task else "CC BY-NC 4.0",
        "source_file_sha256": "b" * 64,
        "corpus_kind": kind,
        "category": category,
        "domain": category if task else "everyday_emotion",
        "emotion": "" if task else category,
        "split": "train",
        "target_turn_count": turns,
        "target_question_turns": sum(
            "ask" in stage or "invite" in stage for stage in stages
        ),
        "target_speaker_pattern": ["user", "assistant"] * (turns // 2),
        "target_length_pattern": ["medium"] * turns,
        "dialogue_stages": stages,
        "response_style": "concise_practical" if task else "warm_grounded",
        "difficulty": "easy" if turns <= 4 else "medium",
        "source_slot_types": ["time"] if task else [],
        "surface_text_generated": False,
        "blueprint_version": "conversation-blueprint-v1",
    }


def test_surface_pilot_is_original_balanced_unique_and_deterministic(tmp_path: Path) -> None:
    blueprints = tmp_path / "blueprints"
    blueprints.mkdir()
    rows = []
    for domain in TASK_DOMAINS:
        rows.extend(_blueprint("task_oriented", domain, index) for index in range(48))
    for emotion in EMOTIONS:
        rows.extend(
            _blueprint("empathetic_conversation", emotion, index)
            for index in range(10)
        )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=BLUEPRINT_SCHEMA),
        blueprints / "blueprints.parquet",
    )
    scenarios = (
        Path(__file__).parents[1]
        / "data/conversation/original/scenarios.json"
    )

    first = build_conversation_surface_pilot(
        blueprints, scenarios, tmp_path / "first", pilot_size=128, seed=7
    )
    second = build_conversation_surface_pilot(
        blueprints, scenarios, tmp_path / "second", pilot_size=128, seed=7
    )

    assert first["files"]["conversations.parquet"]["sha256"] == second["files"]["conversations.parquet"]["sha256"]
    assert first["surface_text"]["model_generated"] is False
    assert first["surface_text"]["source_utterances_accessed"] is False
    assert first["counts"]["by_task"] == {
        "empathetic_dialogue": 64,
        "practical_dialogue": 64,
    }
    assert first["audit"]["unique_rendered_ratio"] == 1.0
    assert first["audit"]["unique_prompt_ratio"] == 1.0
    assert first["audit"]["unique_final_response_ratio"] == 1.0
    assert first["audit"]["placeholder_leaks"] == 0
    assert first["audit"]["length_contract_match_ratio"] >= 0.95
    assert first["audit"]["question_contract_match_ratio"] == 1.0
    assert first["audit"]["task_context_contract_match_ratio"] == 1.0
    assert first["audit"]["source_card_split_overlap"] == 0
    assert first["audit"]["split_holdout_unit"] == "scenario_card_id"

    output_rows = pq.read_table(tmp_path / "first/conversations.parquet").to_pylist()
    assert len(output_rows) == 128
    assert {row["language"] for row in output_rows} == {"en"}
    assert {row["license"] for row in output_rows} == {"CC BY-NC 4.0"}
    assert all(row["source_urls"] == [] for row in output_rows)
    assert all(row["messages"][0]["role"] == "user" for row in output_rows)
    assert all(row["messages"][-1]["role"] == "assistant" for row in output_rows)
    assert all(
        json.loads(row["answer_json"])["dialogue_stages"][-1]
        != "acknowledge_goal"
        for row in output_rows
        if row["task"] == "practical_dialogue" and len(row["messages"]) == 2
    )
    assert all("{" not in row["rendered_text"] for row in output_rows)
    assert all(" i " not in f" {row['rendered_text']} " for row in output_rows)

    audit = json.loads((tmp_path / "first/audit.json").read_text())
    assert audit["rows"] == 128


def test_response_style_changes_the_selected_surface_form() -> None:
    scenarios = json.loads(
        (
            Path(__file__).parents[1]
            / "data/conversation/original/scenarios.json"
        ).read_text()
    )
    blueprint = _blueprint("task_oriented", "coffee_ordering", 3)
    blueprint["target_length_pattern"] = ["medium"] * len(blueprint["dialogue_stages"])
    concise = {**blueprint, "response_style": "concise_practical"}
    stepwise = {**blueprint, "response_style": "stepwise_helpful"}
    concise_messages, _ = _task_messages(
        concise, scenarios["task_scenarios"]["coffee_ordering"], rank=3
    )
    stepwise_messages, _ = _task_messages(
        stepwise, scenarios["task_scenarios"]["coffee_ordering"], rank=3
    )
    assert concise_messages != stepwise_messages
