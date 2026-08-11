from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from string import Formatter

import pytest

from complexity_card_corpus.tasks.core import _compose_subcards
from complexity_card_corpus.variable_by import (
    VariableBy2D,
    aggregate_static_text_progress,
    analyze_static_text_progress,
    analyze_template_density,
    brainstorming_variable_by,
    casual_variable_by,
)
from complexity_card_corpus.variable_by.templates import (
    CRITIQUE_TEMPLATES,
    EMPATHY_TEMPLATES,
    GROUNDED_QA_TEMPLATES,
    REASONING_TEMPLATES,
    SAFETY_ANSWER_TEMPLATES,
)
from complexity_card_corpus.variable_by.reservoirs import (
    GroundedQAFacts,
    grounded_qa_variable_by,
)
from complexity_card_corpus.variable_by.brainstorm_templates import (
    BRAINSTORM_GOAL_TEMPLATES,
)


_STATIC_LINE_CEILINGS = {
    "action.py": 0,
    "communication.py": 0,
    "extraction.py": 0,
    "knowledge.py": 1,
    "reasoning.py": 0,
    "safety.py": 0,
}

_TEMPLATE_LITERAL_CEILINGS = {
    "action.py": 726,
    "communication.py": 1_301,
    "extraction.py": 61,
    "knowledge.py": 483,
    "reasoning.py": 0,
    "safety.py": 272,
}


def _row(scenario: str = "variable-by-2d") -> dict:
    return {"scenario_id": f"scenario:{scenario}"}


def test_variable_by_2d_deals_synonyms_for_the_requested_sense() -> None:
    matrix = VariableBy2D(
        {
            "for": {
                "purpose": ("to", "in order to", "so as to"),
                "beneficiary": ("for", "intended for", "serving"),
            },
            "participant": {
                "learner": ("learners", "students", "class participants"),
            },
        }
    )

    first = matrix.deal("stable-seed")
    second = matrix.deal("stable-seed")

    assert first == second
    assert first["for"]["purpose"] in matrix.variable_for("for", "purpose")
    assert first["for"]["beneficiary"] in matrix.variable_for(
        "for", "beneficiary"
    )
    assert first["participant"]["learner"] in matrix.variable_for(
        "participant", "learner"
    )


def test_compose_subcards_replaces_2d_fields_without_adding_variable_labels() -> None:
    matrix = VariableBy2D(
        {
            "for": {"purpose": ("to", "in order to", "so as to")},
            "audience": {
                "common_noun": ("learners", "students", "class participants")
            },
        }
    )
    card = _compose_subcards(
        _row(),
        0,
        "variable-by-test",
        (
            (
                "Use the activity {for[purpose]} support {audience[common_noun]}.",
                "Choose this format {for[purpose]} include {audience[common_noun]}.",
                "Apply the method {for[purpose]} guide {audience[common_noun]}.",
            ),
        ),
        pool_names=("instruction",),
        variable_by=matrix,
    )

    assert "{" not in card and "}" not in card
    assert "Variable:" not in card
    topology = next(
        topology
        for topology in card.deck_topologies
        if topology.name == "variable-by-test"
    )
    assert "for[purpose]" in topology.variable_by
    assert "audience[common_noun]" in topology.variable_by


def test_variable_by_discovers_cells_and_rejects_unknown_template_senses() -> None:
    matrix = VariableBy2D(
        {
            "linker": {"sequence": ("Then", "Next")},
            "label": {"boundary": ("Boundary", "Safety boundary")},
        }
    )

    assert matrix.validate_templates(
        (
            "{label[boundary]}: pause. {linker[sequence]}, verify.",
            "{linker[sequence]}, continue through the trusted channel.",
        )
    ) == ("label[boundary]", "linker[sequence]")
    with pytest.raises(ValueError, match=r"unknown cells: linker\[purpose\]"):
        matrix.validate_templates(("Use this {linker[purpose]} verify.",))


def test_variable_by_resolves_variables_inside_variables() -> None:
    matrix = VariableBy2D(
        {
            "scenario": {"count": ("6",), "total": ("24",)},
            "calculation": {
                "equation": ("{scenario[count]} × 4 = {scenario[total]}",),
            },
            "explanation": {
                "result": ("The verified equation is {calculation[equation]}.",),
            },
        }
    )

    dealt = matrix.deal("nested-calculation")

    assert dealt["calculation"]["equation"] == "6 × 4 = 24"
    assert dealt["explanation"]["result"] == (
        "The verified equation is 6 × 4 = 24."
    )
    assert matrix.dependency_graph()["explanation[result]"] == (
        "calculation[equation]",
    )
    assert matrix.expand_dependencies(("explanation[result]",)) == (
        "explanation[result]",
        "calculation[equation]",
        "scenario[count]",
        "scenario[total]",
    )


def test_casual_variable_by_nests_surface_topic_and_context_cells() -> None:
    registry = json.loads(
        (
            Path(__file__).parents[1]
            / "data/conversation/original/casual-conversation-decks-v1.json"
        ).read_text()
    )
    matrix = casual_variable_by(
        registry["topic_cards"][0],
        registry["context_cards"][0],
        registry["surface_decks"],
    )
    dealt = matrix.deal("casual-variable-by-test")

    assert matrix.dependency_graph()["surface[user_opening]"] == (
        "topic[opening]",
        "context[opening]",
    )
    assert "topic[reply_lower]" in matrix.expand_dependencies(
        ("surface[assistant_follow_up]",)
    )
    assert "{" not in dealt["surface"]["user_opening"]
    assert "}" not in dealt["surface"]["assistant_closing"]


def test_variable_by_rejects_unknown_and_cyclic_nested_variables() -> None:
    unknown = VariableBy2D(
        {"explanation": {"result": ("Use {calculation[missing]}.",)}}
    )
    with pytest.raises(ValueError, match="reservoir references unknown cells"):
        unknown.deal("unknown")

    cyclic = VariableBy2D(
        {
            "first": {"value": ("{second[value]}",)},
            "second": {"value": ("{first[value]}",)},
        }
    )
    with pytest.raises(ValueError, match="cyclic variable_by dependency"):
        cyclic.deal("cycle")


def test_dataset_templates_are_multi_slot_skeletons_not_static_sentences() -> None:
    template_groups = (
        *SAFETY_ANSWER_TEMPLATES.values(),
        *EMPATHY_TEMPLATES.values(),
        *REASONING_TEMPLATES.values(),
        *CRITIQUE_TEMPLATES.values(),
        *BRAINSTORM_GOAL_TEMPLATES.values(),
        *GROUNDED_QA_TEMPLATES.values(),
    )
    for templates in template_groups:
        for template in templates:
            parsed = tuple(Formatter().parse(template))
            fields = tuple(field for _text, field, _spec, _conversion in parsed if field)
            literal_words = sum(
                len(re.findall(r"[A-Za-z]+", text)) for text, *_rest in parsed
            )
            assert len(fields) >= 2, template
            assert literal_words <= 1, template


def _grounded_facts() -> GroundedQAFacts:
    return GroundedQAFacts(
        code="QA2048",
        year=2021,
        battery_hours=14,
        return_days="30",
        exposure_hours=8,
        temperature_change=5,
        owner="Maya",
        delivery_day=24,
        train_number=582,
        departure_hour=16,
        departure_minute="35",
        platform=7,
        python_minor="12",
        release_major=4,
        release_minor=7,
        longest_battery=13,
        other_battery=9,
        status_minute="20",
        ticket_minute="23",
        available_region="EU",
        ticket_region="US",
        failed_operation="upload a file",
        event_day=18,
        event_room=312,
        energy_kwh=486,
        energy_rate=27,
        course_number=340,
        maintenance_day=17,
        sensor_count=9,
        measured_value=63,
        notice_days=28,
        sample_count=19,
        ph_value=7,
        quote_units=64,
        quote_price=72,
        tested_pages=26,
        operating_limit=58,
    )


def test_grounded_qa_is_a_nested_axis_by_sense_reservoir() -> None:
    matrix = grounded_qa_variable_by("product_specs", _grounded_facts())
    graph = matrix.dependency_graph()

    assert graph["source[passage]"] == (
        "source[documented]",
        "source[context]",
        "source[absence]",
    )
    assert graph["answer[complete]"] == (
        "answer[documented]",
        "fact[additional]",
        "boundary[unknown]",
    )
    assert {
        "unknown[subject]",
        "source[absence_clause]",
    } <= set(matrix.expand_dependencies(("answer[complete]",)))

    dealt = matrix.deal("grounded-contract")
    assert "14 hours" in dealt["source"]["passage"]
    assert "unknown" in dealt["answer"]["complete"].lower()
    assert "{" not in dealt["answer"]["complete"]


def test_grounded_unknown_boundaries_have_no_dominant_trigram() -> None:
    domains = (
        "product_specs",
        "policy_excerpt",
        "science_passage",
        "historical_note",
        "project_brief",
        "travel_information",
        "technical_documentation",
        "comparison_table",
        "conflicting_service_reports",
        "public_event_notice",
        "energy_bill",
        "course_catalog",
        "maintenance_log",
        "environmental_report",
        "software_release_note",
        "contract_clause",
        "lab_report",
        "procurement_quote",
        "accessibility_statement",
        "equipment_manual",
    )
    boundaries = [
        grounded_qa_variable_by(domain, _grounded_facts())
        .deal(f"boundary-collision:{domain}:{index}")["boundary"]["unknown"]
        for domain in domains
        for index in range(64)
    ]
    message_counts: Counter[tuple[str, str, str]] = Counter()
    for boundary in boundaries:
        tokens = re.findall(r"[a-z]+", boundary.lower())
        message_counts.update(set(zip(tokens, tokens[1:], tokens[2:], strict=False)))

    top_trigram, top_count = message_counts.most_common(1)[0]
    assert top_count / len(boundaries) < 0.12, {
        "trigram": " ".join(top_trigram),
        "coverage": top_count / len(boundaries),
    }


def test_task_reservoir_contracts_never_pass_locals_as_an_api() -> None:
    source_root = Path(__file__).parents[1] / "src" / "complexity_card_corpus"
    offenders = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
        if "locals()" in path.read_text(encoding="utf-8")
    }

    assert not offenders, offenders


@pytest.mark.parametrize(
    "domain",
    (
        "names",
        "lesson_activity",
        "event_plan",
        "feature_ideas",
        "writing_prompts",
        "low_cost_activity",
        "outreach",
        "workflow",
    ),
)
def test_brainstorming_variable_by_keeps_grammar_senses_compatible(
    domain: str,
) -> None:
    matrix = brainstorming_variable_by(
        domain,
        scale=48,
        days=21,
        rounds=6,
        setting="a pilot setting",
        signal="successful completions",
    )

    for index in range(64):
        dealt = matrix.deal(f"grammar:{domain}:{index}")
        assert dealt["measurement"]["signal"] == dealt["scenario"]["signal"]
        assert dealt["audience"]["common_noun"] in matrix.cards(
            "audience", "common_noun"
        )
        assert dealt["linker"]["measurement"] in {
            "track",
            "measure",
            "compare",
        }
        assert dealt["linker"]["duration"] in {"within", "over", "across"}
        assert dealt["unit"]["trial_round"] in {
            "rounds",
            "test cycles",
            "pilot rounds",
        }
        sentence = (
            f"A trial with 48 {dealt['audience']['common_noun']} could "
            f"{dealt['linker']['measurement']} successful completions through "
            f"6 {dealt['unit']['trial_round']} "
            f"{dealt['linker']['duration']} 21 days."
        )
        assert "  " not in sentence
        assert sentence.endswith("days.")
    assert matrix.dependency_graph()["measurement[signal]"] == (
        "scenario[signal]",
    )


@pytest.mark.parametrize(
    "table",
    (
        {},
        {"": {"purpose": ("to",)}},
        {"for": {}},
        {"for": {"": ("to",)}},
        {"for": {"purpose": ()}},
        {"for": {"purpose": ("to", "to")}},
    ),
)
def test_variable_by_2d_rejects_invalid_tables(table: dict) -> None:
    with pytest.raises(ValueError):
        VariableBy2D(table)


def test_static_text_progress_tracks_remaining_percentage_by_file() -> None:
    tasks_root = Path(__file__).parents[1] / "src" / "complexity_card_corpus" / "tasks"
    paths = tuple(
        tasks_root / filename
        for filename in (
            "action.py",
            "communication.py",
            "extraction.py",
            "knowledge.py",
            "reasoning.py",
            "safety.py",
        )
    )
    progress = analyze_static_text_progress(paths)
    aggregate = aggregate_static_text_progress(progress)
    observed = {row.filename: row for row in progress}

    regressions = {
        filename: (row.static_lines, _STATIC_LINE_CEILINGS[filename])
        for filename, row in observed.items()
        if row.static_lines > _STATIC_LINE_CEILINGS[filename]
    }
    assert not regressions, regressions
    percentages = {
        filename: f"{row.static_percent:.1f}%"
        for filename, row in observed.items()
    }
    assert all(row.static_percent < 5.0 for row in progress), percentages
    assert aggregate.static_lines <= 1, observed
    assert aggregate.static_percent <= 0.5, percentages
    assert all(0.0 <= row.static_ratio <= 1.0 for row in progress)


def test_static_text_progress_recognizes_fstrings_and_2d_format_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "templates.py"
    source.write_text(
        "STATIC = 'This sentence remains completely static across every generated card.'\n"
        "DYNAMIC = 'This template supports {audience[common_noun]} across every "
        "generated card.'\n"
        "def render(count):\n"
        "    return f'This template supports {count} participants across every "
        "generated card.'\n",
        encoding="utf-8",
    )

    progress = analyze_static_text_progress((source,))[0]

    assert progress.static_lines == 1
    assert progress.variable_lines == 2
    assert progress.static_percent == pytest.approx(100 / 3)


def test_template_density_does_not_treat_one_identifier_as_variable_prose(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weak_template.py"
    source.write_text(
        "def render(code):\n"
        "    return f\"Name candidate {code}'s main evidence or clarity problem, "
        "then rewrite it in exactly two sentences.\"\n",
        encoding="utf-8",
    )

    density = analyze_template_density(source)

    assert density.literal_words >= 13
    assert density.variable_fields == 1
    assert density.static_percent > 90.0


def test_template_literal_migration_budget_does_not_regress() -> None:
    tasks_root = Path(__file__).parents[1] / "src" / "complexity_card_corpus" / "tasks"
    observed = {
        filename: analyze_template_density(tasks_root / filename).literal_words
        for filename in _TEMPLATE_LITERAL_CEILINGS
    }

    assert all(
        observed[filename] <= ceiling
        for filename, ceiling in _TEMPLATE_LITERAL_CEILINGS.items()
    ), observed
    assert observed["reasoning.py"] / 4_080 < 0.05
