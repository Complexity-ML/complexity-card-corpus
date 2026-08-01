from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from ..card_staticity import audit_card_staticity
from ..english_morphology import audit_verb_phrases
from ..scenario_integrity import (
    creation_hash as _creation_hash,
    verification_hash as _verification_hash,
)
from ..scenario_surface import audit_scenario_surface
from .schema import (
    MAX_MEAN_SENTENCE_WORDS,
    MAX_QUESTION_RATE,
    MAX_TRANSITIONS_PER_SENTENCE,
    MIN_MEAN_SENTENCE_WORDS,
    MIN_QUESTION_RATE,
    MIN_TRANSITIONS_PER_SENTENCE,
    MIN_UNIQUE_SENTENCE_RATE,
    SCENARIO_PROVENANCE,
    ScenarioForgeRegistry,
)


def audit_scenarios(
    rows: list[dict[str, Any]], registry: ScenarioForgeRegistry
) -> dict[str, Any]:
    expected_total = sum(family.target for family in registry.families)
    if len(rows) != expected_total:
        raise ValueError(f"expected {expected_total} scenarios, found {len(rows)}")
    if len({row["scenario_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate Scenario Forge IDs")
    if len({row["semantic_signature"] for row in rows}) != len(rows):
        raise ValueError("duplicate Scenario Forge semantic signatures")
    if len({row["semantic_payload"] for row in rows}) != len(rows):
        raise ValueError("duplicate Scenario Forge semantic payloads")
    if len({row["situation"] for row in rows}) != len(rows):
        raise ValueError("duplicate Scenario Forge situations")
    for field in ("title", "goal"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"duplicate Scenario Forge {field} values")
    if any(row["model_generated_dialogue"] for row in rows):
        raise ValueError("Scenario Forge must not contain model-generated dialogue")
    if any(row["provenance"] != SCENARIO_PROVENANCE for row in rows):
        raise ValueError("Scenario Forge provenance mismatch")
    family_specs = {family.family_id: family for family in registry.families}
    atom_ids: dict[str, dict[str, dict[str, str]]] = {}
    for family in registry.families:
        atom_ids[family.family_id] = {
            "constraint": {
                atom.label: atom.atom_id for atom in family.constraints
            },
            "state": {atom.label: atom.atom_id for atom in family.states},
            "outcome": {atom.label: atom.atom_id for atom in family.outcomes},
            "fallback": {atom.label: atom.atom_id for atom in family.fallbacks},
        }
    for row in rows:
        expected_creation = _creation_hash(row["semantic_signature"])
        if row["creation_hash"] != expected_creation:
            raise ValueError(f"creation hash mismatch for {row['scenario_id']}")
        if row["scenario_id"] != f"scenario:{expected_creation[:24]}":
            raise ValueError(f"scenario ID mismatch for {row['scenario_id']}")
        if row["verification_hash"] != _verification_hash(row):
            raise ValueError(f"verification hash mismatch for {row['scenario_id']}")
        if len(row["source_structure_keys"]) != len(set(row["source_structure_keys"])):
            raise ValueError(f"duplicate source cards for {row['scenario_id']}")
        if len(row["source_structure_links"]) != len(
            set(row["source_structure_links"])
        ):
            raise ValueError(f"duplicate source links for {row['scenario_id']}")
        if len(row["source_structure_keys"]) != 8:
            raise ValueError(f"incomplete source card hand for {row['scenario_id']}")
        if len(row["source_structure_links"]) != 12:
            raise ValueError(f"incomplete source graph for {row['scenario_id']}")
        adjacency = {card: set() for card in row["source_structure_keys"]}
        for link in row["source_structure_links"]:
            source, target = link.split("->", maxsplit=1)
            if source not in adjacency or target not in adjacency:
                raise ValueError(f"source graph references an unknown card for {row['scenario_id']}")
            adjacency[source].add(target)
            adjacency[target].add(source)
        if any(not neighbours for neighbours in adjacency.values()):
            raise ValueError(f"orphan source card for {row['scenario_id']}")
        reached = set()
        frontier = [row["source_structure_keys"][0]]
        while frontier:
            card = frontier.pop()
            if card in reached:
                continue
            reached.add(card)
            frontier.extend(adjacency[card] - reached)
        if reached != set(adjacency):
            raise ValueError(f"disconnected source graph for {row['scenario_id']}")
        family = family_specs[row["family"]]
        if row["domain"] not in {domain.domain_id for domain in family.domains}:
            raise ValueError(f"unknown source domain for {row['scenario_id']}")
        ids = atom_ids[row["family"]]
        constraint_id = ids["constraint"][row["constraint"]]
        state_id = ids["state"][row["state"]]
        outcome_id = ids["outcome"][row["desired_outcome"]]
        fallback_id = ids["fallback"][row["fallback"]]
        expected_keys = [
            f"family:{row['family']}",
            f"domain:{row['domain']}",
            f"intent:{row['intent']}",
            f"constraint:{constraint_id}",
            f"state:{state_id}",
            f"outcome:{outcome_id}",
            f"fallback:{fallback_id}",
            f"risk:{row['risk_level']}",
        ]
        expected_links = [
            f"family:{row['family']}->domain:{row['domain']}",
            f"family:{row['family']}->intent:{row['intent']}",
            f"domain:{row['domain']}->intent:{row['intent']}",
            f"domain:{row['domain']}->constraint:{constraint_id}",
            f"domain:{row['domain']}->state:{state_id}",
            f"intent:{row['intent']}->outcome:{outcome_id}",
            f"constraint:{constraint_id}->outcome:{outcome_id}",
            f"constraint:{constraint_id}->fallback:{fallback_id}",
            f"state:{state_id}->outcome:{outcome_id}",
            f"state:{state_id}->fallback:{fallback_id}",
            f"risk:{row['risk_level']}->state:{state_id}",
            f"risk:{row['risk_level']}->fallback:{fallback_id}",
        ]
        if row["source_structure_keys"] != expected_keys:
            raise ValueError(f"source card hand mismatch for {row['scenario_id']}")
        if row["source_structure_links"] != expected_links:
            raise ValueError(f"source graph mismatch for {row['scenario_id']}")

    expected_by_family = {
        family.family_id: family.target for family in registry.families
    }
    actual_by_family = Counter(row["family"] for row in rows)
    if dict(actual_by_family) != expected_by_family:
        raise ValueError(
            f"family allocation mismatch: expected {expected_by_family}, "
            f"found {dict(actual_by_family)}"
        )

    payload_contracts = {
        "practical_action": {"requested_action", "missing_fact"},
        "explanation_learning": {
            "learning_goal",
            "misconception_risk",
            "check_for_understanding",
        },
        "troubleshooting": {
            "diagnostic_goal",
            "observed_symptom",
            "stop_condition",
        },
        "writing_transformation": {
            "transformation",
            "source_state",
            "audience_result",
        },
        "planning_comparison": {
            "planning_goal",
            "options_state",
            "decision_rule",
        },
        "conversation_empathy": {
            "conversational_goal",
            "emotion_state",
            "support_boundary",
        },
        "safety_uncertainty": {
            "safe_goal",
            "risk_state",
            "safety_boundary",
            "safe_alternative",
        },
        "grounded_qa": {"question_goal", "evidence_state", "source_boundary"},
        "summarization_synthesis": {
            "summary_goal",
            "source_state",
            "retention_check",
        },
        "extraction_classification": {
            "extraction_goal",
            "record_state",
            "schema_check",
        },
        "reasoning_verification": {
            "reasoning_goal",
            "problem_state",
            "verification_check",
        },
        "critique_revision": {"critique_goal", "draft_state", "revision_check"},
        "brainstorming_creativity": {
            "ideation_goal",
            "idea_state",
            "selection_check",
        },
        "context_clarification": {
            "clarification_goal",
            "ambiguity_state",
            "scope_check",
        },
    }
    common_payload = {
        "domain_context",
        "subject",
        "constraint",
        "current_state",
        "success_condition",
        "fallback",
    }
    for row in rows:
        payload = json.loads(row["semantic_payload"])
        required = common_payload | payload_contracts[row["family"]]
        if not required <= set(payload):
            missing = sorted(required - set(payload))
            raise ValueError(
                f"scenario {row['scenario_id']} is missing payload fields: {missing}"
            )

    domain_counts: dict[str, dict[str, int]] = defaultdict(dict)
    axis_coverage: dict[str, dict[str, int]] = {}
    compatibility_violations: list[str] = []
    for family in registry.families:
        family_rows = [row for row in rows if row["family"] == family.family_id]
        constraint_by_label = {
            value.label: value.atom_id for value in family.constraints
        }
        state_by_label = {value.label: value.atom_id for value in family.states}
        outcome_by_label = {value.label: value.atom_id for value in family.outcomes}
        fallback_by_label = {value.label: value.atom_id for value in family.fallbacks}
        domain_by_id = {value.domain_id: value for value in family.domains}
        for row in family_rows:
            intent_id = row["intent"]
            constraint_id = constraint_by_label[row["constraint"]]
            state_id = state_by_label[row["state"]]
            outcome_id = outcome_by_label[row["desired_outcome"]]
            fallback_id = fallback_by_label[row["fallback"]]
            domain = domain_by_id[row["domain"]]
            if outcome_id not in family.compatibility.intent_outcomes[intent_id]:
                compatibility_violations.append(f"{row['scenario_id']}:intent_outcome")
            if intent_id not in family.compatibility.domain_intents[domain.domain_id]:
                compatibility_violations.append(f"{row['scenario_id']}:domain_intent")
            if (
                constraint_id
                not in family.compatibility.domain_constraints[domain.domain_id]
            ):
                compatibility_violations.append(
                    f"{row['scenario_id']}:domain_constraint"
                )
            if outcome_id not in family.compatibility.state_outcomes[state_id]:
                compatibility_violations.append(f"{row['scenario_id']}:state_outcome")
            if (
                fallback_id not in family.compatibility.state_fallbacks[state_id]
                or fallback_id
                not in family.compatibility.risk_fallbacks[domain.risk_level]
            ):
                compatibility_violations.append(
                    f"{row['scenario_id']}:risk_state_fallback"
                )
        counts = Counter(row["domain"] for row in family_rows)
        domain_counts[family.family_id] = dict(sorted(counts.items()))
        if max(counts.values()) - min(counts.values()) > 1:
            raise ValueError(f"unbalanced domains in family {family.family_id}")
        coverage = {
            "domains": len({row["domain"] for row in family_rows}),
            "intents": len({row["intent"] for row in family_rows}),
            "constraints": len({row["constraint"] for row in family_rows}),
            "states": len({row["state"] for row in family_rows}),
            "outcomes": len({row["desired_outcome"] for row in family_rows}),
        }
        expected_coverage = {
            "domains": len(family.domains),
            "intents": len(family.intents),
            "constraints": len(family.constraints),
            "states": len(family.states),
            "outcomes": len(family.outcomes),
        }
        if coverage != expected_coverage:
            raise ValueError(
                f"incomplete semantic coverage in {family.family_id}: "
                f"expected {expected_coverage}, found {coverage}"
            )
        axis_coverage[family.family_id] = coverage

    if compatibility_violations:
        raise ValueError(
            "Scenario Forge compatibility violations: "
            + ", ".join(compatibility_violations[:10])
        )

    split_counts = Counter(row["split"] for row in rows)
    expected_validation = round(len(rows) * registry.validation_percent / 100)
    if split_counts["validation"] != expected_validation:
        raise ValueError(
            f"expected exactly {expected_validation} validation rows, found "
            f"{split_counts['validation']}"
        )
    train_groups = {
        (row["family"], row["domain"], row["intent"])
        for row in rows
        if row["split"] == "train"
    }
    validation_groups = {
        (row["family"], row["domain"], row["intent"])
        for row in rows
        if row["split"] == "validation"
    }
    leaking_groups = train_groups & validation_groups
    if leaking_groups:
        raise ValueError(
            "validation domain-intent groups leak into training: "
            + ", ".join(map(str, sorted(leaking_groups)[:10]))
        )

    surface_audit = audit_scenario_surface(rows)
    if surface_audit["issues"]:
        raise ValueError(
            "Scenario Forge surface composition lint failed: "
            + ", ".join(
                f"{issue['scenario_id']}:{issue['kind']}"
                for issue in surface_audit["issues"][:10]
            )
        )
    surface_stats = surface_audit["stats"]
    if surface_stats["unique_sentence_rate"] < MIN_UNIQUE_SENTENCE_RATE:
        raise ValueError(
            "Scenario Forge surface language is too repetitive: "
            f"{surface_stats['unique_sentence_rate']:.3f} < "
            f"{MIN_UNIQUE_SENTENCE_RATE:.3f}"
        )
    if not MIN_QUESTION_RATE <= surface_stats["question_rate"] <= MAX_QUESTION_RATE:
        raise ValueError(
            "Scenario Forge question rate is outside the intended range: "
            f"{surface_stats['question_rate']:.3f}"
        )
    if not (
        MIN_MEAN_SENTENCE_WORDS
        <= surface_stats["mean_sentence_words"]
        <= MAX_MEAN_SENTENCE_WORDS
    ):
        raise ValueError(
            "Scenario Forge mean sentence length is outside the post-training "
            f"target: {surface_stats['mean_sentence_words']:.3f}"
        )
    if not (
        MIN_TRANSITIONS_PER_SENTENCE
        <= surface_stats["transitions_per_sentence"]
        <= MAX_TRANSITIONS_PER_SENTENCE
    ):
        raise ValueError(
            "Scenario Forge transition density is outside the post-training "
            f"target: {surface_stats['transitions_per_sentence']:.3f}"
        )

    morphology_audit = audit_verb_phrases(
        [intent.label for family in registry.families for intent in family.intents]
    )
    source_card_staticity = audit_card_staticity(
        [
            {
                key.split(":", 1)[0]: key.split(":", 1)[1]
                for key in row["source_structure_keys"]
            }
            for row in rows
        ]
    )

    return {
        "scenarios": len(rows),
        "unique_ids": len({row["scenario_id"] for row in rows}),
        "unique_semantic_signatures": len({row["semantic_signature"] for row in rows}),
        "unique_semantic_payloads": len({row["semantic_payload"] for row in rows}),
        "unique_situations": len({row["situation"] for row in rows}),
        "unique_titles": len({row["title"] for row in rows}),
        "unique_goals": len({row["goal"] for row in rows}),
        "unique_triggers": len({row["trigger"] for row in rows}),
        "unique_creation_hashes": len({row["creation_hash"] for row in rows}),
        "unique_verification_hashes": len({row["verification_hash"] for row in rows}),
        "source_graph": {
            "cards_per_scenario": 8,
            "links_per_scenario": 12,
            "orphan_cards": 0,
            "connected_scenarios": len(rows),
            "minimum_card_degree": 2,
            "unique_cards": len(
                {
                    card
                    for row in rows
                    for card in row["source_structure_keys"]
                }
            ),
            "unique_links": len(
                {
                    link
                    for row in rows
                    for link in row["source_structure_links"]
                }
            ),
        },
        "card_staticity": source_card_staticity,
        "model_generated_dialogue_rows": 0,
        "family_counts": dict(sorted(actual_by_family.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "axis_coverage": dict(sorted(axis_coverage.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_holdout_unit": "family+domain+intent",
        "split_group_overlap": 0,
        "validation_family_counts": dict(
            sorted(
                Counter(
                    row["family"] for row in rows if row["split"] == "validation"
                ).items()
            )
        ),
        "risk_counts": dict(sorted(Counter(row["risk_level"] for row in rows).items())),
        "surface_stats": surface_stats,
        "surface_language_audit": {
            key: value for key, value in surface_audit.items() if key != "stats"
        },
        "morphology_audit": morphology_audit,
        "payload_contract_match_ratio": 1.0,
        "compatibility_match_ratio": 1.0,
    }
