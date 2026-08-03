from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MIN_TANK_CAPACITY_RESERVE_RATIO = 1.50


def audit_scenario_tanks(registry_path: Path) -> dict[str, Any]:
    """Measure authored raw material and unused compatible capacity per family."""
    from .build import load_scenario_registry

    root = json.loads(registry_path.read_text())
    registry = load_scenario_registry(registry_path)
    include_by_stem = {
        Path(include).stem: include for include in root.get("includes", [])
    }
    tanks: dict[str, dict[str, Any]] = {}
    for family in registry.families:
        capacity = family.semantic_signature_capacity()
        atom_counts = {
            "domains": len(family.domains),
            "intents": len(family.intents),
            "constraints": len(family.constraints),
            "states": len(family.states),
            "outcomes": len(family.outcomes),
            "fallbacks": len(family.fallbacks),
            "response_contract_rules": len(family.response_contract),
        }
        reserve_ratio = capacity / family.weight
        tanks[family.family_id] = {
            "path": include_by_stem.get(family.family_id),
            "allocation_weight": family.weight,
            "raw_atom_count": sum(atom_counts.values()),
            "raw_atom_counts": atom_counts,
            "compatible_signature_capacity": capacity,
            "unused_signature_capacity_at_baseline": capacity - family.weight,
            "capacity_reserve_ratio": round(reserve_ratio, 6),
            "hydrated_for_scale": reserve_ratio
            >= MIN_TANK_CAPACITY_RESERVE_RATIO,
        }
    return {
        "tank_count": len(tanks),
        "minimum_capacity_reserve_ratio": MIN_TANK_CAPACITY_RESERVE_RATIO,
        "all_tanks_hydrated_for_scale": all(
            tank["hydrated_for_scale"] for tank in tanks.values()
        ),
        "tanks_requiring_authored_material": sorted(
            tank_id
            for tank_id, tank in tanks.items()
            if not tank["hydrated_for_scale"]
        ),
        "tanks": dict(sorted(tanks.items())),
    }
