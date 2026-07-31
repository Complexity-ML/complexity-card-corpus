from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_published_source_is_complexity_original_cc_by_nc() -> None:
    metadata_paths = sorted((ROOT / "data/source").glob("*/dataset.json"))
    assert metadata_paths

    for path in metadata_paths:
        metadata = json.loads(path.read_text())
        assert metadata["license"] == "CC BY-NC 4.0", path
        assert metadata["source"].startswith("Complexity original"), path


def test_release_registries_do_not_include_third_party_datasets() -> None:
    assert not any((ROOT / "data/mosaic").glob("*"))
    assert not (ROOT / "data/conversation/sources.json").exists()

    scenario_registry = json.loads(
        (ROOT / "data/scenario-forge/scenario-forge-v1.json").read_text()
    )
    assert scenario_registry["metadata"]["license"] == "CC BY-NC 4.0"
    assert scenario_registry["metadata"]["source"].startswith(
        "Complexity original"
    )
