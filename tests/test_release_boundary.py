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


def test_release_has_no_retired_external_dataset_pipeline() -> None:
    assert not any((ROOT / "data/mosaic").glob("*"))
    assert not (ROOT / "data/conversation/sources.json").exists()
    for module in ("mosaic.py", "mosaic_stream.py", "oasst1.py"):
        assert not (ROOT / "src/complexity_card_corpus" / module).exists()

    cli_source = (ROOT / "src/complexity_card_corpus/cli.py").read_text()
    for command in (
        "build-mosaic",
        "build-mosaic-shards",
        "tokenize-mosaic-shards",
        "import-oasst1",
        "package-posttrain-hf",
    ):
        assert command not in cli_source


def test_scenario_forge_registry_is_complexity_original() -> None:

    scenario_registry = json.loads(
        (ROOT / "data/scenario-forge/scenario-forge-v1.json").read_text()
    )
    assert scenario_registry["metadata"]["license"] == "CC BY-NC 4.0"
    assert scenario_registry["metadata"]["source"].startswith("Complexity original")
