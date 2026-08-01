from __future__ import annotations

import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_software_and_dataset_licenses_have_explicit_disjoint_scopes() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "Apache-2.0"
    assert set(project["license-files"]) == {"LICENSE", "NOTICE"}

    reuse = tomllib.loads((ROOT / "REUSE.toml").read_text())
    scopes = {
        annotation["SPDX-License-Identifier"]: set(annotation["path"])
        for annotation in reuse["annotations"]
    }
    assert "src/**" in scopes["Apache-2.0"]
    assert "tests/**" in scopes["Apache-2.0"]
    assert "data/**" not in scopes["Apache-2.0"]
    assert scopes["CC-BY-NC-4.0"] == {"data/**"}
    assert (ROOT / "LICENSE").read_text().startswith(
        "                                 Apache License"
    )
    assert "CC BY-NC 4.0" in (ROOT / "DATASET_LICENSE.md").read_text()


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
    for module in ("conversation.py", "mosaic.py", "mosaic_stream.py", "oasst1.py"):
        assert not (ROOT / "src/complexity_card_corpus" / module).exists()

    cli_source = (ROOT / "src/complexity_card_corpus/cli.py").read_text()
    for command in (
        "build-mosaic",
        "build-mosaic-shards",
        'add_parser("build-conversation")',
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


def test_post_training_registry_does_not_include_fantasy_card_collections() -> None:
    registry = json.loads(
        (ROOT / "data/scenario-forge/scenario-forge-v1.json").read_text()
    )
    serialized = json.dumps(registry).lower()
    for dataset_id in (
        "aethoria-v1",
        "aethoria-grand-archive-v1",
        "prismwilds-v1",
        "prismwilds-grand-codex-v1",
    ):
        assert dataset_id not in serialized
