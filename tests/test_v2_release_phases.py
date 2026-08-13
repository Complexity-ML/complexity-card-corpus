from __future__ import annotations

import json

from complexity_card_corpus.v2 import release


def _row() -> dict[str, object]:
    return {
        "example_id": "v2:test:one",
        "task": "casual_conversation",
        "mode": "chat",
        "difficulty": "easy",
        "domain": "social",
        "language": "en",
        "split": "train",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "prompt": "hello",
        "response": "Hello!",
        "reasoning_envelope": False,
        "reasoning_trace": "",
        "final_response": "Hello!",
        "source_representation": json.dumps(
            {
                "case_id": "anchor:test",
                "facts": {},
                "prompt_subcards": ["hello"],
                "answer_subcards": ["Hello!"],
                "variable_by": [],
                "deck_name": "test",
                "variable_indices": {},
                "variable_card_counts": {},
                "dependency_graph": {},
                "validator": {"kind": "exact", "expected": "Hello!"},
            }
        ),
        "source": "test",
        "license": "CC BY-NC 4.0",
        "version": "2.0.0",
    }


def test_v2_build_never_runs_audits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(release, "render_complete_v2", lambda: [_row()])
    monkeypatch.setattr(
        release,
        "v2_generation_progress",
        lambda: {"example_limit": None, "registered_capacity": 1},
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an audit ran during build")

    for name in (
        "audit_v2_behavior",
        "audit_v2_integrity",
        "audit_v2_distribution",
        "audit_v2_near_duplicates",
        "audit_v2_lengths",
        "audit_v2_splits",
        "audit_v2_tokenization",
    ):
        monkeypatch.setattr(release, name, forbidden)

    manifest = release.build_v2_release(tmp_path / "build")

    assert manifest["examples"] == 1
    assert manifest["quality_status"] == "not_run"
    assert manifest["tests_executed_during_build"] is False
    assert manifest["statistical_audits_executed_during_build"] is False


def test_v2_audit_is_a_separate_phase(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(release, "render_complete_v2", lambda: [_row()])
    monkeypatch.setattr(release, "v2_generation_progress", lambda: {})
    root = tmp_path / "build"
    release.build_v2_release(root)
    calls = []

    def passed(name):
        def run(_rows):
            calls.append(name)
            return {"passed": True}

        return run

    monkeypatch.setattr(release, "audit_v2_behavior", passed("behavior"))
    monkeypatch.setattr(release, "audit_v2_integrity", passed("integrity"))
    monkeypatch.setattr(release, "audit_v2_distribution", passed("distribution"))
    monkeypatch.setattr(release, "audit_v2_near_duplicates", passed("near"))
    monkeypatch.setattr(release, "audit_v2_lengths", passed("length"))
    monkeypatch.setattr(release, "audit_v2_splits", passed("split"))
    monkeypatch.setattr(
        release,
        "audit_v2_family_roadmap",
        lambda _rows, tokenizer_root=None, require_splits=False: {
            "complete_gate_contract": True,
            "split_audit": {"passed": True},
            "families": {"casual_conversation": {"priority": "PASS"}},
            "priority_counts": {"PASS": 1},
            "rows": 1,
            "train_rows": 1,
        },
    )
    monkeypatch.setattr(release, "roadmap_markdown", lambda _roadmap: "# V2")

    report = release.audit_v2_release(root)

    assert report["passed"] is True
    assert calls == ["behavior", "integrity", "distribution", "near", "length", "split"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["phase_status"] == "audited"
    assert manifest["quality_status"] == "passed"


def test_v2_token_manifest_is_release_ready_and_portable(
    monkeypatch,
    tmp_path,
) -> None:
    class Encoding:
        n_vocab = 32_000
        eot_token = 0

        @staticmethod
        def encode(text, disallowed_special=()):
            del disallowed_special
            return [1 + byte % 255 for byte in text.encode()]

        @staticmethod
        def encode_single_token(_token):
            return 0

    monkeypatch.setattr(release, "render_complete_v2", lambda: [_row()])
    monkeypatch.setattr(release, "v2_generation_progress", lambda: {})
    monkeypatch.setattr(
        release,
        "load_encoding",
        lambda _root: (
            Encoding(),
            {"encoding_name": "test-32k", "eos_token": "<|endoftext|>"},
        ),
    )
    artifact = tmp_path / "artifact"
    release.build_v2_release(artifact)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["quality_status"] = "passed"
    manifest["audit"] = {"sha256": "audit-sha"}
    manifest_path.write_text(json.dumps(manifest))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")

    token_manifest = release.tokenize_v2_release(
        artifact,
        tokenizer,
        tmp_path / "tokenized",
    )

    assert token_manifest["release_quality"] == {
        "ready": True,
        "assistant_only_loss": True,
        "reasoning_envelope_version": "card-corpus-v2-think-final-v1",
        "source_audit_sha256": "audit-sha",
    }
    assert token_manifest["source"] == {
        "format": "complexity-card-corpus-v2-release-v1",
        "examples": 1,
        "projected_sha256": manifest["projected"]["sha256"],
    }
    assert str(tmp_path) not in json.dumps(token_manifest)
