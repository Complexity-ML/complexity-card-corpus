from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import tiktoken
from tokenizers import Tokenizer as HuggingFaceTokenizer


IGNORE_INDEX = -100


class HuggingFaceEncodingAdapter:
    def __init__(self, tokenizer: HuggingFaceTokenizer, eos_token: str):
        self._tokenizer = tokenizer
        self.n_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
        eos_id = tokenizer.token_to_id(eos_token)
        if eos_id is None:
            raise ValueError(
                f"EOS token is absent from tokenizer vocabulary: {eos_token!r}"
            )
        self.eot_token = int(eos_id)

    def encode(self, text: str, disallowed_special=()) -> list[int]:
        del disallowed_special
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode_single_token(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(token)
        return int(token_id)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(
            [int(token) for token in tokens],
            skip_special_tokens=False,
        )


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def load_encoding(tokenizer_root: Path):
    config_path = tokenizer_root / "tiktoken_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        cache_dir = tokenizer_root / config.get("cache_dir", ".")
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir.resolve())
        return tiktoken.get_encoding(config["encoding_name"]), config

    tokenizer_path = tokenizer_root / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            "Tokenizer must contain tiktoken_config.json or tokenizer.json: "
            f"{tokenizer_root}"
        )
    config: dict[str, Any] = {}
    tokenizer_config_path = tokenizer_root / "tokenizer_config.json"
    special_tokens_path = tokenizer_root / "special_tokens_map.json"
    if tokenizer_config_path.exists():
        config.update(json.loads(tokenizer_config_path.read_text()))
    if special_tokens_path.exists():
        special_tokens = json.loads(special_tokens_path.read_text())
        for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
            config.setdefault(name, special_tokens.get(name))
    eos_token = config.get("eos_token")
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError(f"Hugging Face tokenizer has no EOS token: {tokenizer_root}")
    tokenizer = HuggingFaceTokenizer.from_file(str(tokenizer_path))
    config["encoding_name"] = config.get(
        "encoding_name", f"hf:{tokenizer_root.name}"
    )
    config["backend"] = "huggingface_tokenizers"
    return HuggingFaceEncodingAdapter(tokenizer, eos_token), config


__all__ = (
    "IGNORE_INDEX",
    "HuggingFaceEncodingAdapter",
    "directory_sha256",
    "file_sha256",
    "load_encoding",
)
