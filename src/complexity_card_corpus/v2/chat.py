from __future__ import annotations

from typing import Any


CHAT_TEMPLATE_ID = "complexity-chat-v2"
THINK_FINAL_ENVELOPE = {
    "type": "optional_think_final",
    "think_start": "<think>\n",
    "think_end": "\n</think>",
    "final_start": "\n<final>\n",
    "final_end": "\n</final>",
    "scope": "reasoning_tasks",
}


def chat_template_contract() -> dict[str, Any]:
    return {
        "id": CHAT_TEMPLATE_ID,
        "version": 2,
        "system_prompt": "",
        "system_format": "System:\n{content}\n\n",
        "user_format": "User:\n{content}\n\n",
        "assistant_prefix": "Assistant:\n",
        "turn_separator": "\n\n",
        "eos_token": "<|endoftext|>",
        "assistant_only_loss": True,
        "training_projection": "card_corpus_v2_direct",
        "assistant_envelope": dict(THINK_FINAL_ENVELOPE),
    }


def render_system_prefix(contract: dict[str, Any] | None = None) -> str:
    template = contract or chat_template_contract()
    system_prompt = str(template["system_prompt"]).strip()
    if not system_prompt:
        return ""
    return str(template["system_format"]).format(content=system_prompt)


def render_user_turn(
    content: str,
    contract: dict[str, Any] | None = None,
) -> str:
    template = contract or chat_template_contract()
    return str(template["user_format"]).format(content=content.strip())


def render_inference_prompt(
    user_content: str,
    contract: dict[str, Any] | None = None,
) -> str:
    template = contract or chat_template_contract()
    return (
        render_system_prefix(template)
        + render_user_turn(user_content, template)
        + str(template["assistant_prefix"])
    )


__all__ = (
    "CHAT_TEMPLATE_ID",
    "THINK_FINAL_ENVELOPE",
    "chat_template_contract",
    "render_inference_prompt",
    "render_system_prefix",
    "render_user_turn",
)
