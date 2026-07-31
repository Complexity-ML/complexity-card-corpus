from __future__ import annotations

from typing import Any


CHAT_TEMPLATE_ID = "complexity-chat-v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are Complexity, a concise and grounded assistant. Answer the user "
    "directly. Use provided evidence when present and do not invent missing facts."
)


def chat_template_contract() -> dict[str, Any]:
    """Return the portable prompt contract used by SFT and inference."""

    return {
        "id": CHAT_TEMPLATE_ID,
        "version": 1,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "system_format": "System:\n{content}\n\n",
        "user_format": "User:\n{content}\n\n",
        "assistant_prefix": "Assistant:\n",
        "turn_separator": "\n\n",
        "eos_token": "<|endoftext|>",
        "assistant_only_loss": True,
    }


def render_system_prefix(contract: dict[str, Any] | None = None) -> str:
    template = contract or chat_template_contract()
    return template["system_format"].format(content=template["system_prompt"])


def render_user_turn(content: str, contract: dict[str, Any] | None = None) -> str:
    template = contract or chat_template_contract()
    return template["user_format"].format(content=content.strip())


def render_inference_prompt(
    user_content: str,
    contract: dict[str, Any] | None = None,
) -> str:
    template = contract or chat_template_contract()
    return (
        render_system_prefix(template)
        + render_user_turn(user_content, template)
        + template["assistant_prefix"]
    )
