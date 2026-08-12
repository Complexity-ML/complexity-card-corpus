from __future__ import annotations

from typing import Any


CHAT_TEMPLATE_ID = "complexity-chat-v2"
DEFAULT_SYSTEM_PROMPT = ""
THINK_FINAL_ENVELOPE = {
    "type": "optional_think_final",
    "think_start": "<think>\n",
    "think_end": "\n</think>",
    "final_start": "\n<final>\n",
    "final_end": "\n</final>",
    "scope": "reasoning_tasks",
}


def chat_template_contract() -> dict[str, Any]:
    """Return the portable prompt contract used by SFT and inference."""

    return {
        "id": CHAT_TEMPLATE_ID,
        "version": 2,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "system_format": "System:\n{content}\n\n",
        "user_format": "User:\n{content}\n\n",
        "assistant_prefix": "Assistant:\n",
        "turn_separator": "\n\n",
        "eos_token": "<|endoftext|>",
        "assistant_only_loss": True,
        "training_projection": "naturalize_card_hand_preserve_assistant_turns",
        "assistant_envelope": dict(THINK_FINAL_ENVELOPE),
    }


def render_system_prefix(contract: dict[str, Any] | None = None) -> str:
    template = contract or chat_template_contract()
    system_prompt = str(template["system_prompt"]).strip()
    if not system_prompt:
        return ""
    return template["system_format"].format(content=system_prompt)


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
