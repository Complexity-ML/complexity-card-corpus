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


def render_history_prefix(
    messages: list[dict[str, Any]],
    contract: dict[str, Any] | None = None,
) -> str:
    """Serialize context turns while leaving the final assistant target open."""

    template = contract or chat_template_contract()
    rendered = ""
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if not content:
            raise ValueError("chat history contains an empty turn")
        if role == "system":
            rendered += str(template["system_format"]).format(content=content)
        elif role == "user":
            rendered += str(template["user_format"]).format(content=content)
        elif role == "assistant":
            rendered += (
                str(template["assistant_prefix"])
                + content
                + str(template["eos_token"])
            )
        else:
            raise ValueError(f"unsupported chat history role {role!r}")
    return rendered + str(template["assistant_prefix"])


def validate_training_messages(messages: list[dict[str, Any]]) -> None:
    """Validate one conversation with exactly one model-facing final target.

    Earlier assistant turns are legitimate masked context.  The final assistant
    turn is the only supervised target produced by the V2 tokenizer.
    """

    if not messages:
        raise ValueError("training conversation is empty")
    roles = [str(message.get("role", "")) for message in messages]
    contents = [str(message.get("content", "")).strip() for message in messages]
    if any(not content for content in contents):
        raise ValueError("training conversation contains an empty turn")
    if any(role not in {"system", "user", "assistant"} for role in roles):
        raise ValueError("training conversation contains an unsupported role")
    system_indexes = [index for index, role in enumerate(roles) if role == "system"]
    if system_indexes not in ([], [0]):
        raise ValueError("a system turn is only allowed once at the beginning")
    dialogue = roles[1:] if system_indexes else roles
    if not dialogue or dialogue[0] != "user":
        raise ValueError("training dialogue must begin with a user turn")
    if dialogue[-1] != "assistant":
        raise ValueError("training dialogue must end with the supervised assistant turn")
    expected = "user"
    for role in dialogue:
        if role != expected:
            raise ValueError("training user and assistant turns must alternate")
        expected = "assistant" if expected == "user" else "user"


def conversation_context_text(messages: list[dict[str, Any]]) -> str:
    """Return a role-aware signature of every turn preceding the target."""

    if not messages:
        return ""
    context = messages[:-1] if str(messages[-1].get("role", "")) == "assistant" else messages
    return "\n\n".join(
        f"{str(message.get('role', '')).casefold()}: {str(message.get('content', '')).strip()}"
        for message in context
        if str(message.get("content", "")).strip()
    )


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
    "conversation_context_text",
    "render_inference_prompt",
    "render_history_prefix",
    "render_system_prefix",
    "render_user_turn",
    "validate_training_messages",
)
