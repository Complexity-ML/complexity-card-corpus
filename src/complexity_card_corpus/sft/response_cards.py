from __future__ import annotations

from collections.abc import Mapping

from ..training_cards import TrainingCards


def _ordered_roles(
    clauses: Mapping[str, str],
    response_order: str,
) -> list[str]:
    requested = response_order.split(">")
    ordered = [role for role in requested if role in clauses and clauses[role].strip()]
    ordered.extend(
        role
        for role, text in clauses.items()
        if role not in ordered and text.strip()
    )
    return ordered


def render_response_card_hand(
    clauses: Mapping[str, str],
    *,
    cards: TrainingCards,
) -> str:
    """Compose already-grounded clauses with an invisible structure-card hand.

    This function never creates a fact or conclusion. It only deals an order and
    layout over clauses produced by the family renderer. Lexical bridge variants
    stay in the family renderer because their grammar depends on semantic roles.
    """

    roles = _ordered_roles(clauses, cards.response_order)
    rendered = [clauses[role].strip() for role in roles]
    if not rendered:
        return ""
    if cards.response_layout == "line_breaks":
        return "\n".join(rendered)
    if cards.response_layout == "spaced_lines":
        return "\n\n".join(rendered)
    if cards.response_layout == "opening_break":
        if len(rendered) == 1:
            return rendered[0]
        return f"{rendered[0]}\n\n{' '.join(rendered[1:])}"
    if cards.response_layout == "bullets":
        return "\n".join(f"- {text}" for text in rendered)
    if cards.response_layout == "numbered":
        return "\n".join(
            f"{index}. {text}" for index, text in enumerate(rendered, start=1)
        )
    return " ".join(rendered)


def card_variant(cards: TrainingCards, size: int, *, offset: int = 0) -> int:
    """Map the response hand to a small phrase deck without another RNG."""

    if size < 1:
        raise ValueError("response phrase deck cannot be empty")
    material = "|".join(
        (
            cards.response_bridge,
            cards.response_opening,
            cards.response_layout,
            str(offset),
        )
    )
    # The strings are deliberately stable public card IDs. Summing their bytes
    # gives deterministic cross-process selection without coupling to row order.
    return sum(material.encode("utf-8")) % size
