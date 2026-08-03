from __future__ import annotations

import hashlib

from ..english_morphology import correct_indefinite_articles
from ..training_cards import TrainingCards
from .language import _inline_sentence, _sentence


LINK_MOVES = (
    "clarification",
    "objection",
    "correction",
    "follow_up",
    "validation",
)


# Each move is a real conversational edge: the first user has supplied the
# situation and evidence, but not yet the requested outcome or hard boundary.
# Five independently authored cards per edge give a 25-card shared linkage
# reservoir without inventing a fact about the underlying scenario.
_LINK_ASSISTANT_CARDS: dict[str, tuple[str, ...]] = {
    "clarification": (
        "I have the facts about {subject}, including that {state} What result should the answer produce, and which limit must it preserve?",
        "The available context for {subject} is clear so far: {state} Please clarify the intended outcome and the non-negotiable boundary.",
        "I can work from this evidence, but two details are still missing. What should be accomplished for {subject}, and what constraint must remain unchanged?",
        "Before I give a specific answer, tell me the desired result for {subject} and the one condition I should not cross.",
        "This establishes the current situation for {subject}, but not the requested destination. What outcome and boundary should guide the response?",
    ),
    "objection": (
        "I should not turn these facts into an action yet. For {subject}, what outcome do you want, and what restriction has priority?",
        "There is one issue with proceeding immediately: the context does not define success or the acceptable boundary. Please supply both for {subject}.",
        "I can help, but acting on the evidence alone would add an assumption. What result and hard limit should govern the answer about {subject}?",
        "I would pause before recommending anything. The facts describe {subject}, while the goal and constraint still need to come from you.",
        "The evidence is usable, but it is not permission to choose an outcome. Please name the result and the boundary for {subject}.",
    ),
    "correction": (
        "One correction before continuing: the supplied facts describe {subject}, but they do not define the goal or boundary. Please provide those separately.",
        "I should separate context from instruction here. For {subject}, what should the response achieve, and which constraint must it retain?",
        "The current record tells me what is happening, not what you want done. Please add the intended result and the limit for {subject}.",
        "To avoid silently broadening the request, I need the outcome and the non-negotiable condition stated explicitly for {subject}.",
        "I can confirm the context, but I should not infer the objective from it. What result and boundary complete the request about {subject}?",
    ),
    "follow_up": (
        "I understand the current position for {subject}. What should happen next, and which condition must the next step respect?",
        "That gives me the factual starting point. Please follow up with the intended outcome and the boundary for {subject}.",
        "I have enough context to continue once you name the result and the limit that should guide the next response.",
        "The next useful turn is to define success and the constraint around it. What are they for {subject}?",
        "I can take this forward, but first tell me the desired destination and the rule that must remain in force.",
    ),
    "validation": (
        "The context is internally usable: {state} To validate a final answer, I still need the target outcome and its boundary for {subject}.",
        "I can validate the supplied facts, but not yet a recommendation. What result and constraint should the recommendation be checked against?",
        "The evidence gives a starting point for {subject}. Please provide the success condition and the limit so I can validate the response against both.",
        "Before closing the reasoning, I need two checks from you: the desired outcome and the boundary that must not be violated.",
        "The factual part is ready. What outcome should count as complete, and which condition should the answer use as its final validation check?",
    ),
}


_USER_OPENING_CARDS = (
    "I need help with a bounded situation. {situation}\n\nHere is the available evidence: {data}",
    "Please consider this context before deciding anything. {situation}\n\nThe current record says: {data}",
    "A concrete case needs a careful response. {situation}\n\nThese facts are confirmed: {data}",
    "I want to separate the facts from the decision. {situation}\n\nWhat is documented so far: {data}",
    "This request starts from a specific condition. {situation}\n\nUse the following information: {data}",
    "Before I name the outcome, review this situation. {situation}\n\nThe supplied evidence is: {data}",
    "The starting point is clear but incomplete. {situation}\n\nI can confirm these details: {data}",
    "Work from this factual base first. {situation}\n\nThe relevant record contains: {data}",
    "I have a case that should remain grounded. {situation}\n\nThe known information is: {data}",
    "Start with the circumstances rather than an assumption. {situation}\n\nThe evidence available to us is: {data}",
    "Here is the condition I am dealing with. {situation}\n\nThe factual material reads: {data}",
    "I would like a response anchored in this context. {situation}\n\nThese are the established facts: {data}",
    "A decision is pending around the following situation. {situation}\n\nThe current source provides: {data}",
    "Let me give the evidence before the request. {situation}\n\nThe usable details are: {data}",
    "This is the factual side of the problem. {situation}\n\nThe record currently establishes: {data}",
    "Please hold off on conclusions while reading this. {situation}\n\nWhat we know at present: {data}",
    "I want the next step to respect this context. {situation}\n\nThe confirmed information follows: {data}",
    "The request concerns one limited scenario. {situation}\n\nIts available data is: {data}",
    "Consider the evidence before choosing a direction. {situation}\n\nThe documented starting point is: {data}",
    "I am sharing the context in two parts. {situation}\n\nFirst, the factual record: {data}",
    "A careful answer should begin here. {situation}\n\nThe source material confirms: {data}",
    "The following case needs a bounded interpretation. {situation}\n\nOnly these facts are available: {data}",
    "Use this situation as context, not as permission. {situation}\n\nThe supporting information is: {data}",
    "I can describe the problem before defining success. {situation}\n\nThe evidence I have is: {data}",
    "One issue is ready for a grounded review. {situation}\n\nIts confirmed details are: {data}",
)


_USER_UPDATE_CARDS = (
    "The result I need is this: {goal}\n\nPlease keep this constraint: {rule}",
    "Here is the outcome I want: {goal}\n\nThe answer must respect this boundary: {rule}",
    "Please {lower_goal}\n\nOne non-negotiable condition applies: {rule}",
    "Success means the following: {goal}\n\nUse this limit when deciding: {rule}",
    "The requested result is: {goal}\n\nKeep the response inside this rule: {rule}",
    "That is the right question. {goal}\n\nThe hard boundary is: {rule}",
    "For the next step, {lower_goal}\n\nDo not relax this condition: {rule}",
    "The answer should accomplish this: {goal}\n\nIt must also preserve this constraint: {rule}",
    "Now I can define success: {goal}\n\nKeep this restriction in place: {rule}",
    "The direction I want is the following: {goal}\n\nTreat this condition as fixed: {rule}",
    "Use this as the completion criterion: {goal}\n\nThe response may not cross this boundary: {rule}",
    "My intended destination is clear now: {goal}\n\nRetain this non-negotiable limit: {rule}",
    "Please work toward this result: {goal}\n\nApply this rule throughout: {rule}",
    "The useful outcome would be: {goal}\n\nOne guardrail remains mandatory: {rule}",
    "Here is what a complete answer should deliver: {goal}\n\nPreserve this condition while doing so: {rule}",
    "I can add the missing instruction now: {goal}\n\nThe hard constraint is: {rule}",
    "The next response should achieve this: {goal}\n\nJudge it against this limit: {rule}",
    "This is the result I am asking for: {goal}\n\nDo not change this boundary: {rule}",
    "The requested endpoint is now explicit: {goal}\n\nThe following rule still governs it: {rule}",
    "Please complete the request in this way: {goal}\n\nRemain inside this constraint: {rule}",
    "What I need from the answer is: {goal}\n\nThe controlling condition is: {rule}",
    "I want to resolve the case as follows: {goal}\n\nKeep this requirement unchanged: {rule}",
    "The final response has one job: {goal}\n\nIt must honor this restriction: {rule}",
    "Use the following outcome as the target: {goal}\n\nThe fixed boundary is: {rule}",
    "The missing objective is this: {goal}\n\nThe applicable guardrail is: {rule}",
)


def _stable_index(value: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % size


def dialogue_link_move(cards: TrainingCards, example_id: str) -> str:
    aliases = {
        "clarification_resolved": "clarification",
        "constraint_update": "correction",
        "continued_request": "follow_up",
    }
    move = aliases.get(cards.dialogue_state, cards.dialogue_state)
    if move in LINK_MOVES:
        return move
    return LINK_MOVES[_stable_index(f"dialogue-link:{example_id}", len(LINK_MOVES))]


def render_dialogue_link(
    *,
    cards: TrainingCards,
    example_id: str,
    subject: str,
    state: str,
    goal: str,
    rule: str,
) -> tuple[str, str, str]:
    """Deal one compatible assistant link and the user's grounded update."""

    move = dialogue_link_move(cards, example_id)
    assistant_deck = _LINK_ASSISTANT_CARDS[move]
    variant = _stable_index(
        f"dialogue-link-card:{move}:{example_id}", len(assistant_deck)
    )
    clean_subject = subject.strip().rstrip(".") or "the request"
    clean_state = state.strip().rstrip(".")
    if not clean_state:
        clean_state = "the current facts establish a bounded starting point"
    assistant = assistant_deck[variant].format(
        subject=clean_subject,
        state=_inline_sentence(clean_state).rstrip(".!?"),
    )
    update_variant = _stable_index(
        f"dialogue-user-update:{move}:{example_id}", len(_USER_UPDATE_CARDS)
    )
    clean_goal = _sentence(goal)
    user = _USER_UPDATE_CARDS[update_variant].format(
        goal=clean_goal,
        lower_goal=clean_goal[:1].lower() + clean_goal[1:],
        rule=_sentence(rule),
    )
    return (
        move,
        correct_indefinite_articles(assistant),
        correct_indefinite_articles(user),
    )


def render_dialogue_opening(
    *, example_id: str, situation: str, data: str
) -> str:
    card = _USER_OPENING_CARDS[
        _stable_index(f"dialogue-user-opening:{example_id}", len(_USER_OPENING_CARDS))
    ]
    return correct_indefinite_articles(
        card.format(situation=situation, data=data)
    )


def preserve_linked_dialogue(example_id: str, *, share: int = 5) -> bool:
    """Retain one deterministic card dialogue in ``share`` as true multi-turn.

    The remaining card variants stay complete two-turn instructions. This
    keeps a useful mixture of direct requests and linked conversations instead
    of teaching the model to ask a clarification on every task.
    """

    if share < 2:
        raise ValueError("linked-dialogue share denominator must be at least two")
    return _stable_index(f"preserve-linked-dialogue:{example_id}", share) == 0


def dialogue_link_card_count() -> int:
    return (
        sum(len(cards) for cards in _LINK_ASSISTANT_CARDS.values())
        + len(_USER_OPENING_CARDS)
        + len(_USER_UPDATE_CARDS)
    )
