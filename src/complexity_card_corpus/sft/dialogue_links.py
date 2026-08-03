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


# Family cards keep the conversational edge relevant to the actual task.  The
# selected card IDs live in conditioning metadata; only the rendered sentences
# reach the model.
_FAMILY_OPENING_CARDS: dict[str, tuple[str, ...]] = {
    "brainstorming_creativity": (
        "I want several genuinely different possibilities. {situation}\n\nThe brief currently says: {data}",
        "I have a creative choice to make rather than one answer to recover. {situation}\n\nHere is the working brief: {data}",
    ),
    "context_clarification": (
        "I know part of what I need, but one detail is still open. {situation}\n\nThis is what is established: {data}",
        "I would rather clarify the point that changes the answer than guess. {situation}\n\nThe available information is: {data}",
    ),
    "conversation_empathy": (
        "I want to talk through this without rushing to fix it. {situation}\n\nThe experience I can describe is: {data}",
        "Something about this deserves a careful response. {situation}\n\nHere is the part I can put into words: {data}",
    ),
    "critique_revision": (
        "I want to improve this while keeping its intended meaning. {situation}\n\nHere is the material under review: {data}",
        "A revision is needed, but not every change would help. {situation}\n\nThe current version is: {data}",
    ),
    "explanation_learning": (
        "I am trying to make one idea understandable rather than merely define it. {situation}\n\nThe learning material is: {data}",
        "The explanation should connect the idea to a concrete case. {situation}\n\nWhat the learner has available is: {data}",
    ),
    "extraction_classification": (
        "I need a structured result without inventing missing values. {situation}\n\nThe record to inspect is: {data}",
        "Please separate what the record contains from what its schema requires. {situation}\n\nHere is the source record: {data}",
    ),
    "grounded_qa": (
        "I need an answer that stays within the supplied evidence. {situation}\n\nThe relevant source says: {data}",
        "Please answer from the record rather than general background knowledge. {situation}\n\nThis is the evidence available: {data}",
    ),
    "planning_comparison": (
        "I have alternatives to compare before choosing a plan. {situation}\n\nThe options and facts are: {data}",
        "I want the trade-off to be explicit rather than intuitive. {situation}\n\nHere is the decision material: {data}",
    ),
    "practical_action": (
        "I need a practical next step that fits the circumstances. {situation}\n\nThe usable facts are: {data}",
        "I want to turn this situation into a bounded action. {situation}\n\nHere is what is currently possible: {data}",
    ),
    "reasoning_verification": (
        "I want the result calculated and checked independently. {situation}\n\nThe given inputs are: {data}",
        "A result is needed, but I also want to see that it holds up. {situation}\n\nUse these values: {data}",
    ),
    "safety_uncertainty": (
        "I need a safe response without pretending the uncertainty is resolved. {situation}\n\nWhat is known right now is: {data}",
        "Please keep immediate risk separate from what still needs verification. {situation}\n\nThe available information is: {data}",
    ),
    "summarization_synthesis": (
        "I need the important points brought together for a specific use. {situation}\n\nThe source material is: {data}",
        "Please compress this without losing the decision or open question. {situation}\n\nHere is the material to synthesize: {data}",
    ),
    "troubleshooting": (
        "I want to isolate the problem before changing several things at once. {situation}\n\nThe observed behavior is: {data}",
        "Something is failing, and I need the next diagnostic step. {situation}\n\nThese are the current observations: {data}",
    ),
    "writing_transformation": (
        "I need this rewritten for a clearer purpose while preserving the facts. {situation}\n\nThe source text is: {data}",
        "Please adapt the wording without changing what the text commits to. {situation}\n\nHere is the original material: {data}",
    ),
}


_FAMILY_LINK_CARDS: dict[str, tuple[str, ...]] = {
    "brainstorming_creativity": (
        "What should make the alternatives meaningfully different, and which part of the brief is fixed?",
        "Which criterion should separate the ideas before I recommend one?",
        "I can generate options once the creative boundary and selection test are explicit.",
    ),
    "context_clarification": (
        "Which unresolved detail would materially change the requested result?",
        "What format or scope should I use instead of choosing one silently?",
        "I can preserve the current interpretation while you clarify the one decisive point.",
    ),
    "conversation_empathy": (
        "Would acknowledgment, reflection, or one optional next step feel most useful right now?",
        "I hear the weight of this. Is there one part you want me to stay with first?",
        "I can respond without assuming a motive; tell me whether you want listening or practical support.",
    ),
    "critique_revision": (
        "Which weakness matters most, and what meaning must the revision preserve?",
        "Who is the audience, and what should remain unchanged while the draft improves?",
        "I can prioritize the revision once the main purpose is explicit.",
    ),
    "explanation_learning": (
        "What should the learner be able to explain or apply after this example?",
        "Which part is already understood, and where does the mechanism stop being clear?",
        "I can connect the concept to the example once the learner's current gap is explicit.",
    ),
    "extraction_classification": (
        "Which fields or labels define the required output, and how should missing values be represented?",
        "Should conflicts be preserved, normalized, or returned for review?",
        "I can structure the record once the schema and missing-value rule are fixed.",
    ),
    "grounded_qa": (
        "What exact question should the supplied evidence answer?",
        "Should I report only the supported result, or also state what the source leaves unresolved?",
        "I can answer once the requested scope is separated from unsupported background knowledge.",
    ),
    "planning_comparison": (
        "Which criterion has priority, and which constraint rules out an otherwise attractive option?",
        "What trade-off should decide between the available paths?",
        "I can compare the options once success and the fallback are explicit.",
    ),
    "practical_action": (
        "What outcome should the next step achieve, and what boundary must it respect?",
        "Who can act, and what should be verified before the action proceeds?",
        "I can sequence the work once the immediate result and guardrail are clear.",
    ),
    "reasoning_verification": (
        "What quantity or claim should be established, and which independent check is allowed?",
        "Should the answer preserve a particular precision or unit convention?",
        "I can calculate the result once the target and verification rule are explicit.",
    ),
    "safety_uncertainty": (
        "Is there an immediate risk now, and what safe support is available if the situation worsens?",
        "What is confirmed, and which uncertain part should not be acted on yet?",
        "I can give a bounded next step once the present danger and escalation path are clear.",
    ),
    "summarization_synthesis": (
        "Who is the summary for, and which decision or open point must remain visible?",
        "What length and emphasis would make the synthesis useful?",
        "I can compress the material once the audience and priority are explicit.",
    ),
    "troubleshooting": (
        "What changed most recently, and what single observation would isolate the failure?",
        "Which safe test can distinguish the leading explanations without creating another variable?",
        "I can propose the next diagnostic step once the last known-good state is clear.",
    ),
    "writing_transformation": (
        "Who will read the result, and what tone or meaning must remain intact?",
        "Which part may change freely, and which factual commitment must be preserved?",
        "I can rewrite the text once its audience and non-negotiable meaning are explicit.",
    ),
}


_FAMILY_UPDATE_LEADS: dict[str, tuple[str, ...]] = {
    "brainstorming_creativity": ("Use this selection test", "Keep this part of the brief fixed"),
    "context_clarification": ("The missing detail is", "Use this provisional scope"),
    "conversation_empathy": ("What would help right now is", "Please keep this boundary"),
    "critique_revision": ("The revision should prioritize", "Preserve this meaning"),
    "explanation_learning": ("The learning goal is", "Use this understanding check"),
    "extraction_classification": ("The required schema is", "Apply this missing-value rule"),
    "grounded_qa": ("The exact question is", "Keep the answer within this source limit"),
    "planning_comparison": ("The deciding priority is", "Keep this fallback available"),
    "practical_action": ("The immediate outcome is", "Keep this guardrail in place"),
    "reasoning_verification": ("The value to establish is", "Use this verification rule"),
    "safety_uncertainty": ("The current status is", "Do not cross this safety boundary"),
    "summarization_synthesis": ("The summary is for", "Keep this point visible"),
    "troubleshooting": ("The last relevant change is", "Use only this diagnostic boundary"),
    "writing_transformation": ("The intended audience is", "Preserve this meaning"),
}


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
    task: str = "",
) -> tuple[str, str, str]:
    """Deal one compatible assistant link and the user's grounded update."""

    move = dialogue_link_move(cards, example_id)
    assistant_deck = _FAMILY_LINK_CARDS.get(task, _LINK_ASSISTANT_CARDS[move])
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
    if task in _FAMILY_UPDATE_LEADS:
        leads = _FAMILY_UPDATE_LEADS[task]
        lead = leads[_stable_index(f"family-update:{task}:{example_id}", len(leads))]
        user = f"{lead}: {clean_goal}\n\n{_sentence(rule)}"
    return (
        move,
        correct_indefinite_articles(assistant),
        correct_indefinite_articles(user),
    )


def render_dialogue_opening(
    *, example_id: str, situation: str, data: str, task: str = ""
) -> str:
    deck = _FAMILY_OPENING_CARDS.get(task, _USER_OPENING_CARDS)
    card = deck[
        _stable_index(f"dialogue-user-opening:{task}:{example_id}", len(deck))
    ]
    return correct_indefinite_articles(
        card.format(situation=situation, data=data)
    )


def preserve_linked_dialogue(
    example_id: str,
    *,
    share: int = 5,
    natural_depth: str | None = None,
) -> bool:
    """Retain one deterministic card dialogue in ``share`` as true multi-turn.

    The remaining card variants stay complete two-turn instructions. This
    keeps a useful mixture of direct requests and linked conversations instead
    of teaching the model to ask a clarification on every task.
    """

    if natural_depth is not None:
        if natural_depth not in {"direct", "linked"}:
            raise ValueError(f"unsupported natural dialogue depth: {natural_depth}")
        return natural_depth == "linked"
    if share < 2:
        raise ValueError("linked-dialogue share denominator must be at least two")
    return _stable_index(f"preserve-linked-dialogue:{example_id}", share) == 0


def dialogue_link_card_count() -> int:
    return (
        sum(len(cards) for cards in _LINK_ASSISTANT_CARDS.values())
        + len(_USER_OPENING_CARDS)
        + len(_USER_UPDATE_CARDS)
    )


def family_dialogue_card_count(task: str) -> int:
    """Return the family-specific natural opening/link/update reservoir size."""

    return (
        len(_FAMILY_OPENING_CARDS[task])
        + len(_FAMILY_LINK_CARDS[task])
        + len(_FAMILY_UPDATE_LEADS[task])
    )
