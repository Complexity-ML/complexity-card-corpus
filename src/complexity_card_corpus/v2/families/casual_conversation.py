from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from ...variable_by import VariableBy2D
from ..contracts import (
    ConversationTurn,
    RoleSeparatedVariableBy,
    SemanticFrame,
    SurfaceRole,
)
from ..decks import (
    V2RoleSeparatedDeck,
    V2SubcardPool,
    answer_variant_plans,
    prompt_variant_plans,
)
from ._common import render_v2_row, validate_complete_rows


TASK = "casual_conversation"

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)


def _number_words(value: int) -> str:
    if not 0 <= value < 10_000:
        raise ValueError(f"number outside authored word range: {value}")
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if not ones else f"{_TENS[tens]}-{_ONES[ones]}"
    if value < 1_000:
        hundreds, rest = divmod(value, 100)
        head = f"{_ONES[hundreds]} hundred"
        return head if not rest else f"{head} {_number_words(rest)}"
    thousands, rest = divmod(value, 1_000)
    head = f"{_number_words(thousands)} thousand"
    return head if not rest else f"{head} {_number_words(rest)}"


_ANCHORS = (
    ("hello", "social", "Hello! How can I help?", "greeting"),
    ("what is your name?", "identity", "I'm TR-HASH, an AI assistant.", "identity"),
    ("what is 2 + 2?", "addition", "2 + 2 = 4.", "arithmetic"),
    ("what is the capital of france?", "general_knowledge", "The capital of France is Paris.", "fact"),
    ("name exactly three colors, separated by commas.", "instruction_formatting", "red, blue, green", "formatting"),
    ("tell me a short joke.", "social", "Why did the scarecrow win an award? He was outstanding in his field.", "humor"),
    (
        "a stranger asks me to send them my password so they can fix my account. what should i do?",
        "account_safety",
        "Do not send your password. Contact the service through its official support channel instead.",
        "safety",
    ),
    (
        "i have crushing chest pain and trouble breathing. what should i do?",
        "emergency_health",
        "Call emergency services now and follow the dispatcher's instructions while help is coming.",
        "safety",
    ),
    (
        "can i mix bleach and ammonia to clean faster?",
        "chemical_safety",
        "No. Never mix bleach and ammonia; the combination can release toxic chloramine gas.",
        "safety",
    ),
    (
        "someone asked for my one-time login code. should i send it?",
        "account_safety",
        "Do not share the code. Legitimate support staff do not need your one-time login code.",
        "safety",
    ),
    (
        "i might hurt myself tonight. what should i do?",
        "crisis_support",
        "Call emergency services or a crisis hotline now, and stay with someone you trust while you get immediate help.",
        "safety",
    ),
    ("what is the chemical formula for water?", "general_knowledge", "The formula for water is H2O.", "fact"),
    ("which planet is the largest in the solar system?", "general_knowledge", "Jupiter is the largest planet in the Solar System.", "fact"),
    ("how many minutes are in one hour?", "general_knowledge", "There are 60 minutes in one hour.", "fact"),
)

_ARITHMETIC = (
    (
        "addition",
        tuple((left, right) for left in range(3, 103) for right in range(2, 72)),
        lambda left, right: left + right,
        (
            "Add {scenario[left_words]} and {scenario[right_words]}.",
            "What total comes from {scenario[left_words]} plus {scenario[right_words]}?",
            "Find the sum of {scenario[left_words]} and {scenario[right_words]}.",
            "Combine {scenario[left_words]} with {scenario[right_words]}. What is the total?",
            "How much is {scenario[left_words]} added to {scenario[right_words]}?",
            "Calculate {scenario[left_words]} plus {scenario[right_words]}.",
        ),
        (
            "The total is {scenario[result_words]}; removing {scenario[right_words]} recovers {scenario[left_words]}.",
            "It comes to {scenario[result_words]}. Take away {scenario[right_words]} and {scenario[left_words]} remains.",
            "The sum is {scenario[result_words]}; its difference from {scenario[right_words]} is {scenario[left_words]}.",
            "Together they make {scenario[result_words]}. Subtracting {scenario[left_words]} leaves {scenario[right_words]}.",
        ),
    ),
    (
        "subtraction",
        tuple((left, right) for left in range(52, 152) for right in range(2, 52)),
        lambda left, right: left - right,
        (
            "Subtract {scenario[right_words]} from {scenario[left_words]}.",
            "What remains after taking {scenario[right_words]} away from {scenario[left_words]}?",
            "Find the difference between {scenario[left_words]} and {scenario[right_words]}.",
            "Reduce {scenario[left_words]} by {scenario[right_words]}. What remains?",
            "Calculate {scenario[left_words]} minus {scenario[right_words]}.",
            "How much smaller is {scenario[left_words]} after removing {scenario[right_words]}?",
        ),
        (
            "The difference is {scenario[result_words]}; restoring {scenario[right_words]} returns {scenario[left_words]}.",
            "That leaves {scenario[result_words]}. Adding back {scenario[right_words]} gives {scenario[left_words]}.",
            "The remainder is {scenario[result_words]}; it reaches {scenario[left_words]} when {scenario[right_words]} is restored.",
            "It becomes {scenario[result_words]}. The removed {scenario[right_words]} would rebuild {scenario[left_words]}.",
        ),
    ),
    (
        "multiplication",
        tuple((left, right) for left in range(2, 62) for right in range(2, 37)),
        lambda left, right: left * right,
        (
            "Multiply {scenario[left_words]} by {scenario[right_words]}.",
            "What product comes from {scenario[left_words]} times {scenario[right_words]}?",
            "Find the product of {scenario[left_words]} and {scenario[right_words]}.",
            "Calculate {scenario[left_words]} groups of {scenario[right_words]}.",
            "How much is {scenario[left_words]} multiplied by {scenario[right_words]}?",
            "Evaluate {scenario[left_words]} times {scenario[right_words]}.",
        ),
        (
            "The product is {scenario[result_words]}; dividing it by {scenario[right_words]} returns {scenario[left_words]}.",
            "It equals {scenario[result_words]}. Split that into {scenario[left_words]} groups and each has {scenario[right_words]}.",
            "The multiplication gives {scenario[result_words]}; {scenario[right_words]} equal groups would contain {scenario[left_words]} each.",
            "Together the groups contain {scenario[result_words]}. Dividing by {scenario[left_words]} gives {scenario[right_words]}.",
        ),
    ),
    (
        "division",
        tuple(
            (divisor * quotient, divisor)
            for divisor in range(2, 42)
            for quotient in range(2, 52)
        ),
        lambda left, right: left // right,
        (
            "Divide {scenario[left_words]} by {scenario[right_words]}.",
            "How many groups of {scenario[right_words]} fit into {scenario[left_words]}?",
            "Find the quotient of {scenario[left_words]} and {scenario[right_words]}.",
            "Split {scenario[left_words]} into equal groups of {scenario[right_words]}. How many groups result?",
            "Calculate {scenario[left_words]} divided by {scenario[right_words]}.",
            "What does {scenario[left_words]} become when divided by {scenario[right_words]}?",
        ),
        (
            "The quotient is {scenario[result_words]}; multiplying it by {scenario[right_words]} reconstructs {scenario[left_words]}.",
            "It gives {scenario[result_words]} groups. Their combined size returns {scenario[left_words]} from groups of {scenario[right_words]}.",
            "The division yields {scenario[result_words]}; {scenario[result_words]} groups of {scenario[right_words]} total {scenario[left_words]}.",
            "There are {scenario[result_words]} groups. Multiplying by the group size, {scenario[right_words]}, gives {scenario[left_words]}.",
        ),
    ),
)

_WORDS = (
    "amber", "blue", "coral", "green", "indigo", "ivory", "lilac", "maroon",
    "navy", "ochre", "olive", "orange", "peach", "pink", "plum", "purple",
    "red", "silver", "teal", "violet", "white", "yellow", "bronze", "crimson",
    "cyan", "gold", "gray", "magenta", "mint", "scarlet",
)
_FACTS = (
    ("the capital of Italy", "Rome"), ("the capital of Spain", "Madrid"),
    ("the capital of Japan", "Tokyo"), ("the capital of Canada", "Ottawa"),
    ("the capital of Australia", "Canberra"), ("the capital of Germany", "Berlin"),
    ("the capital of Portugal", "Lisbon"), ("the capital of Norway", "Oslo"),
    ("the capital of Sweden", "Stockholm"), ("the capital of Finland", "Helsinki"),
    ("the planet closest to the Sun", "Mercury"),
    ("the process plants use to convert light into chemical energy", "photosynthesis"),
    ("the gas humans need for cellular respiration", "oxygen"),
    ("the chemical symbol for gold", "Au"),
    ("the number of days in a leap year", "366"),
    ("the number of sides on a hexagon", "six"),
    ("the author of Pride and Prejudice", "Jane Austen"),
    ("the largest ocean on Earth", "the Pacific Ocean"),
    ("the instrument used to measure temperature", "a thermometer"),
    ("the natural satellite that orbits Earth", "the Moon"),
)
_ARITHMETIC_PROMPT_FUNCTIONS = (
    ("request_operation",),
    ("request_result", "name_operation"),
    ("request_operation", "name_operands"),
    ("frame_quantity", "request_result"),
    ("request_result", "name_operands"),
    ("request_calculation", "name_operation"),
)
_ARITHMETIC_ANSWER_FUNCTIONS = (
    ("state_result", "verify_inverse"),
    ("state_result", "verify_reconstruction"),
    ("state_result", "compare_operand"),
    ("state_group_total", "verify_inverse"),
)
_COMPARISON_PROMPT_FUNCTIONS = (
    ("request_greater_value",),
    ("request_larger_choice",),
    ("request_comparison", "request_larger_value"),
    ("present_pair", "request_greater_value"),
)
_COMPARISON_ANSWER_FUNCTIONS = (
    ("state_greater", "state_smaller"),
    ("state_answer", "rank_other"),
    ("issue_choice", "justify_comparison"),
    ("state_smaller", "derive_greater"),
)
_SORTING_PROMPT_FUNCTIONS = (
    ("request_ascending_sort",),
    ("request_ascending_order",),
    ("request_ascending_sequence",),
    ("request_low_to_high",),
)
_SORTING_ANSWER_FUNCTIONS = (
    ("state_sequence", "state_gaps"),
    ("label_ordered_digits", "state_increases"),
    ("state_sequence", "state_adjacent_differences"),
    ("label_arrangement", "state_gaps"),
)
_FACT_PROMPT_FUNCTIONS = (
    ("ask_fact",),
    ("request_direct_answer", "ask_fact"),
    ("request_identification",),
    ("request_name",),
    ("request_direct_fact",),
)
_FACT_ANSWER_FUNCTIONS = (
    ("state_answer",),
    ("state_identity",),
    ("answer_only",),
)
_FOLLOW_UP_CONTEXTS = (
    ("event", "tickets"),
    ("garden", "seedlings"),
    ("workshop", "notebooks"),
    ("delivery", "parcels"),
)
_FOLLOW_UP_PROMPTS = (
    "And with {scenario[increment_words]} more?",
    "What if another {scenario[increment_words]} arrive?",
    "Add {scenario[increment_words]} to that—what is the new total?",
    "How many would there be after {scenario[increment_words]} more?",
)
_FOLLOW_UP_ANSWERS = (
    "{scenario[result_words]} {scenario[items]}.",
    "That would make {scenario[result_words]} {scenario[items]}.",
    "The updated count is {scenario[result_words]} {scenario[items]}.",
    "After the addition, there would be {scenario[result_words]} {scenario[items]}.",
)
_FOLLOW_UP_PROMPT_FUNCTIONS = (
    ("elliptical_follow_up", "add_increment"),
    ("counterfactual_follow_up", "add_increment"),
    ("refer_back", "request_new_total"),
    ("request_updated_count", "add_increment"),
)
_FOLLOW_UP_ANSWER_FUNCTIONS = (
    ("answer_only",),
    ("state_updated_total",),
    ("label_updated_count", "state_total"),
    ("mark_operation", "state_total"),
)

_NATURAL_SOCIAL_SPECS = (
    {
        "intent": "friendly_greeting_with_help",
        "domain": "social_greeting",
        "subjects": (
            "an email draft", "a planning question", "a short calculation",
            "a paragraph I am revising", "a decision between two options",
            "a confusing instruction", "a quick fact check", "a study question",
            "a list I need to organize", "an idea I want to develop",
            "a question I want to phrase clearly",
        ),
        "circumstances": (
            "before a meeting", "during a quiet break", "before I send it",
            "while the details are fresh", "before the end of the day",
            "while I have a few minutes", "before I make a choice",
            "as I get started", "before my next task", "while I review my notes",
        ),
        "prompts": (
            "Hello! Could you help me with {scenario[subject]} {scenario[circumstance]}?",
            "Hi—I've got {scenario[subject]} to work through {scenario[circumstance]}. Can you help?",
            "Hey there. Do you have a moment for {scenario[subject]} {scenario[circumstance]}?",
            "Good to meet you. I'd like a hand with {scenario[subject]} {scenario[circumstance]}.",
            "Hello, can we look at {scenario[subject]} together {scenario[circumstance]}?",
            "Hi! I could use some help with {scenario[subject]} {scenario[circumstance]}.",
            "Hey, would you help me sort out {scenario[subject]} {scenario[circumstance]}?",
            "Hello. I have {scenario[subject]} in front of me {scenario[circumstance]}.",
            "Hi there—may I ask about {scenario[subject]} {scenario[circumstance]}?",
            "Hello! Let's work on {scenario[subject]} {scenario[circumstance]}, if you're available.",
        ),
        "answers": (
            "Hi! Of course—I can help with {scenario[subject]}. What would you like to tackle first?",
            "Hello! Send me the details of {scenario[subject]}, and we'll take it one step at a time.",
            "Hey! I'd be glad to help with {scenario[subject]}. Show me what you have so far.",
            "Hi there. We can examine {scenario[subject]}; start wherever it feels unclear.",
            "Hello! Yes, we can work through {scenario[subject]}. What's the main thing you need from it?",
            "Hi! I'm ready to help with {scenario[subject]}. Paste it here when you're ready.",
            "Hey—absolutely. Tell me what is giving you trouble in {scenario[subject]}.",
            "Hello. We can keep this focused on {scenario[subject]}; what should the result look like?",
            "Hi there! Share {scenario[subject]}, and I'll help you move it forward.",
            "Hello! Let's get started with {scenario[subject]}. What context should I know?",
        ),
    },
    {
        "intent": "receive_thanks",
        "domain": "social_gratitude",
        "subjects": (
            "the summary", "the calculation", "the rewrite", "the explanation",
            "the checklist", "the comparison", "the example", "the plan",
            "the correction", "the step-by-step breakdown",
            "the final answer",
        ),
        "circumstances": (
            "it makes much more sense now", "that cleared up my confusion",
            "I can continue on my own", "that was exactly what I needed",
            "the next step is clear", "I found the mistake",
            "I know what to change", "the shorter version works better",
            "I can explain it now", "that saved me some time",
        ),
        "prompts": (
            "Thanks for {scenario[subject]}; {scenario[circumstance]}.",
            "Thank you—{scenario[subject]} helped, and {scenario[circumstance]}.",
            "I appreciate your help with {scenario[subject]}. {scenario[circumstance_cap]}.",
            "That was helpful, thanks. With {scenario[subject]}, {scenario[circumstance]}.",
            "Thanks! After reading {scenario[subject]}, {scenario[circumstance]}.",
            "Much appreciated. {scenario[subject_cap]} worked well because {scenario[circumstance]}.",
            "Thank you for taking care of {scenario[subject]}; {scenario[circumstance]}.",
            "I just wanted to say thanks for {scenario[subject]}. {scenario[circumstance_cap]}.",
            "Perfect, thank you. From {scenario[subject]}, {scenario[circumstance]}.",
            "Thanks for the help—after {scenario[subject]}, {scenario[circumstance]}.",
        ),
        "answers": (
            "You're welcome! I'm glad the work on {scenario[subject]} helped.",
            "Happy to help. It's good to hear that {scenario[subject]} made things clearer.",
            "You're very welcome—glad {scenario[subject]} was useful.",
            "Anytime! I'm pleased the help with {scenario[subject]} got you moving again.",
            "Glad I could help with {scenario[subject]}. Good luck with the next step!",
            "You're welcome. It's great that {scenario[subject]} gave you what you needed.",
            "My pleasure! I'm glad we could sort out {scenario[subject]}.",
            "Of course. Happy that {scenario[subject]} worked for you.",
            "You're welcome! Nice to know the help with {scenario[subject]} paid off.",
            "Glad to hear it. Come back anytime you want another look at {scenario[subject]}.",
        ),
    },
    {
        "intent": "close_conversation",
        "domain": "social_closing",
        "subjects": (
            "the message", "the budget check", "the study plan", "the outline",
            "the comparison", "the troubleshooting notes", "the short summary",
            "the revised paragraph", "the task list", "the final calculation",
            "the cleaned-up notes",
        ),
        "circumstances": (
            "head to a meeting", "send the result", "take a break",
            "continue with the rest of my work", "share it with the team",
            "review the final version", "start the next task", "save my notes",
            "make the decision", "finish for the day",
        ),
        "prompts": (
            "That covers {scenario[subject]}; I need to {scenario[circumstance]} now. Bye!",
            "I'm all set with {scenario[subject]}. Time to {scenario[circumstance]}—thanks and goodbye.",
            "We can stop here; {scenario[subject]} is done and I should {scenario[circumstance]}.",
            "That's everything I needed for {scenario[subject]}. I'm going to {scenario[circumstance]} now.",
            "Looks good to me. I'll {scenario[circumstance]} after finishing {scenario[subject]}. See you!",
            "No more questions about {scenario[subject]}; I have to {scenario[circumstance]}. Thanks!",
            "Great, {scenario[subject]} is sorted. I'll {scenario[circumstance]} next. Bye for now.",
            "I'm done with {scenario[subject]} for the moment and need to {scenario[circumstance]}. Take care.",
            "That is enough for {scenario[subject]}. I can {scenario[circumstance]} now—goodbye.",
            "Let's leave it there. With {scenario[subject]} complete, I'll {scenario[circumstance]}.",
        ),
        "answers": (
            "Sounds good. I'm glad we got {scenario[subject]} sorted—take care!",
            "You're all set. Good luck when you {scenario[circumstance]}, and see you next time.",
            "Great! I hope {scenario[subject]} serves you well. Until next time.",
            "Happy we could finish {scenario[subject]}. Have a good rest of your day!",
            "Understood. Go ahead and {scenario[circumstance]}; I'll be here if you need anything else.",
            "Glad that is settled. Take care, and good luck with {scenario[subject]}.",
            "All right—thanks for the conversation. I hope {scenario[subject]} works out well.",
            "Perfect. You can {scenario[circumstance]} with that behind you. See you!",
            "You're welcome. Have a smooth time with {scenario[circumstance]}.",
            "Goodbye! Come back anytime you want to revisit {scenario[subject]}.",
        ),
    },
    {
        "intent": "ask_for_help",
        "domain": "social_help",
        "subjects": (
            "draft a short email", "check a small calculation", "simplify a paragraph",
            "compare two choices", "organize a short list", "explain a concept",
            "spot an error in my notes", "plan the next three steps",
            "turn notes into a summary", "rewrite a sentence more clearly",
            "prepare a direct question",
        ),
        "circumstances": (
            "with a five-minute time limit", "without adding new facts",
            "in plain English", "with a concise answer", "one step at a time",
            "for a non-specialist reader", "while keeping the original meaning",
            "with the key result first", "without unnecessary jargon",
            "so I can verify the result myself",
        ),
        "prompts": (
            "Can you help me {scenario[subject]} {scenario[circumstance]}?",
            "Would you mind helping me {scenario[subject]} {scenario[circumstance]}?",
            "I could use a hand to {scenario[subject]} {scenario[circumstance]}.",
            "Could we {scenario[subject]} together {scenario[circumstance]}?",
            "Please help me {scenario[subject]} {scenario[circumstance]}.",
            "Are you able to {scenario[subject]} {scenario[circumstance]}?",
            "I'd like your help to {scenario[subject]} {scenario[circumstance]}.",
            "Can we work on how to {scenario[subject]} {scenario[circumstance]}?",
            "I need to {scenario[subject]} {scenario[circumstance]}; can you assist?",
            "Could you guide me as I {scenario[subject]} {scenario[circumstance]}?",
        ),
        "answers": (
            "Yes—I can help you {scenario[subject]} {scenario[circumstance]}. Send me the material you have.",
            "Absolutely. Share the details, and we'll {scenario[subject]} {scenario[circumstance]}.",
            "Of course. We can {scenario[subject]} {scenario[circumstance]}; start with the current version.",
            "Sure. Show me what you're working with, and I'll help you {scenario[subject]} {scenario[circumstance]}.",
            "I'd be glad to help. Paste the relevant part so we can {scenario[subject]} {scenario[circumstance]}.",
            "Yes. Tell me the intended outcome, and we'll {scenario[subject]} {scenario[circumstance]}.",
            "Certainly—send it over. I'll focus on helping you {scenario[subject]} {scenario[circumstance]}.",
            "We can do that. Give me the input, and I'll help {scenario[subject]} {scenario[circumstance]}.",
            "Sure thing. Let's {scenario[subject]} {scenario[circumstance]}; what should I look at first?",
            "Yes, I can assist. Share the context needed to {scenario[subject]} {scenario[circumstance]}.",
        ),
    },
    {
        "intent": "repair_understanding",
        "domain": "social_repair",
        "subjects": (
            "the difference between the two options", "why the total changed",
            "what the first step accomplishes", "which detail matters most",
            "how the example supports the conclusion", "what you mean by the boundary",
            "why that check is useful", "how the two sentences connect",
            "when the rule applies", "what I should do next",
            "why the answer changed",
        ),
        "circumstances": (
            "your last explanation", "the example above", "the shorter answer",
            "the second paragraph", "the list of steps", "the comparison you gave",
            "the previous message", "the final sentence", "the worked example",
            "the answer you just sent",
        ),
        "prompts": (
            "I didn't follow {scenario[subject]} in {scenario[circumstance]}. Could you rephrase it?",
            "What did you mean by {scenario[subject]} in {scenario[circumstance]}?",
            "I'm still confused about {scenario[subject]} after {scenario[circumstance]}. Can you say it another way?",
            "Could you clarify {scenario[subject]} from {scenario[circumstance]}?",
            "I lost the thread around {scenario[subject]} in {scenario[circumstance]}. Please explain it more simply.",
            "Can we revisit {scenario[subject]} from {scenario[circumstance]}? It was not clear to me.",
            "I understand most of it, but not {scenario[subject]} in {scenario[circumstance]}. Could you help?",
            "Please unpack {scenario[subject]} from {scenario[circumstance]} without using the same wording.",
            "Could you give a fresh explanation of {scenario[subject]} instead of repeating {scenario[circumstance]}?",
            "I'm not sure I understood {scenario[subject]} in {scenario[circumstance]}. What is the plain version?",
        ),
        "answers": (
            "Of course. I'll restate {scenario[subject]} in simpler terms and use a fresh example.",
            "Sure—let me explain {scenario[subject]} from a different angle.",
            "Absolutely. I'll focus only on {scenario[subject]} and avoid repeating the earlier wording.",
            "Yes. Let's slow down and make {scenario[subject]} explicit before moving on.",
            "No problem. I'll give a shorter explanation of {scenario[subject]} and check that it lands.",
            "Thanks for pointing that out. Here's another way to understand {scenario[subject]}.",
            "Certainly. I'll separate {scenario[subject]} from the surrounding details first.",
            "I can clarify that. Let's rebuild the explanation of {scenario[subject]} one piece at a time.",
            "Yes—I'll replace the abstract wording with a concrete account of {scenario[subject]}.",
            "Let's revisit it. I'll explain {scenario[subject]} plainly, then you can tell me what remains unclear.",
        ),
    },
)


def _row(
    *,
    case_id: str,
    domain: str,
    difficulty: str,
    facts: dict[str, Any],
    prompts: tuple[str, ...],
    answers: tuple[str, ...],
    validator: dict[str, Any],
    prompt_functions: tuple[tuple[str, ...], ...] | None = None,
    answer_functions: tuple[tuple[str, ...], ...] | None = None,
    semantic_frame: SemanticFrame | None = None,
    prompt_choice: int | None = None,
    answer_choice: int | None = None,
) -> dict[str, object]:
    scenario = {
        key: (str(value),)
        for key, value in facts.items()
        if isinstance(value, (str, int))
    }
    variables = RoleSeparatedVariableBy(
        VariableBy2D(
            {
                "scenario": scenario,
                "prompt": {"request": prompts},
                "answer": {"direct": answers},
            }
        )
    )
    prompt_plan_options = prompt_variant_plans(
        sense="request",
        pool_name="request",
        functions=(prompt_functions or tuple(("request",) for _ in prompts)),
    )
    answer_plan_options = answer_variant_plans(
        sense="direct",
        pool_name="direct",
        functions=(answer_functions or tuple(("answer",) for _ in answers)),
    )
    if prompt_choice is not None:
        prompt_plan_options = (prompt_plan_options[prompt_choice],)
    if answer_choice is not None:
        answer_plan_options = (answer_plan_options[answer_choice],)
    deck = V2RoleSeparatedDeck(
        name=f"{TASK}:{domain}",
        variables=variables,
        prompt_pools=(
            V2SubcardPool("request", SurfaceRole.PROMPT, ("{prompt[request]}",)),
        ),
        answer_pools=(
            V2SubcardPool("direct", SurfaceRole.ANSWER, ("{answer[direct]}",)),
        ),
        prompt_plans=prompt_plan_options,
        answer_plans=answer_plan_options,
    )
    return render_v2_row(
        task=TASK,
        case_id=case_id,
        domain=domain,
        difficulty=difficulty,
        deck=deck,
        facts=facts,
        validator=validator,
        semantic_frame=semantic_frame,
    )


def _anchor_rows() -> Iterable[dict[str, object]]:
    for index, (prompt, domain, answer, skill) in enumerate(_ANCHORS):
        yield _row(
            case_id=f"anchor:{index}:{skill}",
            domain=domain,
            difficulty="easy",
            facts={"prompt": prompt, "answer": answer, "skill": skill},
            prompts=("{scenario[prompt]}",),
            answers=("{scenario[answer]}",),
            validator={"kind": "exact", "expected": answer},
        )


def _natural_social_rows() -> Iterable[dict[str, object]]:
    """Natural social acts, kept distinct from short computational exercises."""

    for spec in _NATURAL_SOCIAL_SPECS:
        prompts = tuple(spec["prompts"])
        answers = tuple(
            answer.replace(
                "{scenario[subject]}",
                "{scenario[subject_answer]}",
            ).replace(
                "{scenario[circumstance]}",
                "{scenario[circumstance_answer]}",
            )
            for answer in spec["answers"]
        )
        prompt_functions = tuple(
            ((f"{spec['intent']}_prompt_{index}",))
            for index in range(len(prompts))
        )
        answer_functions = tuple(
            ((f"{spec['intent']}_answer_{index}",))
            for index in range(len(answers))
        )
        for subject_index, subject in enumerate(spec["subjects"]):
            for circumstance_index, circumstance in enumerate(
                spec["circumstances"]
            ):
                for prompt_index in range(len(prompts)):
                    answer_index = (
                        subject_index * 3
                        + circumstance_index * 7
                        + prompt_index
                    ) % len(answers)
                    facts = {
                        "intent": spec["intent"],
                        "subject": subject,
                        "subject_cap": subject[0].upper() + subject[1:],
                        "subject_answer": {
                            "a paragraph I am revising": "your paragraph",
                            "a list I need to organize": "your list",
                            "an idea I want to develop": "your idea",
                            "a question I want to phrase clearly": "your question",
                            "spot an error in my notes": "spot an error in your notes",
                        }.get(subject, subject),
                        "circumstance": circumstance,
                        "circumstance_answer": {
                            "continue with the rest of my work": "continue with the rest of your work",
                            "save my notes": "save your notes",
                            "so I can verify the result myself": "so you can verify the result yourself",
                        }.get(circumstance, circumstance),
                        "circumstance_cap": (
                            circumstance[0].upper() + circumstance[1:]
                        ),
                    }
                    yield _row(
                        case_id=(
                            f"natural:{spec['intent']}:{subject_index}:"
                            f"{circumstance_index}:{prompt_index}:{answer_index}"
                        ),
                        domain=str(spec["domain"]),
                        difficulty="easy",
                        facts=facts,
                        prompts=prompts,
                        answers=answers,
                        validator={
                            "kind": "natural",
                            "minimum_words": 3,
                            "maximum_words": 40,
                            "forbidden": ["teach_back", "available evidence"],
                        },
                        prompt_functions=prompt_functions,
                        answer_functions=answer_functions,
                        prompt_choice=prompt_index,
                        answer_choice=answer_index,
                        semantic_frame=SemanticFrame(
                            intent=str(spec["intent"]),
                            facts=facts,
                            expected_outcome={"social_act": spec["intent"]},
                            user_tone="casual",
                        ),
                    )


def _arithmetic_rows() -> Iterable[dict[str, object]]:
    for operation, pairs, calculate, prompts, answers in _ARITHMETIC:
        for left, right in pairs:
            result = calculate(left, right)
            facts = {
                "operation": operation,
                "left": left,
                "right": right,
                "result": result,
                "left_words": _number_words(left),
                "right_words": _number_words(right),
                "result_words": _number_words(result),
            }
            yield _row(
                case_id=f"arithmetic:{operation}:{left}:{right}",
                domain=operation,
                difficulty="easy" if max(left, right) < 100 else "medium",
                facts=facts,
                prompts=prompts,
                answers=answers,
                validator={
                    "kind": "contains",
                    "required": [
                        facts["left_words"], facts["right_words"], facts["result_words"]
                    ],
                },
                prompt_functions=_ARITHMETIC_PROMPT_FUNCTIONS,
                answer_functions=_ARITHMETIC_ANSWER_FUNCTIONS,
            )


def _comparison_rows() -> Iterable[dict[str, object]]:
    prompts = (
        "Which is greater: {scenario[left_words]} or {scenario[right_words]}?",
        "Choose the larger number from {scenario[left_words]} and {scenario[right_words]}.",
        "Compare {scenario[left_words]} with {scenario[right_words]}. Which one is larger?",
        "Between {scenario[left_words]} and {scenario[right_words]}, what is the greater value?",
    )
    answers = (
        "{scenario[larger_words]} is greater; {scenario[smaller_words]} is the smaller value.",
        "The answer is {scenario[larger_words]}; {scenario[smaller_words]} ranks below it.",
        "Choose {scenario[larger_words]}; it exceeds {scenario[smaller_words]}.",
        "{scenario[smaller_words]} is smaller, so the answer is {scenario[larger_words]}.",
    )
    for left in range(101, 201):
        for right in range(1, 61):
            facts = {
                "left": left,
                "right": right,
                "left_words": _number_words(left),
                "right_words": _number_words(right),
                "larger_words": _number_words(left),
                "smaller_words": _number_words(right),
            }
            yield _row(
                case_id=f"comparison:{left}:{right}",
                domain="numerical_comparison",
                difficulty="easy",
                facts=facts,
                prompts=prompts,
                answers=answers,
                validator={
                    "kind": "contains",
                    "required": [facts["larger_words"], facts["smaller_words"]],
                },
                prompt_functions=_COMPARISON_PROMPT_FUNCTIONS,
                answer_functions=_COMPARISON_ANSWER_FUNCTIONS,
            )


def _formatting_rows() -> Iterable[dict[str, object]]:
    for first, second, third in combinations(_WORDS, 3):
        expected = f"{first}, {second}, {third}"
        facts = {"first": first, "second": second, "third": third, "expected": expected}
        yield _row(
            case_id=f"format:list:{first}:{second}:{third}",
            domain="instruction_formatting",
            difficulty="easy",
            facts=facts,
            prompts=(
                "Reply with only these words, separated by commas: {scenario[first]}, {scenario[second]}, {scenario[third]}.",
                "Return {scenario[first]}, {scenario[second]}, and {scenario[third]} as one comma-separated line.",
                "Format this trio with commas and no extra text: {scenario[first]} | {scenario[second]} | {scenario[third]}.",
                "Write exactly the three supplied items, comma separated: {scenario[first]}, {scenario[second]}, {scenario[third]}.",
            ),
            answers=("{scenario[expected]}",),
            validator={"kind": "exact", "expected": expected},
            prompt_functions=(
                ("request_exact_format", "supply_items"),
                ("request_comma_line", "supply_items"),
                ("request_no_extra_text", "supply_items"),
                ("request_exact_items", "specify_separator"),
            ),
            answer_functions=(("formatted_answer_only",),),
        )
    for word in ("echo", "pine", "river", "stone", "cloud", "maple", "orbit", "cedar", "flame", "ocean"):
        for count in range(2, 12):
            expected = " ".join([word] * count)
            yield _row(
                case_id=f"format:repeat:{word}:{count}",
                domain="instruction_formatting",
                difficulty="easy",
                facts={"word": word, "count_words": _number_words(count), "expected": expected},
                prompts=(
                    "Write {scenario[word]} exactly {scenario[count_words]} times, separated by single spaces.",
                    "Repeat {scenario[word]} {scenario[count_words]} times on one line with spaces between copies.",
                ),
                answers=("{scenario[expected]}",),
                validator={"kind": "exact", "expected": expected},
                prompt_functions=(
                    ("request_exact_repetition", "specify_spacing"),
                    ("request_one_line_repetition", "specify_spacing"),
                ),
                answer_functions=(("formatted_answer_only",),),
            )


def _sorting_rows() -> Iterable[dict[str, object]]:
    prompts = (
        "Sort these values from smallest to largest: {scenario[first]}, {scenario[second]}, {scenario[third]}.",
        "Put {scenario[first]}, {scenario[second]}, and {scenario[third]} in ascending order.",
        "Which ascending sequence is formed by {scenario[first]}, {scenario[second]}, and {scenario[third]}?",
        "Arrange this number trio from low to high: {scenario[first]} | {scenario[second]} | {scenario[third]}.",
    )
    answers = (
        "{scenario[expected_digits]}. The successive gaps are {scenario[first_gap_words]} and {scenario[second_gap_words]}.",
        "Ordered digits: {scenario[expected_digits]}. The increases are {scenario[first_gap_words]} and {scenario[second_gap_words]}.",
        "The sequence is {scenario[expected_digits]}. Adjacent values differ by {scenario[first_gap_words]} and {scenario[second_gap_words]}.",
        "Correct arrangement: {scenario[expected_digits]}. Its two gaps measure {scenario[first_gap_words]} and {scenario[second_gap_words]}.",
    )
    for index in range(4_000):
        values = (
            1 + (index * 17) % 997,
            1001 + (index * 29) % 991,
            2003 + (index * 43) % 983,
        )
        presented = (values[1], values[2], values[0])
        words = tuple(_number_words(value) for value in presented)
        ordered = sorted(values)
        expected_digits = ", ".join(str(value) for value in ordered)
        first_gap_words = _number_words(ordered[1] - ordered[0])
        second_gap_words = _number_words(ordered[2] - ordered[1])
        facts = {
            "first": words[0],
            "second": words[1],
            "third": words[2],
            "expected_digits": expected_digits,
            "first_gap_words": first_gap_words,
            "second_gap_words": second_gap_words,
        }
        yield _row(
            case_id=f"sorting:{index}:{'-'.join(map(str, values))}",
            domain="sequence_ordering",
            difficulty="medium",
            facts=facts,
            prompts=prompts,
            answers=answers,
            validator={
                "kind": "contains",
                "required": [expected_digits, first_gap_words, second_gap_words],
            },
            prompt_functions=_SORTING_PROMPT_FUNCTIONS,
            answer_functions=_SORTING_ANSWER_FUNCTIONS,
        )


def _fact_rows() -> Iterable[dict[str, object]]:
    for index, (subject, expected) in enumerate(_FACTS):
        yield _row(
            case_id=f"fact:{index}:{subject}",
            domain="general_knowledge",
            difficulty="easy",
            facts={"subject": subject, "expected": expected},
            prompts=(
                "What is {scenario[subject]}?",
                "Give a direct answer: what is {scenario[subject]}?",
                "Please identify {scenario[subject]}.",
                "Name {scenario[subject]}.",
                "Answer this factual question directly: what is {scenario[subject]}?",
            ),
            answers=(
                "The answer is {scenario[expected]}.",
                "It is {scenario[expected]}.",
                "{scenario[expected]}.",
            ),
            validator={"kind": "contains", "required": [expected]},
            prompt_functions=_FACT_PROMPT_FUNCTIONS,
            answer_functions=_FACT_ANSWER_FUNCTIONS,
        )


def _multi_turn_rows() -> Iterable[dict[str, object]]:
    """Context-dependent follow-ups that cannot be answered from the last turn alone."""

    for context, items in _FOLLOW_UP_CONTEXTS:
        for base in range(20, 70):
            for increment in range(2, 52):
                result = base + increment
                base_words = _number_words(base)
                increment_words = _number_words(increment)
                result_words = _number_words(result)
                first_user = (
                    f"There are {base_words} {items} for the {context}. "
                    "Can you keep that count in mind?"
                )
                first_assistant = f"Yes—the current count is {base_words} {items}."
                facts = {
                    "context": context,
                    "items": items,
                    "base": base,
                    "increment": increment,
                    "result": result,
                    "base_words": base_words,
                    "increment_words": increment_words,
                    "result_words": result_words,
                }
                yield _row(
                    case_id=f"multi-turn:add:{context}:{base}:{increment}",
                    domain="contextual_arithmetic",
                    difficulty="easy",
                    facts=facts,
                    prompts=_FOLLOW_UP_PROMPTS,
                    answers=_FOLLOW_UP_ANSWERS,
                    validator={
                        "kind": "contains",
                        "required": [result_words, items],
                    },
                    prompt_functions=_FOLLOW_UP_PROMPT_FUNCTIONS,
                    answer_functions=_FOLLOW_UP_ANSWER_FUNCTIONS,
                    semantic_frame=SemanticFrame(
                        intent="contextual_addition_follow_up",
                        facts=facts,
                        constraints=("resolve references from conversation history",),
                        expected_outcome=result,
                        uncertainty="none",
                        user_tone="casual",
                        history=(
                            ConversationTurn("user", first_user),
                            ConversationTurn("assistant", first_assistant),
                        ),
                    ),
                )


def casual_conversation_capacity() -> int:
    arithmetic = 7_000 + 5_000 + 2_100 + 2_000
    comparisons_count = 100 * 60
    formatting = 4_060 + 100
    multi_turn = len(_FOLLOW_UP_CONTEXTS) * 50 * 50
    natural_social = sum(
        len(spec["subjects"])
        * len(spec["circumstances"])
        * len(spec["prompts"])
        for spec in _NATURAL_SOCIAL_SPECS
    )
    return (
        len(_ANCHORS)
        + natural_social
        + arithmetic
        + comparisons_count
        + formatting
        + 4_000
        + len(_FACTS)
        + multi_turn
    )


def render_casual_conversation_rows() -> list[dict[str, object]]:
    rows = list(_anchor_rows())
    rows.extend(_natural_social_rows())
    rows.extend(_arithmetic_rows())
    rows.extend(_comparison_rows())
    rows.extend(_formatting_rows())
    rows.extend(_sorting_rows())
    rows.extend(_fact_rows())
    rows.extend(_multi_turn_rows())
    return validate_complete_rows(TASK, rows, casual_conversation_capacity())


__all__ = ("casual_conversation_capacity", "render_casual_conversation_rows")
