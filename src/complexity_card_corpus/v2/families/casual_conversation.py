from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool
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


def _row(
    *,
    case_id: str,
    domain: str,
    difficulty: str,
    facts: dict[str, Any],
    prompts: tuple[str, ...],
    answers: tuple[str, ...],
    validator: dict[str, Any],
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
    deck = V2RoleSeparatedDeck(
        name=f"{TASK}:{domain}",
        variables=variables,
        prompt_pools=(
            V2SubcardPool("request", SurfaceRole.PROMPT, ("{prompt[request]}",)),
        ),
        answer_pools=(
            V2SubcardPool("direct", SurfaceRole.ANSWER, ("{answer[direct]}",)),
        ),
    )
    return render_v2_row(
        task=TASK,
        case_id=case_id,
        domain=domain,
        difficulty=difficulty,
        deck=deck,
        facts=facts,
        validator=validator,
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
        )


def casual_conversation_capacity() -> int:
    arithmetic = 7_000 + 5_000 + 2_100 + 2_000
    comparisons_count = 100 * 60
    formatting = 4_060 + 100
    return len(_ANCHORS) + arithmetic + comparisons_count + formatting + 4_000 + len(_FACTS)


def render_casual_conversation_rows() -> list[dict[str, object]]:
    rows = list(_anchor_rows())
    rows.extend(_arithmetic_rows())
    rows.extend(_comparison_rows())
    rows.extend(_formatting_rows())
    rows.extend(_sorting_rows())
    rows.extend(_fact_rows())
    return validate_complete_rows(TASK, rows, casual_conversation_capacity())


__all__ = ("casual_conversation_capacity", "render_casual_conversation_rows")
