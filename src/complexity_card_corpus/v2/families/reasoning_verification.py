from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SemanticFrame, SurfaceRole, ThinkingBudget
from ..decks import (
    V2RoleSeparatedDeck,
    V2SubcardPool,
    answer_variant_plans,
    prompt_variant_plans,
    thinking_variant_plans,
)
from ._common import render_v2_row, validate_complete_rows
TASK = "reasoning_verification"
_KINDS = ("calculate", "verify_correct", "verify_incorrect")


@dataclass(frozen=True)
class _Operation:
    name: str
    symbol: str
    pairs: tuple[tuple[int, int], ...]

    def result(self, left: int, right: int) -> int:
        if self.name == "addition":
            return left + right
        if self.name == "subtraction":
            return left - right
        if self.name == "multiplication":
            return left * right
        if self.name == "division":
            return left // right
        raise ValueError(f"unknown operation {self.name!r}")


_OPERATIONS = (
    _Operation(
        "addition",
        "+",
        tuple((left, right) for left in range(10, 210) for right in range(2, 82)),
    ),
    _Operation(
        "subtraction",
        "−",
        tuple((left, right) for left in range(60, 260) for right in range(2, 52)),
    ),
    _Operation(
        "multiplication",
        "×",
        tuple((left, right) for left in range(2, 102) for right in range(2, 52)),
    ),
    _Operation(
        "division",
        "÷",
        tuple(
            (divisor * quotient, divisor)
            for divisor in range(2, 52)
            for quotient in range(2, 102)
        ),
    ),
)

_CALCULATE_PROMPTS = (
    "What is {scenario[expression]}?",
    "Calculate {scenario[expression]}.",
    "Evaluate {scenario[expression]}.",
    "Solve {scenario[expression]}.",
    "Find the value of {scenario[expression]}.",
    "Work out {scenario[expression]}.",
    "Compute {scenario[expression]}.",
    "Give the result of {scenario[expression]}.",
    "Perform this calculation: {scenario[expression]}.",
    "Determine the value: {scenario[expression]}.",
    "Use ordinary arithmetic to evaluate {scenario[expression]}.",
    "Return the value of {scenario[expression]}.",
)
_VERIFY_PROMPTS = (
    "Someone calculated {scenario[expression]} as {scenario[candidate]}. Is that correct?",
    "Check this claim: {scenario[expression]} = {scenario[candidate]}.",
    "Verify whether {scenario[expression]} equals {scenario[candidate]}.",
    "Is {scenario[candidate]} the right value for {scenario[expression]}?",
    "Review the proposed result {scenario[candidate]} for {scenario[expression]}.",
    "Decide whether this equation is true: {scenario[expression]} = {scenario[candidate]}.",
    "A calculation gives {scenario[candidate]} for {scenario[expression]}. Check it.",
    "Confirm or correct this result: {scenario[expression]} = {scenario[candidate]}.",
    "Does {scenario[expression]} really equal {scenario[candidate]}?",
    "Assess the answer {scenario[candidate]} for {scenario[expression]}.",
    "Test the proposed equality {scenario[expression]} = {scenario[candidate]}.",
    "Please verify the stated result for {scenario[expression]}: {scenario[candidate]}.",
)
_CALCULATE_ANSWERS = (
    "{scenario[result]}.",
    "The answer is {scenario[result]}.",
    "It equals {scenario[result]}.",
    "That gives {scenario[result]}.",
    "The result is {scenario[result]}.",
    "The value is {scenario[result]}.",
    "You get {scenario[result]}.",
    "It comes to {scenario[result]}.",
    "The calculation yields {scenario[result]}.",
    "The expression evaluates to {scenario[result]}.",
    "Solving it gives {scenario[result]}.",
    "Direct computation gives {scenario[result]}.",
    "Numerically, it is {scenario[result]}.",
    "The exact value is {scenario[result]}.",
    "The computed result is {scenario[result]}.",
    "After calculating, I get {scenario[result]}.",
    "This works out to {scenario[result]}.",
    "Arithmetic gives {scenario[result]}.",
    "The final value is {scenario[result]}.",
    "Computing it produces {scenario[result]}.",
    "The operation returns {scenario[result]}.",
    "Its value comes out as {scenario[result]}.",
    "Evaluation gives {scenario[result]}.",
    "The numerical answer is {scenario[result]}.",
    "Calculated directly: {scenario[result]}.",
    "The outcome is {scenario[result]}.",
    "I get {scenario[result]} from the calculation.",
    "The expression has value {scenario[result]}.",
    "This equals {scenario[result]}.",
    "The arithmetic result is {scenario[result]}.",
)
_CORRECT_ANSWERS = (
    "Yes. {scenario[candidate]} is correct.",
    "Correct—the value is {scenario[result]}.",
    "Yes; the calculation gives {scenario[result]}.",
    "That result is right: {scenario[result]}.",
    "The equality is true.",
    "Yes, {scenario[expression]} equals {scenario[result]}.",
    "The proposed value is accurate.",
    "No correction is needed; the answer is {scenario[result]}.",
    "The arithmetic checks out at {scenario[result]}.",
    "Confirmed. The value is {scenario[result]}.",
    "Yes—the stated answer matches the calculation.",
    "The claim is correct, with a result of {scenario[result]}.",
    "Verification confirms {scenario[result]}.",
    "That equation holds.",
    "That answer checks out.",
    "Yes. Recalculation also gives {scenario[result]}.",
    "The stated value is correct.",
    "Confirmed: the computation returns {scenario[result]}.",
    "The result needs no change.",
    "Yes, the answer {scenario[candidate]} is valid.",
    "A fresh calculation agrees with {scenario[result]}.",
    "The equation is accurate as written.",
    "This is correct; the value remains {scenario[result]}.",
    "The check succeeds, so {scenario[candidate]} stands.",
    "Yes. The numbers produce {scenario[result]}.",
    "The proposed answer passes the arithmetic check.",
    "Recomputing confirms the same value, {scenario[result]}.",
    "The equality is valid.",
    "That calculation is accurate.",
    "Yes; {scenario[candidate]} is the correct answer.",
)
_INCORRECT_ANSWERS = (
    "No. The correct result is {scenario[result]}, not {scenario[candidate]}.",
    "That is incorrect; the value should be {scenario[result]}.",
    "The proposed answer is wrong. It equals {scenario[result]}.",
    "A correction is needed: {scenario[result]}.",
    "No—recalculation gives {scenario[result]}.",
    "The equality is false; the answer is {scenario[result]}.",
    "{scenario[candidate]} does not check out. The corrected value is {scenario[result]}.",
    "That number misses the mark; use {scenario[result]}.",
    "Correcting the arithmetic gives {scenario[result]}.",
    "The claim is inaccurate. The computation returns {scenario[result]}.",
    "No. Replace {scenario[candidate]} with {scenario[result]}.",
    "The proposed value fails verification; it should be {scenario[result]}.",
    "That result needs correction to {scenario[result]}.",
    "A fresh calculation produces {scenario[result]}, so the claim is false.",
    "The equation is not valid as written; its value is {scenario[result]}.",
    "Rechecking shows {scenario[result]} rather than {scenario[candidate]}.",
    "The answer is incorrect. The exact value is {scenario[result]}.",
    "No; the arithmetic result comes to {scenario[result]}.",
    "Verification rejects {scenario[candidate]} and confirms {scenario[result]}.",
    "The stated value is off; the calculation yields {scenario[result]}.",
    "This does not hold. The answer must be {scenario[result]}.",
    "The numbers give {scenario[result]}, so {scenario[candidate]} is incorrect.",
    "This equation fails. Use {scenario[result]} instead.",
    "That answer is not accurate; {scenario[result]} is correct.",
    "A direct check gives {scenario[result]}, not the proposed value.",
    "The calculation disagrees with {scenario[candidate]}; it gives {scenario[result]}.",
    "No. The expression evaluates to {scenario[result]}.",
    "The result requires a fix: {scenario[result]}.",
    "Recomputation corrects the answer to {scenario[result]}.",
    "The claim is false because the value is {scenario[result]}.",
)
_WRONG_OFFSETS = (-9, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 9)
_CALCULATE_THINKING_APPROACHES = (
    "Resolve the arithmetic directly: {thinking[action]}.",
    "Work from the two operands: {thinking[action]}.",
    "Apply the indicated operation: {thinking[action]}.",
    "Evaluate the numerical relation: {thinking[action]}.",
    "Use the operand values themselves: {thinking[action]}.",
    "Carry out the computation once: {thinking[action]}.",
    "Follow the arithmetic definition: {thinking[action]}.",
    "Compute from left to right here: {thinking[action]}.",
    "Determine the quantity numerically: {thinking[action]}.",
    "Process the expression as written: {thinking[action]}.",
    "Obtain the value from the operation: {thinking[action]}.",
    "Calculate using the given integers: {thinking[action]}.",
    "Reduce the expression to one value: {thinking[action]}.",
    "Perform the required integer operation: {thinking[action]}.",
    "Derive the number from the operands: {thinking[action]}.",
    "Evaluate this instance without approximation: {thinking[action]}.",
)
_VERIFY_THINKING_APPROACHES = (
    "Compute independently before judging the claim: {thinking[action]}.",
    "Ignore the candidate initially and recalculate: {thinking[action]}.",
    "Establish the arithmetic value first: {thinking[action]}.",
    "Compare against an independent computation: {thinking[action]}.",
    "Recompute the expression from its operands: {thinking[action]}.",
    "Test the candidate against fresh arithmetic: {thinking[action]}.",
    "Set the proposed number aside and evaluate: {thinking[action]}.",
    "Judge the equality using a separate calculation: {thinking[action]}.",
    "Find the actual value before giving a verdict: {thinking[action]}.",
    "Check the assertion from the underlying operation: {thinking[action]}.",
    "Derive a reference value for comparison: {thinking[action]}.",
    "Rebuild the result directly from the inputs: {thinking[action]}.",
    "Assess the claim after computing the expression: {thinking[action]}.",
    "Use an independent arithmetic route: {thinking[action]}.",
    "Calculate a trustworthy comparison value: {thinking[action]}.",
    "Verify from the operands rather than the assertion: {thinking[action]}.",
)
_THINKING_CHECKS = (
    "The inverse relation checks it: {scenario[check_expression]}.",
    "A reverse operation confirms it since {scenario[check_expression]}.",
    "Verification through the inverse equation produces {scenario[check_expression]}.",
    "The inverse arithmetic is consistent: {scenario[check_expression]}.",
    "Checking backward produces {scenario[check_expression]}.",
    "The reverse equality also holds: {scenario[check_expression]}.",
    "An inverse calculation returns the input: {scenario[check_expression]}.",
    "The operands are recovered by the check {scenario[check_expression]}.",
    "A backward calculation validates the number: {scenario[check_expression]}.",
    "The result survives the inverse test {scenario[check_expression]}.",
    "The corresponding reverse equation is {scenario[check_expression]}.",
    "Cross-checking with the inverse yields {scenario[check_expression]}.",
    "The original operand reappears in {scenario[check_expression]}.",
    "A second arithmetic direction gives {scenario[check_expression]}.",
    "The inverse form reproduces the starting value: {scenario[check_expression]}.",
    "A numerical cross-check is available: {scenario[check_expression]}.",
)
_THINKING_COMPOSITIONS = (
    "{thinking[approach]} {thinking[check]}",
    "{thinking[check]} {thinking[approach]}",
)
_CONTEXT_CALCULATE_PROMPTS = (
    "{scenario[problem]}",
    "Please solve this applied calculation: {scenario[problem]}",
    "Work through this quantity question: {scenario[problem]}",
    "Find the requested amount here: {scenario[problem]}",
    "Use the stated quantities to answer this: {scenario[problem]}",
    "Calculate the outcome in this situation: {scenario[problem]}",
    "Determine the missing quantity: {scenario[problem]}",
    "State the requested amount for this scenario: {scenario[problem]}",
    "Use the quantities in this case: {scenario[problem]}",
    "What result follows from these quantities? {scenario[problem]}",
    "Compute the requested total from the details: {scenario[problem]}",
    "Answer the practical number question: {scenario[problem]}",
)
_CONTEXT_VERIFY_PROMPTS = (
    "{scenario[problem]} A teammate proposes {scenario[candidate]}. Is that correct?",
    "Check this applied result. {scenario[problem]} The proposed answer is {scenario[candidate]}.",
    "{scenario[problem]} Verify whether {scenario[candidate]} is the right answer.",
    "Review the numerical claim in this situation: {scenario[problem]} Someone answered {scenario[candidate]}.",
    "Decide whether the suggested value works. {scenario[problem]} The suggestion is {scenario[candidate]}.",
    "Independently verify this case: {scenario[problem]} The stated result is {scenario[candidate]}.",
    "{scenario[problem]} Should the answer remain {scenario[candidate]}, or be corrected?",
    "Test the proposed outcome against the quantities. {scenario[problem]} Proposed outcome: {scenario[candidate]}.",
    "Assess this calculation before accepting it. {scenario[problem]} Candidate answer: {scenario[candidate]}.",
    "Recalculate the scenario and judge the claim. {scenario[problem]} The claim gives {scenario[candidate]}.",
    "Confirm or correct the submitted result. {scenario[problem]} Submitted result: {scenario[candidate]}.",
    "Use a fresh calculation to check this answer: {scenario[problem]} Answer under review: {scenario[candidate]}.",
)
_CONTEXT_CALCULATE_ANSWERS = (
    "{scenario[outcome]}",
    "Computed directly: {scenario[outcome]}",
    "Using the provided figures: {scenario[outcome]}",
    "The calculation gives this outcome: {scenario[outcome]}",
    "After evaluating the quantities: {scenario[outcome]}",
    "Numerically: {scenario[outcome]}",
    "The applied result is clear: {scenario[outcome]}",
    "The quantities produce this result: {scenario[outcome]}",
    "Working through the operation gives: {scenario[outcome]}",
    "The resulting quantity is established: {scenario[outcome]}",
    "Direct arithmetic leads here: {scenario[outcome]}",
    "The scenario resolves to this: {scenario[outcome]}",
)
_CONTEXT_CORRECT_ANSWERS = (
    "Yes. {scenario[outcome]}",
    "Correct. {scenario[outcome]}",
    "The value checks out. {scenario[outcome]}",
    "Confirmed. {scenario[outcome]}",
    "The submitted value is valid. {scenario[outcome]}",
    "The equality is true. {scenario[outcome]}",
    "A separate tally agrees. {scenario[outcome]}",
    "The reported quantity is right. {scenario[outcome]}",
    "The candidate matches the calculation. {scenario[outcome]}",
    "No correction is needed. {scenario[outcome]}",
    "The reported figure stands. {scenario[outcome]}",
    "Verification succeeds. {scenario[outcome]}",
)
_CONTEXT_INCORRECT_ANSWERS = (
    "No. {scenario[outcome]} The suggested number was {scenario[candidate]}.",
    "That is incorrect. {scenario[outcome]}",
    "The candidate is wrong; {scenario[outcome]}",
    "The claim is false. {scenario[outcome]}",
    "A correction is required: {scenario[outcome]}",
    "Replace {scenario[candidate]} with the verified quantity. {scenario[outcome]}",
    "Verification rejects {scenario[candidate]}. {scenario[outcome]}",
    "The submitted figure fails the check. {scenario[outcome]}",
    "The proposed amount is off. {scenario[outcome]}",
    "A fresh computation disagrees with {scenario[candidate]}. {scenario[outcome]}",
    "The reported value needs a fix. {scenario[outcome]}",
    "Recalculation corrects the result: {scenario[outcome]}",
)
_CALCULATE_PROMPT_FUNCTIONS = (
    ("present_problem",),
    ("request_applied_calculation", "present_problem"),
    ("request_quantity_reasoning", "present_problem"),
    ("request_amount", "present_problem"),
    ("require_given_quantities", "present_problem"),
    ("request_outcome", "present_problem"),
    ("request_missing_quantity", "present_problem"),
    ("request_scenario_amount", "present_problem"),
    ("request_case_calculation", "present_problem"),
    ("request_derived_result", "present_problem"),
    ("request_total", "present_problem"),
    ("request_practical_answer", "present_problem"),
)
_VERIFY_PROMPT_FUNCTIONS = (
    ("present_problem", "present_candidate", "request_verdict"),
    ("request_check", "present_problem", "present_candidate"),
    ("present_problem", "request_candidate_verification"),
    ("request_claim_review", "present_problem", "present_candidate"),
    ("request_candidate_decision", "present_problem"),
    ("request_independent_verification", "present_problem", "present_candidate"),
    ("present_problem", "request_confirm_or_correct"),
    ("request_outcome_test", "present_problem", "present_candidate"),
    ("request_pre_acceptance_check", "present_problem", "present_candidate"),
    ("request_recalculation", "present_problem", "request_verdict"),
    ("request_confirm_or_correct", "present_problem", "present_candidate"),
    ("request_fresh_calculation", "present_problem", "present_candidate"),
)
_CALCULATE_ANSWER_FUNCTIONS = (
    ("state_outcome",),
    ("mark_direct_computation", "state_outcome"),
    ("attribute_figures", "state_outcome"),
    ("state_calculation_result", "state_outcome"),
    ("mark_evaluation", "state_outcome"),
    ("mark_numeric_result", "state_outcome"),
    ("signal_clarity", "state_outcome"),
    ("attribute_quantities", "state_outcome"),
    ("mark_operation", "state_outcome"),
    ("state_established_quantity", "state_outcome"),
    ("attribute_arithmetic", "state_outcome"),
    ("resolve_scenario", "state_outcome"),
)
_CORRECT_ANSWER_FUNCTIONS = (
    ("affirm", "state_outcome"),
    ("confirm", "state_outcome"),
    ("report_successful_check", "state_outcome"),
    ("confirm", "state_outcome"),
    ("validate_candidate", "state_outcome"),
    ("affirm_equality", "state_outcome"),
    ("report_independent_agreement", "state_outcome"),
    ("validate_quantity", "state_outcome"),
    ("match_candidate", "state_outcome"),
    ("reject_correction", "state_outcome"),
    ("retain_result", "state_outcome"),
    ("report_verification_success", "state_outcome"),
)
_INCORRECT_ANSWER_FUNCTIONS = (
    ("reject", "state_correct_outcome", "name_bad_candidate"),
    ("reject", "state_correct_outcome"),
    ("reject_candidate", "state_correct_outcome"),
    ("reject_claim", "state_correct_outcome"),
    ("require_correction", "state_correct_outcome"),
    ("replace_candidate", "state_correct_outcome"),
    ("report_verification_failure", "state_correct_outcome"),
    ("reject_submitted_figure", "state_correct_outcome"),
    ("mark_amount_error", "state_correct_outcome"),
    ("report_disagreement", "state_correct_outcome"),
    ("require_fix", "state_correct_outcome"),
    ("report_recalculation", "state_correct_outcome"),
)
_THINKING_FUNCTIONS = (
    ("derive_independently", "inverse_check"),
    ("inverse_check", "derive_independently"),
)
_ACTORS = (
    "Amina", "Ben", "Chloe", "Diego", "Elena", "Farah", "Gavin", "Hana",
    "Imani", "Jonah", "Keira", "Leo", "Marta", "Nikhil", "Omar", "Priya",
    "Quinn", "Rosa", "Samir", "Talia", "Uma", "Victor", "Wren", "Xavier",
    "Yara", "Zane", "Anika", "Bruno", "Celine", "Darius", "Esme", "Felix",
)
_ITEMS = (
    ("map", "maps"),
    ("sample", "samples"),
    ("notebook", "notebooks"),
    ("battery", "batteries"),
    ("ticket", "tickets"),
    ("seedling", "seedlings"),
    ("sensor", "sensors"),
    ("meal", "meals"),
    ("parcel", "parcels"),
    ("badge", "badges"),
    ("filter", "filters"),
    ("blanket", "blankets"),
    ("folder", "folders"),
    ("marker", "markers"),
    ("kit", "kits"),
    ("lamp", "lamps"),
    ("voucher", "vouchers"),
    ("tablet", "tablets"),
    ("chair", "chairs"),
    ("bottle", "bottles"),
    ("panel", "panels"),
    ("booklet", "booklets"),
    ("crate", "crates"),
    ("label", "labels"),
    ("token", "tokens"),
    ("packet", "packets"),
    ("module", "modules"),
    ("poster", "posters"),
    ("tool", "tools"),
    ("cable", "cables"),
    ("survey", "surveys"),
    ("container", "containers"),
)
_SETTINGS = (
    "the harbor annex",
    "the north workshop",
    "the riverside depot",
    "the hilltop clinic",
    "the central library",
    "the east greenhouse",
    "the station archive",
    "the lakeside shelter",
    "the community hall",
    "the field laboratory",
    "the transit office",
    "the coastal warehouse",
    "the school studio",
    "the mobile kitchen",
    "the research outpost",
    "the neighborhood hub",
    "the museum store",
    "the repair garage",
    "the training center",
    "the public garden",
    "the mountain cabin",
    "the market pavilion",
    "the health kiosk",
    "the volunteer base",
)
_DOMAIN_PROJECTS = (
    ("inventory", "seasonal inventory"),
    ("event_planning", "community event"),
    ("field_research", "field study"),
    ("education", "training session"),
    ("logistics", "delivery schedule"),
    ("community_service", "relief program"),
)


def _context(case_index: int, operation: _Operation, left: int, right: int) -> dict[str, str]:
    actor = _ACTORS[case_index % len(_ACTORS)]
    item_index = (case_index // len(_ACTORS)) % len(_ITEMS)
    singular, plural = _ITEMS[item_index]
    setting = _SETTINGS[
        (case_index // (len(_ACTORS) * len(_ITEMS))) % len(_SETTINGS)
    ]
    domain, project = _DOMAIN_PROJECTS[
        (
            case_index
            // (len(_ACTORS) * len(_ITEMS) * len(_SETTINGS))
        )
        % len(_DOMAIN_PROJECTS)
    ]
    problem = {
        "addition": (
            f"At {setting}, {actor} recorded {left} {plural}, then added {right} "
            f"for the {project}. How many {plural} are recorded altogether?"
        ),
        "subtraction": (
            f"At {setting}, {actor} began the {project} with {left} {plural} and "
            f"used {right}. How many {plural} remain?"
        ),
        "multiplication": (
            f"For the {project}, {actor} prepared {left} stations at {setting}, "
            f"placing {right} {plural} at each one. How many {plural} were placed?"
        ),
        "division": (
            f"At {setting}, {actor} divided {left} {plural} equally among {right} "
            f"stations for the {project}. How many {plural} went to each station?"
        ),
    }[operation.name]
    result = operation.result(left, right)
    outcome = {
        "addition": (
            f"The confirmed {project} total for {actor} at {setting} is "
            f"{result} {plural}."
        ),
        "subtraction": (
            f"The remaining {project} stock for {actor} at {setting} is "
            f"{result} {plural}."
        ),
        "multiplication": (
            f"The complete {project} placement for {actor} at {setting} uses "
            f"{result} {plural}."
        ),
        "division": (
            f"The {project} allocation for {actor} at {setting} is "
            f"{result} {plural} per station."
        ),
    }[operation.name]
    return {
        "actor": actor,
        "item": singular,
        "items": plural,
        "setting": setting,
        "domain": domain,
        "project": project,
        "problem": problem,
        "outcome": outcome,
    }


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _thinking_actions(
    operation: str,
    left: int,
    right: int,
    result: int,
    *,
    context: dict[str, str],
) -> tuple[str, ...]:
    values = {
        "addition": (
            f"adding {left} to {right} reaches {result}",
            f"summing {left} with {right} results in {result}",
            f"combining {left} with {right} gives {result}",
            f"incrementing {left} with {right} reaches {result}",
            f"{left} plus {right} totals {result}",
            f"the two addends produce {result}",
            f"accumulating {right} onto {left} gives {result}",
            f"their arithmetic sum comes to {result}",
        ),
        "subtraction": (
            f"subtracting {right} from {left} leaves {result}",
            f"the gap from {left} down to {right} equals {result}",
            f"removing {right} from {left} gives {result}",
            f"decreasing {left} by {right} leaves {result}",
            f"{left} minus {right} comes to {result}",
            f"the minuend and subtrahend produce {result}",
            f"taking away {right} leaves {result}",
            f"their arithmetic difference equals {result}",
        ),
        "multiplication": (
            f"multiplying {left} by {right} yields {result}",
            f"multiplying factors {left} and {right} gives {result}",
            f"scaling {left} by {right} gives {result}",
            f"{left} times {right} comes to {result}",
            f"repeated groups combine into {result}",
            f"the two factors produce {result}",
            f"forming the product gives {result}",
            f"their multiplication evaluates to {result}",
        ),
        "division": (
            f"dividing {left} by {right} returns {result}",
            f"dividing dividend {left} by divisor {right} gives quotient {result}",
            f"partitioning {left} into sets of {right} leaves {result} sets",
            f"{left} divided by {right} comes to {result}",
            f"equal groups give a count of {result}",
            f"the dividend and divisor produce {result}",
            f"forming the quotient gives {result}",
            f"their exact division evaluates to {result}",
        ),
    }
    try:
        return tuple(
            (
                f"for {context['actor']}'s {context['project']} at "
                f"{context['setting']}, {action}"
            )
            for action in values[operation]
        )
    except KeyError as error:
        raise ValueError(f"unknown operation {operation!r}") from error


def reasoning_verification_capacity() -> int:
    return sum(len(operation.pairs) for operation in _OPERATIONS) * len(_KINDS)


def _selected_cases(seed: int) -> Iterable[tuple[_Operation, str, int, int]]:
    for operation in _OPERATIONS:
        for kind in _KINDS:
            pairs = sorted(
                operation.pairs,
                key=lambda pair: _digest(
                    f"{seed}:{operation.name}:{kind}:{pair[0]}:{pair[1]}"
                ),
            )
            yield from (
                (operation, kind, left, right)
                for left, right in pairs
            )


def _deck(
    operation: _Operation,
    kind: str,
    left: int,
    right: int,
    *,
    seed: int,
    case_index: int,
) -> tuple[V2RoleSeparatedDeck, dict[str, str]]:
    result = operation.result(left, right)
    expression = f"{left} {operation.symbol} {right}"
    check_expressions = {
        "addition": f"{result} − {right} = {left}",
        "subtraction": f"{result} + {right} = {left}",
        "multiplication": f"{result} ÷ {right} = {left}",
        "division": f"{result} × {right} = {left}",
    }
    if kind == "verify_incorrect":
        offset = _WRONG_OFFSETS[
            int.from_bytes(
                _digest(f"wrong:{seed}:{operation.name}:{left}:{right}")[:8],
                "big",
            )
            % len(_WRONG_OFFSETS)
        ]
        candidate = result + offset
    else:
        candidate = result
    context = _context(case_index, operation, left, right)
    prompts = (
        _CONTEXT_CALCULATE_PROMPTS
        if kind == "calculate"
        else _CONTEXT_VERIFY_PROMPTS
    )
    answers = {
        "calculate": _CONTEXT_CALCULATE_ANSWERS,
        "verify_correct": _CONTEXT_CORRECT_ANSWERS,
        "verify_incorrect": _CONTEXT_INCORRECT_ANSWERS,
    }[kind]
    approaches = (
        _CALCULATE_THINKING_APPROACHES
        if kind == "calculate"
        else _VERIFY_THINKING_APPROACHES
    )
    variables = RoleSeparatedVariableBy(
        VariableBy2D(
            {
                "scenario": {
                    "left": (str(left),),
                    "right": (str(right),),
                    "expression": (expression,),
                    "result": (str(result),),
                    "candidate": (str(candidate),),
                    "check_expression": (check_expressions[operation.name],),
                    "problem": (context["problem"],),
                    "outcome": (context["outcome"],),
                    "actor": (context["actor"],),
                    "item": (context["item"],),
                    "items": (context["items"],),
                    "setting": (context["setting"],),
                    "project": (context["project"],),
                    "domain": (context["domain"],),
                },
                "prompt": {"request": prompts},
                "thinking": {
                    "action": _thinking_actions(
                        operation.name,
                        left,
                        right,
                        result,
                        context=context,
                    ),
                    "approach": approaches,
                    "check": _THINKING_CHECKS,
                    "analysis": _THINKING_COMPOSITIONS,
                },
                "answer": {"direct": answers},
            }
        )
    )
    deck = V2RoleSeparatedDeck(
        name=f"{TASK}:{operation.name}:{kind}",
        variables=variables,
        prompt_pools=(
            V2SubcardPool("request", SurfaceRole.PROMPT, ("{prompt[request]}",)),
        ),
        answer_pools=(
            V2SubcardPool("direct", SurfaceRole.ANSWER, ("{answer[direct]}",)),
        ),
        thinking_pools=(
            V2SubcardPool(
                "analysis",
                SurfaceRole.THINKING,
                ("{thinking[analysis]}",),
            ),
        ),
        prompt_plans=prompt_variant_plans(
            sense="request",
            pool_name="request",
            functions=(
                _CALCULATE_PROMPT_FUNCTIONS
                if kind == "calculate"
                else _VERIFY_PROMPT_FUNCTIONS
            ),
        ),
        answer_plans=answer_variant_plans(
            sense="direct",
            pool_name="direct",
            functions={
                "calculate": _CALCULATE_ANSWER_FUNCTIONS,
                "verify_correct": _CORRECT_ANSWER_FUNCTIONS,
                "verify_incorrect": _INCORRECT_ANSWER_FUNCTIONS,
            }[kind],
        ),
        thinking_plans=thinking_variant_plans(
            sense="analysis",
            pool_name="analysis",
            functions=_THINKING_FUNCTIONS,
            budget=ThinkingBudget.VERIFICATION,
        ),
    )
    return deck, {
        "operation": operation.name,
        "kind": kind,
        "left": str(left),
        "right": str(right),
        "result": str(result),
        "candidate": str(candidate),
        **context,
    }


def render_reasoning_verification_rows(*, seed: int = 42) -> list[dict[str, object]]:
    """Render the complete independently authored V2 reasoning family."""

    rows: list[dict[str, object]] = []
    for case_index, (operation, kind, left, right) in enumerate(
        _selected_cases(seed)
    ):
        case_id = f"{operation.name}:{kind}:{left}:{right}"
        deck, facts = _deck(
            operation,
            kind,
            left,
            right,
            seed=seed,
            case_index=case_index,
        )
        rows.append(
            render_v2_row(
                task=TASK,
                case_id=case_id,
                domain=facts["domain"],
                difficulty="easy" if max(left, right) < 100 else "medium",
                deck=deck,
                facts=facts,
                validator={"kind": "arithmetic"},
                semantic_frame=SemanticFrame(
                    intent=(
                        "calculate_quantity"
                        if facts["kind"] == "calculate"
                        else "verify_quantity"
                    ),
                    facts=facts,
                    constraints=("exact integer arithmetic", "independent check"),
                    expected_outcome=facts["result"],
                    uncertainty=(
                        "candidate_under_review"
                        if facts["kind"] != "calculate"
                        else "none"
                    ),
                    user_tone="neutral",
                ),
            )
        )
    return validate_complete_rows(TASK, rows, reasoning_verification_capacity())


__all__ = (
    "reasoning_verification_capacity",
    "render_reasoning_verification_rows",
)
