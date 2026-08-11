from __future__ import annotations

from collections.abc import Callable


def reasoning_case(
    domain: str,
    code: str,
    units: int,
    each: int,
    extra: int,
    *,
    number: Callable[[str, int, int], int],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    """Compute one reasoning case and localize its semantic surfaces."""

    if domain == "shopping_arithmetic":
        result = units * each + extra
        data = f"Problem {code}: {units} items cost ${each} each, plus a ${extra} delivery fee."
        equation = f"{units} × {each} + {extra} = {result}"
        total, check = (
            f"${result}",
            f"the item subtotal is ${units * each}, and adding ${extra} gives ${result}",
        )
        components = (
            f"Each of the {units} items costs ${each}; the flat ${extra} delivery fee is added once, not per item.",
            f"The ${extra} delivery fee applies once in total, separately from the {units} × ${each} subtotal of item costs.",
            f"The ${extra} fee is a flat charge on the whole order, independent of the {units} × ${each} item total.",
        )
    elif domain == "schedule_math":
        result = units * each + extra
        data = f"Problem {code}: {units} sessions last {each} minutes each, followed by a {extra}-minute break."
        equation = f"{units} × {each} + {extra} = {result}"
        total, check = (
            f"{result} minutes",
            f"removing the {extra}-minute break leaves {units * each} session minutes",
        )
        components = (
            f"Each of the {units} sessions runs exactly {each} minutes; the {extra}-minute break is added once at the very end.",
            f"The {extra}-minute break is added once, kept fully separate from the {units} × {each} minutes of total session time.",
            f"The {extra}-minute break is a single addition at the end, not repeated across the {units} sessions.",
        )
    elif domain == "unit_conversion":
        result = units * 100
        data = f"Problem {code}: convert {units} metres to centimetres using 1 metre = 100 centimetres."
        equation = f"{units} × 100 = {result}"
        total, check = (
            f"{result} centimetres",
            f"dividing {result} by 100 returns {units} metres",
        )
        components = (
            f"Each of the {units} whole metres converts to exactly 100 centimetres, since 1 metre always equals 100 centimetres exactly, with no rounding needed.",
            f"The fixed conversion factor of 100 centimetres per metre applies uniformly across all {units} whole metres, with no rounding involved at any step.",
            f"This conversion is exact for any whole number of metres, including these {units}, since the 100:1 ratio never changes.",
            f"Multiplying {units} metres by the exact factor of 100 yields centimetres without an estimate or rounding step.",
            f"The unit identity 1 metre = 100 centimetres fixes the result for all {units} metres exactly.",
            f"Because the conversion factor is defined exactly, the {units}-metre quantity scales directly by 100.",
        )
    elif domain == "proportions":
        result = units * each
        data = f"Problem {code}: one batch uses {each} cups; keep the ratio for {units} batches."
        equation = f"{units} × {each} = {result}"
        total, check = (
            f"{result} cups",
            f"{result} divided by {units} returns {each} cups per batch",
        )
        components = (
            f"Each of the {units} batches uses the exact same {each}-cup ratio as the original single batch, entirely unscaled and unchanged.",
            f"The {each}-cup ratio per batch stays exactly fixed across all {units} separate batches, without any adjustment to the original recipe.",
            f"Scaling to {units} batches multiplies the count of batches, not the {each}-cup ratio within each one.",
            f"Keeping {each} cups in every batch makes the total the product of that fixed amount and {units} batches.",
            f"The per-batch quantity remains {each} cups, so only the number of batches changes in the calculation.",
            f"Multiplication preserves the original {each}-cup proportion across each of the {units} identical batches.",
        )
    elif domain == "table_comparison":
        result = max(units * each, units * extra)
        data = f"Problem {code}: table A reports {units} × {each}; table B reports {units} × {extra}. Compare the totals."
        equation = f"max({units} × {each}, {units} × {extra}) = {result}"
        total, check = (
            f"{result}",
            f"computing both products independently confirms {result} as the larger entry",
        )
        components = (
            f"Table A's total is {units} × {each}, while table B's total is {units} × {extra}; only the larger of the two totals is reported.",
            f"Both table entries share the exact same {units} count but differ only in their separately reported per-unit rate value.",
            f"The shared {units} count means the comparison between tables A and B comes down entirely to {each} versus {extra}.",
        )
    elif domain == "sequence_pattern":
        result = units + 3 * each
        data = f"Problem {code}: the sequence is {units}, {units + each}, {units + 2 * each}, __; use the constant difference."
        equation = f"{units} + 3 × {each} = {result}"
        total, check = f"{result}", f"each adjacent pair differs by {each}"
        components = (
            f"Each term after the starting value {units} increases by the exact same fixed difference of {each}, which confirms an arithmetic pattern applies throughout this sequence.",
            f"The constant difference of {each} between consecutive terms is confirmed separately by each adjacent pair of the first three listed sequence values.",
            f"A constant per-term increase of {each} starting from {units} is what makes this sequence arithmetic rather than some other pattern.",
        )
    elif domain == "logical_constraints":
        result = each - 1 + units
        data = f"Problem {code}: A must occur immediately before B; B is at slot {each}; C is at slot {units}. Find A's slot and add it to C's slot."
        equation = f"({each} - 1) + {units} = {result}"
        total, check = (
            f"{result}",
            f"slot {each - 1} is occupied by A, immediately before B at slot {each}",
        )
        components = (
            f"A's slot follows only from B's fixed position at slot {each}; C's slot {units} is independent of that constraint.",
            f"A's slot is fixed relative to B first, and only then added to C's separately given slot {units}.",
            f"The constraint linking A to B at slot {each} says nothing about C, whose slot {units} is given separately.",
            f"Placing A one position before B fixes A at {each - 1}; the stated slot {units} for C remains a separate input.",
            f"First derive A from B's slot {each}, then combine that derived position with C's independent slot {units}.",
            f"Only A depends on B's position: C already has slot {units} and enters the final addition unchanged.",
        )
    elif domain == "work_allocation":
        people = number(f"allocation-people:{code}", 3, 9)
        rounds = number(f"allocation-rounds:{code}", 2, 6)
        items_per_round = number(f"allocation-items:{code}", 2, 12)
        items_per_person = rounds * items_per_round
        item_count = people * items_per_person
        data = (
            f"Problem {code}: distribute {item_count} items equally among {people} "
            f"people, then divide each person's share equally across {rounds} rounds."
        )
        equation = (
            f"{item_count} / {people} = {items_per_person}; "
            f"{items_per_person} / {rounds} = {items_per_round}"
        )
        total, check = (
            f"{items_per_person} items per person and {items_per_round} items per person per round",
            f"{people} people × {rounds} rounds × {items_per_round} items = {item_count} items",
        )
        components = (
            f"The equal split happens twice: first among {people} people, then across {rounds} rounds for each person's share.",
            f"Each person's {items_per_person}-item share divides into {rounds} equal rounds of {items_per_round} items apiece.",
            f"Dividing by {people} people and then {rounds} rounds gives sequential splits of the same {item_count} items.",
        )
    else:
        result = units
        total_outcomes = units + each
        data = f"Problem {code}: a bag has {units} blue and {each} amber tokens; one token is drawn uniformly."
        equation = f"{units} / ({units} + {each}) = {units}/{total_outcomes}"
        total, check = (
            f"{units}/{total_outcomes} probability of blue",
            f"the favorable and total counts are {units} and {total_outcomes}",
        )
        components = (
            f"The {total_outcomes} total tokens include both the {units} blue and {each} amber tokens counted together in one draw.",
            f"Every one of the {total_outcomes} tokens has an equal chance of being drawn, since none are distinguished beyond their color.",
            f"The {units} blue and {each} amber tokens together make up all {total_outcomes} possible outcomes of the single draw.",
        )
    return data, equation, total, check, components
