from __future__ import annotations


def brainstorm_checks(
    domain_label: str,
) -> tuple[
    dict[str, str | tuple[str, ...]],
    dict[str, str | tuple[str, ...]],
    str,
    tuple[str, ...],
]:
    constraint_checks = {
        "Make the options meaningfully different rather than cosmetic rewrites.": (
            f"The {domain_label} options differ in mechanism rather than wording alone.",
            f"Each {domain_label} proposal uses a distinct way to achieve the brief.",
            f"The alternatives change how the {domain_label} result would be produced, not merely how it is described.",
            f"Different operating approaches separate the three {domain_label} candidates.",
            f"The retained {domain_label} ideas pursue the goal through genuinely different methods.",
            f"Choosing among these {domain_label} options changes the underlying approach as well as the wording.",
        ),
        "Avoid ideas that create unnecessary safety, privacy, or exclusion risks.": (
            f"None of the {domain_label} options requires sensitive personal data or an avoidable safety risk.",
            f"The proposed {domain_label} approaches work without collecting private information or exposing people to needless harm.",
            f"Safety, privacy, and access remain protected across the retained {domain_label} ideas.",
            f"No {domain_label} candidate depends on risky conduct, personal records, or an exclusionary condition.",
            f"Each option meets the {domain_label} brief without introducing an unnecessary privacy or safety burden.",
            f"The selection filters out {domain_label} ideas that would create avoidable harm or unequal access.",
        ),
        "Explain briefly how each retained option meets the named criteria.": (
            f"Each description states how its {domain_label} option fits the brief.",
            f"The rationale for every {domain_label} candidate connects directly to the requested criteria.",
            f"Each retained idea includes a concise account of why it qualifies for the {domain_label} brief.",
            f"The descriptions make the criterion fit of all three {domain_label} options explicit.",
            f"Every {domain_label} proposal is paired with a brief explanation of the requirements it meets.",
            f"The response links each candidate's design to the named {domain_label} conditions.",
        ),
        "Keep every option feasible within the stated resources.": (
            (
                f"All three {domain_label} options stay within the resources named in the brief.",
                f"Each proposed {domain_label} approach fits the available time, capacity, and materials.",
                f"Resource limits remain satisfied across the complete {domain_label} candidate set.",
                f"None of the retained {domain_label} proposals requires capacity outside the brief.",
                f"The three {domain_label} ideas are feasible with only the resources already supplied.",
                f"Every {domain_label} candidate remains deliverable under the stated resource boundary.",
            )
        ),
        "Keep the intended audience visible in each option.": (
            (
                f"Each {domain_label} option remains directed to the audience named in the brief.",
                f"The intended audience is explicit throughout all three {domain_label} proposals.",
                f"Every retained {domain_label} idea identifies who it is designed to serve.",
                f"Audience fit stays visible when the {domain_label} alternatives are compared.",
                f"None of the {domain_label} options loses the people specified by the brief.",
                f"The three {domain_label} candidates keep their intended participants in scope.",
            )
        ),
        "Keep the proposal small enough to test and revise.": (
            (
                f"The selected {domain_label} option can be tested at the stated scale before expansion.",
                f"A bounded pilot can evaluate the chosen {domain_label} proposal before any wider rollout.",
                f"The recommended {domain_label} idea remains small enough for an initial test and later revision.",
                f"Testing the leading {domain_label} option first does not require committing to expansion.",
                f"The selected proposal supports a limited {domain_label} trial whose results can guide changes.",
                f"A first-stage test keeps the {domain_label} choice reversible at the supplied scale.",
            )
        ),
    }
    outcome_checks = {
        "The remaining ideas are feasible within the available resources.": (
            f"The three retained {domain_label} ideas remain feasible under the stated limits.",
            f"Available resources are sufficient for every surviving {domain_label} proposal.",
            f"Each shortlisted {domain_label} approach can be delivered with the capacity already specified.",
            f"The retained alternatives require no {domain_label} resources beyond those in the brief.",
            f"All remaining {domain_label} ideas fit the supplied time, materials, and operating limits.",
            f"Feasibility is preserved across the final {domain_label} shortlist without expanding the resource boundary.",
        ),
        "The main trade-off of each leading option is visible.": (
            (
                f"The {domain_label} alternatives emphasize different strengths, making the choice explicit.",
                f"Different advantages remain visible across the {domain_label} options, so the recommendation has a clear basis.",
                f"The leading {domain_label} choices expose distinct trade-offs rather than collapsing into one idea.",
                f"Each {domain_label} alternative offers a different benefit, allowing the final choice to be justified directly.",
                f"Comparison preserves the separate strengths of the {domain_label} candidates and clarifies why one leads.",
                f"The trade-offs among the {domain_label} proposals stay visible in the resulting selection.",
            )
        ),
        "Each retained option satisfies the stated criteria.": (
            f"Each retained {domain_label} option satisfies the stated criteria.",
            f"All shortlisted {domain_label} proposals meet the requirements in the brief.",
            f"Every surviving idea clears the full set of {domain_label} conditions.",
            f"The named criteria hold for each of the retained {domain_label} alternatives.",
            f"No selected {domain_label} candidate falls outside the requested requirements.",
            f"Criterion checks pass across the complete {domain_label} shortlist.",
        ),
        "One idea is developed into a small testable proposal.": (
            (
                f"The selected {domain_label} idea is the smallest concrete proposal to test first.",
                f"One leading {domain_label} option has been reduced to a bounded initial trial.",
                f"The recommendation turns a {domain_label} candidate into a specific testable proposal.",
                f"A single {domain_label} idea now has a concrete first experiment rather than a broad rollout.",
                f"The chosen option defines the smallest useful {domain_label} pilot for gathering evidence.",
                f"Development stops at a practical first test of the selected {domain_label} approach.",
            )
        ),
        "Compatible strengths are combined without preserving their conflicts.": (
            (
                f"The {domain_label} selection keeps compatible strengths without combining conflicting requirements.",
                f"The chosen {domain_label} proposal joins only features that can coexist under the brief.",
                f"Useful elements are retained in the final {domain_label} idea while incompatible demands stay separate.",
                f"The recommendation combines complementary {domain_label} strengths without importing their conflicts.",
                f"Only mutually workable features survive in the selected {domain_label} approach.",
                f"The final {domain_label} option integrates compatible benefits and leaves contradictory requirements out.",
            )
        ),
        "The candidate options differ in a meaningful and useful way.": (
            f"The candidate {domain_label} options differ in a way that changes how the brief would be carried out.",
            f"Each {domain_label} candidate offers a materially different route to the requested outcome.",
            f"The alternatives create useful {domain_label} choices rather than superficial wording changes.",
            f"Selecting a different candidate would change the practical {domain_label} approach.",
            f"The three proposals cover distinct and useful ways to execute the {domain_label} brief.",
            f"Meaningful operational differences separate the available {domain_label} candidates.",
        ),
    }
    default_constraint = (
        f"The {domain_label} options remain bounded by the explicit brief."
    )
    default_outcome = (
        f"The selected {domain_label} option is concrete enough to test first.",
        f"The recommended {domain_label} proposal has a specific first test.",
        f"A bounded trial can now evaluate the leading {domain_label} idea.",
        f"The chosen {domain_label} approach is defined clearly enough for an initial pilot.",
        f"The final {domain_label} selection translates into a practical first experiment.",
        f"The leading option supplies an actionable starting test for {domain_label}.",
    )
    return constraint_checks, outcome_checks, default_constraint, default_outcome
