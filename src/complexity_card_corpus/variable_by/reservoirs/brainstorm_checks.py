from __future__ import annotations


def brainstorm_checks(
    domain_label: str,
) -> tuple[dict[str, str], dict[str, str], str, str]:
    constraint_checks = {
        "Make the options meaningfully different rather than cosmetic rewrites.": (
            f"The {domain_label} options differ in mechanism rather than wording alone."
        ),
        "Avoid ideas that create unnecessary safety, privacy, or exclusion risks.": (
            f"None of the {domain_label} options requires sensitive personal data or an avoidable safety risk."
        ),
        "Explain briefly how each retained option meets the named criteria.": (
            f"Each description states how its {domain_label} option fits the brief."
        ),
        "Keep every option feasible within the stated resources.": (
            f"All three {domain_label} options stay within the resources named in the brief."
        ),
        "Keep the intended audience visible in each option.": (
            f"Each {domain_label} option remains directed to the audience named in the brief."
        ),
        "Keep the proposal small enough to test and revise.": (
            f"The selected {domain_label} option can be tested at the stated scale before expansion."
        ),
    }
    outcome_checks = {
        "The remaining ideas are feasible within the available resources.": (
            f"The three retained {domain_label} ideas remain feasible under the stated limits."
        ),
        "The main trade-off of each leading option is visible.": (
            f"The {domain_label} alternatives emphasize different strengths, making the choice explicit."
        ),
        "Each retained option satisfies the stated criteria.": (
            "Each retained option satisfies the stated criteria."
        ),
        "One idea is developed into a small testable proposal.": (
            f"The selected {domain_label} idea is the smallest concrete proposal to test first."
        ),
        "Compatible strengths are combined without preserving their conflicts.": (
            f"The {domain_label} selection keeps compatible strengths without combining conflicting requirements."
        ),
        "The candidate options differ in a meaningful and useful way.": (
            f"The candidate {domain_label} options differ in a way that changes how the brief would be carried out."
        ),
    }
    default_constraint = (
        f"The {domain_label} options remain bounded by the explicit brief."
    )
    default_outcome = (
        f"The selected {domain_label} option is concrete enough to test first."
    )
    return constraint_checks, outcome_checks, default_constraint, default_outcome
