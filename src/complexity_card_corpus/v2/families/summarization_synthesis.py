from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool
from ._axes import PEOPLE, SITES
from ._common import render_v2_row, validate_complete_rows


TASK = "summarization_synthesis"
_CASES = (
    ("operations", "a ventilation fault interrupted the morning room inspection", "replaced a loose fan guard and repeated the safety check", "the room reopened with normal airflow", "add guard clearance to the monthly checklist"),
    ("operations", "two workshops were assigned to the same room", "moved the smaller workshop and corrected the shared calendar", "both sessions started on time", "enable conflict alerts for recurring bookings"),
    ("logistics", "a medical parcel missed its transfer scan", "located it in the inbound cage and restored its tracking entry", "the parcel reached the clinic that afternoon", "require a scan before cages are moved"),
    ("logistics", "an inventory count differed from the ordering system", "recounted the sealed stock and removed a duplicated receipt", "the verified balance matched the shelves", "review automated receipts before replenishment"),
    ("training", "new staff could not follow the outdated login guide", "rewrote the access steps and tested them with a new starter", "the next group completed setup without assistance", "review the guide after every account-system change"),
    ("training", "a workshop left no time for practice", "shortened the lecture and added a guided exercise", "participants completed the core task during the session", "reserve one third of future sessions for practice"),
    ("research", "field readings varied after equipment transport", "recalibrated the optical sensor before each location", "the repeated measurements became consistent", "log calibration values with every site visit"),
    ("research", "the survey report mixed quotations with interpretation", "separated participant statements from analyst conclusions", "reviewers could trace each conclusion to evidence", "use distinct evidence and analysis sections"),
    ("finance", "several reimbursements lacked readable receipts", "asked claimants for replacements before approval", "supported claims were paid without exceptions", "check document quality at submission"),
    ("finance", "the forecast used an expired supplier quote", "obtained a current estimate and recalculated the total", "the revised budget stayed within its ceiling", "record quotation expiry dates"),
    ("software", "a release increased error rates for mobile users", "disabled the feature flag and restored the prior path", "error rates returned to baseline", "add mobile traffic to the pre-release check"),
    ("software", "a nightly job processed one batch twice", "removed the duplicate trigger and reran the affected validation", "the corrected output contained one record per source", "alert on overlapping job starts"),
    ("community", "visitors could not find the accessible entrance", "added route details to the notice and placed signs outside", "all registered visitors reached the venue independently", "include an accessibility route in every event brief"),
    ("community", "volunteers arrived at different check-in doors", "named one entrance in the shift message and roster", "the next handoff began without delays", "confirm the arrival point in every reminder"),
    ("communications", "an outage notice did not identify the affected service", "published a correction naming the service and alternative", "support calls about the notice decreased", "require impact and workaround fields before release"),
    ("communications", "a newsletter repeated an outdated event date", "corrected the web and email copies and noted the change", "readers received one consistent date", "verify linked dates during final review"),
)
_FOCI = {
    "executive": "prioritize the issue, intervention, and outcome for a decision-maker",
    "action": "prioritize what was done and the next preventive action",
    "impact": "prioritize the observable result and who benefited",
    "handoff": "prioritize current status and what the next owner must remember",
}
_PROMPTS = (
    "Summarize this record with an {scenario[focus]} focus; {scenario[guidance]}. Record: {scenario[record]}",
    "Produce a two-sentence {scenario[focus]} summary. In particular, {scenario[guidance]}. Source: {scenario[record]}",
    "Synthesize the following notes for a {scenario[focus]} handoff; {scenario[guidance]}. {scenario[record]}",
    "Condense this report without copying its wording. Use an {scenario[focus]} focus and {scenario[guidance]}. Report: {scenario[record]}",
)


def _summary(focus: str, person: str, issue: str, action: str, outcome: str, follow_up: str, site: str, domain: str) -> str:
    if focus == "executive":
        return f"At {site}, {person} addressed {issue} by having the team {action}. The intervention succeeded: {outcome}; this gives the {domain} owner a verified basis for the next decision."
    if focus == "action":
        return f"The team {action} at {site} after {issue}. Its next preventive task is to {follow_up}, giving {person} a checkable {domain} handoff."
    if focus == "impact":
        return f"The key result at {site} was that {outcome}. This followed {person}'s response to {issue} and provides a concrete {domain} measure for later review."
    if focus == "handoff":
        return f"Current status at {site}: {outcome}. The next {domain} owner should {follow_up}, with {person}'s earlier response to {issue} retained as the handoff context."
    raise ValueError(focus)


def summarization_synthesis_capacity() -> int:
    return len(_CASES) * len(_FOCI)


def render_summarization_synthesis_rows() -> list[dict[str, object]]:
    rows = []
    for case_index, (domain, issue, action, outcome, follow_up) in enumerate(_CASES):
        person = PEOPLE[case_index % len(PEOPLE)]
        site = SITES[case_index % len(SITES)]
        record = (
            f"Incident: {issue}. Owner: {person} at {site}. Response: the team "
            f"{action}. Observed result: {outcome}. Recommendation: {follow_up}."
        )
        for focus, guidance in _FOCI.items():
            target = _summary(focus, person, issue, action, outcome, follow_up, site, domain)
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {"focus": (focus,), "guidance": (guidance,), "record": (record,)},
                        "prompt": {"summary_request": _PROMPTS},
                        "answer": {"summary": (target,)},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{case_index}:{focus}", variables=variables,
                prompt_pools=(V2SubcardPool("summary_request", SurfaceRole.PROMPT, ("{prompt[summary_request]}",)),),
                answer_pools=(V2SubcardPool("summary", SurfaceRole.ANSWER, ("{answer[summary]}",)),),
            )
            case_id = f"{domain}:{case_index}:{focus}"
            rows.append(
                render_v2_row(
                    task=TASK, case_id=case_id, domain=domain, difficulty="medium",
                    deck=deck,
                    facts={"focus": focus, "issue": issue, "action": action, "outcome": outcome, "follow_up": follow_up, "site": site, "person": person},
                    validator={"kind": "exact", "expected": target},
                )
            )
    return validate_complete_rows(TASK, rows, summarization_synthesis_capacity())


__all__ = ("render_summarization_synthesis_rows", "summarization_synthesis_capacity")
