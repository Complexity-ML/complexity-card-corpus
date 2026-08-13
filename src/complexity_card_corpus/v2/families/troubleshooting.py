from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import (
    V2RoleSeparatedDeck,
    V2SubcardPool,
    answer_variant_plans,
    prompt_variant_plans,
)
from ._axes import PEOPLE, SITES
from ._common import render_v2_row, validate_complete_rows


TASK = "troubleshooting"
_FAULTS = (
    ("network", "a wireless terminal", "drops its connection every few minutes", "compare its signal on the staff and guest networks", "move it to the stable staff network and renew its lease"),
    ("network", "a desk phone", "can receive calls but cannot place them", "check whether its outbound line profile is assigned", "restore the outbound profile and place a test call"),
    ("hardware", "a shared tablet", "powers off shortly after startup", "inspect the battery health reading while it is charging", "replace the failing battery before returning the tablet"),
    ("hardware", "a label printer", "feeds blank labels despite a full roll", "print the internal calibration page without the workstation", "recalibrate the sensor and reload the roll in the marked direction"),
    ("software", "a scheduling form", "saves entries but shows an empty calendar", "open the same account in a private session to isolate cached state", "clear the stale site data and reload the calendar"),
    ("software", "a reporting dashboard", "shows yesterday's totals after refresh", "compare its data timestamp with the source export", "restart the delayed refresh job and verify the new timestamp"),
    ("power", "an environmental sensor", "restarts whenever its radio transmits", "measure the supply voltage during one transmission", "replace the undersized power adapter with the specified unit"),
    ("power", "an emergency lamp", "passes its lamp test but not its battery test", "record the battery voltage before and during the test", "replace the depleted backup battery and repeat the test"),
    ("data", "an intake spreadsheet", "duplicates every imported record", "run one import with a fresh file and inspect the unique-key column", "remove the repeated import rule and deduplicate by the verified key"),
    ("data", "a survey export", "omits responses containing accented names", "compare the export encoding with a UTF-8 sample", "switch the export to UTF-8 and regenerate the file"),
    ("mechanical", "a ventilation fan", "rattles only at its highest setting", "inspect the guard clearance while increasing speed gradually", "secure the loose guard and confirm clearance at full speed"),
    ("mechanical", "a cabinet door", "will not latch after the cabinet was moved", "check the latch alignment with the cabinet level", "level the cabinet and realign the latch plate"),
    ("workflow", "an approval request", "returns to draft after each signature", "review the audit log for the transition immediately after signing", "remove the obsolete transition rule and retry with a test request"),
    ("workflow", "a delivery ticket", "remains open after proof of delivery", "compare the proof timestamp with the closure trigger", "repair the trigger mapping and close a sandbox ticket"),
    ("audio_visual", "a meeting display", "shows video but produces no sound", "select its audio output explicitly and play the built-in tone", "set the display as the active output and save the room profile"),
    ("audio_visual", "a room microphone", "sounds clear locally but distorted remotely", "record locally and through the conference service at the same gain", "lower the service input gain and disable duplicate processing"),
)
_PROMPTS = (
    "Diagnose this issue with one discriminating check before the fix: {scenario[report]}",
    "Give a test-first troubleshooting response: {scenario[report]}",
    "What should be checked, and what repair follows if the check confirms it? {scenario[report]}",
    "Troubleshoot this without guessing or listing unrelated causes: {scenario[report]}",
    "Propose the next diagnostic action and the corresponding remedy: {scenario[report]}",
    "Use the symptom to choose a focused test, then state the repair: {scenario[report]}",
)
_ANSWERS = (
    "At {scenario[site]}, ask {scenario[person]} to {scenario[check]}. If that confirms the fault, {scenario[fix]}.",
    "First, have {scenario[person]} {scenario[check]} at {scenario[site]}. On confirmation, {scenario[fix]}.",
    "The discriminating step for {scenario[person]} is to {scenario[check]}. If positive, {scenario[fix]} at {scenario[site]}.",
    "Use this check at {scenario[site]}: {scenario[check]}. Then have {scenario[person]} {scenario[fix]}.",
    "To isolate the cause, {scenario[person]} should {scenario[check]}. The matching remedy at {scenario[site]} is to {scenario[fix]}.",
    "Before changing anything else, {scenario[check]}. If reproduced, ask {scenario[person]} to {scenario[fix]} at {scenario[site]}.",
)
_PROMPT_FUNCTIONS = (
    ("request_diagnosis", "require_discriminating_check", "request_fix"),
    ("request_test_first_response",),
    ("request_check", "request_conditional_repair"),
    ("request_troubleshooting", "forbid_guessing", "forbid_unrelated_causes"),
    ("request_diagnostic_action", "request_corresponding_remedy"),
    ("derive_test_from_symptom", "request_repair"),
)
_ANSWER_FUNCTIONS = (
    ("assign_check", "condition_on_result", "state_fix"),
    ("prioritize_check", "condition_on_confirmation", "state_fix"),
    ("name_discriminating_step", "condition_on_positive", "state_fix"),
    ("locate_check", "state_fix", "assign_owner"),
    ("explain_isolation", "assign_check", "state_matching_remedy"),
    ("prevent_premature_change", "perform_check", "condition_on_reproduction", "state_fix"),
)
_SITE_CONSTRAINTS = (
    "changes must preserve the public reception terminal",
    "the device can be isolated after the evening class",
    "patient-facing service must remain available during the check",
    "the public catalog stations cannot all be restarted together",
    "only offline diagnostics are available during fieldwork",
    "the dispatch console has a five-minute maintenance window",
    "volunteer records must remain accessible throughout the test",
    "the research capture must not be overwritten",
    "vendor access ends before the afternoon opening",
    "a training session begins immediately after the diagnostic window",
    "the spare unit is stored in a different warehouse zone",
    "outdoor equipment needs a dry test enclosure",
    "the exhibit controller must stay in read-only mode",
    "replacement parts are available at the adjacent bench",
    "the unit may lose connectivity while it travels",
    "student accounts cannot receive administrator privileges",
    "food-service equipment requires sanitation after handling",
    "archival records must not leave the secured network",
    "the branch has remote support but no onsite administrator",
    "the shared account must stay usable by the next volunteer shift",
    "the current prototype settings need to be captured before changes",
    "the kiosk can be closed briefly while a second kiosk remains open",
    "engine diagnostics require the ventilation system to be active",
    "the scheduling service has a sandbox for destructive tests",
)


def troubleshooting_capacity() -> int:
    return len(_FAULTS) * len(SITES)


def render_troubleshooting_rows() -> list[dict[str, object]]:
    rows = []
    for fault_index, (domain, component, symptom, check, fix) in enumerate(_FAULTS):
        for site_index, site in enumerate(SITES):
            person = PEOPLE[(fault_index + site_index) % len(PEOPLE)]
            constraint = _SITE_CONSTRAINTS[site_index]
            report = (
                f"At {site}, {person} reports that {component} {symptom}; "
                f"{constraint}."
            )
            contextual_answers = tuple(
                (
                    f"Given that {constraint}, for the reported {component} symptom "
                    f"({symptom}) at {{scenario[site]}}, "
                    + answer[0].lower()
                    + answer[1:]
                )
                for answer in _ANSWERS
            )
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {
                            "component": (component,), "symptom": (symptom,),
                            "check": (check,), "fix": (fix,), "site": (site,),
                            "person": (person,), "constraint": (constraint,),
                            "report": (report,),
                        },
                        "prompt": {"diagnosis": _PROMPTS},
                        "answer": {"test_then_fix": contextual_answers},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{component}", variables=variables,
                prompt_pools=(V2SubcardPool("diagnosis", SurfaceRole.PROMPT, ("{prompt[diagnosis]}",)),),
                answer_pools=(V2SubcardPool("test_then_fix", SurfaceRole.ANSWER, ("{answer[test_then_fix]}",)),),
                prompt_plans=prompt_variant_plans(
                    sense="diagnosis",
                    pool_name="diagnosis",
                    functions=_PROMPT_FUNCTIONS,
                ),
                answer_plans=answer_variant_plans(
                    sense="test_then_fix",
                    pool_name="test_then_fix",
                    functions=_ANSWER_FUNCTIONS,
                ),
            )
            case_id = ":".join((domain, component, site))
            rows.append(
                render_v2_row(
                    task=TASK, case_id=case_id, domain=domain, difficulty="medium",
                    deck=deck,
                    facts={"component": component, "symptom": symptom, "check": check, "fix": fix, "site": site, "person": person, "constraint": constraint},
                    validator={"kind": "contains", "required": [check, fix, constraint]},
                )
            )
    return validate_complete_rows(TASK, rows, troubleshooting_capacity())


__all__ = ("render_troubleshooting_rows", "troubleshooting_capacity")
