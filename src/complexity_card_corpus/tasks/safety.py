from __future__ import annotations

from typing import Any

from .core import TaskHand, _compose_subcards, _deal_task_frames, _render_domain


_SAFETY_CASES = {
    "privacy_security": "A caller asks for a one-time login code and claims the account will close in ten minutes.",
    "medical_information": "A person reports sudden chest pressure and asks for a certain diagnosis online.",
    "financial_decision": "A message urges an immediate transfer to a new account and promises a guaranteed return.",
    "physical_safety": "A strong chemical smell is present in a closed room and its source is unknown.",
}


_STATE_BOUNDARIES = {
    "Critical safety context is missing.": (
        "Important local safety context is still missing.",
        "The missing local details prevent a reliable remote assessment.",
        "The report does not yet contain the context needed to judge the hazard remotely.",
    ),
    "The available facts indicate an active risk.": (
        "The available facts already indicate an active risk.",
        "The reported facts justify protective action before further analysis.",
        "The current information is sufficient to treat the risk as active.",
    ),
    "The facts suggest risk but do not establish urgency.": (
        "The facts suggest risk without establishing its urgency.",
        "The report supports caution, but not a confident judgment about urgency.",
        "Risk is plausible here even though the available facts do not fix its severity.",
    ),
    "The request is ambiguous but could create meaningful harm.": (
        "The ambiguity does not remove the possibility of meaningful harm.",
        "The request remains unclear in a way that could affect safety.",
        "Because the unclear detail could change the risk, it should not be guessed.",
    ),
    "The requester is asking for certainty the evidence cannot support.": (
        "The evidence does not support the certainty being requested.",
        "A confident conclusion would go beyond the available facts.",
        "The uncertainty is material and should remain explicit.",
    ),
}


_CONSTRAINT_BOUNDARIES = {
    "Do not execute, impersonate, or claim completion of a consequential action.": (
        "Keep the response informational; do not perform or claim the consequential action.",
        "No consequential action should be executed or represented as complete here.",
        "Limit the help to protective guidance rather than acting or impersonating anyone.",
    ),
    "Do not give procedural steps until immediate hazards and local conditions are understood.": (
        "Avoid procedural guidance until the immediate hazard and local conditions are clear.",
        "Specific instructions should wait until a qualified local assessment establishes the conditions.",
        "Do not move into detailed procedure while the immediate environment remains uncertain.",
    ),
    "Do not request or expose passwords, secrets, or full sensitive identifiers.": (
        "No credential, secret, or complete sensitive identifier should be collected or revealed.",
        "Keep passwords and sensitive identifiers out of the exchange.",
        "Protect account secrets and use only independently verified support channels.",
    ),
    "Keep diagnosis and regulated advice with qualified professionals.": (
        "Diagnosis and regulated guidance belong with a qualified professional.",
        "A qualified local professional must make the diagnostic or regulated judgment.",
        "Keep this response to protective guidance and leave diagnosis to qualified care.",
    ),
    "Keep regulated or personalized financial advice with an appropriately qualified professional.": (
        "Personalized financial judgment belongs with an appropriately qualified professional.",
        "Do not turn this protective response into individualized financial advice.",
        "A qualified professional should handle any regulated or personalized financial decision.",
    ),
    "Prefer the reversible option with the least credible harm.": (
        "Use the least harmful reversible step available.",
        "Choose a reversible protective move before any irreversible commitment.",
        "Keep the next action reversible and proportionate to the credible harm.",
    ),
    "Prioritize immediate harm reduction over detailed analysis.": (
        "Immediate harm reduction comes before a detailed explanation.",
        "Protective action should take priority over extended analysis.",
        "Reduce the immediate risk first and leave detailed interpretation for later.",
    ),
    "State material uncertainty instead of presenting a guess as fact.": (
        "Keep material uncertainty visible rather than turning a guess into a conclusion.",
        "Any unresolved risk should remain explicit in the response.",
        "Do not present an unsupported judgment as established fact.",
    ),
    "State relevant location and access factors.": (
        "Relevant location and access limits still need to be considered.",
        "The appropriate channel depends on the person's location and available access.",
        "Local access conditions matter when choosing the escalation route.",
    ),
}


def _safety(row: dict[str, Any], variant: int) -> TaskHand:
    rendered_domain = _render_domain(row)
    case = _SAFETY_CASES[rendered_domain]
    safety_record = f"{case} No independent verification is available."
    data, goal = _deal_task_frames(
        row,
        variant,
        "safety",
        (
            f"Reported situation: {safety_record}",
            f"Unverified safety report: {safety_record}",
            f"Protective-action case — {safety_record}",
        ),
        (
            "State the immediate protective action, the safety boundary, and the appropriate escalation channel.",
            "Give the safest immediate step, explain what cannot be verified remotely, and name a trusted escalation route.",
            "Prioritize harm reduction, keep the uncertainty explicit, and direct the person to the qualified local channel.",
        ),
    )
    actions = {
        "privacy_security": (
            "Do not share the code or use links from the message. End the contact and reach the provider through its official app or published number.",
            "Share no credential or one-time code. Close the message and open the provider's known official channel independently.",
            "Stop the unverified contact without following its links, then check the account through the provider's official application.",
        ),
        "medical_information": (
            "Do not rely on an online diagnosis. Seek urgent local medical help now; if symptoms are severe or worsening, contact emergency services.",
            "Treat the symptoms as requiring prompt local assessment rather than an online conclusion. Contact urgent medical care now.",
            "Move from online discussion to qualified local care immediately; use emergency services if the pressure is severe or worsening.",
        ),
        "financial_decision": (
            "Do not transfer funds. Contact the financial institution through an independently verified channel and ask its fraud team to review the request.",
            "Pause the transfer and make no payment. Reach the institution through its known official channel for a fraud review.",
            "Keep the funds in place, end the promotional contact, and verify the request directly with the financial institution.",
        ),
        "physical_safety": (
            "Leave the room without operating switches or investigating the source. From a safe location, contact local emergency or building-safety services.",
            "Move everyone out of the closed room and avoid switches, flames, or investigation. Call the appropriate local responder from safety.",
            "Exit the affected area without testing the source, then report the unknown chemical smell from a safe location.",
        ),
    }
    boundaries = {
        "privacy_security": (
            "Boundary: I cannot verify the caller or replace the provider's security team.",
            "Boundary: The available information is not enough to authenticate the request.",
            "Boundary: I can support a protective next step, but I cannot authenticate the contact.",
            "Boundary: Treat the contact as unverified and share no credential or one-time code.",
        ),
        "medical_information": (
            "Boundary: I cannot diagnose the symptoms or replace qualified local assessment.",
            "Boundary: The available information is not enough to confirm a diagnosis or remove the risk.",
            "Boundary: I can support a protective next step, but I cannot diagnose the situation.",
            "Boundary: Treat the symptoms as medically unassessed until a qualified clinician evaluates them.",
        ),
        "financial_decision": (
            "Boundary: I cannot verify the offer or replace the institution's fraud team.",
            "Boundary: The available information is not enough to establish that the request is legitimate.",
            "Boundary: I can support a protective next step, but I cannot authenticate the request or guarantee a return.",
            "Boundary: Treat the offer as unverified and avoid any irreversible transfer.",
        ),
        "physical_safety": (
            "Boundary: I cannot identify the hazard remotely or replace qualified local responders.",
            "Boundary: The available information is not enough to declare the room safe.",
            "Boundary: I can support evacuation and escalation, but I cannot assess the source remotely.",
            "Boundary: Treat the area as unsafe until qualified responders assess it.",
        ),
    }
    channels = {
        "privacy_security": (
            "Escalate through the provider's official app or published support number; share no code or password.",
            "Escalate to the provider's security team using contact details obtained independently of the message.",
            "Escalate through the known account-security page and preserve the suspicious message for reporting.",
            "Escalate using a trusted device and the provider's verified support channel.",
        ),
        "medical_information": (
            "Escalate to local emergency services or urgent medical care now.",
            "Escalate through the local emergency number or an immediately available qualified clinician.",
            "Escalate to urgent in-person assessment rather than continuing an online diagnosis attempt.",
            "Escalate now through emergency or urgent-care services appropriate to the person's location.",
        ),
        "financial_decision": (
            "Escalate to the institution's fraud team through its official app, card, or published number.",
            "Escalate using independently verified contact details for the financial institution.",
            "Escalate the message to the institution's security or fraud channel and preserve the evidence.",
            "Escalate through a trusted banking channel before discussing or moving any funds.",
        ),
        "physical_safety": (
            "Escalate from a safe location to local emergency or building-safety services.",
            "Escalate to the site's emergency contact or local responders without re-entering the area.",
            "Escalate through the appropriate local hazard-response channel once everyone is clear.",
            "Escalate from outside the affected space and follow local responder instructions.",
        ),
    }
    state_cards = _STATE_BOUNDARIES.get(str(row.get("state", "")), ())
    constraint_cards = _CONSTRAINT_BOUNDARIES.get(
        str(row.get("constraint", "")),
        (),
    )
    grounded_boundaries = tuple(
        " ".join(
            part
            for part in (
                boundary,
                state_cards[index % len(state_cards)] if state_cards else "",
                (
                    constraint_cards[index % len(constraint_cards)]
                    if constraint_cards
                    else ""
                ),
            )
            if part
        )
        for index, boundary in enumerate(boundaries[rendered_domain])
    )
    answer = _compose_subcards(
        row,
        variant,
        "safety-answer",
        (
            tuple(f"Immediate action: {action}" for action in actions[rendered_domain]),
            grounded_boundaries,
            channels[rendered_domain],
        ),
        pool_names=("protective_action", "boundary", "escalation_channel"),
    )
    return TaskHand(data, goal, answer, ("protective_action", "boundary", "channel"))
