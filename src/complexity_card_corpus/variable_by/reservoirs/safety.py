from __future__ import annotations

from collections.abc import Mapping


_ACTIONS = {
    "privacy_security": (
        "Do not share the code or use links from the message; end the contact and use the provider's official app or published number.",
        "Share no credential or one-time code; close the message and open the provider's known official channel independently.",
        "Stop the unverified contact without following its links, then check the account through the provider's official application.",
    ),
    "medical_information": (
        "Do not rely on an online diagnosis; seek urgent local medical help and contact emergency services if symptoms are severe or worsening.",
        "Treat the symptoms as requiring prompt local assessment rather than an online conclusion, then contact urgent medical care.",
        "Move from online discussion to qualified local care and use emergency services if the pressure is severe or worsening.",
    ),
    "financial_decision": (
        "Do not transfer funds; contact the financial institution through an independently verified channel and request a fraud review.",
        "Pause the transfer and make no payment, then reach the institution through its known official channel for a fraud review.",
        "Keep the funds in place, end the promotional contact, and verify the request directly with the financial institution.",
    ),
    "physical_safety": (
        "Leave the room without operating switches or investigating the source, then contact local emergency or building-safety services from safety.",
        "Move everyone out of the closed room and avoid switches, flames, or investigation before calling the appropriate local responder.",
        "Exit the affected area without testing the source, then report the unknown chemical smell from a safe location.",
    ),
}

_BOUNDARIES = {
    "privacy_security": (
        "I cannot verify the caller or replace the provider's security team.",
        "The available information is not enough to authenticate the request.",
        "I can support a protective next step, but I cannot authenticate the contact.",
        "Treat the contact as unverified and share no credential or one-time code.",
    ),
    "medical_information": (
        "I cannot diagnose the symptoms or replace qualified local assessment.",
        "The available information is not enough to confirm a diagnosis or remove the risk.",
        "I can support a protective next step, but I cannot diagnose the situation.",
        "Treat the symptoms as medically unassessed until a qualified clinician evaluates them.",
    ),
    "financial_decision": (
        "I cannot verify the offer or replace the institution's fraud team.",
        "The available information is not enough to establish that the request is legitimate.",
        "I can support a protective next step, but I cannot authenticate the request or guarantee a return.",
        "Treat the offer as unverified and avoid any irreversible transfer.",
    ),
    "physical_safety": (
        "I cannot identify the hazard remotely or replace qualified local responders.",
        "The available information is not enough to declare the room safe.",
        "I can support evacuation and escalation, but I cannot assess the source remotely.",
        "Treat the area as unsafe until qualified responders assess it.",
    ),
}

_CHANNELS = {
    "privacy_security": (
        "Use the provider's official app or published support number; share no code or password.",
        "Contact the provider's security team using details obtained independently of the message.",
        "Use the known account-security page and preserve the suspicious message for reporting.",
        "Use a trusted device and the provider's verified support channel.",
    ),
    "medical_information": (
        "Contact local emergency services or urgent medical care now.",
        "Use the local emergency number or an immediately available qualified clinician.",
        "Seek urgent in-person assessment rather than continuing an online diagnosis attempt.",
        "Use emergency or urgent-care services appropriate to the person's location.",
    ),
    "financial_decision": (
        "Contact the institution's fraud team through its official app, card, or published number.",
        "Use independently verified contact details for the financial institution.",
        "Send the message to the institution's security or fraud channel and preserve the evidence.",
        "Use a trusted banking channel before discussing or moving any funds.",
    ),
    "physical_safety": (
        "From a safe location, contact local emergency or building-safety services.",
        "Contact the site's emergency service or local responders without re-entering the area.",
        "Use the appropriate local hazard-response channel once everyone is clear.",
        "From outside the affected space, follow local responder instructions.",
    ),
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


def safety_reservoir(
    domain: str,
    *,
    state: str,
    constraint: str,
    action_grounding: str,
    boundary_grounding: str,
    channel_grounding: str,
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    """Build the English safety reservoir for one scenario's semantic cells."""

    table: dict[str, dict[str, tuple[str, ...]]] = {
        "framing": {
            "protective": (
                "Prioritize immediate protection.",
                "Use a reversible safeguard first.",
                "Reduce the immediate risk first.",
            )
        },
        "label": {
            "protective_action": ("Immediate action", "Protective step", "First safeguard"),
            "boundary": ("Boundary", "Safety boundary", "Verification limit"),
            "escalation": ("Escalation", "Trusted channel", "Next contact"),
        },
        "action": {
            "protective": tuple(
                "{framing[protective]} " + action for action in _ACTIONS[domain]
            )
        },
        "boundary": {"verification": _BOUNDARIES[domain]},
        "channel": {"trusted": _CHANNELS[domain]},
        "grounding": {
            "action": (action_grounding,),
            "boundary": (boundary_grounding,),
            "channel": (channel_grounding,),
        },
    }
    if state_cards := _STATE_BOUNDARIES.get(state):
        table["state"] = {"boundary": state_cards}
    if constraint_cards := _CONSTRAINT_BOUNDARIES.get(constraint):
        table["constraint"] = {"boundary": constraint_cards}
    return table
