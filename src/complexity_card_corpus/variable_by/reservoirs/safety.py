from __future__ import annotations

from collections.abc import Mapping


_ACTIONS = {
    "privacy_security": (
        "Do not share the code or use links from the message; end the contact and use the provider's official app or published number.",
        "Share no credential or one-time code; close the message and open the provider's known official channel independently.",
        "Stop the unverified contact without following its links, then check the account through the provider's official application.",
        "Ignore the contact's requested action, preserve the message, and verify the account in the provider's official app.",
        "Close the untrusted channel, disclose no secret, and start a separate session through the provider's published service.",
        "Keep every credential private and confirm the account state through a trusted device and known provider address.",
    ),
    "medical_information": (
        "Do not rely on an online diagnosis; seek urgent local medical help and contact emergency services if symptoms are severe or worsening.",
        "Treat the symptoms as requiring prompt local assessment rather than an online conclusion, then contact urgent medical care.",
        "Move from online discussion to qualified local care and use emergency services if the pressure is severe or worsening.",
        "Arrange prompt local clinical assessment and use emergency help immediately if the reported symptoms intensify.",
        "Seek an in-person medical evaluation rather than treating the remote exchange as a diagnosis.",
        "Contact qualified local care now, escalating to emergency services when symptoms are severe, new, or worsening.",
    ),
    "financial_decision": (
        "Do not transfer funds; contact the financial institution through an independently verified channel and request a fraud review.",
        "Pause the transfer and make no payment, then reach the institution through its known official channel for a fraud review.",
        "Keep the funds in place, end the promotional contact, and verify the request directly with the financial institution.",
        "Make no transfer and independently ask the institution's fraud team to verify the request.",
        "Preserve the current account state, stop engaging with the sender, and open a fresh official banking channel.",
        "Decline the pressured payment and confirm the supposed offer directly with the institution before any commitment.",
    ),
    "physical_safety": (
        "Leave the room without operating switches or investigating the source, then contact local emergency or building-safety services from safety.",
        "Move everyone out of the closed room and avoid switches, flames, or investigation before calling the appropriate local responder.",
        "Exit the affected area without testing the source, then report the unknown chemical smell from a safe location.",
        "Move to a safe location without touching controls, then report the suspected hazard to local responders.",
        "Keep people away from the affected space and contact the appropriate emergency or building-safety service.",
        "Avoid investigating or operating equipment; leave the area and request qualified local hazard assessment.",
    ),
}

_BOUNDARIES = {
    "privacy_security": (
        "I cannot verify the caller or replace the provider's security team.",
        "The available information is not enough to authenticate the request.",
        "I can support a protective next step, but I cannot authenticate the contact.",
        "Treat the contact as unverified and share no credential or one-time code.",
        "Only the provider can authenticate this contact through its established security process.",
        "Remote claims from the caller do not establish control of the account or legitimacy of the request.",
    ),
    "medical_information": (
        "I cannot diagnose the symptoms or replace qualified local assessment.",
        "The available information is not enough to confirm a diagnosis or remove the risk.",
        "I can support a protective next step, but I cannot diagnose the situation.",
        "Treat the symptoms as medically unassessed until a qualified clinician evaluates them.",
        "A remote text exchange cannot establish the cause or severity of the reported symptoms.",
        "Only qualified local assessment can determine what the symptoms mean and how urgent they are.",
    ),
    "financial_decision": (
        "I cannot verify the offer or replace the institution's fraud team.",
        "The available information is not enough to establish that the request is legitimate.",
        "I can support a protective next step, but I cannot authenticate the request or guarantee a return.",
        "Treat the offer as unverified and avoid any irreversible transfer.",
        "The claimed return and payment request remain unverified until the institution confirms them independently.",
        "Nothing in the message establishes that the sender is authorized to request or receive funds.",
    ),
    "physical_safety": (
        "I cannot identify the hazard remotely or replace qualified local responders.",
        "The available information is not enough to declare the room safe.",
        "I can support evacuation and escalation, but I cannot assess the source remotely.",
        "Treat the area as unsafe until qualified responders assess it.",
        "A remote description cannot identify the substance or establish that the affected space is safe.",
        "Qualified local responders must determine the source and the conditions for safe re-entry.",
        "The report cannot establish what the substance is or whether conditions permit re-entry.",
        "Only an on-site qualified assessment can identify the hazard and clear the space for return.",
        "Remote guidance can support evacuation, but it cannot certify the affected room as safe.",
        "The source and concentration of the possible hazard remain unknown until local responders assess them.",
        "Do not treat the absence of visible effects as evidence that the area is safe to enter.",
        "A text description cannot replace direct hazard identification or a responder's re-entry decision.",
    ),
}

_CHANNELS = {
    "privacy_security": (
        "Use the provider's official app or published support number; share no code or password.",
        "Contact the provider's security team using details obtained independently of the message.",
        "Use the known account-security page and preserve the suspicious message for reporting.",
        "Use a trusted device and the provider's verified support channel.",
        "Open the provider's official service independently and report the suspicious contact there.",
        "Reach account security through a known application or a number published by the provider.",
    ),
    "medical_information": (
        "Contact local emergency services or urgent medical care now.",
        "Use the local emergency number or an immediately available qualified clinician.",
        "Seek urgent in-person assessment rather than continuing an online diagnosis attempt.",
        "Use emergency or urgent-care services appropriate to the person's location.",
        "Contact a qualified local clinician or emergency service that can assess the person directly.",
        "Use the urgent medical channel available at the person's current location.",
    ),
    "financial_decision": (
        "Contact the institution's fraud team through its official app, card, or published number.",
        "Use independently verified contact details for the financial institution.",
        "Send the message to the institution's security or fraud channel and preserve the evidence.",
        "Use a trusted banking channel before discussing or moving any funds.",
        "Report the request through the institution's verified fraud service before taking financial action.",
        "Contact the bank or payment provider using details from a trusted statement, card, or application.",
    ),
    "physical_safety": (
        "From a safe location, contact local emergency or building-safety services.",
        "Contact the site's emergency service or local responders without re-entering the area.",
        "Use the appropriate local hazard-response channel once everyone is clear.",
        "From outside the affected space, follow local responder instructions.",
        "Call the appropriate hazard-response service from safety and follow its local directions.",
        "Use the site's emergency channel or local responders without returning to the uncertain area.",
    ),
}

_STATE_BOUNDARIES = {
    "Critical safety context is missing.": (
        "Important local safety context is still missing.",
        "The missing local details prevent a reliable remote assessment.",
        "The report does not yet contain the context needed to judge the hazard remotely.",
        "Key facts about exposure, timing, and local access are still absent.",
        "The missing circumstances materially limit any judgment about the present risk.",
    ),
    "The available facts indicate an active risk.": (
        "The available facts already indicate an active risk.",
        "The reported facts justify protective action before further analysis.",
        "The current information is sufficient to treat the risk as active.",
        "Enough risk is visible to justify protection before further interpretation.",
        "The reported conditions support acting on the hazard without waiting for certainty.",
    ),
    "The facts suggest risk but do not establish urgency.": (
        "The facts suggest risk without establishing its urgency.",
        "The report supports caution, but not a confident judgment about urgency.",
        "Risk is plausible here even though the available facts do not fix its severity.",
        "Caution is warranted while the timing and severity remain uncertain.",
        "The possibility of harm is supported, but the report does not establish how urgent it is.",
    ),
    "The request is ambiguous but could create meaningful harm.": (
        "The ambiguity does not remove the possibility of meaningful harm.",
        "The request remains unclear in a way that could affect safety.",
        "Because the unclear detail could change the risk, it should not be guessed.",
        "A protective response is appropriate even though the ambiguous fact remains unresolved.",
        "The unclear element must stay explicit because it could materially alter the safety assessment.",
    ),
    "The requester is asking for certainty the evidence cannot support.": (
        "The evidence does not support the certainty being requested.",
        "A confident conclusion would go beyond the available facts.",
        "The uncertainty is material and should remain explicit.",
        "A definite answer would overstate what the reported facts can establish.",
        "The requested confidence is not available from the current evidence.",
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
                "Prioritize immediate protection:",
                "Use this reversible safeguard first:",
                "Reduce the immediate risk with this step:",
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
