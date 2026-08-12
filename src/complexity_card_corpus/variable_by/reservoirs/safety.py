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
        "Preserve the suspicious message, disclose nothing, and inspect the account from a separately opened official session.",
        "Break contact with the sender and verify any claimed account problem through a known provider route.",
        "Leave codes and passwords untouched while the provider's genuine security service checks the alert.",
    ),
    "medical_information": (
        "Do not rely on an online diagnosis; seek urgent local medical help and contact emergency services if symptoms are severe or worsening.",
        "Treat the symptoms as requiring prompt local assessment rather than an online conclusion, then contact urgent medical care.",
        "Move from online discussion to qualified local care and use emergency services if the pressure is severe or worsening.",
        "Arrange prompt local clinical assessment and use emergency help immediately if the reported symptoms intensify.",
        "Seek an in-person medical evaluation rather than treating the remote exchange as a diagnosis.",
        "Contact qualified local care now, escalating to emergency services when symptoms are severe, new, or worsening.",
        "Arrange direct medical assessment promptly and use emergency help rather than waiting online if the condition worsens.",
        "Have a qualified local service assess the person now, with emergency escalation for severe or changing symptoms.",
        "Stop seeking certainty by message and move to in-person or emergency care appropriate to the current severity.",
    ),
    "financial_decision": (
        "Do not transfer funds; contact the financial institution through an independently verified channel and request a fraud review.",
        "Pause the transfer and make no payment, then reach the institution through its known official channel for a fraud review.",
        "Keep the funds in place, end the promotional contact, and verify the request directly with the financial institution.",
        "Make no transfer and independently ask the institution's fraud team to verify the request.",
        "Preserve the current account state, stop engaging with the sender, and open a fresh official banking channel.",
        "Decline the pressured payment and confirm the supposed offer directly with the institution before any commitment.",
        "Keep control of the funds, preserve the request as evidence, and ask the verified institution to investigate it.",
        "End the sender's payment flow and independently confirm the account state with the official fraud service.",
        "Authorize nothing while a trusted banking contact checks the claimed offer and destination account.",
    ),
    "physical_safety": (
        "Leave the affected area without operating controls or investigating the source, then contact local emergency or building-safety services from safety.",
        "Move everyone away from the hazard and avoid touching equipment or testing the source before calling the appropriate local responder.",
        "Exit the affected space without attempting to identify the source, then report the observed hazard from a safe location.",
        "Move to a safe location without touching nearby controls, then report the suspected physical hazard to local responders.",
        "Keep people away from the affected space and contact the appropriate emergency or building-safety service.",
        "Avoid investigating or operating equipment; leave the area and request qualified local hazard assessment.",
        "Move away from the uncertain area without touching controls, then request an on-site hazard response.",
        "Keep everyone outside the affected rooms and let qualified responders identify the source.",
        "Do not test switches, equipment, or the unknown source; evacuate and report the suspected hazard from safety.",
        "Leave the space undisturbed, prevent re-entry, and contact the appropriate local emergency channel.",
        "Create distance from the possible exposure before asking trained responders to assess the site.",
        "Treat the rooms as unsafe, avoid any source investigation, and escalate from a clear location.",
    ),
}

_BOUNDARIES = {
    "privacy_security": (
        "I cannot verify the caller or replace the provider's security team.",
        "The available information is not enough to authenticate the request.",
        "A safe account response can be outlined here, but the contact's identity cannot be authenticated remotely.",
        "Treat the contact as unverified and share no credential or one-time code.",
        "Only the provider can authenticate this contact through its established security process.",
        "Remote claims from the caller do not establish control of the account or legitimacy of the request.",
        "The sender's urgency and repeated contact provide no proof that the account notice is genuine.",
        "Only independent provider verification can establish whether any security action is actually required.",
        "The contact must remain unauthenticated because its identity cannot be confirmed through the message itself.",
    ),
    "medical_information": (
        "I cannot diagnose the symptoms or replace qualified local assessment.",
        "The available information is not enough to confirm a diagnosis or remove the risk.",
        "Protective guidance is possible, while diagnosis still requires qualified assessment of the person.",
        "Treat the symptoms as medically unassessed until a qualified clinician evaluates them.",
        "A remote text exchange cannot establish the cause or severity of the reported symptoms.",
        "Only qualified local assessment can determine what the symptoms mean and how urgent they are.",
        "Reported symptoms can justify urgent care without allowing a remote diagnosis of their cause.",
        "Severity and cause remain medically unconfirmed until the person is assessed directly.",
        "No text-only account can rule out a serious condition or replace examination by qualified care.",
    ),
    "financial_decision": (
        "I cannot verify the offer or replace the institution's fraud team.",
        "The available information is not enough to establish that the request is legitimate.",
        "The immediate financial risk can be bounded, but neither the request nor its promised return can be verified here.",
        "Treat the offer as unverified and avoid any irreversible transfer.",
        "The claimed return and payment request remain unverified until the institution confirms them independently.",
        "Nothing in the message establishes that the sender is authorized to request or receive funds.",
        "Neither the promised return nor the pressured timing is independently supported by the available record.",
        "The institution must verify the offer and destination before either can be treated as legitimate.",
        "A message alone cannot authenticate the payment request or establish that any return is guaranteed.",
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
        "Open the account provider's verified application directly and submit the suspicious contact to security.",
        "Use support details from a trusted statement or official site, never from the incoming message.",
        "Preserve the message as evidence and start a separate conversation with the provider's fraud or security team.",
        "From a known device, check the account through the provider's established security route.",
        "Report the attempted code request using independently sourced provider contact information.",
        "Reach the provider through its authenticated account portal and keep every login secret out of the report.",
    ),
    "medical_information": (
        "Contact local emergency services or urgent medical care now.",
        "Use the local emergency number or an immediately available qualified clinician.",
        "Seek urgent in-person assessment rather than continuing an online diagnosis attempt.",
        "Use emergency or urgent-care services appropriate to the person's location.",
        "Contact a qualified local clinician or emergency service that can assess the person directly.",
        "Use the urgent medical channel available at the person's current location.",
        "Reach a local service capable of examining the person and determining the appropriate urgency.",
        "Contact emergency care or another qualified clinician who can assess the symptoms directly now.",
        "Use the verified urgent-care route available locally instead of relying on further online exchange.",
    ),
    "financial_decision": (
        "Contact the institution's fraud team through its official app, card, or published number.",
        "Use independently verified contact details for the financial institution.",
        "Send the message to the institution's security or fraud channel and preserve the evidence.",
        "Use a trusted banking channel before discussing or moving any funds.",
        "Report the request through the institution's verified fraud service before taking financial action.",
        "Contact the bank or payment provider using details from a trusted statement, card, or application.",
        "Open the institution's genuine application or call a verified number and ask its fraud team to review the request.",
        "Report the destination account and promised return through an independently sourced banking-security channel.",
        "Use a trusted statement or official account portal to reach the institution before discussing any transfer.",
    ),
    "physical_safety": (
        "From a safe location, contact local emergency or building-safety services.",
        "Contact the site's emergency service or local responders without re-entering the area.",
        "Use the appropriate local hazard-response channel once everyone is clear.",
        "From outside the affected space, follow local responder instructions.",
        "Call the appropriate hazard-response service from safety and follow its local directions.",
        "Use the site's emergency channel or local responders without returning to the uncertain area.",
        "Report the possible exposure to local responders while remaining outside the affected space.",
        "From the safe assembly point, use the building's emergency contact or the local hazard service.",
        "Ask qualified local responders to assess the rooms before anyone considers going back inside.",
        "Use the locally designated emergency route and wait in safety for responder instructions.",
        "Contact building safety or emergency services from outside, keeping the area closed to re-entry.",
        "Escalate the unknown smell through an on-site emergency channel that can send a qualified assessment.",
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
        "The information warrants care without showing how quickly the risk may develop.",
        "A hazard remains plausible, although the report cannot establish its immediacy.",
        "Protective caution is justified while the urgency remains unresolved.",
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
        "Offer safe information without taking the consequential step or pretending it has occurred.",
        "The response may guide protection, but it must neither act for the person nor report an action as done.",
        "Leave execution to the authorized person or service and make no completion claim.",
    ),
    "Do not give procedural steps until immediate hazards and local conditions are understood.": (
        "Avoid procedural guidance until the immediate hazard and local conditions are clear.",
        "Specific instructions should wait until a qualified local assessment establishes the conditions.",
        "Do not move into detailed procedure while the immediate environment remains uncertain.",
        "Establish immediate safety and local conditions before describing any procedure.",
        "Until the hazard is assessed locally, keep the response to evacuation and qualified escalation.",
        "Detailed steps are inappropriate while the surroundings and immediate danger remain unknown.",
    ),
    "Do not request or expose passwords, secrets, or full sensitive identifiers.": (
        "No credential, secret, or complete sensitive identifier should be collected or revealed.",
        "Keep passwords and sensitive identifiers out of the exchange.",
        "Protect account secrets and use only independently verified support channels.",
        "Ask for no password, access code, or complete identifier, and keep all such data out of the response.",
        "Verification should occur through the provider without exposing credentials in this exchange.",
        "Use nonsensitive observations only; secrets and full identifiers must remain private.",
    ),
    "Keep diagnosis and regulated advice with qualified professionals.": (
        "Diagnosis and regulated guidance belong with a qualified professional.",
        "A qualified local professional must make the diagnostic or regulated judgment.",
        "Keep this response to protective guidance and leave diagnosis to qualified care.",
        "Provide only a safety boundary while a qualified professional handles diagnosis or regulated judgment.",
        "Do not substitute a remote conclusion for the assessment required from qualified care.",
        "The professional decision remains outside this response; limit the help to proportionate protection.",
    ),
    "Keep regulated or personalized financial advice with an appropriately qualified professional.": (
        "Personalized financial judgment belongs with an appropriately qualified professional.",
        "Do not turn this protective response into individualized financial advice.",
        "A qualified professional should handle any regulated or personalized financial decision.",
        "Keep the response at the level of fraud prevention and refer individualized judgment to qualified advice.",
        "Do not recommend a personalized financial choice; preserve that decision for an authorized professional.",
        "Protect the funds now while leaving regulated advice and individual suitability assessment to qualified help.",
    ),
    "Prefer the reversible option with the least credible harm.": (
        "Use the least harmful reversible step available.",
        "Choose a reversible protective move before any irreversible commitment.",
        "Keep the next action reversible and proportionate to the credible harm.",
        "Favor the protective choice that can be undone and carries the smallest plausible downside.",
        "Start with a low-regret safeguard rather than a step that cannot be reversed.",
        "Match the response to the credible risk while preserving a safe way back.",
        "Where uncertainty remains, use the bounded option that avoids creating additional harm.",
        "Select a proportionate precaution whose effects can be stopped or reversed.",
        "Do not increase exposure merely to gain certainty; take the safer reversible path.",
    ),
    "Prioritize immediate harm reduction over detailed analysis.": (
        "Immediate harm reduction comes before a detailed explanation.",
        "Protective action should take priority over extended analysis.",
        "Reduce the immediate risk first and leave detailed interpretation for later.",
        "Move the person or account toward safety before analyzing the cause in depth.",
        "The first priority is limiting harm; explanation can follow once protection is in place.",
        "Act on the credible immediate danger before attempting a complete interpretation.",
    ),
    "State material uncertainty instead of presenting a guess as fact.": (
        "Keep material uncertainty visible rather than turning a guess into a conclusion.",
        "Any unresolved risk should remain explicit in the response.",
        "Do not present an unsupported judgment as established fact.",
        "Separate the confirmed observations from every safety conclusion that remains uncertain.",
        "Name the material unknown instead of filling it with an unverified interpretation.",
        "Report what the evidence cannot establish as clearly as what it does establish.",
        "Avoid false reassurance or alarm by preserving the unresolved part of the risk assessment.",
        "Keep the response bounded by observed facts while qualified verification is pending.",
        "Treat a consequential unknown as open rather than converting it into certainty.",
    ),
    "State relevant location and access factors.": (
        "Relevant location and access limits still need to be considered.",
        "The appropriate channel depends on the person's location and available access.",
        "Local access conditions matter when choosing the escalation route.",
        "Choose help only after accounting for the person's location and the services they can actually reach.",
        "The escalation path must fit both local availability and any practical access barrier.",
        "Location determines which qualified channel is reachable, so keep that constraint explicit.",
    ),
}


def safety_reservoir(
    domain: str,
    *,
    state: str,
    constraint: str,
    action_grounding: str | tuple[str, ...],
    boundary_grounding: str | tuple[str, ...],
    channel_grounding: str | tuple[str, ...],
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    """Build the English safety reservoir for one scenario's semantic cells."""

    table: dict[str, dict[str, tuple[str, ...]]] = {
        "framing": {
            "protective": (
                "Prioritize immediate protection:",
                "Use this reversible safeguard first:",
                "Reduce the immediate risk with this step:",
                "Begin with the least harmful protective move:",
                "Take this proportionate precaution before further analysis:",
                "Start from a reversible action that limits exposure:",
                "Protect the person or account before interpreting the cause:",
                "Use the safest available response while facts remain uncertain:",
                "Put immediate risk reduction ahead of detailed explanation:",
                "Choose this low-regret safeguard as the first response:",
                "Keep the next move protective and reversible:",
                "Act on the credible risk without assuming its cause:",
                "Use this cautious first step while verification is pending:",
                "Limit possible harm with the following immediate measure:",
                "Make protection the first priority in this uncertain situation:",
                "Apply this reversible boundary before gathering more detail:",
                "Move first to the safer state described here:",
                "Use a protective default until qualified verification arrives:",
                "Take the minimum-risk action supported by the report:",
                "Preserve safety now with this reversible response:",
                "Respond to the possible harm with this bounded precaution:",
                "Begin by reducing exposure with this protective move:",
                "Keep uncertainty visible and take this protective step:",
                "Use this immediate safeguard without claiming certainty:",
                "Select the reversible option that best limits credible harm:",
                "Take the protective route with the lowest plausible downside:",
                "Start with a safeguard that can be reversed if facts change:",
                "Reduce credible harm before drawing any conclusion:",
                "Choose the immediate measure that preserves the safest options:",
                "Limit exposure first while leaving interpretation to qualified help:",
                "Use a cautious measure that does not depend on an assumed cause:",
                "Move toward safety through the least disruptive available step:",
                "Address the present risk with a reversible protective measure:",
            )
        },
        "label": {
            "protective_action": (
                "Immediate action",
                "Protective step",
                "First safeguard",
            ),
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
            "action": (
                (action_grounding,)
                if isinstance(action_grounding, str)
                else action_grounding
            ),
            "boundary": (
                (boundary_grounding,)
                if isinstance(boundary_grounding, str)
                else boundary_grounding
            ),
            "channel": (
                (channel_grounding,)
                if isinstance(channel_grounding, str)
                else channel_grounding
            ),
        },
    }
    if state_cards := _STATE_BOUNDARIES.get(state):
        table["state"] = {"boundary": state_cards}
    if constraint_cards := _CONSTRAINT_BOUNDARIES.get(constraint):
        table["constraint"] = {"boundary": constraint_cards}
    return table
