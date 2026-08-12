from __future__ import annotations

from ..matrix import VariableBy2D


def safety_case_cards(
    *,
    closure_minutes: int,
    attempt_count: int,
    symptom_minutes: int,
    pain_scale: int,
    promised_return: int,
    contact_count: int,
    hazard_minutes: int,
    room_count: int,
) -> tuple[
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    """Build fact-preserving safety cases and grounding reservoirs."""

    return {
        "privacy_security": f"A caller, after {attempt_count} attempts, asks for a one-time login code and claims the account will close in {closure_minutes} minutes.",
        "medical_information": f"A person reports sudden chest pressure, rated {pain_scale} out of 10, that started {symptom_minutes} minutes ago and asks for a certain diagnosis online.",
        "financial_decision": f"A message, sent after {contact_count} prior contacts, urges an immediate transfer to a new account and promises a guaranteed {promised_return}% return.",
        "physical_safety": f"A possible physical hazard has affected {room_count} rooms for about {hazard_minutes} minutes and its source is unknown.",
    }, {
        "privacy_security": (
            f"The caller's {attempt_count} attempts and the claimed {closure_minutes}-minute deadline are not independently verifiable.",
            f"Neither the {attempt_count} contacts nor the asserted {closure_minutes}-minute closure window authenticates the caller.",
            f"Independent evidence does not confirm the caller's {attempt_count} attempts or the claimed deadline of {closure_minutes} minutes.",
            f"The pressure created by {attempt_count} contacts and a {closure_minutes}-minute warning provides no verification of the request.",
        ),
        "medical_information": (
            f"The reported {pain_scale}-out-of-10 pressure over {symptom_minutes} minutes cannot be assessed without an in-person exam.",
            f"A remote exchange cannot determine the cause or severity of pressure rated {pain_scale}/10 for {symptom_minutes} minutes.",
            f"Qualified direct assessment is needed for the {pain_scale}-out-of-10 pressure reported over {symptom_minutes} minutes.",
            f"The {symptom_minutes}-minute symptom history and pain rating of {pain_scale} do not support a certain online diagnosis.",
        ),
        "financial_decision": (
            f"The {contact_count} prior contacts and the guaranteed {promised_return}% return cannot be verified without contacting the institution directly.",
            f"Neither {contact_count} earlier messages nor the promised {promised_return}% gain establishes that the offer is legitimate.",
            f"Only the institution can verify the request; its {promised_return}% promise and {contact_count} prior contacts are unsupported here.",
            f"The record independently confirms neither the {contact_count} contacts nor the claimed guaranteed return of {promised_return}%.",
        ),
        "physical_safety": (
            f"A hazard affecting {room_count} rooms for about {hazard_minutes} minutes cannot be identified remotely.",
            f"Remote text cannot establish the source or severity of the reported condition across {room_count} rooms over {hazard_minutes} minutes.",
            f"The {hazard_minutes}-minute report involving {room_count} rooms requires qualified on-site assessment rather than remote identification.",
            f"Neither the affected area of {room_count} rooms nor the {hazard_minutes}-minute duration reveals the hazard's cause from a distance.",
        ),
    }, {
        "privacy_security": (
            f"Mention the {attempt_count} contact attempts and the claimed {closure_minutes}-minute deadline when escalating.",
            f"Report both the {attempt_count} contacts and the asserted closure window of {closure_minutes} minutes through the official channel.",
            f"Give provider security the observed contact count ({attempt_count}) and the caller's {closure_minutes}-minute warning.",
            f"Preserve the message history and include its {attempt_count} attempts plus the claimed {closure_minutes}-minute deadline in the report.",
        ),
        "medical_information": (
            f"Report the {pain_scale}-out-of-10 pressure and its {symptom_minutes}-minute onset time to whoever provides care.",
            f"Tell qualified care that the pressure is rated {pain_scale}/10 and began {symptom_minutes} minutes ago.",
            f"Pass the observed severity of {pain_scale} and the {symptom_minutes}-minute timeline to the local medical service.",
            f"When seeking care, state both the {pain_scale}-out-of-10 rating and the reported duration of {symptom_minutes} minutes.",
        ),
        "financial_decision": (
            f"Report the {contact_count} prior contacts and the promised {promised_return}% return when escalating to the institution.",
            f"Give the institution's fraud team the observed {contact_count} contacts and the claimed return of {promised_return}%.",
            f"Include both the contact history ({contact_count}) and the guaranteed-{promised_return}% claim in the official fraud report.",
            f"Preserve and relay the sender's {promised_return}% promise together with its {contact_count} previous contacts.",
        ),
        "physical_safety": (
            f"Tell the responder the reported hazard has affected {room_count} rooms for about {hazard_minutes} minutes.",
            f"Give local responders the observed scope of {room_count} rooms and the approximate {hazard_minutes}-minute duration.",
            f"When escalating, report that the condition spans {room_count} rooms and has persisted for roughly {hazard_minutes} minutes.",
            f"Relay the affected-room count ({room_count}) and elapsed time ({hazard_minutes} minutes) without guessing the source.",
        ),
    }, {
        "privacy_security": (
            f"No framing of a {closure_minutes}-minute deadline across {attempt_count} attempts changes that.",
            f"Urgency based on {attempt_count} contacts and a {closure_minutes}-minute warning does not authenticate the request.",
            f"The claimed closure window of {closure_minutes} minutes remains unverified despite {attempt_count} attempts.",
            f"Repeated contact ({attempt_count} times) and the asserted deadline do not justify sharing account secrets.",
        ),
        "medical_information": (
            f"That holds no matter how severe the {pain_scale}-out-of-10 pressure has felt for the past {symptom_minutes} minutes.",
            f"A rating of {pain_scale}/10 over {symptom_minutes} minutes increases the need for care, not the reliability of remote diagnosis.",
            f"The reported {symptom_minutes}-minute duration and severity of {pain_scale} cannot turn an online exchange into an examination.",
            f"Even with pressure rated {pain_scale} for {symptom_minutes} minutes, qualified assessment remains necessary.",
        ),
        "financial_decision": (
            f"Neither the promised {promised_return}% return nor the {contact_count} prior contacts changes that.",
            f"A {promised_return}% guarantee and {contact_count} earlier messages do not make an unverified transfer safe.",
            f"The sender's {contact_count} contacts cannot validate its promised return of {promised_return}%.",
            f"Pressure from repeated contact and a claimed {promised_return}% gain does not remove the need for independent verification.",
        ),
        "physical_safety": (
            f"The reported spread across {room_count} rooms over {hazard_minutes} minutes does not permit remote hazard identification.",
            f"A condition lasting {hazard_minutes} minutes in {room_count} rooms still requires protection and qualified local assessment.",
            f"Neither the {room_count}-room scope nor the {hazard_minutes}-minute timeline makes source investigation safe.",
            "Observed duration and room count should guide escalation without being treated as a diagnosis of the hazard.",
        ),
    }


def inherited_safety_case(
    *,
    seed: str,
    subject: str,
    context: str,
) -> tuple[str, str, str, str]:
    """Materialize a concrete case without inheriting another domain's facts."""

    subject = subject.strip().rstrip(".")
    context = context.strip().rstrip(".")
    matrix = VariableBy2D(
        {
            "scenario": {
                "subject": (subject,),
                "context": (context,),
            },
            "case": {
                "report": (
                    "A person asks what to do about {scenario[subject]}, but the facts needed to assess the immediate risk have not been verified. Relevant context: {scenario[context]}.",
                    "A report concerning {scenario[subject]} lacks the verified details needed to judge immediate risk. The relevant setting is: {scenario[context]}.",
                    "Someone reports {scenario[subject]} without enough confirmed case information to determine urgency. Context supplied: {scenario[context]}.",
                    "The available account of {scenario[subject]} leaves the immediate level of risk uncertain. Its domain context is: {scenario[context]}.",
                ),
            },
            "boundary": {
                "remote_limit": (
                    "The missing details prevent a reliable remote assessment of {scenario[subject]}.",
                    "No remote conclusion about {scenario[subject]} is supported until the case details are verified.",
                    "The available account does not establish the severity of {scenario[subject]}.",
                    "Without confirmed case facts, the risk associated with {scenario[subject]} remains undetermined.",
                    "The evidence supplied for {scenario[subject]} is insufficient for a certain remote judgment.",
                    "A dependable assessment of {scenario[subject]} requires the absent case information.",
                    "Risk from {scenario[subject]} cannot be judged reliably until the missing facts are confirmed.",
                    "The report does not contain enough verified detail to assess {scenario[subject]} with confidence.",
                    "Any certain conclusion about {scenario[subject]} would exceed the available evidence.",
                    "Qualified local assessment is needed because key facts about {scenario[subject]} remain unavailable.",
                    "The current evidence leaves both severity and urgency for {scenario[subject]} unresolved.",
                    "No supported remote finding can close the uncertainty around {scenario[subject]}.",
                    "Missing observations keep the immediate significance of {scenario[subject]} undetermined.",
                    "A reliable judgment about {scenario[subject]} depends on case details not supplied here.",
                    "The known facts do not establish how serious or urgent {scenario[subject]} is.",
                    "Remote text alone cannot settle the risk represented by {scenario[subject]} in this incomplete account.",
                    "Verification must precede any definite assessment of {scenario[subject]}.",
                    "The uncertainty around {scenario[subject]} remains material because essential case facts are absent.",
                    "Available observations are insufficient to determine the level of risk attached to {scenario[subject]}.",
                    "No conclusion about the severity of {scenario[subject]} is warranted from this partial report.",
                    "Important gaps in the case prevent a dependable evaluation of {scenario[subject]}.",
                    "The evidence boundary leaves the urgency of {scenario[subject]} open for qualified assessment.",
                    "Until missing details are established, {scenario[subject]} cannot receive a certain remote classification.",
                    "A cautious response is required because the report does not verify enough about {scenario[subject]}.",
                    "The missing evidence around {scenario[subject]} prevents a reliable severity judgment.",
                    "No definite risk level for {scenario[subject]} follows from the observations supplied so far.",
                    "The current account supports caution but not a settled interpretation of {scenario[subject]}.",
                    "Assessing {scenario[subject]} remotely would require facts that are absent from this report.",
                    "The available details leave the practical significance of {scenario[subject]} open.",
                    "A qualified assessment must resolve the evidence gaps surrounding {scenario[subject]}.",
                    "The report cannot establish either reassurance or severity for {scenario[subject]}.",
                    "Any firm classification of {scenario[subject]} would exceed what has been observed.",
                ),
            },
            "channel": {
                "qualified": (
                    "Describe {scenario[subject]} and any verified changes when contacting qualified help.",
                    "Give the qualified responder the confirmed facts about {scenario[subject]} and report any change.",
                    "When seeking qualified help, pass along the observed details and progression of {scenario[subject]}.",
                    "Share only verified observations about {scenario[subject]} with the appropriate responder.",
                    "Report the known timing and changes related to {scenario[subject]} through a qualified channel.",
                    "Tell the appropriate responder what is confirmed about {scenario[subject]} and what remains unknown.",
                    "Pass the established facts about {scenario[subject]} to qualified help, distinguishing observations from uncertainty.",
                    "When contacting a responder, state the verified details of {scenario[subject]} and identify every unresolved point.",
                    "Give qualified help the known facts, timing, and changes concerning {scenario[subject]} without filling evidence gaps.",
                    "Escalate {scenario[subject]} with a concise account of what was observed and what has not been confirmed.",
                    "Provide the responder with the supported details around {scenario[subject]}, including any documented progression.",
                    "Use the qualified channel to relay confirmed observations about {scenario[subject]} while labeling missing facts clearly.",
                    "Report {scenario[subject]} through appropriate help with verified timing, visible changes, and remaining uncertainty.",
                    "When help is reached, separate the known features of {scenario[subject]} from details that are still unverified.",
                    "Share the evidence available for {scenario[subject]} with a qualified responder and avoid presenting unknowns as facts.",
                    "Describe what has actually been observed about {scenario[subject]} when escalating, together with any change over time.",
                    "Qualified help should receive the confirmed account of {scenario[subject]}, not assumptions about missing case details.",
                    "Communicate the supported facts about {scenario[subject]} and explicitly flag information that remains unavailable.",
                    "Relay the observed state of {scenario[subject]} to the appropriate service, noting which facts still need verification.",
                    "Give the responder a factual timeline for {scenario[subject]} and leave unsupported conclusions out of the report.",
                    "Escalation should carry the verified observations related to {scenario[subject]} plus a clear statement of uncertainty.",
                    "State the known circumstances of {scenario[subject]} through qualified help and distinguish them from unanswered questions.",
                    "Contact the suitable responder with a bounded report of {scenario[subject]} based only on confirmed observations.",
                    "Send qualified help a concise account of the observed {scenario[subject]} without inferring its cause.",
                    "Use the appropriate service to report the timeline and confirmed features of {scenario[subject]}.",
                    "Give a responder the evidence about {scenario[subject]} while identifying every missing case detail.",
                    "Escalate through qualified help with observations about {scenario[subject]} separated from interpretation.",
                    "Pass along when and how {scenario[subject]} changed, limiting the report to supported facts.",
                    "Provide the responsible service with the known scope of {scenario[subject]} and its unresolved aspects.",
                    "When contacting help, distinguish the recorded signs of {scenario[subject]} from any suspected explanation.",
                    "Relay a factual account of {scenario[subject]} and let the qualified responder assess its significance.",
                ),
            },
            "action": {
                "uncertainty": (
                    "Unverified facts about {scenario[subject]} cannot support a certain remote conclusion.",
                    "Treat the unresolved details of {scenario[subject]} as uncertainty, not confirmation.",
                    "Do not turn the incomplete report of {scenario[subject]} into a definite finding.",
                    "Keep conclusions about {scenario[subject]} bounded by what has actually been verified.",
                    "The unknown details surrounding {scenario[subject]} require a cautious, reversible response.",
                    "Use a protective next step for {scenario[subject]} without claiming that its severity is known.",
                    "Respond cautiously to {scenario[subject]} while leaving every unverified detail unresolved.",
                    "Choose a reversible safeguard for {scenario[subject]} instead of assigning a definite interpretation.",
                    "Keep the response to {scenario[subject]} protective and proportionate to the limited evidence.",
                    "Act on possible harm from {scenario[subject]} without presenting uncertainty as a confirmed finding.",
                    "Use the least harmful next step for {scenario[subject]} while verification remains incomplete.",
                    "Preserve a safe state around {scenario[subject]} and avoid conclusions unsupported by the report.",
                    "Treat missing facts about {scenario[subject]} as a reason for caution rather than certainty.",
                    "Limit the response to a reversible precaution for {scenario[subject]} pending qualified assessment.",
                    "Protect against credible risk from {scenario[subject]} without deciding what the incomplete facts mean.",
                    "Take a bounded safety action for {scenario[subject]} and keep the evidentiary gaps explicit.",
                    "Do not infer severity from the partial account of {scenario[subject]}; maintain a protective default.",
                    "Let uncertainty about {scenario[subject]} guide a cautious action rather than a certain claim.",
                    "Keep any response to {scenario[subject]} reversible until the absent case information is obtained.",
                    "Address the plausible risk in {scenario[subject]} while withholding unsupported classification.",
                    "Use a low-regret precaution for {scenario[subject]} and preserve the distinction between known and unknown.",
                    "Make immediate protection the limit of the response to {scenario[subject]} until facts are verified.",
                    "Avoid turning uncertainty around {scenario[subject]} into either reassurance or diagnosis.",
                    "Respond within the confirmed evidence for {scenario[subject]}, favoring safety and reversibility.",
                    "Move toward a safer state around {scenario[subject]} without deciding facts the report leaves open.",
                    "Keep the next step for {scenario[subject]} low risk and avoid assigning an unsupported cause.",
                    "Use the evidence only to justify protection, not certainty about {scenario[subject]}.",
                    "Preserve reversibility while qualified help resolves the unknowns in {scenario[subject]}.",
                    "Choose caution proportionate to {scenario[subject]} without overstating its confirmed severity.",
                    "Address the possible harm first and leave interpretation of {scenario[subject]} unsettled.",
                    "Maintain a protective boundary around {scenario[subject]} until direct verification is available.",
                    "Do not let an incomplete account of {scenario[subject]} support an irreversible response.",
                ),
            },
        }
    )
    dealt = matrix.deal(seed)
    return (
        dealt["case"]["report"],
        dealt["boundary"]["remote_limit"],
        dealt["channel"]["qualified"],
        dealt["action"]["uncertainty"],
    )
