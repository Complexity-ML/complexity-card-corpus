from __future__ import annotations

from collections.abc import Mapping


_QUOTES = {
    "work_stress": "I keep thinking about the unfinished work even when I try to rest.",
    "relationship_tension": "I keep replaying our conversation and wondering what I should have said.",
    "uncertain_change": "The change may be good, yet I am scared of losing what feels familiar.",
    "social_mistake": "I made a mistake in front of everyone and cannot stop thinking about it.",
    "achievement": "I am proud of finishing, but I also feel strangely empty now.",
    "loss_disappointment": "I prepared for this outcome for months, and now I feel both sad and stuck.",
    "caregiving_stress": "I care about helping them, but I am tired and feel guilty whenever I need a break.",
    "grief_update": "Some days I can discuss the loss; on others, an ordinary reminder makes it immediate again.",
    "creative_rejection": "I know one rejection does not define my work, but it makes the whole project feel pointless.",
    "work_conflict": "I want to address this with my colleague, but I worry the conversation will become defensive again.",
}

_ACKNOWLEDGMENTS = {
    "work_stress": (
        "It makes sense that unfinished work keeps pulling at your attention while you try to rest.",
        "That sounds exhausting: your body is off duty, but your mind is still tracking the unfinished work.",
        "You are describing the strain of carrying work beyond the hours you meant to give it.",
    ),
    "relationship_tension": (
        "Replaying a tense conversation can leave you searching for a perfect response that was unavailable then.",
        "It sounds as though the conversation ended, but the uncertainty around it did not.",
        "Wondering what you should have said can be painful when the relationship matters to you.",
    ),
    "uncertain_change": (
        "Hope and fear can sit together when change offers something new and asks you to release the familiar.",
        "It makes sense to see possible good in the change while grieving the certainty you have now.",
        "You do not have to treat excitement and fear as evidence that one of them is false.",
    ),
    "social_mistake": (
        "A public mistake can feel much larger from inside the moment than it looks to everyone else.",
        "The embarrassment sounds vivid, and replaying it may be keeping the moment active.",
        "It is understandable that being seen making a mistake would stay with you for a while.",
    ),
    "achievement": (
        "Finishing something important can bring pride and a surprising sense of emptiness at the same time.",
        "You reached the finish line, and the structure around the effort seems to have disappeared suddenly.",
        "Pride does not cancel the flat feeling that can follow a long-awaited achievement.",
    ),
    "loss_disappointment": (
        "After months of preparation, this outcome can carry both grief and uncertainty about what comes next.",
        "It makes sense to feel sad and stuck when so much effort was tied to a different outcome.",
        "The disappointment sounds heavy precisely because the preparation mattered to you.",
    ),
    "caregiving_stress": (
        "Caring deeply for someone and needing rest can both be true; exhaustion does not erase your care.",
        "The caregiving matters to you while the constant demand is wearing down your energy.",
        "The guilt around taking a break may add another burden to an already tiring role.",
    ),
    "grief_update": (
        "Grief can shift without a steady schedule, and an ordinary reminder can make the loss feel newly present.",
        "It makes sense that talking feels possible one day while a reminder overwhelms you on another.",
        "The return of an intense feeling does not make the quieter days false.",
    ),
    "creative_rejection": (
        "A rejection can make the effort behind a project feel invisible without defining the work itself.",
        "It sounds painful to have one response cast doubt over a project carrying so much attention.",
        "Knowing intellectually that rejection is limited does not make its immediate disappointment less real.",
    ),
    "work_conflict": (
        "Wanting to repair the working relationship while fearing another defensive exchange is a difficult tension.",
        "You want a constructive conversation, not another round of the same conflict.",
        "Your hesitation makes sense when the previous exchange left safe discussion uncertain.",
    ),
}

_REFLECTIONS = {
    "The emotion is immediate and physically activating.": (
        "You can slow the pace before trying to decide anything.",
        "A brief pause may be more useful than pushing toward an answer.",
        "The immediate task can simply be to make the moment less demanding.",
        "Nothing has to be resolved while the reaction still feels this immediate.",
        "Making room for one quieter moment may be enough before deciding what comes next.",
    ),
    "The person feels ready for a small constructive step.": (
        "If you want to move, the next step can stay small and reversible.",
        "Readiness does not require taking on the whole situation at once.",
        "One modest action can be enough for now.",
        "Feeling ready can mean choosing only the smallest useful move.",
        "A limited next step can honor that readiness without creating pressure.",
    ),
    "The person holds two conflicting feelings at once.": (
        "Both reactions can have room without forcing one to cancel the other.",
        "You do not have to choose which of the two feelings is valid.",
        "Mixed feelings can be acknowledged before any decision is made.",
        "The two responses can coexist without requiring an immediate verdict.",
        "Holding both feelings for now is a valid alternative to resolving the tension.",
    ),
    "The person is repeatedly replaying the event.": (
        "The replay does not have to produce a perfect explanation tonight.",
        "Noticing the same moment return is different from having to solve it each time.",
        "The event can matter without requiring another full review right now.",
        "A recurring thought does not create an obligation to analyze it again immediately.",
        "You can notice the replay and still choose not to follow it through another cycle.",
    ),
    "The speaker expresses several emotions without one clear request.": (
        "You do not need to sort every feeling before naming what would help.",
        "It is fine if the immediate need is clearer than the full explanation.",
        "Several feelings can be present before one request takes shape.",
        "A useful need can be named even while the emotions remain mixed.",
        "The lack of one clear label does not prevent you from asking for support.",
    ),
}

_AGENCY = (
    "You can give yourself time before deciding what the experience means.",
    "There is no need to force an immediate solution or a more acceptable feeling.",
    "You can choose whether you want reflection, company, or one small next step.",
    "The pace and direction of the next conversation remain yours.",
    "You remain free to pause or continue with only the part that feels manageable.",
    "Any next step can stay proportionate to the energy and clarity you have now.",
    "You can decide later whether this needs action, reflection, or simply more time.",
    "It is reasonable to keep the next choice open until you know what would help.",
    "You may set the pace without explaining or defending that pace to anyone here.",
    "Only the amount you want to explore needs to enter this conversation.",
    "You can stop at acknowledgment and leave practical decisions for another moment.",
    "The next move can remain optional rather than becoming another demand.",
    "You can choose one manageable part without committing to address the rest today.",
    "There is room to change direction if a suggested step does not feel useful.",
    "You may ask for listening now and reconsider practical help later.",
    "No immediate conclusion is required for your experience to be taken seriously.",
    "You can keep what feels private outside the conversation and still receive support.",
    "The decision to continue, pause, or shift topics remains under your control.",
    "A response can be useful without pushing you toward a particular interpretation.",
    "You are allowed to leave the question open while the situation settles.",
    "Support can follow your stated need instead of choosing a goal on your behalf.",
    "You can accept only the part of a suggestion that fits your present capacity.",
    "Nothing here requires turning a difficult moment into a plan immediately.",
    "You retain the option to revisit this when the timing feels more workable.",
)

_QUESTIONS = (
    "What would feel most useful to name first?",
    "Would you rather stay with the feeling or consider one gentle next step?",
    "What part of this do you most want another person to understand?",
    "What kind of support would feel least demanding right now?",
    "Would naming the hardest part make this feel more manageable?",
    "Do you want reflection, quiet company, or help choosing one small step?",
    "Would you prefer to explore the reaction or simply have it heard?",
    "What would make the next few minutes feel a little gentler?",
    "Would a small practical step help, or would listening be more useful?",
    "What would respecting your own pace look like in this moment?",
    "Which part deserves acknowledgment even if nothing is decided yet?",
    "Would it help more to describe the moment or to focus on what you need now?",
    "Is there one part you want reflected back without trying to resolve it?",
    "Would you like space to unpack this, or would a brief acknowledgment fit better?",
    "What kind of response would feel supportive without adding pressure?",
    "Is the most useful thing right now clarity, company, or a pause?",
    "Would you like to stay with what happened or shift toward the present moment?",
    "What would make this conversation feel more under your control?",
    "Is there a small part of the experience that feels safe enough to name?",
    "Would a question help, or would you prefer a simple reflection?",
    "What support would match the amount of energy you have available?",
    "Would you rather identify a need or let the mixed feelings remain as they are?",
)


def empathy_reservoir(
    domain: str, state: str
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    domain_label = domain.replace("_", " ")
    return {
        "domain": {"label": (domain_label,)},
        "label": {"source": ("Person says", "Conversation excerpt", "Message")},
        "quote": {"person": (_QUOTES[domain],)},
        "goal": {
            "acknowledge": (
                "Respond to the {domain[label]} experience with acknowledgment.",
                "Offer a grounded response to the {domain[label]} experience.",
                "Acknowledge the feeling in this {domain[label]} message.",
            )
        },
        "constraint": {
            "agency": (
                "Preserve the speaker's agency without diagnosis or pressure.",
                "Leave the speaker's next choice open and avoid diagnosis.",
                "Validate without imposing a conclusion or a solution.",
            ),
            "question": (
                "Ask at most one gentle question.",
                "Open no more than one optional question.",
                "Use a single gentle question at most.",
            ),
        },
        "acknowledgment": {"domain": _ACKNOWLEDGMENTS[domain]},
        "reflection": {
            "state": _REFLECTIONS.get(
                state,
                ("You can take this one part at a time.",),
            )
        },
        "agency": {"choice": _AGENCY},
        "question": {"optional": _QUESTIONS},
    }
