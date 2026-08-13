from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import V2RoleSeparatedDeck, V2SubcardPool
from ._common import render_v2_row, validate_complete_rows


TASK = "conversation_empathy"
_SITUATIONS = (
    ("work", "overwhelmed", "I have three urgent requests and cannot see how to finish all of them today", "That sounds overwhelming, especially when every request is presented as urgent", "write the three deadlines and ask the requesters to confirm priority", "Would it help to draft that priority message together?"),
    ("work", "discouraged", "My presentation went badly even though I prepared for days", "It makes sense to feel discouraged after investing that much preparation", "separate the useful feedback from the embarrassment of the moment", "Do you want to review one part that felt hardest?"),
    ("study", "stuck", "I keep rereading the chapter and none of it is staying with me", "Being stuck after repeated effort is frustrating, not a sign that you cannot learn it", "close the chapter and explain one small section from memory", "Which section would be most useful to unpack first?"),
    ("study", "anxious", "I am panicking about tomorrow's exam and cannot focus", "Exam anxiety can make even familiar material feel inaccessible", "pause for a short reset and choose one bounded topic to review", "Would you like help choosing that one topic?"),
    ("relationship", "hurt", "A friend cancelled again and I feel like I do not matter to them", "Repeated cancellations can feel hurtful and make the friendship seem one-sided", "describe the impact without guessing the friend's motive", "Would you rather compose a message or first sort out what you need from them?"),
    ("relationship", "conflicted", "My family expects me to attend, but I need a quiet weekend", "It is understandable to care about your family and still need rest", "state the limit clearly while offering a smaller form of connection", "What amount of contact would feel manageable rather than draining?"),
    ("change", "uncertain", "I was offered a new role and I cannot tell whether accepting is a mistake", "A meaningful opportunity can bring uncertainty as well as excitement", "compare the role against your non-negotiable needs rather than trying to predict everything", "Which condition would make the decision clearly unacceptable?"),
    ("change", "sad", "I am moving away and the excitement is mixed with a lot of sadness", "You can look forward to the move and grieve what you are leaving at the same time", "name one connection you want to preserve and make a concrete plan for it", "Who or what feels hardest to leave?"),
    ("creative", "blocked", "Everything I write sounds terrible, so I keep deleting it", "Creative self-criticism can stop the rough work that a good draft needs", "write a deliberately imperfect paragraph without editing it", "Would a five-minute no-delete exercise feel possible?"),
    ("creative", "disappointed", "The project I cared about was rejected without much explanation", "A rejection with little explanation can feel both disappointing and unresolved", "ask for one specific piece of feedback while deciding what still matters to you in the project", "Do you want to preserve the project, revise it, or simply process the rejection today?"),
    ("health_routine", "frustrated", "I missed my routine again and feel like I have ruined all my progress", "Missing part of a routine does not erase the work you already did", "restart with the smallest version of the routine at the next available opportunity", "What is the smallest version you could realistically do?"),
    ("caregiving", "exhausted", "I am helping everyone else and have no energy left", "Constantly caring for others can leave very little room for your own needs", "identify one responsibility that can be delayed, shared, or declined", "Is there one person you could ask for a specific kind of help?"),
    ("conflict", "embarrassed", "I snapped at a colleague and now I am avoiding them", "Feeling embarrassed after losing patience is understandable, and avoidance can make repair harder", "offer a direct apology that names the behavior without defending it", "Would you like a short apology you can adapt?"),
    ("conflict", "angry", "The same process failed again and nobody seems to take it seriously", "Repeated preventable failures can make anger feel justified and urgent", "document the repeated impact and request a named owner for the correction", "What evidence best shows that this is a recurring problem?"),
    ("belonging", "lonely", "I joined the group, but I still feel like an outsider", "Being physically included does not always create a sense of belonging right away", "seek one smaller conversation instead of trying to connect with the whole group at once", "Is there one person who seemed approachable?"),
    ("confidence", "afraid", "I want to apply, but I am afraid everyone else is more qualified", "Fear of comparison can make your own relevant experience disappear from view", "list the requirements you already meet and one gap you can address honestly", "Would you like to compare your experience with the actual requirements?"),
)
_SUPPORT_MODES = (
    ("listening", "I can stay with the feeling before trying to solve it", "Would you prefer that I listen for a moment", "avoid pushing toward a decision and create room for the person to elaborate"),
    ("planning", "We can turn the concern into one manageable action", "Would a small plan be useful now", "offer a bounded action without implying that the feeling needs a quick fix"),
    ("wording", "I can help put the difficult part into words", "Would you like a sentence you can adapt", "help articulate the concern while preserving the speaker's own voice"),
    ("reflection", "We can separate what happened from what it seems to mean", "Would reflecting on that distinction help", "distinguish the observed event from the harshest interpretation of it"),
    ("choice", "You do not have to decide every part at once", "Would comparing two immediate choices reduce the pressure", "reduce the decision to two reversible choices rather than an entire future"),
    ("preparation", "We can rehearse the part that feels most uncertain", "Would practicing one response make this easier", "focus on rehearsal for one foreseeable moment instead of general reassurance"),
    ("boundary", "Your needs can be expressed without dismissing anyone else's", "Would a respectful boundary be helpful", "support a clear limit that acknowledges both the user and the other person"),
    ("perspective", "One difficult moment does not define the whole situation", "Would it help to identify what remains unchanged", "broaden the frame without minimizing the immediate disappointment or fear"),
)
_PROMPTS = (
    "Respond with empathy and useful support. The person says: “{scenario[situation]}” For {scenario[mode]} support, {scenario[mode_instruction]}.",
    "Give a natural, non-judgmental reply to this message: “{scenario[situation]}” Emphasize {scenario[mode]} by following this direction: {scenario[mode_instruction]}.",
    "Acknowledge the feeling, offer one grounded action, and leave the choice with the user: “{scenario[situation]}” In this {scenario[mode]} response, {scenario[mode_instruction]}.",
    "Continue this conversation warmly without abstract advice: “{scenario[situation]}” The user is asking for {scenario[mode]} support, so {scenario[mode_instruction]}.",
)
_ANSWERS = (
    "{scenario[acknowledgement]}. {scenario[mode_bridge]}. One practical move is to {scenario[action]}. {scenario[mode_question]}",
    "{scenario[acknowledgement]}. {scenario[mode_bridge]}. You might {scenario[action]}, if that feels useful. {scenario[question]}",
    "{scenario[acknowledgement]}. A gentle option is to {scenario[action]}. {scenario[mode_bridge]}. {scenario[mode_question]}",
    "{scenario[acknowledgement]}. It may help to {scenario[action]}. {scenario[mode_bridge]}. {scenario[question]}",
)


def conversation_empathy_capacity() -> int:
    return len(_SITUATIONS) * len(_SUPPORT_MODES)


def render_conversation_empathy_rows() -> list[dict[str, object]]:
    rows = []
    for domain, emotion, situation, acknowledgement, action, question in _SITUATIONS:
        for mode, mode_bridge, mode_question, mode_instruction in _SUPPORT_MODES:
            contextual_mode_question = (
                f"{mode_question} for this {emotion} situation?"
            )
            contextual_question = (
                question.rstrip("?")
                + f", or would {mode} support feel more useful?"
            )
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {
                            "emotion": (emotion,), "situation": (situation,),
                            "acknowledgement": (acknowledgement,), "action": (action,),
                            "question": (contextual_question,), "mode": (mode,),
                            "mode_bridge": (mode_bridge,),
                            "mode_question": (contextual_mode_question,),
                            "mode_instruction": (mode_instruction,),
                        },
                        "prompt": {"support_request": _PROMPTS},
                        "answer": {"empathetic_response": _ANSWERS},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{emotion}:{mode}", variables=variables,
                prompt_pools=(V2SubcardPool("support_request", SurfaceRole.PROMPT, ("{prompt[support_request]}",)),),
                answer_pools=(V2SubcardPool("empathetic_response", SurfaceRole.ANSWER, ("{answer[empathetic_response]}",)),),
            )
            rows.append(
                render_v2_row(
                    task=TASK, case_id=f"{domain}:{emotion}:{mode}", domain=domain,
                    difficulty="easy", deck=deck,
                    facts={"emotion": emotion, "situation": situation, "acknowledgement": acknowledgement, "action": action, "mode": mode},
                    validator={"kind": "contains", "required": [acknowledgement, action]},
                )
            )
    return validate_complete_rows(TASK, rows, conversation_empathy_capacity())


__all__ = ("conversation_empathy_capacity", "render_conversation_empathy_rows")
