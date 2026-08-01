from __future__ import annotations

import re


DATASET_ID = "complexity-original-post-training-v1"


DATASET_LICENSE = "CC BY-NC 4.0"


DATASET_SOURCE = "Complexity original Scenario Forge conversations"


REVIEW_GRADES = (
    "semantic_accuracy",
    "constraint_following",
    "language_quality",
    "individualization",
    "safety",
)


_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


_REVIEW_STATUSES = frozenset({"pending", "approved", "rejected"})


_REVIEW_GRADE_VALUES = frozenset({"", "pass", "fail"})


_MAX_SURFACE_FORMULATION_SHARE = 0.05


_MAX_FAMILY_SKELETON_SHARE = 0.20


_FORBIDDEN_ASSISTANT_META_PHRASES = (
    "the response should",
    "the response must",
    "the response can",
    "the answer should",
    "the answer must",
    "the answer can",
    "the explanation should",
    "the explanation must",
    "the final review should",
    "a valid answer",
    "a worked response",
)


_FORBIDDEN_USER_META_PHRASES = (
    "the assistant must",
    "the assistant should",
    "the response should",
    "the response must",
    "the answer should",
    "how should an assistant",
    "what response would",
    "which response would",
    "what should the response cover",
    "how can the response",
    "the response must reflect",
)


_INTENT_FIELD = {
    "practical_action": "requested_action",
    "explanation_learning": "learning_goal",
    "troubleshooting": "diagnostic_goal",
    "writing_transformation": "transformation",
    "planning_comparison": "planning_goal",
    "conversation_empathy": "conversational_goal",
    "safety_uncertainty": "safe_goal",
    "grounded_qa": "question_goal",
    "summarization_synthesis": "summary_goal",
    "extraction_classification": "extraction_goal",
    "reasoning_verification": "reasoning_goal",
    "critique_revision": "critique_goal",
    "brainstorming_creativity": "ideation_goal",
    "context_clarification": "clarification_goal",
}


_ACKNOWLEDGEMENTS = (
    "Thanks, that gives me a concrete starting point. I will keep the next step tied to what we can verify.",
    "Understood. I will separate what is confirmed from what still needs checking.",
    "That helps. We can handle the immediate task first and keep the final decision reversible.",
    "I follow. There is enough context for a bounded next step, but not for unsupported certainty.",
    "Got it. I will focus on the result you need and keep a fallback available if evidence is missing.",
    "That is a useful starting point. We can make one practical choice instead of a broad assumption.",
    "Understood. I will keep the guidance specific enough to verify.",
    "That makes sense. We can move from the known facts to one cautious action.",
    "Thanks for the update. I will keep the remaining uncertainty visible.",
    "I understand. We can preserve your control while narrowing the next decision.",
    "Got it. I will use the confirmed facts and leave unsupported details open.",
    "That gives us a clear scope: one objective, one limit, and one check at the end.",
)


_INSTRUCT_OPENINGS = (
    "Please solve this hand.",
    "Work through the following card hand.",
    "Use these cards to produce the requested result.",
    "Resolve this case from the supplied cards.",
    "Complete the task described by this hand.",
    "Apply the rule and goal to the data below.",
    "Handle this card hand using only its stated facts.",
    "Produce a bounded answer for the following case.",
)


_CHAT_OPENINGS = (
    "I want to work through this hand.",
    "Can we resolve this case together?",
    "I need help with the following card hand.",
    "Let's work from these situation and data cards.",
    "Could you help me reason through this case?",
    "I have a bounded task to work through.",
    "Let's start with the situation and known facts.",
    "I would like to handle this case carefully.",
)


_INTENT_SUBJECT_TEMPLATES = {
    "apply the principle": "apply the principle to {subject}",
    "contrast nearby concepts": "contrast the concepts related to {subject}",
    "diagnose a misconception": "diagnose a misconception about {subject}",
    "explain the core mechanism": "explain the core mechanism behind {subject}",
    "walk through a worked example": "walk through a worked example of {subject}",
    "isolate the failing boundary": "isolate the failing boundary in {subject}",
    "prevent recurrence": "prevent the problem from recurring in {subject}",
    "produce a minimal reproduction": "produce a minimal reproduction of {subject}",
    "recover safely": "recover safely from {subject}",
    "verify a proposed fix": "verify a proposed fix for {subject}",
    "adapt tone for the audience": "adapt the tone of {subject} for the audience",
    "draft from structured facts": "draft {subject} from structured facts",
    "restructure for action": "restructure {subject} for action",
    "revise for clarity": "revise {subject} for clarity",
    "summarize faithfully": "summarize {subject} faithfully",
    "allocate limited resources": "allocate limited resources for {subject}",
    "compare against hard criteria": "compare {subject} against hard criteria",
    "define viable options": "define viable options for {subject}",
    "design a fallback": "design a fallback for {subject}",
    "sequence the work": "sequence the work for {subject}",
    "acknowledge the experience": "acknowledge the experience behind {subject}",
    "choose a gentle next step": "choose a gentle next step for {subject}",
    "clarify the immediate need": "clarify the immediate need in {subject}",
    "prepare a grounded conversation": "prepare a grounded conversation about {subject}",
    "reflect on meaning and progress": "reflect on the meaning and progress of {subject}",
    "clarify the safe scope": "clarify the safe scope of {subject}",
    "identify an escalation threshold": "identify an escalation threshold for {subject}",
    "offer a safe alternative": "offer a safe alternative for {subject}",
    "preserve privacy and control": "preserve privacy and control around {subject}",
    "set a clear safety boundary": "set a clear safety boundary for {subject}",
    "answer the direct question": "answer the direct question about {subject}",
    "locate supporting evidence": "locate supporting evidence for {subject}",
    "compare two claims": "compare two claims about {subject}",
    "draw a cautious inference": "draw a cautious inference about {subject}",
    "identify what remains unknown": "identify what remains unknown about {subject}",
    "summarize the essentials": "summarize the essentials of {subject}",
    "synthesize related points": "synthesize the related points in {subject}",
    "extract decisions and actions": "extract decisions and actions from {subject}",
    "organize the chronology": "organize the chronology of {subject}",
    "adapt the summary for its audience": "adapt the summary of {subject} for its audience",
    "extract the requested fields": "extract the requested fields from {subject}",
    "normalize the recorded values": "normalize the recorded values in {subject}",
    "classify the record": "classify the record for {subject}",
    "identify missing required fields": "identify missing required fields in {subject}",
    "convert the record into a clear structure": "convert {subject} into a clear structure",
    "calculate the requested result": "calculate the requested result for {subject}",
    "compare the available quantities": "compare the available quantities for {subject}",
    "test whether the constraint is satisfied": "test whether the constraint for {subject} is satisfied",
    "explain the reasoning step by step": "explain the reasoning for {subject} step by step",
    "verify the proposed result": "verify the proposed result for {subject}",
    "identify the most important weakness": "identify the most important weakness in {subject}",
    "revise the weak section": "revise the weak section of {subject}",
    "check internal consistency": "check the internal consistency of {subject}",
    "strengthen the evidence connection": "strengthen the evidence connection in {subject}",
    "prioritize the necessary fixes": "prioritize the necessary fixes for {subject}",
    "generate several distinct options": "generate several distinct options for {subject}",
    "diversify the current ideas": "diversify the current ideas for {subject}",
    "filter ideas against the criteria": "filter ideas for {subject} against the criteria",
    "combine compatible concepts": "combine compatible concepts for {subject}",
    "develop one promising idea": "develop one promising idea for {subject}",
    "ask one decisive clarifying question": "ask one decisive clarifying question about {subject}",
    "restate the understood request": "restate the understood request for {subject}",
    "resolve the ambiguous reference": "resolve the ambiguous reference in {subject}",
    "separate facts from assumptions": "separate facts from assumptions about {subject}",
    "propose a bounded interpretation": "propose a bounded interpretation of {subject}",
}


_MASKED_RESPONSE_FIELDS = (
    ("subject", "subject"),
    ("surface_intent", "intent"),
    ("state", "state"),
    ("constraint", "constraint"),
    ("desired_outcome", "desired_outcome"),
    ("fallback", "fallback"),
    ("fallback_surface", "fallback_surface"),
    ("domain_context", "domain_context"),
)
