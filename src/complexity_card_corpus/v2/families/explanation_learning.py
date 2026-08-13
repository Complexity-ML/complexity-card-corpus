from __future__ import annotations

from ...variable_by import VariableBy2D
from ..contracts import RoleSeparatedVariableBy, SurfaceRole
from ..decks import (
    V2RoleSeparatedDeck,
    V2SubcardPool,
    answer_variant_plans,
    prompt_variant_plans,
)
from ._common import render_v2_row, validate_complete_rows


TASK = "explanation_learning"
_CONCEPTS = (
    ("mathematics", "a percentage", "it describes a part per hundred", "25% means 25 out of every 100", "convert it to a fraction over 100 or a decimal"),
    ("mathematics", "an average", "it redistributes a total evenly across the number of observations", "a total of 20 across four values has an average of 5", "multiply the average by the count to recover the total"),
    ("physics", "mass versus weight", "mass measures matter while weight is the force of gravity on that mass", "an object keeps its mass on the Moon but weighs less there", "compare the same object's mass and weight under different gravity"),
    ("physics", "electrical resistance", "it limits current for a given voltage", "a larger resistance produces less current when voltage stays fixed", "use the relation current equals voltage divided by resistance"),
    ("biology", "natural selection", "heritable differences that affect reproduction change how common traits become", "better-camouflaged insects may leave more offspring", "track trait frequencies across generations instead of treating adaptation as an individual's choice"),
    ("biology", "a vaccine", "it trains immune recognition before exposure to the disease", "immune memory can respond faster when the pathogen appears later", "explain how immune preparation changes the response to later exposure"),
    ("computing", "a cache", "it keeps recently useful data closer to where it is needed", "a browser can reuse an image instead of downloading it again", "remove stale cached data and check whether a newer source version appears"),
    ("computing", "encryption", "it transforms readable data into a form that requires a key to interpret", "an encrypted file remains unintelligible without the decryption key", "verify that protection depends on a secret key rather than an encoding format alone"),
    ("networks", "latency", "it is the delay before data completes a trip rather than the amount sent per second", "a connection can have high bandwidth yet still react slowly", "measure round-trip time separately from throughput"),
    ("networks", "DNS", "it maps human-readable names to network addresses", "a resolver can find the address associated with a website name", "test name resolution separately from destination-service availability"),
    ("economics", "opportunity cost", "choosing one option gives up the best available alternative", "spending an hour on one task means not using that hour on the next-best task", "compare the chosen benefit with what the strongest rejected option offered"),
    ("economics", "compound interest", "each period can earn interest on earlier interest as well as principal", "reinvested interest makes later gains larger than early gains at the same rate", "calculate each period from the updated balance"),
    ("research", "a control group", "it provides a comparison for changes that might have happened without the tested intervention", "two similar groups can differ mainly in whether they receive the treatment", "compare the treated change with a credible untreated baseline"),
    ("research", "correlation versus causation", "two variables moving together does not by itself show that one produces the other", "ice-cream sales and sunburns can rise together because hot weather affects both", "look for confounders and evidence from a causal design"),
    ("language", "active voice", "the grammatical subject performs the action", "‘The team repaired the sensor’ names the actor directly", "identify who does the verb rather than only who receives its effect"),
    ("civics", "separation of powers", "government authority is divided so institutions can limit one another", "a legislature makes laws while courts can review their application", "verify that no single branch controls every core function unchecked"),
    ("earth_science", "the water cycle", "water moves between Earth's surface and atmosphere through evaporation, condensation, and precipitation", "sunlight can evaporate lake water that later falls as rain", "trace a drop of water through evaporation, condensation, and precipitation"),
)
_AUDIENCES = (
    ("a curious twelve-year-old", "use familiar words and one concrete comparison", "Imagine explaining the change with objects on a kitchen table"),
    ("a new employee", "connect the idea to a workplace decision", "Link the idea to a choice someone could face during a normal shift"),
    ("an adult returning to study", "define the term before using its formal relation", "Treat the definition as a foundation before adding the technical detail"),
    ("a technically confident reader", "state the mechanism precisely without unnecessary background", "Focus on the variables and the relation that produces the observed result"),
    ("a reader learning English", "use short sentences and explain specialist vocabulary", "Keep each step separate so the new term has one clear meaning"),
    ("someone revising for a quiz", "highlight the distinction most likely to be tested", "Remember the contrast by asking which condition would make the statement false"),
    ("a community workshop participant", "tie the idea to an everyday situation", "Connect the mechanism to a shared activity that can be observed directly"),
    ("a skeptical reader", "include a check that could reveal a mistaken interpretation", "Look for an observation that would differ if the explanation were wrong"),
)
_PROMPTS = (
    "Explain {scenario[concept]} for {scenario[audience]}; {scenario[instruction]}.",
    "Teach this concept clearly to {scenario[audience]}: {scenario[concept]}. Please {scenario[instruction]}.",
    "Give a compact learning explanation of {scenario[concept]}. The reader is {scenario[audience]}, so {scenario[instruction]}.",
    "Help {scenario[audience]} understand {scenario[concept]}; {scenario[instruction]}.",
    "Describe how {scenario[concept]} works for {scenario[audience]}. In the explanation, {scenario[instruction]}.",
    "Build an intuitive but accurate explanation of {scenario[concept]} for {scenario[audience]}; {scenario[instruction]}.",
)
_ANSWERS = (
    "{scenario[concept_cap]} works this way: {scenario[mechanism]}. {scenario[bridge]}. For example, {scenario[example]}. A useful check is to {scenario[check]}.",
    "The central idea is that {scenario[mechanism]}. {scenario[bridge]}. Consider this example: {scenario[example]}. To test your understanding, {scenario[check]}.",
    "For {scenario[audience]}, the simplest accurate explanation is that {scenario[mechanism]}. {scenario[bridge]}. One concrete case is this: {scenario[example]}. You can verify the idea when you {scenario[check]}.",
    "Start with the mechanism: {scenario[mechanism]}. {scenario[bridge]}. The example ‘{scenario[example]}’ shows it in action. The distinction remains clear if you {scenario[check]}.",
    "In plain terms, {scenario[mechanism]}. {scenario[bridge]}. Here is the idea in use: {scenario[example]}. As a self-check, {scenario[check]}.",
    "A reliable way to understand {scenario[concept]} is to notice that {scenario[mechanism]}. {scenario[bridge]}. For instance, {scenario[example]}. A mistaken reading can be caught if you {scenario[check]}.",
)
_PROMPT_FUNCTIONS = (
    ("request_explanation", "specify_audience", "adapt_instruction"),
    ("request_teaching", "specify_audience", "adapt_instruction"),
    ("request_compact_explanation", "specify_audience", "adapt_instruction"),
    ("request_understanding", "specify_audience", "adapt_instruction"),
    ("request_mechanism", "specify_audience", "adapt_instruction"),
    ("request_intuitive_accuracy", "specify_audience", "adapt_instruction"),
)
_ANSWER_FUNCTIONS = (
    ("define_mechanism", "adapt_bridge", "give_example", "give_check"),
    ("state_core_idea", "adapt_bridge", "give_example", "test_understanding"),
    ("adapt_audience", "define_mechanism", "give_example", "verify"),
    ("lead_with_mechanism", "adapt_bridge", "show_example", "preserve_distinction"),
    ("plain_language_mechanism", "adapt_bridge", "apply_example", "self_check"),
    ("frame_reliable_method", "define_mechanism", "give_example", "catch_misreading"),
)

_MISCONCEPTIONS = {
    "a percentage": "a percentage is not a raw count unless the reference total is known",
    "an average": "an average does not reveal how widely the individual values differ",
    "mass versus weight": "mass and weight are not interchangeable names for the same property",
    "electrical resistance": "resistance does not create current when no voltage is applied",
    "natural selection": "an individual does not deliberately acquire the inherited trait it needs",
    "a vaccine": "a vaccine does not guarantee that exposure can never cause infection",
    "a cache": "cached content is not automatically the newest authoritative content",
    "encryption": "changing a file's visible encoding is not the same as protecting it with a secret key",
    "latency": "high bandwidth does not guarantee a short response delay",
    "DNS": "successful name resolution does not prove that the destination service is healthy",
    "opportunity cost": "the cost is not every rejected option added together",
    "compound interest": "the same interest amount is not added each period when the balance changes",
    "a control group": "a control group is not useful if it differs systematically before treatment",
    "correlation versus causation": "a strong association alone does not identify the causal direction",
    "active voice": "active voice does not require every sentence to begin with a person's name",
    "separation of powers": "divided authority does not mean the institutions never interact",
    "the water cycle": "water does not disappear permanently when it evaporates",
}
_BOUNDARIES = {
    "a percentage": "the denominator or reference whole must remain explicit",
    "an average": "the total and the number of observations must refer to the same set",
    "mass versus weight": "the local gravitational field matters only to weight",
    "electrical resistance": "the comparison assumes voltage is held constant",
    "natural selection": "the trait must be heritable and connected to reproductive difference",
    "a vaccine": "the explanation concerns immune preparation rather than treatment of every later symptom",
    "a cache": "the speed benefit must be balanced against rules for refreshing stale entries",
    "encryption": "the key and threat model determine what protection is actually provided",
    "latency": "delay and throughput must be measured separately",
    "DNS": "name lookup and service availability are separate stages",
    "opportunity cost": "the relevant alternative is the best forgone one",
    "compound interest": "interest must remain in the balance for compounding to occur",
    "a control group": "the groups need a credible basis for comparison",
    "correlation versus causation": "confounding variables and reverse direction remain possible",
    "active voice": "the grammatical actor must perform the stated verb",
    "separation of powers": "the exact institutional powers depend on the constitutional system",
    "the water cycle": "the cycle moves water between reservoirs but does not create new water",
}
_LEARNING_MOVES = (
    ("define", "build a precise definition before adding examples", "state the defining relation and separate it from illustrations", "Can the learner identify the definition without relying on the example?"),
    ("contrast", "contrast the concept with its nearest common confusion", "name the boundary that distinguishes the two ideas", "Can the learner explain why the common confusion fails?"),
    ("predict", "use the mechanism to predict what changes next", "connect a changed condition to its expected consequence", "Can the learner predict an outcome before seeing it?"),
    ("diagnose", "diagnose a deliberately mistaken interpretation", "locate the exact assumption that makes the interpretation fail", "Can the learner repair the mistaken interpretation in one sentence?"),
    ("apply", "apply the idea to a fresh but concrete situation", "transfer the mechanism rather than copying the original example", "Can the learner use the idea in a new situation?"),
    ("explain", "construct a short causal explanation", "link each cause to the next observable effect", "Can the learner account for the result without skipping the mechanism?"),
    ("compare", "compare two cases that differ in one important condition", "hold unrelated details stable while changing the decisive condition", "Can the learner name which changed condition matters?"),
    ("verify", "verify the claim with an independent check", "use the check to test the result rather than merely repeat it", "Can the learner produce evidence that could disconfirm the claim?"),
    ("teach_back", "prepare a teach-back in the learner's own words", "retain the technical meaning while changing the wording", "Can the learner explain it accurately without reciting the source?"),
    ("summarize", "create a compact memory structure", "retain the mechanism, boundary, and one diagnostic cue", "Can the learner recover all three parts from the summary?"),
    ("counterexample", "use a counterexample to expose where an overbroad claim fails", "construct a nearby case that violates the mistaken generalization", "Can the learner use the counterexample to state the concept's boundary?"),
)
_STUDY_CONTEXTS = (
    ("quiz preparation", "the learner must distinguish tempting answer choices", "a definition, a contrast, and a quick check", "test the idea against one plausible distractor"),
    ("workplace onboarding", "the learner needs to recognize the idea during an ordinary shift", "a decision cue and one safe example", "ask what action would change if the concept were misunderstood"),
    ("independent study", "the learner needs feedback without an instructor present", "a self-check with a recoverable explanation", "answer the check first and then compare the reasoning"),
    ("peer teaching", "the learner will explain the idea to another person", "a plain-language account followed by the precise term", "invite the listener to produce a different valid example"),
    ("community workshop", "participants bring different levels of prior knowledge", "an observable situation and a shared comparison", "collect two predictions before revealing the explanation"),
    ("technical review", "the learner must preserve assumptions and scope", "a mechanism statement with an explicit boundary", "look for a case outside the stated boundary"),
    ("troubleshooting practice", "the concept will guide diagnosis of a failure", "a symptom, a competing explanation, and a discriminating check", "choose a check whose outcomes separate the explanations"),
    ("decision support", "the learner must connect understanding to a choice", "a consequence map and one reversible test", "state what evidence would justify changing the choice"),
)
_DEPTH_GUIDANCE = {
    "concise": "answer in a concise lesson of 26 to 80 words",
    "detailed": "answer in a detailed lesson of 81 to 200 words",
    "extended": "answer in an extended lesson of 201 to 512 words",
}
_EXPANDED_PROMPTS = (
    "Teach {scenario[concept]} to {scenario[audience]} for {scenario[context]}. {scenario[move_instruction]}; {scenario[depth_guidance]}.",
    "Create a {scenario[depth]} lesson about {scenario[concept]} for {scenario[audience]}. The setting is {scenario[context]}, so {scenario[move_instruction]}.",
    "Help {scenario[audience]} learn {scenario[concept]} during {scenario[context]}. Emphasize this learning move: {scenario[move_instruction]}. {scenario[depth_guidance_cap]}.",
    "Prepare teaching material about {scenario[concept]}. It is for {scenario[audience]} in {scenario[context]}; {scenario[move_instruction]} and {scenario[depth_guidance]}.",
    "Explain {scenario[concept]} accurately for {scenario[context]}. Adapt it to {scenario[audience]}, {scenario[move_instruction]}, and {scenario[depth_guidance]}.",
    "Build a lesson on {scenario[concept]} for {scenario[audience]}. In the {scenario[context]} setting, {scenario[move_instruction]}; {scenario[depth_guidance]}.",
)
_EXPANDED_PROMPT_FUNCTIONS = (
    ("request_teaching", "specify_audience", "specify_context", "specify_move", "set_length"),
    ("request_lesson", "set_length", "specify_context", "specify_move"),
    ("request_learning_support", "specify_audience", "specify_move", "set_length"),
    ("request_material", "specify_audience", "specify_context", "set_length"),
    ("request_accuracy", "specify_context", "adapt_audience", "set_length"),
    ("request_lesson", "specify_audience", "specify_context", "set_length"),
)


def _expanded_answer_functions(depth: str) -> tuple[tuple[str, ...], ...]:
    if depth == "concise":
        return (
            ("define", "contextualize", "apply", "verify"),
            ("state_core", "illustrate", "apply", "verify"),
            ("adapt_audience", "illustrate", "supply_artifact", "verify"),
            ("explain_mechanism", "pair_example_artifact", "apply", "confirm_transfer"),
            ("explain_mechanism", "illustrate", "apply_with_artifact", "test"),
            ("state_fact", "apply_example", "supply_artifact", "review"),
        )
    if depth == "detailed":
        return (
            ("define_mechanism", "state_boundary", "contextualize", "apply", "correct_misconception", "verify"),
            ("state_claim", "state_boundary", "correct_misconception", "contextualize", "apply", "verify"),
            ("observe", "explain_mechanism", "state_boundary", "contextualize", "apply", "verify"),
            ("explain_mechanism", "adapt_audience", "state_boundary", "illustrate", "apply", "diagnose", "verify"),
        )
    if depth == "extended":
        return (
            ("define_mechanism", "illustrate", "state_boundary", "contrast_misconception", "apply", "verify", "transfer"),
            ("connect_definition_use_limit", "illustrate", "mark_scope", "contrast", "apply", "retrieve", "transfer"),
            ("build_reasoning_chain", "state_boundary", "compare_predictions", "apply", "verify", "generate_example"),
            ("state_relation", "observe_then_explain", "state_limit", "compare_consequences", "retrieve", "verify", "transfer"),
        )
    raise ValueError(depth)


def _expanded_answers(depth: str) -> tuple[str, ...]:
    if depth == "concise":
        return (
            "For {scenario[audience]}, {scenario[concept]} means that {scenario[mechanism]}. In {scenario[context]}, use {scenario[artifact]} to {scenario[move_action]}. Example: {scenario[example]}. Check {scenario[concept]} here by asking: {scenario[verification_question]}",
            "The core of {scenario[concept]} is that {scenario[mechanism]}. Show {scenario[audience]} the case that {scenario[example]}. During {scenario[context]}, ask them to {scenario[move_action]}. Verify this {scenario[context]} lesson when you {scenario[context_check]}.",
            "Teach {scenario[audience]} that {scenario[mechanism]}. The example ‘{scenario[example]}’ makes {scenario[concept]} concrete. For {scenario[context]}, provide {scenario[artifact]} and have the learner {scenario[move_action]}. Check the {scenario[context]} result by asking: {scenario[verification_question]}",
            "{scenario[concept_cap]} works because {scenario[mechanism]}. For {scenario[audience]}, pair the example ‘{scenario[example]}’ with {scenario[artifact]}. The {scenario[context]} activity is to {scenario[move_action]}. Confirm that {scenario[concept]} transferred when you {scenario[context_check]}.",
            "Explain {scenario[concept]} to {scenario[audience]} through this mechanism: {scenario[mechanism]}. Use the example that {scenario[example]}. In {scenario[context]}, the learner should {scenario[move_action]} using {scenario[artifact]}. Test this {scenario[context]} understanding with: {scenario[verification_question]}",
            "For {scenario[audience]}, start {scenario[concept]} with the fact that {scenario[mechanism]}. Apply it to ‘{scenario[example]}’. Within {scenario[context]}, use {scenario[artifact]} and ask the learner to {scenario[move_action]}. Review the {scenario[context]} attempt when you {scenario[context_check]}.",
        )
    if depth == "detailed":
        return (
            "{scenario[concept_cap]} can be learned as a mechanism, a boundary, and a check. The mechanism is that {scenario[mechanism]}. For {scenario[audience]}, {scenario[bridge_lower]}. The boundary matters because {scenario[boundary]}. In {scenario[context]}, {scenario[context_need]}; the useful teaching artifact is {scenario[artifact]}. Start from the concrete case that {scenario[example]}. Then ask the learner to {scenario[move_action]}. This prevents the common mistake that {scenario[misconception]}. Finish by asking: {scenario[verification_question]} The independent check is to {scenario[check]}. {scenario[move_success]}",
            "Begin the lesson on {scenario[concept]} with this claim: {scenario[mechanism]}. {scenario[bridge]}. That statement has a scope condition: {scenario[boundary]}. The distinction is important because {scenario[misconception]}. For {scenario[context]}, the learner needs {scenario[artifact]} because {scenario[context_need]}. Use the example ‘{scenario[example]}’ and have the learner {scenario[move_action]}. Do not accept a repeated definition as evidence of understanding. Instead, {scenario[context_check]}. A final verification is to {scenario[check]}. {scenario[move_success]}",
            "For {scenario[audience]}, teach {scenario[concept]} through a sequence of observation, mechanism, and test. The observation is that {scenario[example]}. The mechanism explaining it is that {scenario[mechanism]}. {scenario[bridge]}. Keep the explanation accurate by stating that {scenario[boundary]}. Within {scenario[context]}, {scenario[context_need]}, so provide {scenario[artifact]}. The active learning move is to {scenario[move_action]}. This also exposes the misconception that {scenario[misconception]}. Ask, ‘{scenario[verification_question]}’ Then verify by having the learner {scenario[check]}. {scenario[move_success]}",
            "The detailed account of {scenario[concept]} starts with {scenario[mechanism]}. Adapt that idea for {scenario[audience]} by following this bridge: {scenario[bridge_lower]}. Next, make the limit visible: {scenario[boundary]}. A useful instance is that {scenario[example]}. In {scenario[context]}, {scenario[context_need]}; use {scenario[artifact]} to support the explanation. Have the learner {scenario[move_action]}, then challenge the false idea that {scenario[misconception]}. Use this diagnostic question: {scenario[verification_question]} Close with the independent check to {scenario[check]}. {scenario[move_success]}",
        )
    if depth == "extended":
        return (
            "Start with the central meaning of {scenario[concept]}: {scenario[mechanism]}. This is the mechanism the learner must be able to use, not merely repeat. For {scenario[audience]}, {scenario[bridge_lower]}. The concrete case ‘{scenario[example]}’ shows the idea operating in a situation rather than floating as a definition. Ask the learner to identify exactly which part of the case is explained by the mechanism.\n\nNext, establish the boundary: {scenario[boundary]}. This prevents the tempting but inaccurate claim that {scenario[misconception]}. Have the learner compare the accurate account with that mistaken one and name the condition that separates them. In {scenario[context]}, {scenario[context_need]}. A suitable learning aid is {scenario[artifact]}. Use it to {scenario[move_action]}. The activity should make the learner produce a consequence, distinction, or check rather than copy the source wording.\n\nNow pose the diagnostic question, ‘{scenario[verification_question]}’ Let the learner answer before offering feedback. Then {scenario[context_check]}. The independent content check is to {scenario[check]}. These two checks serve different purposes: one tests whether the learner can use the concept in the present setting, while the other tests whether the underlying relation still holds. {scenario[move_success]} If the learner cannot yet do that, return to the mechanism and alter one condition in the example. The {scenario[context]} lesson on {scenario[concept]} is complete when the learner can {scenario[move_action]} without relying on the original wording.",
            "An extended lesson on {scenario[concept]} should connect definition, use, and limitation. The defining mechanism is that {scenario[mechanism]}. For {scenario[audience]}, {scenario[bridge_lower]}. Begin with the example that {scenario[example]}. Ask what the example would look like if the mechanism did not operate; this makes the account testable instead of decorative.\n\nThe next step is to mark its scope. {scenario[boundary_cap]}. Without that boundary, a learner may conclude that {scenario[misconception]}. Put the two readings side by side and ask which observation would distinguish them. Within {scenario[context]}, {scenario[context_need]}. Provide {scenario[artifact]}, then ask the learner to {scenario[move_action]}. This choice aligns the activity with the actual purpose of the setting rather than adding an unrelated exercise.\n\nUse the question ‘{scenario[verification_question]}’ as a pause for retrieval. Do not reveal the answer immediately. Instead, {scenario[context_check]}. After that, use the subject-specific check: {scenario[check]}. A correct response should preserve the mechanism, respect the boundary, and explain why the familiar misconception fails. {scenario[move_success]} For transfer, change one feature of the original example and ask the learner to predict whether the same explanation still applies. If the prediction changes, the learner should identify the condition responsible. In {scenario[context]}, this final transfer task demonstrates usable understanding of {scenario[concept]} in new language.",
            "Teach {scenario[concept]} as a chain of reasoning. First, state that {scenario[mechanism]}. Then help {scenario[audience]} enter the idea through this bridge: {scenario[bridge_lower]}. Use the case ‘{scenario[example]}’ and ask what cause, relation, or rule links the starting condition to the result. The learner should be able to point to that link explicitly.\n\nSecond, protect the explanation from overreach. The necessary boundary is that {scenario[boundary]}. A frequent error is to think that {scenario[misconception]}. Rather than just labeling that view wrong, change one relevant condition and compare the predictions produced by the accurate mechanism and the mistaken one. For {scenario[context]}, {scenario[context_need]}. Build the activity around {scenario[artifact]} and have the learner {scenario[move_action]}.\n\nThird, gather evidence of understanding. Ask, ‘{scenario[verification_question]}’ Then {scenario[context_check]}. Follow with the independent disciplinary check to {scenario[check]}. If the answers disagree, revisit the specific link in the reasoning chain instead of restarting the whole lesson. {scenario[move_success]} Finally, request a new example for {scenario[context]}. The learner should {scenario[move_action]}, identify the boundary of {scenario[concept]}, and explain what observation could disconfirm the prediction.",
            "A robust account of {scenario[concept]} for {scenario[audience]} begins with the relation that matters: {scenario[mechanism]}. {scenario[bridge]}. The example ‘{scenario[example]}’ gives the learner something observable to reason about. Ask them to describe the result first, then identify the mechanism that accounts for it. This order makes gaps in causal understanding easier to see.\n\nAccuracy also requires a limit. In this case, {scenario[boundary]}. The misleading alternative is that {scenario[misconception]}. Invite the learner to produce one consequence of each interpretation. The two consequences should differ in a way that could be observed or checked. In {scenario[context]}, {scenario[context_need]}; therefore use {scenario[artifact]}. The principal activity is to {scenario[move_action]}.\n\nFor retrieval, ask the learner: ‘{scenario[verification_question]}’ Pause before feedback, and then {scenario[context_check]}. For verification of the concept itself, ask them to {scenario[check]}. A strong answer should connect evidence to the mechanism and preserve the stated boundary. {scenario[move_success]} End by asking {scenario[audience]} to {scenario[move_action]} in a new {scenario[context]} case involving {scenario[concept]} and name evidence that could overturn the result.",
        )
    raise ValueError(depth)


def explanation_learning_capacity() -> int:
    base = len(_CONCEPTS) * len(_AUDIENCES)
    expanded = (
        len(_CONCEPTS)
        * len(_AUDIENCES)
        * len(_LEARNING_MOVES)
        * len(_STUDY_CONTEXTS)
        * len(_DEPTH_GUIDANCE)
    )
    return base + expanded


def render_explanation_learning_rows() -> list[dict[str, object]]:
    rows = []
    for domain, concept, mechanism, example, check in _CONCEPTS:
        for audience, instruction, bridge in _AUDIENCES:
            variables = RoleSeparatedVariableBy(
                VariableBy2D(
                    {
                        "scenario": {
                            "concept": (concept,), "concept_cap": (concept[0].upper() + concept[1:],),
                            "mechanism": (mechanism,), "example": (example,),
                            "check": (check,), "audience": (audience,),
                            "instruction": (instruction,),
                            "bridge": (bridge,),
                        },
                        "prompt": {"teaching_request": _PROMPTS},
                        "answer": {"explanation": _ANSWERS},
                    }
                )
            )
            deck = V2RoleSeparatedDeck(
                name=f"{TASK}:{domain}:{concept}", variables=variables,
                prompt_pools=(V2SubcardPool("teaching_request", SurfaceRole.PROMPT, ("{prompt[teaching_request]}",)),),
                answer_pools=(V2SubcardPool("explanation", SurfaceRole.ANSWER, ("{answer[explanation]}",)),),
                prompt_plans=prompt_variant_plans(
                    sense="teaching_request",
                    pool_name="teaching_request",
                    functions=_PROMPT_FUNCTIONS,
                ),
                answer_plans=answer_variant_plans(
                    sense="explanation",
                    pool_name="explanation",
                    functions=_ANSWER_FUNCTIONS,
                ),
            )
            case_id = f"{domain}:{concept}:{audience}"
            rows.append(
                render_v2_row(
                    task=TASK, case_id=case_id, domain=domain, difficulty="medium",
                    deck=deck,
                    facts={"concept": concept, "mechanism": mechanism, "example": example, "check": check, "audience": audience},
                    validator={"kind": "contains", "required": [mechanism, example, check]},
                )
            )
            misconception = _MISCONCEPTIONS[concept]
            boundary = _BOUNDARIES[concept]
            for move, move_instruction, move_action, move_success in _LEARNING_MOVES:
                for context, context_need, artifact, context_check in _STUDY_CONTEXTS:
                    for depth, depth_guidance in _DEPTH_GUIDANCE.items():
                        scenario = {
                            "concept": (concept,),
                            "concept_cap": (concept[0].upper() + concept[1:],),
                            "mechanism": (mechanism,),
                            "example": (example,),
                            "check": (check,),
                            "audience": (audience,),
                            "bridge": (bridge,),
                            "bridge_lower": (bridge[0].lower() + bridge[1:],),
                            "misconception": (misconception,),
                            "boundary": (boundary,),
                            "boundary_cap": (boundary[0].upper() + boundary[1:],),
                            "move": (move,),
                            "move_instruction": (move_instruction,),
                            "move_action": (move_action,),
                            "move_action_cap": (move_action[0].upper() + move_action[1:],),
                            "move_success": (
                                f"Success means the learner can {move_action}.",
                            ),
                            "context": (context,),
                            "context_need": (context_need,),
                            "artifact": (artifact,),
                            "context_check": (context_check,),
                            "verification_question": (move_success,),
                            "depth": (depth,),
                            "depth_guidance": (depth_guidance,),
                            "depth_guidance_cap": (depth_guidance[0].upper() + depth_guidance[1:],),
                        }
                        expanded_variables = RoleSeparatedVariableBy(
                            VariableBy2D(
                                {
                                    "scenario": scenario,
                                    "prompt": {"learning_request": _EXPANDED_PROMPTS},
                                    "answer": {"lesson": _expanded_answers(depth)},
                                }
                            )
                        )
                        expanded_deck = V2RoleSeparatedDeck(
                            name=f"{TASK}:{domain}:{concept}:{depth}",
                            variables=expanded_variables,
                            prompt_pools=(
                                V2SubcardPool(
                                    "learning_request",
                                    SurfaceRole.PROMPT,
                                    ("{prompt[learning_request]}",),
                                ),
                            ),
                            answer_pools=(
                                V2SubcardPool(
                                    "lesson",
                                    SurfaceRole.ANSWER,
                                    ("{answer[lesson]}",),
                                ),
                            ),
                            prompt_plans=prompt_variant_plans(
                                sense="learning_request",
                                pool_name="learning_request",
                                functions=_EXPANDED_PROMPT_FUNCTIONS,
                            ),
                            answer_plans=answer_variant_plans(
                                sense="lesson",
                                pool_name="lesson",
                                functions=_expanded_answer_functions(depth),
                            ),
                        )
                        case_id = ":".join(
                            (domain, concept, audience, move, context, depth)
                        )
                        rows.append(
                            render_v2_row(
                                task=TASK,
                                case_id=case_id,
                                domain=domain,
                                difficulty=(
                                    "easy" if depth == "concise" else "medium"
                                    if depth == "detailed" else "hard"
                                ),
                                deck=expanded_deck,
                                facts={
                                    "concept": concept,
                                    "audience": audience,
                                    "move": move,
                                    "context": context,
                                    "depth": depth,
                                    "mechanism": mechanism,
                                    "example": example,
                                    "boundary": boundary,
                                    "misconception": misconception,
                                    "check": check,
                                },
                                validator={
                                    "kind": "contains",
                                    "required": (
                                        [mechanism, example]
                                        if depth == "concise"
                                        else [
                                            mechanism,
                                            example,
                                            boundary,
                                            misconception,
                                            check,
                                        ]
                                    ),
                                },
                            )
                        )
    return validate_complete_rows(TASK, rows, explanation_learning_capacity())


__all__ = ("explanation_learning_capacity", "render_explanation_learning_rows")
