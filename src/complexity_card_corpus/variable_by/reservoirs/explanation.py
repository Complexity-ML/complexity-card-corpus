from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExplanationFacts:
    app_name: str
    backup_version: int
    bill_type: str
    cell_pair: str
    chamber_size: int
    consumer: str
    correlated_pair: str
    dataset_mean: float
    dns_ttl_seconds: int
    gene_count: int
    grammar_object: str
    grammar_sentence: str
    grammar_subject: str
    habitat_area: int
    heads_count: int
    hours: int
    interest: int
    interview_number: int
    kw: int
    mean_low: int
    mean_mid: int
    mean_outlier: int
    minutes_open: int
    object_name: str
    port_number: int
    principal: int
    producer: str
    rect_h: int
    rect_w: int
    sample_days: int
    site: str
    topic: str
    toss_count: int
    vote_margin: int
    weight_kg: int


def explanation_reservoir(
    facts: ExplanationFacts,
) -> dict[
    str,
    tuple[
        str | tuple[str, ...],
        str | tuple[str, ...],
        tuple[str, ...],
        tuple[str, str, str],
    ],
]:
    """Localize explanation cards from already dealt semantic facts."""

    app_name = facts.app_name
    backup_version = facts.backup_version
    bill_type = facts.bill_type
    cell_pair = facts.cell_pair
    chamber_size = facts.chamber_size
    consumer = facts.consumer
    correlated_pair = facts.correlated_pair
    dataset_mean = facts.dataset_mean
    dns_ttl_seconds = facts.dns_ttl_seconds
    gene_count = facts.gene_count
    grammar_object = facts.grammar_object
    grammar_sentence = facts.grammar_sentence
    grammar_subject = facts.grammar_subject
    habitat_area = facts.habitat_area
    heads_count = facts.heads_count
    hours = facts.hours
    interest = facts.interest
    interview_number = facts.interview_number
    kw = facts.kw
    mean_low = facts.mean_low
    mean_mid = facts.mean_mid
    mean_outlier = facts.mean_outlier
    minutes_open = facts.minutes_open
    object_name = facts.object_name
    port_number = facts.port_number
    principal = facts.principal
    producer = facts.producer
    rect_h = facts.rect_h
    rect_w = facts.rect_w
    sample_days = facts.sample_days
    site = facts.site
    topic = facts.topic
    toss_count = facts.toss_count
    vote_margin = facts.vote_margin
    weight_kg = facts.weight_kg
    lessons: dict[
        str,
        tuple[
            str | tuple[str, ...],
            str | tuple[str, ...],
            tuple[str, ...],
            tuple[str, str, str],
        ],
    ] = {
        "computing": (
            (
                "RAM holds working data temporarily; storage retains files after power is removed.",
                "Active programs use volatile memory for current work, while saved data persists on non-volatile storage.",
                "Information in RAM depends on continued power; a file written to storage survives shutdown.",
            ),
            (
                f"After {minutes_open} minutes of editing, closing {app_name} frees its RAM, but its saved file remains on storage.",
                f"When {app_name} closes after {minutes_open} minutes, its temporary memory is released while the saved document stays on disk.",
                f"A file saved during {minutes_open} minutes in {app_name} persists after exit even though the application's working RAM does not.",
            ),
            (
                f"Why does a saved file survive a restart while an unsaved edit in {app_name} may not?",
                f"After restarting, what makes the saved file persist while an unsaved {app_name} edit disappears?",
                f"Which location preserves the file after power loss, and why does {app_name}'s unsaved work lack that protection?",
            ),
            (
                f"The reverse also holds after {minutes_open} minutes: an unsaved edit in {app_name} sitting only in RAM disappears if the application closes without saving.",
                f"Conversely, an edit kept only in {app_name}'s RAM for {minutes_open} minutes is lost when the app closes without saving it to storage.",
                f"An edit kept only in RAM for {minutes_open} minutes cannot survive a restart, unlike a file that has already been saved to storage.",
            ),
        ),
        "software_resilience": (
            (
                "A backup is a separate restorable copy that preserves a known-good state if an update corrupts or removes working files.",
                "Resilience comes from keeping an independent copy that can actually restore the system after a damaging change.",
                "A usable backup separates known-good data from the live installation and proves it can recreate that state.",
            ),
            (
                f"Before updating {app_name} to version {backup_version}, copy its data and configuration to another location and confirm that the copy can be restored.",
                f"For the version-{backup_version} change to {app_name}, preserve data and settings independently, then run a restoration check before installation.",
                f"Test recovery from a separate copy of {app_name}'s configuration and files before applying update {backup_version}.",
            ),
            (
                f"Why should restore capability be checked before the version-{backup_version} update begins?",
                f"What risk remains if the version-{backup_version} copy exists but has never been restored successfully?",
                f"Before updating {app_name}, what does a successful restore test establish that copying alone does not?",
            ),
            (
                f"The same logic applies before any risky change to {app_name}: verify the version-{backup_version} backup before editing a shared configuration file.",
                f"A related case: checking that a version-{backup_version} backup restores correctly matters just as much before editing a shared configuration file.",
                f"This same precaution applies whenever a change to {app_name} is hard to reverse: first prove that its version-{backup_version} backup can be restored.",
            ),
        ),
        "data_literacy": (
            (
                "The mean uses every value; the median is the middle value after sorting.",
                "A mean divides the sum by the number of observations, whereas a median depends on the center of their sorted order.",
                "Every observation influences the arithmetic average, but only position determines which value is the median.",
                "The median identifies the ordered midpoint; the mean balances the numerical total across all entries.",
            ),
            (
                f"For {mean_low}, {mean_mid}, and {mean_outlier}, the mean is {dataset_mean} while the median is {mean_mid}.",
                f"Sorting the values as {mean_low}, {mean_mid}, {mean_outlier} puts {mean_mid} in the middle, whereas averaging all three gives {dataset_mean}.",
                f"The outlier {mean_outlier} pulls the arithmetic average to {dataset_mean}; the central sorted observation stays {mean_mid}.",
                f"In this three-value set, division of the total by three yields a mean of {dataset_mean}, but position makes {mean_mid} the median.",
            ),
            (
                "Which measure better represents a typical value when one value is extreme?",
                "When one observation is an outlier, should typical value be described by the mean or the median?",
                "Which summary resists the extreme value and stays closer to the middle observation?",
            ),
            (
                f"This same mean-versus-median gap appears in income data, where one earner near ${mean_outlier} thousand pulls the mean well above the typical value.",
                f"Income data shows the identical gap: one high earner pulls the mean well above the median of {mean_mid}, just as in this sample.",
                f"A single extreme outlier near {mean_outlier} can pull the mean far from the median in any dataset, not only in this one example.",
            ),
        ),
        "physical_science": (
            (
                "Mass measures matter; weight is the gravitational force acting on that mass.",
                "An object's amount of matter determines its mass, while local gravity determines the force called weight.",
                "Changing gravity changes weight but not mass, because only weight is a force produced by the gravitational field.",
                "Mass belongs to the object itself; weight describes how strongly gravity pulls on that mass.",
            ),
            (
                f"The same {object_name}, weighing {weight_kg} kg on Earth, keeps its mass on the Moon but weighs less there.",
                f"Moving the {weight_kg}-kg {object_name} from Earth to the Moon leaves its mass unchanged even though the weaker pull reduces its weight.",
                f"On the Moon, the {object_name} still contains the same amount of matter as it did on Earth, but gravity exerts less force on it.",
                f"A balance would assign the {object_name} the same {weight_kg}-kg mass in both places, while a force scale would show a lower lunar weight.",
            ),
            (
                f"What changes on the Moon: the {object_name}'s mass, its weight, or both?",
                f"For the {object_name} on the Moon, which property stays fixed and which responds to weaker gravity?",
                f"Does moving the {weight_kg}-kg {object_name} to the Moon alter its amount of matter or only the gravitational force?",
            ),
            (
                f"A {weight_kg}-kg rock brought to Mars would show the identical pattern: unchanged mass, but reduced weight under Mars's weaker gravity.",
                f"The same {object_name} taken to Mars keeps its mass unchanged, while its weight drops under Mars's weaker gravity.",
                f"Any {weight_kg}-kg object carried to a body with weaker gravity keeps its mass while its weight drops, following the same rule.",
            ),
        ),
        "life_science": (
            (
                "A gene is a DNA sequence; an expressed trait also depends on regulation and environment.",
                "Possessing the same DNA does not guarantee identical traits because cells regulate which genes are active.",
                "Genes provide sequences, while gene expression and environmental conditions shape the resulting characteristics.",
            ),
            (
                f"Two cells can share the same {gene_count} candidate genes while activating different ones, such as {cell_pair}.",
                f"Although {cell_pair} contain the same set of {gene_count} candidate genes, each cell type switches on a different subset.",
                f"Different expression patterns let {cell_pair} perform unlike functions despite sharing {gene_count} candidate genes.",
            ),
            (
                f"Why can {cell_pair} behave differently?",
                f"How can {cell_pair} share DNA yet perform different functions?",
                f"What explains the different behavior of {cell_pair} when their gene set is the same?",
            ),
            (
                f"Identical twins show a related pattern: the same {gene_count}-gene set can still produce different traits through environmental differences over a lifetime.",
                f"A related case is identical twins, whose shared {gene_count}-gene set can still produce different traits through environmental differences over time.",
                f"Gene regulation explains both cases: which of the {gene_count} genes are active, not just which are present, shapes the outcome.",
            ),
        ),
        "mathematics": (
            (
                "Area counts square units inside a shape; perimeter measures the boundary length.",
                "Multiplying side lengths measures a rectangle's interior, whereas adding all sides measures its perimeter.",
                "Area describes enclosed surface in square units; perimeter totals the one-dimensional distance around the edge.",
            ),
            (
                f"A {rect_w} by {rect_h} rectangle has area {rect_w * rect_h} square units and perimeter {2 * (rect_w + rect_h)} units.",
                f"For sides {rect_w} and {rect_h}, multiplication gives {rect_w * rect_h} square units inside and doubling their sum gives a boundary of {2 * (rect_w + rect_h)} units.",
                f"The {rect_w}-by-{rect_h} rectangle encloses {rect_w * rect_h} square units, while its four edges total {2 * (rect_w + rect_h)} units.",
            ),
            (
                "Which quantity changes when only the boundary length changes?",
                "If a rectangle's boundary is altered, how should area and perimeter be checked separately?",
                "Why can changing a side length affect both the enclosed area and the boundary measure?",
            ),
            (
                f"Stretching only the {rect_w}-unit width of the same rectangle increases both its area and its perimeter, unlike simply changing its position.",
                f"Widening the same {rect_h}-unit-tall rectangle changes both its area and its perimeter, unlike sliding it to a new position.",
                f"Moving a {rect_w}-by-{rect_h} rectangle without resizing it changes neither its area nor its perimeter, unlike stretching one side.",
            ),
        ),
        "personal_finance": (
            (
                "Interest is the price of borrowing; principal is the amount borrowed.",
                "Principal names the original debt, and interest is the additional borrowing charge paid with it.",
                "Repayment returns the borrowed principal plus interest charged for the use of that money.",
            ),
            (
                f"A ${principal} principal with ${interest} interest requires ${principal + interest} in total repayment.",
                f"Returning the ${principal} borrowed amount and paying its ${interest} charge produces a ${principal + interest} repayment.",
                f"The debt totals ${principal + interest}: ${principal} restores principal and ${interest} covers interest.",
            ),
            (
                "Which part of the repayment is the borrowing cost?",
                "In the total repayment, what amount pays for borrowing rather than returning principal?",
                "How do principal and interest divide the amount that must be repaid?",
            ),
            (
                f"A larger ${principal * 2} principal at the same rate would produce a larger interest charge, since interest scales with the amount borrowed.",
                f"Borrowing twice the ${principal} principal at the same rate produces a larger interest charge, since interest scales with the amount borrowed.",
                "Interest scales with how much is borrowed, so a smaller principal at the same rate produces a smaller interest charge.",
            ),
        ),
        "civics": (
            (
                "A proposed bill is not a law until the required legislative and approval steps occur.",
                "Passing one legislative stage advances a bill but does not make it enforceable before all required approvals are complete.",
                "A bill becomes law only after the full prescribed process, not merely because one chamber has approved it.",
                "Legislative approval at an intermediate stage moves a proposal forward while leaving later lawmaking requirements outstanding.",
            ),
            (
                f"A {chamber_size}-member legislative chamber can advance {bill_type} by a {vote_margin}-vote margin without making it enforceable law.",
                f"Even after {bill_type} wins by {vote_margin} votes among {chamber_size} legislators, it remains a proposal until the later required stages succeed.",
                f"The chamber's {vote_margin}-vote majority moves {bill_type} forward, but completion of the remaining approval process is still necessary.",
                f"Approval in a chamber of {chamber_size} members is one procedural step for {bill_type}; its {vote_margin}-vote margin alone creates no enforceable rule.",
            ),
            (
                "Does approval at this legislative stage alone make the proposal a law?",
                "After this chamber approves the proposal, what further requirement prevents calling it law yet?",
                "Why is winning this legislative vote insufficient by itself to create enforceable law?",
            ),
            (
                f"A bill that advances in a {chamber_size}-member chamber by {vote_margin} votes but then fails at the next required stage illustrates the same gap.",
                f"A related case: a proposal can win a {vote_margin}-vote margin among {chamber_size} members and still fail a later required approval.",
                f"Passing one stage by {vote_margin} votes in a {chamber_size}-member chamber never guarantees passage through every remaining stage.",
            ),
        ),
        "media_literacy": (
            (
                "A primary source records direct evidence; a secondary source interprets other material.",
                "Original records provide direct evidence, while later analyses explain or evaluate those records.",
                "The direct artifact is primary material; commentary built from it belongs to the secondary layer.",
            ),
            (
                f"A {interview_number}-word original interview about {topic} is primary evidence, while an article analyzing it is secondary.",
                f"For {topic}, the original {interview_number}-word interview is the direct record and the later analytical article is an interpretation.",
                f"The interview itself supplies primary evidence about {topic}; an article discussing that {interview_number}-word record is secondary.",
            ),
            (
                "Which source should be checked for the speaker's exact words?",
                "To verify the speaker's wording, should the original interview or the later analysis control?",
                "Where is direct evidence of the exact statement found: in the primary record or its interpretation?",
            ),
            (
                f"A {interview_number}-word press release quoting the original announcement is primary, while a news article summarizing that release is secondary.",
                f"The original press release, {interview_number} words long, counts as primary evidence, while an article summarizing it afterward is secondary.",
                f"The same distinction applies to any {interview_number}-word record: the original account is primary, and any later interpretation is secondary.",
            ),
        ),
        "probability": (
            (
                "Independent events do not change each other's probabilities; mutually exclusive events cannot occur together.",
                "Independence concerns whether one event affects another's chance, while mutual exclusion concerns whether both can happen at once.",
                "Two outcomes are mutually exclusive when they cannot coincide; two trials are independent when knowing one leaves the other's probability unchanged.",
            ),
            (
                f"In a recorded run of {toss_count} independent coin tosses, {heads_count} landed heads, while any one toss could not be both heads and tails.",
                f"Across {toss_count} separate tosses, the observed {heads_count} heads do not alter the next toss, yet a single toss cannot show heads and tails together.",
                f"The {toss_count} coin trials are independent even though each individual trial has mutually exclusive heads and tails outcomes; {heads_count} heads were observed.",
            ),
            (
                f"Can {toss_count} independent events still occur together?",
                "Does independence prevent events from occurring together, or only prevent one from changing the other's probability?",
                f"What distinguishes the independence of these {toss_count} tosses from the impossibility of one toss being both heads and tails?",
            ),
            (
                f"Drawing cards without replacement is a related case where the draws are not independent, unlike the {toss_count} tosses that produced {heads_count} heads.",
                f"A related case is drawing cards without replacement, where the draws are not independent, unlike the run of {toss_count} coin tosses with {heads_count} heads.",
                f"Whether events are independent depends on the setup, not on the observed {heads_count} heads among {toss_count} trials.",
            ),
        ),
        "ecology": (
            (
                "Energy moves through a food web and is partly lost as heat, while matter is recycled through organisms and the environment.",
                "Matter cycles repeatedly through an ecosystem, whereas usable energy flows onward and eventually disperses as heat.",
                "Organisms reuse the ecosystem's matter, but each energy transfer leaves less energy available at the next feeding level.",
                "A food web circulates nutrients among organisms and surroundings while energy follows a one-way path toward heat loss.",
            ),
            (
                f"Across a {habitat_area}-square-metre habitat, {producer} stores solar energy, {consumer} consumes it, and decomposers return matter to the soil.",
                f"In this {habitat_area}-square-metre habitat, energy passes from {producer} to {consumer}, while decomposers recycle their material into soil nutrients.",
                f"When {consumer} feeds on {producer}, some energy dissipates, but decomposers make their matter available again within the {habitat_area}-square-metre habitat.",
                f"The {producer}, {consumer}, and decomposers share matter in a repeating cycle across {habitat_area} square metres, even as transferred energy leaves as heat.",
            ),
            (
                "In this food-web sequence, which is recycled: energy, matter, or both?",
                "What returns through the ecosystem, and what instead dissipates as heat?",
                "How do matter and energy follow different paths through this food web?",
            ),
            (
                f"A wildfire across the same {habitat_area}-square-metre habitat releases stored energy as heat while returning matter to the soil as ash.",
                f"A related case is a wildfire in a {habitat_area}-square-metre habitat: energy leaves as heat while matter returns to the soil as ash.",
                f"Across the {habitat_area}-square-metre habitat involving {producer} and {consumer}, energy dissipates while matter keeps cycling.",
            ),
        ),
        "electrical_energy": (
            (
                "Power is the rate of energy use, while energy is the accumulated amount used over time.",
                "Kilowatts describe how quickly a device uses energy; kilowatt-hours total that use across a duration.",
                "Power is an instantaneous usage rate, whereas energy grows by combining that rate with operating time.",
            ),
            (
                f"A {kw}-kilowatt device running for {hours} hours uses {kw * hours} kilowatt-hours of energy.",
                f"Multiplying the device's {kw}-kilowatt power by its {hours}-hour runtime gives {kw * hours} kilowatt-hours consumed.",
                f"The device holds a {kw}-kilowatt rate while operating, so after {hours} hours its accumulated use is {kw * hours} kilowatt-hours.",
                f"Running at {kw} kilowatts for {hours} hours adds up to {kw * hours} kilowatt-hours, even though the power rating itself stays fixed.",
            ),
            (
                "What changes if the same device runs twice as long: its power, its energy use, or both?",
                "If operating time doubles while the device stays unchanged, which quantity doubles?",
                "Why does a longer run increase kilowatt-hours without changing the device's kilowatt rating?",
            ),
            (
                f"Running two {kw}-kilowatt devices together for {hours} hours uses the same {kw * hours * 2} kilowatt-hours as one device running for {hours * 2} hours.",
                f"Two {kw}-kilowatt devices run together for {hours} hours use the same total energy as one such device run for {hours * 2} hours.",
                "Power stays constant for a steady device, while total energy use keeps growing the longer it runs.",
            ),
        ),
        "language_grammar": (
            (
                "A subject is the sentence element linked to the main verb's actor or topic; an object receives or completes the verb's action.",
                "The subject names who or what the clause is about, while the object is the element affected or completed by the verb.",
                "In a transitive clause, the subject performs or anchors the action and the object answers whom or what that action reaches.",
                "Verb agreement and clause meaning identify the subject; the noun phrase receiving the action functions as the object.",
            ),
            (
                f"In '{grammar_sentence},' {grammar_subject} is the subject and {grammar_object} is the object.",
                f"The sentence '{grammar_sentence}' places {grammar_subject} in the subject role and {grammar_object} in the object role.",
                f"Reading '{grammar_sentence}' by grammatical function identifies {grammar_subject} as subject and {grammar_object} as the verb's object.",
            ),
            (
                f"What is the object in '{grammar_sentence}'?",
                f"In '{grammar_sentence},' which element receives or completes the verb's action?",
                f"How can the object be identified separately from {grammar_subject}, the subject, in this sentence?",
            ),
            (
                f"In 'The teacher graded the essays,' as in '{grammar_sentence}', teacher is the subject and essays is the object, following the same pattern.",
                f"The same pattern appears in 'The teacher graded the essays' as it does in '{grammar_sentence}': teacher is the subject and essays is the object.",
                f"Any transitive sentence, including '{grammar_sentence}', follows this same pattern: the actor is the subject, and what receives the action is the object.",
            ),
        ),
        "computer_networks": (
            (
                "The Domain Name System translates a human-readable host name into an IP address that a network connection can use.",
                "DNS resolves a site's readable name to the numerical network address needed to contact its server.",
                "Before a connection reaches a named host, a DNS lookup supplies the IP address associated with that name.",
                "A host name is convenient for people; DNS maps it to the IP address used by the network route.",
            ),
            (
                f"A browser can ask for {site}, cache the returned address for {dns_ttl_seconds} seconds, and then connect on port {port_number}.",
                f"To reach {site}, the browser resolves its address, retains that result for {dns_ttl_seconds} seconds, and opens port {port_number}.",
                f"The lookup for {site} returns an IP that may stay cached for {dns_ttl_seconds} seconds before a connection uses port {port_number}.",
                f"For {site}, name resolution supplies an address that may be reused for {dns_ttl_seconds} seconds before traffic reaches port {port_number}.",
            ),
            (
                "Does DNS carry the whole web page, or does it help locate the destination?",
                f"When the browser requests {site}, what role does DNS play before page data is transferred?",
                "How is resolving a host name different from carrying the content returned by that host?",
            ),
            (
                f"A phone book locates a number without carrying the conversation, much like resolving {site}, caching it for {dns_ttl_seconds} seconds, then using port {port_number}.",
                f"A related analogy is a phone book: it locates a number without carrying the conversation, just as DNS resolves {site} before the {dns_ttl_seconds}-second cache period.",
                f"Applied to {site}, the sequence is resolution, a {dns_ttl_seconds}-second reuse window, and finally a connection to port {port_number}.",
                f"A directory lookup is a useful analogy for {site}: find the destination first, retain it temporarily, then contact the service on port {port_number}.",
                f"The {dns_ttl_seconds}-second cache avoids resolving {site} on every request, while port {port_number} identifies the service endpoint after resolution.",
                f"Resolving {site} answers where to connect; keeping that answer briefly and opening port {port_number} are separate later steps.",
            ),
        ),
        "research_methods": (
            (
                "Correlation shows that two measurements vary together; causation requires evidence that changing one factor changes the other.",
                "Two variables can move together without either producing the other; a causal claim needs evidence from a controlled change.",
                "An observed association describes a pattern, whereas causation says that intervening on one factor alters the outcome.",
                "Shared movement is correlational evidence only; establishing cause requires ruling out alternatives and testing an intervention.",
            ),
            (
                f"Over a {sample_days}-day sample, {correlated_pair} rose together because a shared seasonal factor affects both.",
                f"During {sample_days} days of observation, both {correlated_pair} increased as the same seasonal condition changed.",
                f"A common seasonal driver explains why {correlated_pair} followed similar trends across the {sample_days}-day record.",
                f"The {sample_days}-day data links {correlated_pair}, but their joint rise can be accounted for by season rather than a direct effect.",
            ),
            (
                f"Does the example show that one of {correlated_pair} causes the other, or that both share a common cause?",
                f"What evidence is missing before the correlation between {correlated_pair} could support a causal claim?",
                f"Why can the shared seasonal factor explain the movement in {correlated_pair} without either causing the other?",
            ),
            (
                f"A {sample_days}-day vaccine trial avoids this pitfall by randomly assigning who receives the treatment, which correlation alone cannot establish.",
                f"A vaccine trial over {sample_days} days sidesteps this pitfall by randomly assigning treatment, something correlation alone could never establish.",
                f"Randomized assignment over {sample_days} days is what lets a study move from observing correlation to actually establishing causation.",
            ),
        ),
    }
    return lessons
