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
) -> dict[str, tuple[str, str, str, tuple[str, str, str]]]:
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
    lessons: dict[str, tuple[str, str, str, tuple[str, str, str]]] = {
        "computing": (
            "RAM holds working data temporarily; storage retains files after power is removed.",
            f"After {minutes_open} minutes of editing, closing {app_name} frees its RAM, but its saved file remains on storage.",
            f"Why does a saved file survive a restart while an unsaved edit in {app_name} may not?",
            (
                f"The reverse also holds after {minutes_open} minutes: an unsaved edit in {app_name} sitting only in RAM disappears if the application closes without saving.",
                f"Conversely, an edit kept only in {app_name}'s RAM for {minutes_open} minutes is lost when the app closes without saving it to storage.",
                f"An edit kept only in RAM for {minutes_open} minutes cannot survive a restart, unlike a file that has already been saved to storage.",
            ),
        ),
        "software_resilience": (
            "A backup is a separate restorable copy that preserves a known-good state if an update corrupts or removes working files.",
            f"Before updating {app_name} to version {backup_version}, copy its data and configuration to another location and confirm that the copy can be restored.",
            f"Why should restore capability be checked before the version-{backup_version} update begins?",
            (
                f"The same logic applies before any risky change to {app_name}: verify the version-{backup_version} backup before editing a shared configuration file.",
                f"A related case: checking that a version-{backup_version} backup restores correctly matters just as much before editing a shared configuration file.",
                f"This same precaution applies whenever a change to {app_name} is hard to reverse: first prove that its version-{backup_version} backup can be restored.",
            ),
        ),
        "data_literacy": (
            "The mean uses every value; the median is the middle value after sorting.",
            f"For {mean_low}, {mean_mid}, and {mean_outlier}, the mean is {dataset_mean} while the median is {mean_mid}.",
            "Which measure better represents a typical value when one value is extreme?",
            (
                f"This same mean-versus-median gap appears in income data, where one earner near ${mean_outlier} thousand pulls the mean well above the typical value.",
                f"Income data shows the identical gap: one high earner pulls the mean well above the median of {mean_mid}, just as in this sample.",
                f"A single extreme outlier near {mean_outlier} can pull the mean far from the median in any dataset, not only in this one example.",
            ),
        ),
        "physical_science": (
            "Mass measures matter; weight is the gravitational force acting on that mass.",
            f"The same {object_name}, weighing {weight_kg} kg on Earth, keeps its mass on the Moon but weighs less there.",
            f"What changes on the Moon: the {object_name}'s mass, its weight, or both?",
            (
                f"A {weight_kg}-kg rock brought to Mars would show the identical pattern: unchanged mass, but reduced weight under Mars's weaker gravity.",
                f"The same {object_name} taken to Mars keeps its mass unchanged, while its weight drops under Mars's weaker gravity.",
                f"Any {weight_kg}-kg object carried to a body with weaker gravity keeps its mass while its weight drops, following the same rule.",
            ),
        ),
        "life_science": (
            "A gene is a DNA sequence; an expressed trait also depends on regulation and environment.",
            f"Two cells can share the same {gene_count} candidate genes while activating different ones, such as {cell_pair}.",
            f"Why can {cell_pair} behave differently?",
            (
                f"Identical twins show a related pattern: the same {gene_count}-gene set can still produce different traits through environmental differences over a lifetime.",
                f"A related case is identical twins, whose shared {gene_count}-gene set can still produce different traits through environmental differences over time.",
                f"Gene regulation explains both cases: which of the {gene_count} genes are active, not just which are present, shapes the outcome.",
            ),
        ),
        "mathematics": (
            "Area counts square units inside a shape; perimeter measures the boundary length.",
            f"A {rect_w} by {rect_h} rectangle has area {rect_w * rect_h} square units and perimeter {2 * (rect_w + rect_h)} units.",
            "Which quantity changes when only the boundary length changes?",
            (
                f"Stretching only the {rect_w}-unit width of the same rectangle increases both its area and its perimeter, unlike simply changing its position.",
                f"Widening the same {rect_h}-unit-tall rectangle changes both its area and its perimeter, unlike sliding it to a new position.",
                f"Moving a {rect_w}-by-{rect_h} rectangle without resizing it changes neither its area nor its perimeter, unlike stretching one side.",
            ),
        ),
        "personal_finance": (
            "Interest is the price of borrowing; principal is the amount borrowed.",
            f"A ${principal} principal with ${interest} interest requires ${principal + interest} in total repayment.",
            "Which part of the repayment is the borrowing cost?",
            (
                f"A larger ${principal * 2} principal at the same rate would produce a larger interest charge, since interest scales with the amount borrowed.",
                f"Borrowing twice the ${principal} principal at the same rate produces a larger interest charge, since interest scales with the amount borrowed.",
                "Interest scales with how much is borrowed, so a smaller principal at the same rate produces a smaller interest charge.",
            ),
        ),
        "civics": (
            "A proposed bill is not a law until the required legislative and approval steps occur.",
            f"A {chamber_size}-member legislative chamber can advance {bill_type} by a {vote_margin}-vote margin without making it enforceable law.",
            "Does approval at this legislative stage alone make the proposal a law?",
            (
                f"A bill that advances in a {chamber_size}-member chamber by {vote_margin} votes but then fails at the next required stage illustrates the same gap.",
                f"A related case: a proposal can win a {vote_margin}-vote margin among {chamber_size} members and still fail a later required approval.",
                f"Passing one stage by {vote_margin} votes in a {chamber_size}-member chamber never guarantees passage through every remaining stage.",
            ),
        ),
        "media_literacy": (
            "A primary source records direct evidence; a secondary source interprets other material.",
            f"A {interview_number}-word original interview about {topic} is primary evidence, while an article analyzing it is secondary.",
            "Which source should be checked for the speaker's exact words?",
            (
                f"A {interview_number}-word press release quoting the original announcement is primary, while a news article summarizing that release is secondary.",
                f"The original press release, {interview_number} words long, counts as primary evidence, while an article summarizing it afterward is secondary.",
                f"The same distinction applies to any {interview_number}-word record: the original account is primary, and any later interpretation is secondary.",
            ),
        ),
        "probability": (
            "Independent events do not change each other's probabilities; mutually exclusive events cannot occur together.",
            f"In a recorded run of {toss_count} independent coin tosses, {heads_count} landed heads, while any one toss could not be both heads and tails.",
            f"Can {toss_count} independent events still occur together?",
            (
                f"Drawing cards without replacement is a related case where the draws are not independent, unlike the {toss_count} tosses that produced {heads_count} heads.",
                f"A related case is drawing cards without replacement, where the draws are not independent, unlike the run of {toss_count} coin tosses with {heads_count} heads.",
                f"Whether events are independent depends on the setup, not on the observed {heads_count} heads among {toss_count} trials.",
            ),
        ),
        "ecology": (
            "Energy moves through a food web and is partly lost as heat, while matter is recycled through organisms and the environment.",
            f"Across a {habitat_area}-square-metre habitat, {producer} stores solar energy, {consumer} consumes it, and decomposers return matter to the soil.",
            "In this food-web sequence, which is recycled: energy, matter, or both?",
            (
                f"A wildfire across the same {habitat_area}-square-metre habitat releases stored energy as heat while returning matter to the soil as ash.",
                f"A related case is a wildfire in a {habitat_area}-square-metre habitat: energy leaves as heat while matter returns to the soil as ash.",
                f"Across the {habitat_area}-square-metre habitat involving {producer} and {consumer}, energy dissipates while matter keeps cycling.",
            ),
        ),
        "electrical_energy": (
            "Power is the rate of energy use, while energy is the accumulated amount used over time.",
            f"A {kw}-kilowatt device running for {hours} hours uses {kw * hours} kilowatt-hours of energy.",
            "What changes if the same device runs twice as long: its power, its energy use, or both?",
            (
                f"Running two {kw}-kilowatt devices together for {hours} hours uses the same {kw * hours * 2} kilowatt-hours as one device running for {hours * 2} hours.",
                f"Two {kw}-kilowatt devices run together for {hours} hours use the same total energy as one such device run for {hours * 2} hours.",
                "Power stays constant for a steady device, while total energy use keeps growing the longer it runs.",
            ),
        ),
        "language_grammar": (
            "A subject is the sentence element linked to the main verb's actor or topic; an object receives or completes the verb's action.",
            f"In '{grammar_sentence},' {grammar_subject} is the subject and {grammar_object} is the object.",
            f"What is the object in '{grammar_sentence}'?",
            (
                f"In 'The teacher graded the essays,' as in '{grammar_sentence}', teacher is the subject and essays is the object, following the same pattern.",
                f"The same pattern appears in 'The teacher graded the essays' as it does in '{grammar_sentence}': teacher is the subject and essays is the object.",
                f"Any transitive sentence, including '{grammar_sentence}', follows this same pattern: the actor is the subject, and what receives the action is the object.",
            ),
        ),
        "computer_networks": (
            "The Domain Name System translates a human-readable host name into an IP address that a network connection can use.",
            f"A browser can ask for {site}, cache the returned address for {dns_ttl_seconds} seconds, and then connect on port {port_number}.",
            "Does DNS carry the whole web page, or does it help locate the destination?",
            (
                f"A phone book locates a number without carrying the conversation, much like resolving {site}, caching it for {dns_ttl_seconds} seconds, then using port {port_number}.",
                f"A related analogy is a phone book: it locates a number without carrying the conversation, just as DNS resolves {site} before the {dns_ttl_seconds}-second cache period.",
                f"The same lookup-cache-connect pattern applies to {site}: DNS supplies the address, the cache keeps it for {dns_ttl_seconds} seconds, and the connection uses port {port_number}.",
            ),
        ),
        "research_methods": (
            "Correlation shows that two measurements vary together; causation requires evidence that changing one factor changes the other.",
            f"Over a {sample_days}-day sample, {correlated_pair} rose together because a shared seasonal factor affects both.",
            f"Does the example show that one of {correlated_pair} causes the other, or that both share a common cause?",
            (
                f"A {sample_days}-day vaccine trial avoids this pitfall by randomly assigning who receives the treatment, which correlation alone cannot establish.",
                f"A vaccine trial over {sample_days} days sidesteps this pitfall by randomly assigning treatment, something correlation alone could never establish.",
                f"Randomized assignment over {sample_days} days is what lets a study move from observing correlation to actually establishing causation.",
            ),
        ),
    }
    return lessons
