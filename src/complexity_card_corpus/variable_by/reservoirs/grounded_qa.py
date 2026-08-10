from __future__ import annotations

from dataclasses import dataclass

from ..matrix import VariableBy2D


@dataclass(frozen=True)
class GroundedQAFacts:
    """Typed semantic facts required by every grounded-QA domain.

    Keeping this contract explicit makes missing or renamed facts fail at the
    call site instead of surfacing later as an opaque context lookup.
    """

    code: str
    year: int
    battery_hours: int
    return_days: str
    exposure_hours: int
    temperature_change: int
    owner: str
    delivery_day: int
    train_number: int
    departure_hour: int
    departure_minute: str
    platform: int
    python_minor: str
    release_major: int
    release_minor: int
    longest_battery: int
    other_battery: int
    status_minute: str
    ticket_minute: str
    available_region: str
    ticket_region: str
    failed_operation: str
    event_day: int
    event_room: int
    energy_kwh: int
    energy_rate: int
    course_number: int
    maintenance_day: int
    sensor_count: int
    measured_value: int
    notice_days: int
    sample_count: int
    ph_value: int
    quote_units: int
    quote_price: int
    tested_pages: int
    operating_limit: int


@dataclass(frozen=True)
class _GroundedDomainCells:
    documented_source: str
    context_source: str
    absence_source: str
    absence_clause: str
    requested_fact: str
    requested_boundary: str
    documented_answer: str
    unknown_subject: str
    additional_facts: tuple[str, str, str]
    unknown_answers: tuple[str, str, str] | None = None


def _domain_cells(f: GroundedQAFacts) -> dict[str, _GroundedDomainCells]:
    return {
        "product_specs": _GroundedDomainCells(
            f"The Lumen Mini has a rated battery life of {f.battery_hours} hours.",
            "Wi-Fi 6 and USB-C charging are listed as supported features.",
            "No water-resistance rating is listed.",
            "the specification lists no water-resistance rating",
            "State the rated battery life",
            "determine whether water resistance is documented",
            f"The rated battery life is {f.battery_hours} hours.",
            "the product's water-resistance rating",
            (
                "Wi-Fi 6 and USB-C charging are also confirmed.",
                "The confirmed feature list includes Wi-Fi 6 and USB-C charging.",
                "Support for both Wi-Fi 6 and USB-C charging is documented.",
            ),
        ),
        "policy_excerpt": _GroundedDomainCells(
            f"Returns are accepted within {f.return_days} days with proof of purchase.",
            "Opened safety equipment is excluded.",
            "The text gives no holiday extension.",
            "the policy defines no holiday extension",
            "State the ordinary return window",
            "determine whether a holiday extension is defined",
            f"The ordinary return window is {f.return_days} days with proof of purchase.",
            "a holiday extension",
            (
                "Opened safety equipment is excluded from returns.",
                "The policy excludes returns of opened safety equipment.",
                "Opened safety equipment cannot be returned under this policy.",
            ),
        ),
        "science_passage": _GroundedDomainCells(
            f"A {f.year} trial exposed identical samples to light for {f.exposure_hours} hours.",
            f"The treated sample warmed by {f.temperature_change}°C.",
            "The passage does not identify the molecular mechanism.",
            "the passage reports no molecular mechanism",
            "State the observed temperature change",
            "determine whether its molecular mechanism is established",
            f"The treated sample warmed by {f.temperature_change}°C.",
            "the molecular mechanism",
            (
                f"The trial occurred in {f.year} with a {f.exposure_hours}-hour exposure.",
                f"Identical samples received {f.exposure_hours} hours of light in {f.year}.",
                f"The {f.year} trial used identical samples and {f.exposure_hours} hours of exposure.",
            ),
        ),
        "historical_note": _GroundedDomainCells(
            f"The archive records that the bridge opened in {f.year}.",
            "Mayor Elena Voss presided over the opening.",
            "The archive does not name the original architect.",
            "the archive names no original architect",
            "State the bridge's opening year",
            "determine whether the original architect is identified",
            f"The bridge opened in {f.year}.",
            "the identity of the original architect",
            (
                "The archive names Elena Voss as the presiding mayor.",
                "Mayor Elena Voss is recorded as presiding over the opening.",
                "Elena Voss is the mayor associated with the recorded opening.",
            ),
        ),
        "project_brief": _GroundedDomainCells(
            f"The brief assigns the prototype to {f.owner}.",
            f"Delivery is set for day {f.delivery_day} while hosting approval remains pending.",
            "No hosting approver is named.",
            "the brief names no hosting approver",
            "State who owns the prototype",
            "determine whether the hosting approver is known",
            f"{f.owner} owns the prototype.",
            "the identity of the hosting approver",
            (
                f"Delivery is confirmed for day {f.delivery_day}.",
                f"The brief fixes day {f.delivery_day} as the delivery date.",
                f"Day {f.delivery_day} is the documented delivery target.",
            ),
        ),
        "travel_information": _GroundedDomainCells(
            f"Train {f.train_number} departs at {f.departure_hour:02d}:{f.departure_minute} from platform {f.platform}.",
            "Bicycles require a reservation.",
            "The notice gives no information about onboard meals.",
            "the notice does not mention onboard meals",
            "State the train's departure details",
            "determine whether meal service is documented",
            f"Train {f.train_number} departs at {f.departure_hour:02d}:{f.departure_minute} from platform {f.platform}.",
            "onboard meal service",
            (
                "Bicycles require a reservation before boarding.",
                "The notice confirms that bicycles need a reservation.",
                "A reservation is required for bicycles on this train.",
            ),
            unknown_answers=(
                "Meal service isn't specified and remains unknown because the notice does not mention onboard meals.",
                "Meal service isn't specified, so it remains unknown; the notice gives no information about onboard meals.",
                "Meal service isn't specified and stays unknown because no onboard-meal information appears in the notice.",
            ),
        ),
        "technical_documentation": _GroundedDomainCells(
            f"Version {f.release_major}.{f.release_minor} requires Python 3.{f.python_minor}.",
            "Linux arm64 is supported.",
            "Offline activation is not described in this excerpt.",
            "the excerpt does not describe offline activation",
            "State the Python requirement",
            "determine whether offline activation is supported",
            f"The requirement is Python 3.{f.python_minor}.",
            "support for offline activation",
            (
                "Linux arm64 support is confirmed.",
                "The excerpt explicitly supports the Linux arm64 architecture.",
                "Linux arm64 appears in the supported-platform list.",
            ),
        ),
        "comparison_table": _GroundedDomainCells(
            f"Table: Cedar—$48, {f.longest_battery} hours, repairable yes; Flint—$42, {f.other_battery} hours, repairable no.",
            "Vale costs $45 and is repairable, but its battery value is missing.",
            "No battery-life value is supplied for Vale.",
            "the table supplies no battery-life value for Vale",
            "Identify the longest stated battery life",
            "determine whether Vale can be compared on battery life",
            f"Cedar has the longest stated battery life at {f.longest_battery} hours.",
            "Vale's battery-life comparison",
            (
                "Cedar and Vale are repairable; Flint is not.",
                "The repairability column marks Cedar and Vale yes and Flint no.",
                "Repairability is confirmed for Cedar and Vale but not Flint.",
            ),
        ),
        "conflicting_service_reports": _GroundedDomainCells(
            f"At 09:{f.status_minute}, the public status check reports the {f.available_region} endpoint available.",
            f"At 09:{f.ticket_minute}, one {f.ticket_region} account reports it cannot {f.failed_operation}.",
            "The reports cover different regions, scopes, times, and operations.",
            "the reports do not share one time, region, account scope, and operation",
            "Explain what each report establishes",
            "identify what remains unknown and the next direct verification step",
            f"The reports appear to conflict, but their different scopes leave one global service state unknown. Compare the same time window, region, account scope, and operation, then reproduce the attempt to {f.failed_operation} with a direct check.",
            "one global service state",
            (
                f"Compare the same scope and reproduce the attempt to {f.failed_operation}.",
                f"A direct check should align region, account, time, and the attempt to {f.failed_operation}.",
                f"The next verification should reproduce {f.failed_operation} under matching conditions.",
            ),
        ),
        "public_event_notice": _GroundedDomainCells(
            f"The Open Methods workshop starts on August {f.event_day} at 18:30 in Room {f.event_room}.",
            "Step-free access is available.",
            "The notice does not say whether advance registration is required.",
            "the notice gives no advance-registration requirement",
            "State the workshop time and venue",
            "determine whether advance registration is required",
            f"The workshop starts on August {f.event_day} at 18:30 in Room {f.event_room}.",
            "the advance-registration requirement",
            (
                "Step-free access is available.",
                "The notice confirms step-free access.",
                "Availability of step-free access is documented.",
            ),
        ),
        "energy_bill": _GroundedDomainCells(
            f"The statement records {f.energy_kwh} kWh at {f.energy_rate} cents per kWh before fixed charges.",
            "The statement covers a 30-day billing period.",
            "No rebate or credit is listed.",
            "the statement lists no rebate or credit",
            "State the recorded usage and unit rate",
            "determine whether a rebate is documented",
            f"Usage is {f.energy_kwh} kWh at {f.energy_rate} cents per kWh before fixed charges.",
            "a rebate or credit",
            (
                "The billing period spans 30 days.",
                "The statement covers a 30-day period.",
                "A 30-day billing period is documented.",
            ),
        ),
        "course_catalog": _GroundedDomainCells(
            f"Course CS-{f.course_number} requires Introductory Programming.",
            "The course meets on Tuesdays and includes weekly labs.",
            "The entry does not identify the final assessment format.",
            "the entry names no final assessment format",
            "State the course prerequisite",
            "determine whether the final assessment format is documented",
            f"The prerequisite for CS-{f.course_number} is Introductory Programming.",
            "the final assessment format",
            (
                "Tuesday meetings and weekly labs are confirmed.",
                "The entry schedules weekly labs on Tuesdays.",
                "Both the Tuesday meeting time and weekly labs are documented.",
            ),
        ),
        "maintenance_log": _GroundedDomainCells(
            f"On August {f.maintenance_day}, a technician replaced the intake filter after reduced airflow.",
            "A follow-up test restored normal flow.",
            "The log does not establish the original cause of the blockage.",
            "the log establishes no original cause of the blockage",
            "State the maintenance action and test result",
            "determine whether the original cause is established",
            "The intake filter was replaced and the follow-up test restored normal airflow.",
            "the original cause of the blockage",
            (
                f"The replacement is dated August {f.maintenance_day}.",
                f"The work occurred on August {f.maintenance_day}.",
                f"August {f.maintenance_day} is the recorded maintenance date.",
            ),
        ),
        "environmental_report": _GroundedDomainCells(
            f"At the north site, {f.sensor_count} sensors recorded a median of {f.measured_value} units.",
            "The measurement covers the stated survey window.",
            "The report provides no pre-survey baseline.",
            "the report provides no pre-survey baseline",
            "State the reported measurement",
            "determine whether change from baseline can be calculated",
            f"The north-site median is {f.measured_value} units across {f.sensor_count} sensors.",
            "change from the pre-survey baseline",
            (
                "The median is limited to the stated survey window.",
                "Only the recorded survey window is represented.",
                "The measurement is scoped to the north-site survey window.",
            ),
        ),
        "software_release_note": _GroundedDomainCells(
            f"Release {f.release_major}.{f.release_minor} adds export filters.",
            "The release fixes duplicate notifications on Linux and deprecates a legacy import option.",
            "No removal date is given for the legacy import option.",
            "the release note gives no removal date",
            "State the added behavior",
            "determine whether the legacy import removal date is known",
            f"Release {f.release_major}.{f.release_minor} adds export filters.",
            "the legacy import removal date",
            (
                "Duplicate notifications on Linux are fixed.",
                "The release resolves duplicate Linux notifications.",
                "A fix for duplicate notifications on Linux is documented.",
            ),
        ),
        "contract_clause": _GroundedDomainCells(
            f"Clause 8 requires {f.notice_days} calendar days of written notice before termination.",
            "Notices must be sent to the registered office.",
            "The clause does not identify an arbitration venue.",
            "Clause 8 identifies no arbitration venue",
            "State the termination notice period",
            "determine whether an arbitration venue is identified",
            f"The notice period is {f.notice_days} calendar days in writing.",
            "the arbitration venue",
            (
                "Notices must go to the registered office.",
                "The registered office is the required notice destination.",
                "Delivery of notices to the registered office is required.",
            ),
        ),
        "lab_report": _GroundedDomainCells(
            f"The laboratory tested {f.sample_count} samples and recorded a median pH of {f.ph_value}.",
            "Every sample was measured with the same calibrated probe.",
            "The report does not state when the probe was last calibrated.",
            "the report states no last-calibration date",
            "State the median pH",
            "determine whether the last calibration date is documented",
            f"The median pH is {f.ph_value} across {f.sample_count} samples.",
            "the probe's last calibration date",
            (
                f"The same calibrated probe measured all {f.sample_count} samples.",
                "One calibrated probe was used for every sample.",
                f"Probe usage was consistent across all {f.sample_count} samples.",
            ),
        ),
        "procurement_quote": _GroundedDomainCells(
            f"Quote Q-{f.code} offers {f.quote_units} units at ${f.quote_price} each.",
            "The quote remains valid for 21 days and includes tax.",
            "Shipping time is not listed.",
            "the quote lists no shipping time",
            "State the quoted quantity and unit price",
            "determine whether shipping time is documented",
            f"The quote covers {f.quote_units} units at ${f.quote_price} each.",
            "the shipping time",
            (
                "The quote is valid for 21 days with tax included.",
                "Validity lasts 21 days and the price includes tax.",
                "A 21-day validity period and inclusive tax are documented.",
            ),
        ),
        "accessibility_statement": _GroundedDomainCells(
            f"The statement reports a WCAG 2.2 AA review of {f.tested_pages} public pages.",
            "Keyboard navigation is named as tested.",
            "No remediation date is given for remaining issues.",
            "the statement gives no remediation date",
            "State the conformance target and tested scope",
            "determine whether a remediation date is documented",
            f"The target is WCAG 2.2 AA across {f.tested_pages} public pages.",
            "the remediation date for remaining issues",
            (
                "Keyboard navigation was specifically tested.",
                "The tested areas include keyboard navigation.",
                "Testing of keyboard navigation is documented.",
            ),
        ),
        "equipment_manual": _GroundedDomainCells(
            f"The manual permits continuous supervised operation below {f.operating_limit}°C.",
            "A five-minute cooldown is required after an overload warning.",
            "Remote control is not described.",
            "the manual does not describe remote control",
            "State the operating limit and cooldown requirement",
            "determine whether remote control is documented",
            f"The limit is below {f.operating_limit}°C with a five-minute cooldown after an overload warning.",
            "remote-control support",
            (
                "Continuous operation is limited to supervised mode.",
                "Supervised mode is required for continuous operation.",
                "The manual conditions continuous operation on supervision.",
            ),
        ),
    }


def grounded_qa_variable_by(domain: str, facts: GroundedQAFacts) -> VariableBy2D:
    """Build a genuine axis-by-sense reservoir for one grounded-QA case."""

    try:
        cells = _domain_cells(facts)[domain]
    except KeyError as error:
        raise KeyError(f"unknown grounded-QA domain: {domain}") from error

    return VariableBy2D(
        {
            "source": {
                "code": (facts.code,),
                "documented": (cells.documented_source,),
                "context": (cells.context_source,),
                "absence": (cells.absence_source,),
                "absence_clause": (cells.absence_clause,),
                "passage": (
                    "{source[documented]} {source[context]} {source[absence]}",
                    "{source[context]} {source[documented]} {source[absence]}",
                    "{source[documented]} {source[absence]} {source[context]}",
                ),
            },
            "request": {
                "documented": (cells.requested_fact,),
                "boundary": (cells.requested_boundary,),
                "instruction": (
                    "{request[documented]} and {request[boundary]}.",
                    "{request[documented]}; also {request[boundary]}.",
                    "{request[boundary]}, then {request[documented]}.",
                ),
            },
            "answer": {
                "documented": (cells.documented_answer,),
                "complete": (
                    "{answer[documented]} {fact[additional]} {boundary[unknown]}",
                    "{fact[additional]} {answer[documented]} {boundary[unknown]}",
                    "{answer[documented]} {boundary[unknown]} {fact[additional]}",
                ),
            },
            "fact": {"additional": cells.additional_facts},
            "unknown": {"subject": (cells.unknown_subject,)},
            "boundary": {
                "unknown": cells.unknown_answers or (
                    "This record marks {unknown[subject]} unknown; {source[absence_clause]}.",
                    "The status of {unknown[subject]} is unresolved and therefore unknown; {source[absence_clause]}.",
                    "On this evidence, {unknown[subject]} stays unknown: {source[absence_clause]}.",
                    "Available evidence leaves {unknown[subject]} unknown: {source[absence_clause]}.",
                    "Treat {unknown[subject]} as an unknown field; {source[absence_clause]}.",
                    "The supplied record keeps {unknown[subject]} unknown; {source[absence_clause]}.",
                    "For this answer, {unknown[subject]} must remain unknown: {source[absence_clause]}.",
                    "No supported determination exists for {unknown[subject]}, so it is unknown; {source[absence_clause]}.",
                    "Evidence in this source leaves {unknown[subject]} unknown: {source[absence_clause]}.",
                    "The documented limit prevents a known value for {unknown[subject]}; treat it as unknown. {source[absence_clause]}.",
                    "Within this record, {unknown[subject]} cannot be established; {source[absence_clause]}. Its status is unknown.",
                    "A grounded response must leave {unknown[subject]} unknown: {source[absence_clause]}.",
                ),
            },
            "scope": {
                "source": (
                    "Use only supplied Source {source[code]}.",
                    "Rely exclusively on supplied Source {source[code]}.",
                    "Ground the response in supplied Source {source[code]} alone.",
                ),
                "answer": (
                    "Source {source[code]} supports the documented answer.",
                    "The supported result comes from Source {source[code]}.",
                    "The response is grounded in Source {source[code]}.",
                    "Documented evidence comes directly from Source {source[code]}.",
                    "Source {source[code]} supplies the answer's factual basis.",
                    "The known result is supported by Source {source[code]}.",
                    "Only Source {source[code]} grounds the supported finding.",
                    "The documented portion follows from Source {source[code]}.",
                    "Evidence for the known part appears in Source {source[code]}.",
                    "The factual response rests on Source {source[code]}.",
                    "Support for the answer is contained in Source {source[code]}.",
                    "Source {source[code]} establishes the response's known portion.",
                ),
            },
            "constraint": {
                "unknown": (
                    "Leave undocumented details unknown.",
                    "Do not infer absent information.",
                    "Mark unsupported details as unknown.",
                ),
                "grounding": (
                    "{scope[source]} {constraint[unknown]}",
                    "{constraint[unknown]} {scope[source]}",
                    "{scope[source]} Explicitly, {constraint[unknown]}",
                ),
            },
            "label": {
                "source": ("Source", "Evidence", "Supplied record"),
                "request": ("Request", "Question", "Grounded task"),
                "evidence": ("Evidence scope", "Grounding", "Source basis"),
                "documented": ("Documented answer", "Supported finding", "Known result"),
                "additional": ("Corroborating fact", "Additional evidence", "Related fact"),
                "boundary": ("Unknown boundary", "Evidence limit", "Unsupported field"),
                "situation": ("Situation", "Evidence state", "Record status"),
                "rule": ("Rule", "Grounding rule", "Evidence constraint"),
            },
            "situation": {
                "grounded": (
                    "Source {source[code]} answers the documented part while leaving {unknown[subject]} unresolved.",
                    "The supplied record supports the documented part but not {unknown[subject]}.",
                    "The evidence establishes the documented part and leaves {unknown[subject]} unknown.",
                ),
            },
        }
    )
