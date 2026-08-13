from __future__ import annotations


WRITING_CASES = (
    ("operations", "maintenance log", "service record", "replace the loose guard", "secure the unstable guard", "Tuesday noon", "midday Tuesday", "the fan can be tested safely", "the fan test can proceed without exposing staff to the loose part"),
    ("operations", "room schedule", "venue calendar", "resolve the duplicate booking", "remove the room conflict", "Friday morning", "Friday before noon", "both workshops have a confirmed location", "each workshop has a definite room"),
    ("logistics", "delivery record", "shipment entry", "correct the depot code", "replace the incorrect depot identifier", "the next dispatch", "before the next shipment leaves", "the parcel follows the verified route", "the parcel is sent along the confirmed route"),
    ("logistics", "stock report", "inventory summary", "reconcile the missing filters", "account for the unlisted filters", "today's close", "before business closes today", "the order uses an accurate balance", "the next order is based on the true stock level"),
    ("training", "login guide", "access instructions", "replace the outdated steps", "update the obsolete procedure", "the next session", "before the next class", "new staff can activate their accounts", "new employees can complete account activation"),
    ("training", "workshop plan", "session outline", "add a practice interval", "include time for guided practice", "Wednesday", "by Wednesday", "participants can apply the demonstration", "attendees can try the demonstrated task themselves"),
    ("research", "methods note", "procedure description", "name the calibration procedure", "identify the calibration method", "the review meeting", "before the review meeting", "another researcher can repeat the measurement", "a second researcher can reproduce the reading"),
    ("research", "survey report", "study write-up", "separate quotations from interpretations", "distinguish participant words from analysis", "Monday afternoon", "by Monday afternoon", "readers can trace conclusions to evidence", "readers can connect each finding to its supporting material"),
    ("finance", "expense claim", "reimbursement request", "attach the readable receipt", "include a legible copy of the receipt", "month-end review", "before the month-end review", "the reimbursement has documented support", "the payment request has clear evidence"),
    ("finance", "budget sheet", "cost forecast", "update the supplier estimate", "replace the old vendor price", "Thursday", "by Thursday", "the total reflects the current price", "the forecast uses current pricing"),
    ("software", "release note", "deployment notice", "state the rollback condition", "describe when to restore the previous version", "approval", "before approval", "operators know when to restore the prior version", "operators can recognize when a rollback is required"),
    ("software", "issue ticket", "defect report", "include the reproduction steps", "add the sequence that triggers the problem", "triage", "before triage", "the maintainer can observe the failure", "the maintainer can reproduce the defect"),
    ("community", "event notice", "visitor announcement", "clarify the accessible entrance", "identify the step-free entrance", "publication", "before publication", "visitors know the route before arriving", "attendees can plan an accessible route in advance"),
    ("community", "volunteer reminder", "shift message", "specify the check-in location", "name the arrival point", "the morning shift", "before the morning shift", "everyone arrives at the same door", "all volunteers report to one entrance"),
    ("communications", "service alert", "incident message", "identify the affected service", "name the unavailable product", "immediate release", "before the notice goes out", "readers understand the impact", "readers know which service is affected"),
    ("communications", "correction notice", "public correction", "name the original error", "state what was wrong in the earlier version", "today", "before the end of today", "the public can see what changed", "readers can distinguish the correction from the old claim"),
)


WRITING_AUDIENCES = (
    ("The note is for the morning shift.", "for the morning shift"),
    ("A new team member will read it first.", "for a first-time team member"),
    ("The project owner needs it for a decision.", "for the project owner's decision"),
    ("External partners will rely on the wording.", "for external partners"),
    ("A non-specialist reader needs to understand it.", "for a non-specialist reader"),
    ("The next duty manager will use it during handoff.", "for the next duty manager"),
    ("The whole working group will receive it.", "for the full working group"),
    ("The recipient has only a minute to read it.", "for a time-limited recipient"),
)


WRITING_CHANNELS = (
    ("It will be sent in the shared chat.", "in the shared chat"),
    ("It belongs in a short email.", "in a short email"),
    ("It will appear in the task tracker.", "in the task tracker"),
    ("It is going into the handoff document.", "in the handoff document"),
)


__all__ = ("WRITING_AUDIENCES", "WRITING_CASES", "WRITING_CHANNELS")
