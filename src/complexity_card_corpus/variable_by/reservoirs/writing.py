from __future__ import annotations


def writing_cards(
    domain: str,
    *,
    code: str,
    owner: str,
    day: int,
) -> tuple[str, tuple[str, ...]]:
    """Return source notes and faithful rewrite alternatives for one domain."""

    cards = {
        "email": (
            f"notes {code}: send team; review complete; two figures need captions; {owner} owns them; target day {day}; release waits",
            (
                f"Subject: Review {code} next steps\n\nThe review is complete. Two figures still need captions, which {owner} owns for day {day}. Whether to release has not been decided.",
                f"Subject: Remaining work for {code}\n\nReview has finished, but captions are outstanding on two figures. {owner} will complete them before day {day} ends; no release decision has been made.",
                f"Subject: {code} review status\n\nTwo figure captions remain after the completed review, with {owner} responsible through day {day}. Release is still awaiting a decision.",
            ),
        ),
        "project_update": (
            f"update {code}: review complete; captions missing on two figures; owner {owner}; target day {day}; release decision blocked",
            (
                f"Project update {code}: Review is complete. Remaining work: {owner} adds captions to two figures, with completion expected day {day}. Blocker: approval for release is still open.",
                f"Update {code}: The review has finished; two figures still require captions from {owner} no later than day {day}. Release cannot be decided until that work is addressed.",
                f"Status for {code}: Two missing figure captions are the remaining task after review. {owner} owns delivery for day {day}, and the release decision is open.",
            ),
        ),
        "support_reply": (
            f"case {code}: issue reviewed; two screenshots need labels; {owner} will add them before day {day} ends; resolution waits for review",
            (
                f"Support reply {code}: We have completed the issue review. {owner} will label the two remaining screenshots ahead of day {day}. We will confirm resolution after that review.",
                f"Case {code} has been reviewed, with labels still missing from two screenshots. {owner} will add them no later than day {day}; resolution awaits the subsequent check.",
                f"We finished reviewing support case {code}. Two screenshots now need labels from {owner} before day {day}, after which the resolution can be confirmed.",
                f"Review of {code} is complete except for two unlabeled screenshots. {owner} owns that work through day {day}, and confirmation will follow its review.",
            ),
        ),
        "meeting_notes": (
            f"meeting {code}: review complete; two captions outstanding; {owner}; day {day}; no release decision yet",
            (
                f"Meeting {code} — Decision: review complete. Action: {owner} adds two captions, day {day} at the latest. Open item: no release decision has been made.",
                f"Notes for {code}: The review is finished. {owner} owns the two outstanding captions and must complete them before the day {day} cutoff; release remains undecided.",
                f"Meeting record {code}: Review completion was confirmed, while two captions were assigned to {owner} for day {day}. The group did not decide on release.",
            ),
        ),
        "technical_explanation": (
            f"draft {code}: validation complete; two diagrams lack captions; {owner} adds them no later than day {day}; publication waits",
            (
                f"Technical note {code}: Validation is complete, but two diagrams still lack captions. {owner} will add them, deadline day {day}; publication timing remains undecided until they are reviewed.",
                f"Validation for {code} has finished. Two diagram captions remain assigned to {owner} through day {day}, and publication cannot yet be scheduled.",
                f"The {code} draft passed validation with two captions still absent from its diagrams. {owner} will supply them before day {day} ends; publication timing stays open pending review.",
                f"Technical record {code} confirms validation is done. Captioning two diagrams belongs to {owner} before day {day}, while the publication date remains unresolved.",
                f"Two uncaptained diagrams are the only recorded follow-up after validating {code}. {owner} is due to address them on day {day}; no publishing schedule is confirmed.",
                f"For {code}, validation has concluded and two diagram captions remain. {owner} owns completion through day {day}, with release timing still unspecified.",
            ),
        ),
        "public_notice": (
            f"notice {code}: east entrance closed day {day}; inspection; use west entrance; {owner} posts signs; reopening not confirmed",
            (
                f"Public notice {code}: The east entrance will be closed for inspection starting day {day}. Please use the west entrance. {owner} will post directions; the reopening time is not yet confirmed.",
                f"Beginning day {day}, inspection will close the east entrance under notice {code}. Enter through the west side and follow signs posted by {owner}; no reopening time is confirmed.",
                f"Notice {code}: Use the west entrance while the east entrance is inspected from day {day}. {owner} is responsible for signs, and the time of reopening remains unknown.",
                f"Entrance notice {code}: Inspection closes the east doors from day {day}, so visitors should follow {owner}'s signs to the west side. A reopening date has not been set.",
                f"From day {day}, the east entrance is unavailable during inspection. Use the west entrance indicated by {owner}; notice {code} gives no confirmed end time.",
                f"The east side closes for inspection on day {day} under notice {code}. {owner} will direct access through the west entrance until reopening is confirmed.",
                f"Access update {code}: Enter on the west while inspection is under way at the east entrance beginning day {day}. {owner} will post directions, and reopening is unscheduled.",
                f"Inspection starts at the east entrance on day {day}. Follow the west-entrance route signed by {owner}; the closure length remains unconfirmed in notice {code}.",
                f"Notice {code} directs visitors to the west entrance from day {day} because the east entrance is being inspected. {owner} owns signage; no return-to-service time is available.",
            ),
        ),
        "handover_note": (
            f"handover {code}: source review done; two tables pending; {owner} owns checks day {day}; export not started",
            (
                f"Handover {code}: Source review is complete. {owner} will check the two pending tables, cutting off at day {day}. The export has not started.",
                f"For handover {code}, source review has finished and two tables still await checks from {owner} before the day {day} deadline. Export work is not yet under way.",
                f"The sources for {code} are reviewed. {owner} owns validation of the two remaining tables through day {day}; the export remains unstarted.",
            ),
        ),
        "schedule_change": (
            f"schedule {code}: review moved from day {day - 1} to day {day}; room unchanged; {owner} confirms attendees; reason not provided",
            (
                f"Schedule change {code}: The review has moved from day {day - 1} to day {day}; the room is unchanged. {owner} will confirm attendance. No reason for the change was provided.",
                f"The {code} review now takes place on day {day}, one day later than planned, in the same room. {owner} will check attendance; the source gives no reason for rescheduling.",
                f"For {code}, keep the room but replace day {day - 1} with day {day}. Attendance confirmation belongs to {owner}, and the cause of the change is not documented.",
            ),
        ),
        "feedback_message": (
            f"feedback {code}: summary accurate; main decision appears after background; ask {owner} to move it first before day {day} ends; no content change",
            (
                f"Feedback {code}: The summary is accurate, but the main decision appears after the background. {owner}, please move the decision to the opening, aiming for day {day}, without changing the content.",
                f"The content in {code} is accurate, although background currently precedes the main decision. By day {day}, {owner} should move that decision first without editing its substance.",
                f"Revision note {code}: Preserve every claim and bring the main decision ahead of the background. {owner} owns this ordering change for day {day}.",
            ),
        ),
        "procedure_summary": (
            f"procedure {code}: preserve original; duplicate file; {owner} validates copy day {day}; publish only after match; fallback unspecified",
            (
                f"Procedure {code}: Preserve the original, duplicate the file, and have {owner} validate the copy, day {day} being the last acceptable date. Publish only after both versions match; no fallback is specified.",
                f"Keep the original file unchanged and let {owner} validate a duplicate no later than day {day}. Publication can proceed after the two versions match; the source names no fallback.",
                f"Work from a duplicate while preserving the original, with {owner} completing validation no later than day {day}. Wait to publish until the comparison succeeds, and do not invent a fallback procedure.",
                f"Have {owner} check a copied file against the preserved original before day {day} closes. A matching result is required before publication; fallback handling remains unspecified.",
            ),
        ),
        "event_invitation": (
            f"invite {code}: open review session; team audience; room {day}; 15:00 day {day}; reply to {owner} before day {day - 2} ends; agenda pending",
            (
                f"Invitation {code}: Join the team review session in Room {day} at 15:00 on day {day}. Please reply to {owner} no later than day {day - 2}. The agenda is still pending.",
                f"The team review for {code} is scheduled in Room {day} at 15:00 on day {day}. Confirm with {owner} no later than day {day - 2}; an agenda has not yet been issued.",
                f"You are invited to review {code} at 15:00 on day {day}, in Room {day}. Send your response to {owner} before day {day - 2} closes; the agenda remains forthcoming.",
            ),
        ),
        "progress_brief": (
            f"brief {code}: 8 of 10 records checked; two awaiting sources; {owner} requests them day {day}; final count pending",
            (
                f"Progress brief {code}: Eight of ten records are checked. Two still await source documents, which {owner} will request, with day {day} marking the deadline. The final count remains pending.",
                f"Progress brief {code}: Review is complete for eight of the ten records. {owner} must request the two missing source documents no later than day {day}; until they arrive, the total cannot be finalized.",
                f"Progress brief {code}: The review stands at eight completed records out of ten. Two records lack source documents, which {owner} will request no later than day {day}, so the final count remains open.",
                f"Progress brief {code}: Two of ten records are still waiting on source material after eight checks were completed. {owner} owns those requests through day {day}, and no final total is available yet.",
            ),
        ),
    }
    source, rewrites = cards[domain]
    return source, (rewrites,) if isinstance(rewrites, str) else rewrites
