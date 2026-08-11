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
            f"Subject: Review {code} next steps\n\nThe review is complete. Two figures still need captions, which {owner} owns for day {day}. The release decision remains pending.",
        ),
        "project_update": (
            f"update {code}: review complete; captions missing on two figures; owner {owner}; target day {day}; release decision blocked",
            f"Project update {code}: Review is complete. Remaining work: {owner} adds captions to two figures, with completion expected day {day}. Blocker: the release decision remains pending.",
        ),
        "support_reply": (
            f"case {code}: issue reviewed; two screenshots need labels; {owner} will add them by day {day}; resolution waits for review",
            f"Support reply {code}: We have completed the issue review. {owner} will label the two remaining screenshots ahead of day {day}. We will confirm resolution after that review.",
        ),
        "meeting_notes": (
            f"meeting {code}: review complete; two captions outstanding; {owner}; day {day}; no release decision yet",
            f"Meeting {code} — Decision: review complete. Action: {owner} adds two captions, day {day} at the latest. Open item: no release decision has been made.",
        ),
        "technical_explanation": (
            f"draft {code}: validation complete; two diagrams lack captions; {owner} adds them by day {day}; publication waits",
            f"Technical note {code}: Validation is complete, but two diagrams still lack captions. {owner} will add them, deadline day {day}; publication timing remains undecided until they are reviewed.",
        ),
        "public_notice": (
            f"notice {code}: east entrance closed day {day}; inspection; use west entrance; {owner} posts signs; reopening not confirmed",
            f"Public notice {code}: The east entrance will be closed for inspection starting day {day}. Please use the west entrance. {owner} will post directions; the reopening time is not yet confirmed.",
        ),
        "handover_note": (
            f"handover {code}: source review done; two tables pending; {owner} owns checks day {day}; export not started",
            f"Handover {code}: Source review is complete. {owner} will check the two pending tables, cutting off at day {day}. The export has not started.",
        ),
        "schedule_change": (
            f"schedule {code}: review moved from day {day - 1} to day {day}; room unchanged; {owner} confirms attendees; reason not provided",
            f"Schedule change {code}: The review has moved from day {day - 1} to day {day}; the room is unchanged. {owner} will confirm attendance. No reason for the change was provided.",
        ),
        "feedback_message": (
            f"feedback {code}: summary accurate; main decision appears after background; ask {owner} to move it first by day {day}; no content change",
            f"Feedback {code}: The summary is accurate, but the main decision appears after the background. {owner}, please move the decision to the opening, aiming for day {day}, without changing the content.",
        ),
        "procedure_summary": (
            f"procedure {code}: preserve original; duplicate file; {owner} validates copy day {day}; publish only after match; fallback unspecified",
            (
                f"Procedure {code}: Preserve the original, duplicate the file, and have {owner} validate the copy, day {day} being the last acceptable date. Publish only after both versions match; no fallback is specified.",
                f"Keep the original file unchanged and let {owner} validate a duplicate by day {day}. Publication can proceed after the two versions match; the source names no fallback.",
                f"Work from a duplicate while preserving the original, with {owner} completing validation no later than day {day}. Wait to publish until the comparison succeeds, and do not invent a fallback procedure.",
                f"Have {owner} check a copied file against the preserved original by day {day}. A matching result is required before publication; fallback handling remains unspecified.",
            ),
        ),
        "event_invitation": (
            f"invite {code}: open review session; team audience; room {day}; 15:00 day {day}; reply to {owner} by day {day - 2}; agenda pending",
            f"Invitation {code}: Join the team review session in Room {day} at 15:00 on day {day}. Please reply to {owner} by day {day - 2}. The agenda is still pending.",
        ),
        "progress_brief": (
            f"brief {code}: 8 of 10 records checked; two awaiting sources; {owner} requests them day {day}; final count pending",
            (
                f"Progress brief {code}: Eight of ten records are checked. Two still await source documents, which {owner} will request, with day {day} marking the deadline. The final count remains pending.",
                f"Progress brief {code}: Review is complete for eight of the ten records. {owner} must request the two missing source documents by day {day}; until they arrive, the total cannot be finalized.",
                f"Progress brief {code}: The review stands at eight completed records out of ten. Two records lack source documents, which {owner} will request no later than day {day}, so the final count remains open.",
                f"Progress brief {code}: Two of ten records are still waiting on source material after eight checks were completed. {owner} owns those requests through day {day}, and no final total is available yet.",
            ),
        ),
    }
    source, rewrites = cards[domain]
    return source, (rewrites,) if isinstance(rewrites, str) else rewrites
