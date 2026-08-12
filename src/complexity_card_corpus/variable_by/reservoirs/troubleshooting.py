from __future__ import annotations

_ERRORS = {
    "software_install": (
        "macOS 15",
        "installer exits with code 73",
        "the install directory was changed",
        (
            "Compare the installer settings with control run {code}. If the previous "
            "install directory is documented, select it in {scope}; otherwise stop and "
            "request that value"
        ),
        "discard the test configuration and leave the original installer settings unchanged",
    ),
    "network_connection": (
        "a laptop on Wi-Fi",
        "requests time out after DNS lookup",
        "a custom DNS server was enabled",
        (
            "Compare the DNS settings with control run {code}. If the previous resolver "
            "is documented, select it in {scope}; otherwise stop and request that value"
        ),
        "discard the test profile and leave the original network settings unchanged",
    ),
    "file_sync": (
        "a desktop sync client",
        "local changes remain queued",
        "the remote folder was renamed",
        (
            "Read the remote folder listing without modifying it. If the former folder "
            "name is still present, point a disposable sync profile at it; otherwise stop "
            "and preserve the queue"
        ),
        "discard the disposable sync profile and leave the queued changes untouched",
    ),
    "peripheral": (
        "a USB keyboard",
        "the device powers on but sends no input",
        "it was moved through a hub",
        "Bypass the hub and connect the keyboard directly for one test",
        (
            "restore the original hub arrangement and stop with both observations intact",
            "reconnect the keyboard through its original hub and retain the direct and hub-test results",
            "return to the documented hub setup without further changes, preserving both recorded outcomes",
            "undo the direct connection, restore the prior hub path, and keep the two observations for diagnosis",
        ),
    ),
    "web_form": (
        "a current browser",
        "submission returns HTTP 422",
        "a required profile field was removed",
        (
            (
                "Duplicate the draft in {scope}. If the required field's previous value is "
                "documented, restore it only in the duplicate; otherwise stop and request the value"
            ),
            (
            "Within {scope}, open a copy of the submission. Compare the required fields with "
            "control {code}, entering a missing value only when that control documents it; "
                "otherwise preserve the copy and ask for the value"
            ),
            (
                "Preserve the original draft and reproduce the HTTP 422 response from {scope}. "
                "Use the last documented profile value only in that copy, or stop if no prior "
                "value is recorded"
            ),
            (
                "In {scope}, inspect which required field is absent without submitting the original. "
                "Copy a value from control {code} only if it is explicitly recorded there; otherwise "
                "request clarification"
            ),
        ),
        "discard the duplicate draft and leave the original form unchanged",
    ),
    "data_pipeline": (
        "a nightly ETL job",
        "the transform stage reports a schema mismatch",
        "a source column changed type",
        (
            "Use {scope} with a read-only input copy. If control run {code} documents the "
            "previous column type, cast only the copy to that type; otherwise stop and "
            "request the schema"
        ),
        "discard the isolated pipeline copy and leave the source data unchanged",
    ),
    "audio_output": (
        "a laptop with external speakers",
        "the level meter moves but no sound is audible",
        "the preferred output device was changed",
        (
            "Play control tone {code} in {scope}, then select the documented speaker "
            "for that profile only and compare the level meter with the audible result"
        ),
        "restore the former output selection and retain both test observations",
    ),
    "printer_queue": (
        "a shared office printer",
        "the first queued document remains in processing",
        "a new paper-size default was selected",
        (
            "Preserve the source document, pause the queue, and send one disposable "
            "one-page file from {scope} using the paper size recorded in control {code}"
        ),
        "remove only the disposable job and leave the original document and queue intact",
    ),
    "account_login": (
        "a browser login page",
        "the password is accepted but the verification code does not arrive",
        "the notification address was edited",
        (
            "Check the masked delivery destination against control {code} without "
            "changing credentials. If it differs, correct it only through the official "
            "account recovery flow in {scope}"
        ),
        "end the recovery session without altering the current password or security factors",
    ),
    "storage_space": (
        "a user laptop",
        "the system reports less than one gigabyte free",
        "a local snapshot was created",
        (
            "Use the read-only storage summary in {scope} to compare snapshots, temporary "
            "files, and user folders with control {code}; do not delete any category"
        ),
        "close the inspection without deleting files and retain the category totals",
    ),
    "battery_charging": (
        "a tablet at room temperature",
        "the charging indicator stays off",
        "the charging cable was replaced",
        (
            "Test the documented cable from control {code} with the same power outlet, "
            "then test the replacement cable once while recording the indicator state"
        ),
        "disconnect the test cable and restore the documented charging arrangement",
    ),
    "browser_session": (
        "a signed-in web application",
        "one page reloads into a blank state",
        "a browser extension was enabled",
        (
            "Open the same page in {scope} with extensions disabled. Compare its network "
            "result with control {code} without clearing the original profile"
        ),
        "discard the test profile and preserve the original browser session",
    ),
    "email_delivery": (
        "a desktop mail client",
        "one message remains in the outbox",
        "the outgoing server port was changed",
        (
            "Compare the outgoing settings with control {code}, then send a disposable "
            "message to the sender's own address from {scope} using the documented port"
        ),
        "remove only the disposable message and restore the previous test settings",
    ),
    "spreadsheet_formula": (
        "a workbook copy",
        "the monthly total is lower than the visible line items",
        "one row was inserted above the total",
        (
            "Within {scope}, inspect the total formula. Contrast its referenced range with "
            "control {code}, adjusting only the copied workbook if the inserted row is excluded"
        ),
        "discard the workbook copy and preserve the original values and formulas",
    ),
}

_ROLLBACK_ALTERNATIVES = {
    "software_install": (
        "remove only the test setup and restore the documented installer configuration",
        "return the isolated installer profile to its former settings without touching the original installation",
        "abandon the trial configuration and preserve the pre-test installation state",
    ),
    "network_connection": (
        "delete the disposable network profile and keep the established resolver settings",
        "end the DNS trial by restoring the recorded network configuration",
        "remove only the test resolver changes while leaving the original connection untouched",
    ),
    "file_sync": (
        "remove the trial sync profile while preserving every queued local change",
        "end the isolated synchronization test and leave the pending file queue unchanged",
        "discard only the disposable profile, retaining the queued edits in their original state",
    ),
    "web_form": (
        "delete the copied draft and keep the original submission exactly as it was",
        "close the form test without applying any copied value to the source draft",
        "remove only the disposable form copy while preserving the untouched original",
    ),
    "data_pipeline": (
        "discard the read-only pipeline trial and retain the source dataset without modification",
        "end the isolated ETL run while preserving the original schema and data",
        "remove the test pipeline copy and leave every production input unchanged",
    ),
    "audio_output": (
        "return the test profile to its documented audio output and retain both meter observations",
        "undo the temporary speaker choice while preserving the audible and level-meter results",
        "restore the former output route and keep the two test records for comparison",
    ),
    "printer_queue": (
        "delete only the one-page test job and preserve the source document plus existing queue",
        "remove the disposable print request without changing the original queued work",
        "end the print trial by clearing its test page alone and retaining the prior queue state",
    ),
    "account_login": (
        "close the recovery attempt without changing the password or enrolled security methods",
        "end only the diagnostic session and retain all current credentials and factors",
        "leave account authentication unchanged when exiting the temporary recovery flow",
    ),
    "storage_space": (
        "finish the read-only review without removing data and preserve all measured category sizes",
        "close the storage inspection while leaving files untouched and retaining its totals",
        "make no deletion after the test; keep both the user data and recorded usage figures",
    ),
    "battery_charging": (
        "remove the trial cable and return the tablet to its previously documented charging setup",
        "end the cable comparison by restoring the original power arrangement",
        "disconnect only the test lead and preserve the former charger configuration",
    ),
    "browser_session": (
        "remove the isolated browser profile while keeping the signed-in session unchanged",
        "close the extension-free test profile and preserve the original application session",
        "discard only the temporary browser state without clearing the established profile",
    ),
    "email_delivery": (
        "delete the self-addressed test mail and return the outgoing settings to their recorded values",
        "remove only the disposable message while restoring the prior mail-server configuration",
        "end the delivery trial by clearing its test email and preserving the original client settings",
    ),
    "spreadsheet_formula": (
        "remove the test workbook and retain every original value and formula unchanged",
        "close the copied spreadsheet without applying its formula edit to the source file",
        "discard only the isolated workbook, preserving the original ranges and cell contents",
    ),
}

_DIAGNOSTIC_ALTERNATIVES = {
    "software_install": (
        "In {scope}, compare the install path recorded by {code} with the current selection; reuse it only when the control documents it",
        "Read the former directory from {code} and apply it only to a copied installer profile in {scope}",
        "Use {scope} to contrast the selected install location with control {code}, requesting the prior path when it is not recorded",
        "Preserve the current installer while a disposable setup in {scope} tests the directory explicitly documented by {code}",
        "Verify the install path against {code} before changing the test profile; an absent prior value ends the diagnostic",
        "Inside {scope}, reproduce the directory selection from {code} without altering the established installation",
    ),
    "network_connection": (
        "Inspect the resolver saved in control {code} and test that value only inside {scope}, stopping if the former DNS setting is absent",
        "Compare the current DNS entry with {code} from a disposable network profile in {scope}",
        "Use the resolver documented by {code} for one isolated lookup, or stop when no earlier value is available",
        "Keep the active connection unchanged while {scope} tests the control resolver recorded under {code}",
        "From {scope}, run one lookup against the DNS value in {code} and retain both outcomes for comparison",
        "Check whether control {code} supplies the former resolver before entering anything in the temporary profile",
    ),
    "file_sync": (
        "Use a disposable profile in {scope} to inspect whether control {code}'s former remote folder still exists, without changing the queued files",
        "Read the remote directory list from {scope} and compare it with {code}; attach a trial profile only to a folder whose former name is documented",
        "Preserve the pending queue while {scope} checks whether the remote name recorded in {code} is still available",
        "Create a throwaway sync profile in {scope}, point it to the documented folder from {code}, and leave the production queue disconnected",
        "Compare the current remote folder names with control {code} from {scope}; stop before linking anything when the prior name is absent",
        "Use {scope} to test the former remote path from {code} without uploading, renaming, or releasing any queued local change",
        "In {scope}, verify the folder identity against {code} first and connect only the disposable profile when both records agree",
    ),
    "peripheral": (
        "Connect the keyboard without the hub for one isolated input test, recording both the direct result and control {code}",
        "Within {scope}, test one direct keyboard connection; contrast its input record with the hub-based control {code}",
        "Remove the hub only inside {scope}, capture one keyboard-input result, and preserve {code} for comparison",
        "Inside {scope}, attach the keyboard directly once and contrast the observed input with the hub result stored in {code}",
        "Use {scope} for a single no-hub keyboard trial, keeping {code} as the unchanged baseline",
        "Run one keyboard-input check on a direct port in {scope}; record the outcome beside control {code}",
    ),
    "web_form": "Compare a copied form in {scope} with control {code}, restoring a required value only when the control explicitly supplies it",
    "data_pipeline": (
        "Run a read-only copy in {scope} against the schema documented by {code}; cast the copy only if that prior type is recorded",
        "Compare the copied column type with {code} from {scope}, changing the copy only when the control gives an exact schema",
        "Use {scope} for one read-only transform whose input type matches the value explicitly stored in {code}",
        "Preserve the source dataset while a test pipeline in {scope} checks its schema against control {code}",
        "Inspect the mismatch on a read-only input copy and request the earlier type if {code} does not document it",
        "In {scope}, reproduce the transform with the control schema from {code} without casting any production input",
    ),
    "audio_output": "Play the same tone through the output documented in {code} inside {scope}, then compare audible sound with the meter",
    "printer_queue": "Pause the queue and submit one disposable page from {scope} using the paper size stored in control {code}",
    "account_login": "Compare the masked notification destination with {code} and use only the official recovery route in {scope} if they differ",
    "storage_space": "Review category totals in {scope} beside control {code}, keeping snapshots, temporary data, and user files read-only",
    "battery_charging": "Test the cable documented by {code} and the replacement once each from the same outlet, recording the indicator both times",
    "browser_session": "Open the affected page in {scope} without extensions and compare its network record with {code}, leaving the original profile intact",
    "email_delivery": "Use the outgoing port documented by {code} for one self-addressed disposable message from {scope}, without changing the established client",
    "spreadsheet_formula": "In a workbook copy under {scope}, compare the total range with control {code} and adjust it only if the inserted row is omitted",
}


def troubleshooting_cards(
    domain: str,
) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    """Return localized diagnostic cells for one troubleshooting domain."""

    environment, error, change, diagnostic, rollback = _ERRORS[domain]
    diagnostics = (diagnostic,) if isinstance(diagnostic, str) else diagnostic
    diagnostic_alternatives = _DIAGNOSTIC_ALTERNATIVES[domain]
    if isinstance(diagnostic_alternatives, str):
        diagnostic_alternatives = (diagnostic_alternatives,)
    diagnostics = (*diagnostics, *diagnostic_alternatives)
    rollbacks = (rollback,) if isinstance(rollback, str) else rollback
    rollbacks = (*rollbacks, *_ROLLBACK_ALTERNATIVES.get(domain, ()))
    return environment, error, change, diagnostics, rollbacks


def troubleshooting_verification_cards(error: str) -> tuple[str, ...]:
    """Build varied direct-plus-regression verification statements."""

    return (
        f"Direct check: confirm that '{error}' no longer appears. Regression check: repeat the last known-good operation.",
        f"Verify the fix by checking that '{error}' is absent, then rerun the documented good case.",
        f"The direct test passes only when '{error}' disappears; the regression test must also preserve the former good behavior.",
        f"First establish that '{error}' is gone, then repeat the control operation to detect regression.",
        f"A successful repair removes '{error}' while leaving the previously working action unchanged.",
        f"Treat the change as verified only if '{error}' stops and the control behavior remains available.",
        f"Run the direct operation until '{error}' is absent; afterward, execute the former good case once.",
        f"Confirmation requires two observations: no '{error}' on the failing path and no regression on the control path.",
        f"Test the repaired route for '{error}', then compare the last known-good operation with its recorded result.",
        f"The immediate check is that '{error}' does not recur; the separate check is that prior behavior is preserved.",
        f"Observe whether '{error}' has cleared before repeating the successful control action unchanged.",
        f"Validate both sides: the reported error must vanish and the documented baseline must still pass.",
        f"Repeat the original failure test for '{error}', followed by one regression run of the known-good case.",
        f"Accept the diagnosis only when '{error}' is absent and the control operation produces its earlier outcome.",
        f"Check directly for removal of '{error}', then use the baseline action to make sure nothing else broke.",
        f"Verification consists of a clean failing-path retest plus a successful replay of the prior good behavior.",
        f"Confirm the reported route no longer yields '{error}' and that the independent control route remains stable.",
        f"Use absence of '{error}' as the direct signal, and preservation of the former operation as the regression signal.",
        f"Recreate both tests after the change: the failure should clear, while the known-good behavior should not change.",
        f"A clean result needs '{error}' to disappear without altering the outcome of the control operation.",
        f"Inspect the repaired action for the reported error, then rerun the baseline as a separate safeguard.",
        f"Prove the fix by removing '{error}' from the direct test and retaining success in the regression test.",
        f"Finish with two checks: the affected path is clear of '{error}', and the established path still works.",
        f"Record a passing outcome only after the affected action avoids '{error}' and the baseline succeeds again.",
        f"Compare two post-change observations: disappearance of '{error}' and unchanged behavior on the control route.",
        f"The repair qualifies when the former failure is absent and a separate known-good replay remains successful.",
        f"Retest the problem path for '{error}', then verify the original working path without carrying over test state.",
        f"Use one result to show the error cleared and another independent result to show no regression occurred.",
        f"Both gates must pass: the reported operation stops producing '{error}', and the control still matches its baseline.",
        f"After the change, capture evidence from the affected route and from an unchanged successful route.",
        f"Do not mark the issue fixed until a direct retest clears and a distinct regression check also passes.",
    )


def troubleshooting_failure_cards(rollback: str) -> tuple[str, ...]:
    """Build failure branches around one domain-compatible rollback."""

    leads = (
        "If either check fails", "If the direct or regression result is negative",
        "After any failed verification", "When one of the two checks does not pass",
        "If the observed result contradicts the expected outcome", "Should either test expose a problem",
        "If verification remains incomplete", "When the direct check or baseline check fails",
        "If the change does not clear both gates", "On an unsuccessful direct or regression run",
        "If either observation is worse than the control", "When the evidence does not support the repair",
        "If the reported failure remains", "If the former good behavior regresses",
        "When one result cannot be verified", "If the two checks do not both succeed",
        "After a failed comparison with the baseline", "When the bounded test produces an adverse result",
        "If the repair creates a new discrepancy", "If the control path no longer behaves as recorded",
        "When uncertainty remains after both tests", "If either success condition is absent",
        "Following any failed validation step", "If the final observations do not support the change",
    )
    return tuple(f"{lead}, {rollback}." for lead in leads)


def troubleshooting_comparison_cards(code: str) -> tuple[str, ...]:
    """Return varied controlled-comparison steps tied to one baseline log."""

    return (
        f"Repeat the failing operation in the same setup and compare the resulting log with {code}.",
        f"Run the failing action once more under the test condition, then compare both observations with {code}.",
        f"Keep every other variable fixed, repeat the operation, and inspect the difference from control {code}.",
        f"Reproduce the affected action once and contrast its evidence with baseline {code}.",
        f"Under unchanged conditions, rerun the failure and compare the new record against {code}.",
        f"Perform one controlled repetition, preserving all other inputs, then review it beside log {code}.",
        f"Repeat only the affected operation and identify what differs from the successful run {code}.",
        f"Hold the environment constant while recreating the issue, using {code} as the comparison point.",
        f"Create one fresh observation of the failure and place it alongside control evidence {code}.",
        f"Rerun the same action without extra changes, then compare its output with baseline {code}.",
        f"Use identical inputs for one reproduction and inspect the delta from control run {code}.",
        f"Repeat the reported path once, retaining its log for direct comparison with {code}.",
        f"With unrelated variables frozen, reproduce the issue and contrast both records against {code}.",
        f"Generate one bounded failure record, then check which observation differs from log {code}.",
        f"Re-execute the affected step under the same setup and compare the evidence to {code}.",
        f"Keep the test isolated, repeat the operation, and inspect its result beside control {code}.",
        f"Capture one new occurrence of the issue and compare each relevant field with {code}.",
        f"Reproduce without widening scope, then use successful record {code} to locate the changed outcome.",
        f"Repeat the exact failing path once and evaluate its observations against baseline {code}.",
        f"Run a single controlled replay and compare the resulting state with the state recorded in {code}.",
        f"Preserve all fixed inputs while retesting, then examine the new evidence relative to {code}.",
        f"Create a like-for-like repetition of the failure and contrast it with control run {code}.",
        f"Execute the affected operation once more, changing nothing else, and compare it to {code}.",
        f"Use one isolated reproduction to reveal the difference between the failure and baseline {code}.",
        f"Collect one fresh failure trace and evaluate it field by field against control {code}.",
        f"Replay only the affected path, then locate the first observation that diverges from {code}.",
        f"With the setup held constant, contrast one new failing result with the known-good evidence in {code}.",
        f"Produce a single comparable fault record and inspect where its state separates from baseline {code}.",
        f"Retest the reported operation once and use {code} to identify the earliest changed outcome.",
        f"Keep inputs identical while one bounded replay is compared directly with control run {code}.",
        f"Place a newly captured failure beside {code} and examine only the observations that differ.",
        f"Run the same path once under preserved conditions, using baseline {code} to isolate the delta.",
    )


def troubleshooting_opening_cards(code: str) -> tuple[str, ...]:
    """Return varied state-preserving openings tied to a control record."""

    return (
        f"Preserve log {code}, then reproduce once without changing user data.",
        f"Begin by retaining control log {code} and repeating the failure one time.",
        f"Record the current state beside log {code}; make no change before one controlled reproduction.",
        f"Protect the existing data and use log {code} as the comparison baseline.",
        f"Keep all user data unchanged and save {code} before one isolated replay.",
        f"Retain the current evidence under {code}, then reproduce only the reported path.",
        f"Before testing, preserve state and make control record {code} the baseline.",
        f"Store the unchanged starting condition with {code} and perform one bounded reproduction.",
        f"Use {code} to capture the pre-test state while keeping every source value intact.",
        f"First protect the original data, preserve log {code}, and repeat the issue once.",
        f"Make no broad change; retain baseline {code} and run a single controlled replay.",
        f"Save the current configuration as {code} before reproducing the affected operation.",
        f"Start from an untouched state documented in {code}, with one isolated failure test.",
        f"Preserve both data and configuration in control {code} before the first retest.",
        f"Create no new state change until log {code} records the existing condition.",
        f"Hold the original environment fixed, retain {code}, and reproduce once.",
        f"Document the starting state in {code} while leaving user-owned data unmodified.",
        f"Keep a recoverable baseline through log {code}, then test the failing action alone.",
        f"Anchor the diagnosis in preserved record {code} before repeating the problem.",
        f"Retain the known state and its log {code}; limit the next action to one replay.",
        f"Protect existing information, store baseline {code}, and run no more than one reproduction.",
        f"Use the unchanged state in {code} as the starting point for a bounded retest.",
        f"Capture all current observations in {code} before isolating the reported failure.",
        f"Preserve the pre-test condition under {code} and avoid any unrelated modification.",
        f"Before intervention, retain {code} as the unchanged reference for one bounded reproduction.",
        f"Keep the starting evidence recoverable in {code}, then replay only the failing action.",
        f"Establish {code} as the protected baseline before collecting one new failure record.",
        f"Save the observed pre-test condition under {code} and isolate a single repetition.",
        f"Use a preserved snapshot identified by {code} before testing the affected path once.",
        f"Leave source data intact, anchor the comparison in {code}, and reproduce only the reported fault.",
        f"Document an untouched baseline as {code} before running the smallest useful retest.",
        f"Secure the current state in {code}; the next operation should only recreate the known failure.",
    )


def troubleshooting_diagnostic_surfaces(
    diagnostic_step: str,
    scope: str,
) -> tuple[str, ...]:
    """Wrap a diagnostic in varied, grammatical numbered steps."""

    leads = (
        "", "Run this bounded diagnostic: ", f"In {scope}, perform this check: ",
        "Use this isolated test: ", "Next, apply the following diagnostic: ",
        "For the discriminating check, ", "Within the test boundary, ",
        "Use one controlled probe: ", "At the diagnostic stage, ",
        "To isolate the changed condition, ", "For a reversible test, ",
        "Now test the suspected interface. ", "Using only the copied state, ",
        "For the smallest useful experiment, ", "To compare cause and coincidence, ",
        "Under the preserved setup, ", "Use the control-backed diagnostic: ",
        "For one evidence-producing test, ", "Without widening the change, ",
        "At this isolated layer, ", "To test the leading explanation, ",
        "For a low-risk comparison, ", "Inside the disposable environment, ",
        "Change only one variable. Then ",
    )
    return tuple(f"2. {lead}{diagnostic_step}." for lead in leads)


def troubleshooting_first_step_surfaces(step: str) -> tuple[str, ...]:
    """Naturalize the first diagnostic step without a dominant bridge."""

    sentence_step = step[:1].upper() + step[1:]
    return (
        step, f"First, {step}", f"Start with this safeguard: {step}",
        f"Protect the current state first. {sentence_step}", f"Start here: {step}",
        f"Before testing, {step}", f"Preparation comes first. {sentence_step}",
        f"Establish a safe baseline. {sentence_step}", f"Preserve the initial condition. {sentence_step}",
        f"Begin from a recoverable state: {step}", f"Make the baseline explicit. {step}",
        f"The first action protects evidence: {step}", f"Prioritize reversibility at the outset. {step}",
        f"Open with a state-preserving step: {step}", f"Set the control boundary first. {step}",
        f"Keep the original condition available. {sentence_step}", f"The safe starting move is this: {step}",
        f"Retain a comparison point before diagnosis. {sentence_step}", f"Begin by protecting what already works. {sentence_step}",
        f"Create the rollback boundary first. {sentence_step}", f"Secure the current evidence before changing anything. {sentence_step}",
        f"Use this reversible opening step: {step}", f"Start the test from preserved state. {step}",
        f"Hold the baseline steady at the beginning. {sentence_step}",
    )
