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
        "restore the original hub arrangement and stop with both observations intact",
    ),
    "web_form": (
        "a current browser",
        "submission returns HTTP 422",
        "a required profile field was removed",
        (
            "Duplicate the draft in {scope}. If the required field's previous value is "
            "documented, restore it only in the duplicate; otherwise stop and request the value"
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
            "Inspect the total formula in {scope} and compare its referenced range with "
            "control {code}; adjust only the copied workbook if the inserted row is excluded"
        ),
        "discard the workbook copy and preserve the original values and formulas",
    ),
}


def troubleshooting_cards(domain: str) -> tuple[str, str, str, tuple[str, ...], str]:
    """Return localized diagnostic cells for one troubleshooting domain."""

    return _ERRORS[domain]
