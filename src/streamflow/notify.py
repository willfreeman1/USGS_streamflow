"""Plain-text email for the Monday job. Skips send if SMTP is not set."""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from streamflow.config import smtp_settings

logger = logging.getLogger(__name__)


def format_weekly_email(decision: dict, warnings: list[str]) -> tuple[str, str]:
    """Subject and body in ordinary words."""
    kind = decision.get("kind", "ran")
    end = decision.get("end_date", "unknown")
    gap = decision.get("recall_gap")
    gap_pts = None if gap is None else round(100 * float(gap), 1)
    if kind == "failed":
        subject = "[streamflow] Weekly run failed"
    elif kind == "replace":
        subject = f"[streamflow] Weekly run: model retrained (window ending {end})"
    elif kind == "incomplete":
        subject = f"[streamflow] Weekly run: not enough weeks yet ({end})"
    else:
        subject = f"[streamflow] Weekly run: keep the current model (window ending {end})"

    lines = [
        "Streamflow drought monitor — Monday job",
        "",
        f"Window ending: {end}",
        f"Model in use was trained through: {decision.get('installed_through', 'unknown')}",
    ]
    if decision.get("n_weeks") is not None:
        lines.append(f"Known weeks in a row in this window: {decision['n_weeks']}")
    if decision.get("locked_recall") is not None:
        lines.append(
            "Share of real droughts the model caught: "
            f"{100 * float(decision['locked_recall']):.1f}%"
        )
    if decision.get("persistence_recall") is not None:
        lines.append(
            "Share the simple “already dry this week” check caught: "
            f"{100 * float(decision['persistence_recall']):.1f}%"
        )
    if gap_pts is not None:
        lines.append(f"Lead (model minus already-dry): {gap_pts} percentage points")
        lines.append("Retrain the model if that lead falls below 10 points.")
    if decision.get("locked_false_alarm_rate") is not None:
        lines.append(
            "Share of ordinary weeks we flagged: "
            f"{100 * float(decision['locked_false_alarm_rate']):.1f}%"
        )
    if decision.get("flow_shift") is not None and decision["flow_shift"] == decision["flow_shift"]:
        lines.append(
            f"Flow mix vs 1980–1999: {float(decision['flow_shift']):.3f} "
            "(watch if above 0.20 — check that live columns are built the same way)"
        )
    lines += ["", f"Decision: {decision.get('summary', kind)}"]
    if warnings:
        lines += ["", "Warnings:"]
        lines.extend(f"- {w}" for w in warnings)
    return subject, "\n".join(lines) + "\n"


def send_email(subject: str, body: str) -> bool:
    """Send, or log and return False if mail is not configured."""
    settings = smtp_settings()
    if settings is None:
        logger.info("email skipped (ALERT_EMAIL_TO / SMTP_USER / SMTP_PASSWORD not set)")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings["from_addr"]
    msg["To"] = settings["to"]
    msg.set_content(body)
    with smtplib.SMTP(settings["host"], int(settings["port"]), timeout=60) as smtp:
        smtp.starttls()
        smtp.login(settings["user"], settings["password"])
        smtp.send_message(msg)
    logger.info("email sent to %s", settings["to"])
    return True
