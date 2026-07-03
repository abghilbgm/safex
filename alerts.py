"""
SafeX - Alert System
Multi-channel alerting for PPE violations: Telegram, Email, Webhook.
Includes cooldown logic to prevent alert flooding.
"""

import os
import time
import json
import logging
import smtplib
import requests
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from threading import Lock

from utils import Violation, format_timestamp, setup_logger

logger = setup_logger("safex.alerts")


class AlertCooldown:
    """
    Manages alert cooldown to prevent flooding.
    Tracks last alert time per camera and per violation type.
    """

    def __init__(self, cooldown_seconds: int = 60):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: Dict[str, float] = {}
        self._lock = Lock()

    def can_alert(self, key: str) -> bool:
        """Check if enough time has passed since last alert for this key."""
        with self._lock:
            last = self._last_alert.get(key, 0)
            now = time.time()
            if now - last >= self.cooldown_seconds:
                self._last_alert[key] = now
                return True
            return False

    def reset(self, key: str = None):
        """Reset cooldown for a key or all keys."""
        with self._lock:
            if key:
                self._last_alert.pop(key, None)
            else:
                self._last_alert.clear()


class TelegramAlert:
    """
    Send violation alerts via Telegram Bot API.
    Supports text messages and photo snapshots.
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._validate()

    def _validate(self):
        """Validate bot token."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured. Alerts disabled.")

    def send_text(self, message: str) -> bool:
        """Send a text message."""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_photo(self, image_path: str, caption: str = "") -> bool:
        """Send a photo with caption."""
        try:
            url = f"{self.base_url}/sendPhoto"
            with open(image_path, "rb") as photo:
                payload = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": "HTML"
                }
                files = {"photo": photo}
                response = requests.post(url, data=payload, files=files, timeout=30)

            if response.status_code == 200:
                logger.info("Telegram photo alert sent")
                return True
            else:
                logger.error(f"Telegram photo error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Telegram photo send failed: {e}")
            return False


class EmailAlert:
    """
    Send violation alerts via SMTP email.
    """

    def __init__(self, smtp_host: str, smtp_port: int, username: str,
                 password: str, from_addr: str, to_addrs: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send(self, subject: str, body: str, image_path: Optional[str] = None) -> bool:
        """Send an email alert with optional image attachment."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "html"))

            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header("Content-Disposition", "attachment",
                                  filename=os.path.basename(image_path))
                    msg.attach(img)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.info(f"Email alert sent to {self.to_addrs}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False


class WebhookAlert:
    """
    Send violation alerts to a webhook endpoint (Slack, Teams, custom).
    """

    def __init__(self, webhook_url: str, headers: Optional[Dict] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}

    def send(self, payload: dict) -> bool:
        """Send JSON payload to webhook."""
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            if response.status_code in (200, 201, 204):
                logger.info("Webhook alert sent")
                return True
            else:
                logger.error(f"Webhook error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")
            return False


class AlertManager:
    """
    Centralized alert manager. Routes violations to configured channels.
    Handles cooldown, formatting, and dispatch.
    """

    def __init__(self, config: dict):
        self.config = config.get("alerts", {})
        self.enabled = self.config.get("enabled", False)
        self.cooldown = AlertCooldown(
            cooldown_seconds=self.config.get("cooldown_seconds", 60)
        )

        # Initialize channels
        self.telegram: Optional[TelegramAlert] = None
        self.email: Optional[EmailAlert] = None
        self.webhook: Optional[WebhookAlert] = None

        self._init_channels()

    def _init_channels(self):
        """Initialize configured alert channels."""
        # Telegram
        tg_config = self.config.get("telegram", {})
        if tg_config.get("enabled"):
            bot_token = tg_config.get("bot_token") or os.getenv("SAFEX_TELEGRAM_BOT_TOKEN", "")
            chat_id = tg_config.get("chat_id") or os.getenv("SAFEX_TELEGRAM_CHAT_ID", "")
            if bot_token and chat_id:
                self.telegram = TelegramAlert(bot_token, chat_id)
                logger.info("Telegram alerts enabled")

        # Email
        email_config = self.config.get("email", {})
        if email_config.get("enabled"):
            self.email = EmailAlert(
                smtp_host=email_config.get("smtp_host", "smtp.gmail.com"),
                smtp_port=email_config.get("smtp_port", 587),
                username=email_config.get("username", ""),
                password=email_config.get("password") or os.getenv("SAFEX_EMAIL_PASSWORD", ""),
                from_addr=email_config.get("from_addr", ""),
                to_addrs=email_config.get("to_addrs", [])
            )
            logger.info("Email alerts enabled")

        # Webhook
        wh_config = self.config.get("webhook", {})
        if wh_config.get("enabled"):
            self.webhook = WebhookAlert(
                webhook_url=wh_config.get("url", ""),
                headers=wh_config.get("headers")
            )
            logger.info("Webhook alerts enabled")

    def send_violation_alert(
        self,
        violation: Violation,
        camera_name: str = "Unknown",
        snapshot_path: Optional[str] = None
    ):
        """
        Send alert for a violation if cooldown allows.
        
        Args:
            violation: The detected violation
            camera_name: Camera/source identifier
            snapshot_path: Path to violation snapshot image
        """
        if not self.enabled:
            return

        # Cooldown check: key = camera + missing PPE combination
        cooldown_key = f"{camera_name}_{','.join(sorted(violation.missing_ppe))}"
        if not self.cooldown.can_alert(cooldown_key):
            return

        # Format message
        message = self._format_violation_message(violation, camera_name)

        # Dispatch to all channels
        if self.telegram:
            if snapshot_path and os.path.exists(snapshot_path):
                self.telegram.send_photo(snapshot_path, caption=message)
            else:
                self.telegram.send_text(message)

        if self.email:
            subject = f"\u26a0\ufe0f SafeX Alert: {violation.severity.upper()} - {camera_name}"
            body = self._format_email_body(violation, camera_name)
            self.email.send(subject, body, snapshot_path)

        if self.webhook:
            payload = {
                "event": "ppe_violation",
                "camera": camera_name,
                "timestamp": datetime.now().isoformat(),
                "severity": violation.severity,
                "missing_ppe": violation.missing_ppe,
                "video_timestamp": format_timestamp(violation.timestamp),
                "details": violation.to_dict()
            }
            self.webhook.send(payload)

    def _format_violation_message(self, violation: Violation, camera_name: str) -> str:
        """Format violation as alert message."""
        severity_emoji = {
            "critical": "\U0001f534",
            "high": "\U0001f7e0",
            "medium": "\U0001f7e1"
        }.get(violation.severity, "\u26aa")

        missing = ", ".join(v.replace("_", " ").title() for v in violation.missing_ppe)

        return (
            f"{severity_emoji} <b>PPE VIOLATION DETECTED</b>\n\n"
            f"<b>Camera:</b> {camera_name}\n"
            f"<b>Severity:</b> {violation.severity.upper()}\n"
            f"<b>Missing PPE:</b> {missing}\n"
            f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"<b>Video Time:</b> {format_timestamp(violation.timestamp)}\n"
            f"<b>Confidence:</b> {violation.person_bbox.confidence:.0%}"
        )

    def _format_email_body(self, violation: Violation, camera_name: str) -> str:
        """Format violation as HTML email body."""
        missing = ", ".join(v.replace("_", " ").title() for v in violation.missing_ppe)
        severity_color = {
            "critical": "#FF0000",
            "high": "#FF8C00",
            "medium": "#FFD700"
        }.get(violation.severity, "#888")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {severity_color};">\u26a0\ufe0f PPE Violation Detected</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr><td><b>Camera:</b></td><td>{camera_name}</td></tr>
                <tr><td><b>Severity:</b></td><td style="color: {severity_color}; font-weight: bold;">{violation.severity.upper()}</td></tr>
                <tr><td><b>Missing PPE:</b></td><td>{missing}</td></tr>
                <tr><td><b>Detected At:</b></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td><b>Video Timestamp:</b></td><td>{format_timestamp(violation.timestamp)}</td></tr>
                <tr><td><b>Detection Confidence:</b></td><td>{violation.person_bbox.confidence:.0%}</td></tr>
            </table>
            <br>
            <p style="color: #666;">This alert was generated by SafeX PPE Monitoring System.</p>
        </body>
        </html>
        """
