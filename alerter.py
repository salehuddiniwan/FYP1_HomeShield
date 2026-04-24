"""
HomeShield Alerter — WhatsApp notifications via Twilio with snapshot images.

Optimisations over previous version:
  * `alert_log` is now a bounded deque (max 500 entries) instead of a
    list that grew forever — prevents a slow memory leak when the
    system runs for days.
"""
import os
import time
import threading
from collections import deque
from datetime import datetime
from config import Config

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("[WARN] twilio not installed — WhatsApp alerts will be logged only")


_ALERT_LOG_MAX = 500
_SEVERE_TYPES = {"fall_detected", "lying_motionless", "intruder_detected"}
_EVENT_LABELS = {
    "fall_detected":      "FALL DETECTED",
    "lying_motionless":   "PERSON LYING MOTIONLESS",
    "inactivity":         "PROLONGED INACTIVITY",
    "zone_entry":         "CHILD ENTERED RESTRICTED ZONE",
    "intruder_detected":  "INTRUDER DETECTED",
}


class Alerter:
    """Manages alert dispatch with cooldown to prevent spam."""

    def __init__(self):
        self.twilio_client = None
        self.last_alert_time = {}               # event_type:camera_id -> ts
        self.alert_log = deque(maxlen=_ALERT_LOG_MAX)

        if TWILIO_AVAILABLE and Config.TWILIO_ACCOUNT_SID and Config.TWILIO_AUTH_TOKEN:
            try:
                self.twilio_client = TwilioClient(
                    Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN
                )
                print("[INFO] Twilio WhatsApp client initialized")
            except Exception as e:
                print(f"[WARN] Twilio init failed: {e}")

    def should_alert(self, event_type, camera_id):
        """Cooldown check. Returns True if enough time has passed."""
        key = f"{event_type}:{camera_id}"
        last = self.last_alert_time.get(key, 0)
        return (time.time() - last) > Config.ALERT_COOLDOWN_SECONDS

    def send_alert(self, event_type, camera_name, person_category, confidence,
                   snapshot_path=None, camera_id=None, details=""):
        """Send WhatsApp alert. Non-blocking (threaded)."""

        self.last_alert_time[f"{event_type}:{camera_id}"] = time.time()

        severity = "CRITICAL" if event_type in _SEVERE_TYPES else "WARNING"
        label    = _EVENT_LABELS.get(event_type, event_type.upper())
        icon     = "🚨" if severity == "CRITICAL" else "⚠️"

        msg_parts = [
            f"{icon} *HomeShield Alert*\n",
            f"*{severity}: {label}*\n",
            f"📍 Camera: {camera_name}",
            f"👤 Person: {person_category}",
            f"📊 Confidence: {confidence:.0%}",
            f"🕐 Time: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}",
        ]
        if details:
            msg_parts.append(f"ℹ️ Details: {details}")
        msg_parts.append("\n_Please check on your family member immediately._")
        msg = "\n".join(msg_parts)

        alert_info = {
            "event_type":      event_type,
            "camera_name":     camera_name,
            "person_category": person_category,
            "confidence":      round(confidence, 2),
            "snapshot_path":   snapshot_path,
            "details":         details,
            "timestamp":       datetime.now().isoformat(),
            "sent":            False,
        }

        threading.Thread(
            target=self._dispatch,
            args=(msg, snapshot_path, alert_info),
            daemon=True,
        ).start()

        return alert_info

    def _dispatch(self, message, snapshot_path, alert_info):
        """Actually send via Twilio (runs in a background thread)."""
        if not Config.ALERT_PHONE_NUMBERS:
            print(f"[ALERT] No phone numbers configured. Message:\n{message}")
            self.alert_log.append(alert_info)
            return

        if self.twilio_client is None:
            print(f"[ALERT] Twilio not configured. Message:\n{message}")
            self.alert_log.append(alert_info)
            return

        any_sent = False
        for phone in Config.ALERT_PHONE_NUMBERS:
            try:
                to = phone if phone.startswith("whatsapp:") else f"whatsapp:{phone}"
                kwargs = {
                    "body":  message,
                    "from_": Config.TWILIO_WHATSAPP_FROM,
                    "to":    to,
                }

                # Twilio requires a publicly reachable URL for media, so
                # we keep the snapshot local (see README for ngrok setup).
                if snapshot_path and os.path.exists(snapshot_path):
                    pass  # placeholder for media_url when a public URL exists

                self.twilio_client.messages.create(**kwargs)
                any_sent = True
                print(f"[ALERT] WhatsApp sent to {phone}")

            except Exception as e:
                print(f"[ERROR] WhatsApp send failed to {phone}: {e}")

        alert_info["sent"] = any_sent
        self.alert_log.append(alert_info)

    def get_recent_alerts(self, count=20):
        # deque supports slicing via list() conversion of the tail
        return list(self.alert_log)[-count:]
