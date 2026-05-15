import asyncio
import logging
import platform
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel

log = logging.getLogger("mutter.alarms")


class Alarm(BaseModel):
    id: int | None = None
    description: str
    label: str | None = None
    fire_at: str
    fired: bool = False
    created_at: str | None = None


class AlarmStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS alarms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    label TEXT,
                    fire_at TEXT NOT NULL,
                    fired BOOLEAN DEFAULT 0,
                    created_at TEXT NOT NULL
                )"""
            )

    def add_alarm(self, description: str, fire_at: str, label: str | None = None) -> Alarm:
        if fire_at.endswith("Z"):
            fire_at = fire_at[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(fire_at)
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 datetime: {fire_at}")
        if parsed.tzinfo is None:
            raise ValueError(f"alarm_time must include timezone offset: {fire_at}")
        now = datetime.now(parsed.tzinfo)
        if parsed <= now:
            raise ValueError(f"alarm_time is in the past: {fire_at}")
        alarm = Alarm(
            description=description,
            label=label,
            fire_at=fire_at,
            created_at=datetime.now().isoformat(),
        )
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO alarms (description, label, fire_at, fired, created_at) VALUES (?, ?, ?, ?, ?)",
                (alarm.description, alarm.label, alarm.fire_at, alarm.fired, alarm.created_at),
            )
            alarm.id = cursor.lastrowid
        log.info("[alarms] added #%d: %s at %s", alarm.id, alarm.description, alarm.fire_at)
        return alarm

    def get_due(self, now: datetime) -> list[Alarm]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alarms WHERE fired = 0",
            ).fetchall()
        due = []
        for row in rows:
            try:
                fire_at = datetime.fromisoformat(row["fire_at"])
                if fire_at.tzinfo is None:
                    fire_at = fire_at.replace(tzinfo=now.tzinfo)
                if fire_at <= now:
                    due.append(Alarm(**dict(row)))
            except ValueError:
                continue
        return due

    def list_pending(self) -> list[Alarm]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM alarms WHERE fired = 0",
            ).fetchall()
        alarms = [Alarm(**dict(row)) for row in rows]
        alarms.sort(key=lambda a: datetime.fromisoformat(a.fire_at))
        return alarms

    def mark_fired(self, alarm_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("UPDATE alarms SET fired = 1 WHERE id = ?", (alarm_id,))
            return cursor.rowcount > 0

    def cancel_alarm(self, alarm_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM alarms WHERE id = ? AND fired = 0", (alarm_id,))
            return cursor.rowcount > 0


def _fire_alarm(alarm: Alarm) -> bool:
    if platform.system() != "Darwin":
        log.info("[alarms] fired #%d: %s (no notification on %s)", alarm.id, alarm.description, platform.system())
        return True
    escaped = alarm.description.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    notify_script = (
        f'display notification "{escaped}" '
        f'with title "Mutter Alarm" '
        f'sound name "Glass"'
    )
    alert_script = (
        f'display alert "Mutter Alarm" '
        f'message "{escaped}" '
        f'giving up after 20'
    )
    try:
        subprocess.run(["osascript", "-e", notify_script], timeout=5, capture_output=True)
        subprocess.Popen(["osascript", "-e", alert_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log.info("[alarms] fired #%d: %s", alarm.id, alarm.description)
        return True
    except Exception as e:
        log.warning("[alarms] osascript failed for #%d: %s", alarm.id, e)
        return False


async def alarm_loop(store: AlarmStore, tz_name: str) -> None:
    tz = ZoneInfo(tz_name)
    log.info("[alarms] loop running, checking every 15s (tz=%s)", tz_name)
    while True:
        await asyncio.sleep(15)
        try:
            now = datetime.now(tz)
            pending = await asyncio.to_thread(store.list_pending)
            if pending:
                log.debug("[alarms] %d pending, now=%s", len(pending), now.isoformat())
                for a in pending:
                    log.debug("[alarms]   #%d fire_at=%s", a.id, a.fire_at)
            due = await asyncio.to_thread(store.get_due, now)
            if due:
                log.info("[alarms] %d alarm(s) due at %s", len(due), now.strftime("%H:%M:%S"))
            for alarm in due:
                fired = await asyncio.to_thread(_fire_alarm, alarm)
                if fired:
                    await asyncio.to_thread(store.mark_fired, alarm.id)
                else:
                    log.warning("[alarms] notification failed for #%d, will retry", alarm.id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("[alarms] loop error: %s", e)
