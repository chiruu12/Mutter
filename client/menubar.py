import logging
import os
import threading
from pathlib import Path

import httpx
import rumps
from pynput import keyboard

from client.recorder import Recorder

log = logging.getLogger("mutter.menubar")


def _safe_notify(title: str, subtitle: str, message: str) -> None:
    try:
        _safe_notify(title, subtitle, message)
    except RuntimeError:
        log.warning("[menubar] %s: %s", title, message)


def _server_url() -> str:
    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    port = os.environ.get("SERVER_PORT", "7860")
    return f"http://{host}:{port}"


def _hotkey() -> str:
    return os.environ.get("HOTKEY", "cmd+shift+m")


def _hotkey_to_pynput(hotkey: str) -> str:
    parts = hotkey.split("+")
    mapped = []
    for p in parts:
        p = p.strip().lower()
        if p in ("cmd", "command"):
            mapped.append("<cmd>")
        elif p in ("shift",):
            mapped.append("<shift>")
        elif p in ("ctrl", "control"):
            mapped.append("<ctrl>")
        elif p in ("alt", "option"):
            mapped.append("<alt>")
        else:
            mapped.append(p)
    return "+".join(mapped)


class MutterApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Mutter", title="🎙")
        self.server = _server_url()
        self.recorder = Recorder()
        self.is_recording = False
        self.menu = [
            rumps.MenuItem("Record", callback=self.toggle_record),
            None,
            rumps.MenuItem("Recent Tasks", callback=self.show_tasks),
            rumps.MenuItem("Recent Notes", callback=self.show_notes),
            None,
            rumps.MenuItem("Status", callback=self.show_status),
        ]
        self._setup_hotkey()

    def _setup_hotkey(self) -> None:
        pynput_key = _hotkey_to_pynput(_hotkey())
        hotkeys = keyboard.GlobalHotKeys({
            pynput_key: self.toggle_record,
        })
        hotkeys.start()

    def toggle_record(self, sender=None) -> None:
        if not self.is_recording:
            self.is_recording = True
            self.title = "🔴"
            self.recorder.start()
        else:
            self.is_recording = False
            self.title = "🎙"
            wav_path = self.recorder.stop_and_save()
            threading.Thread(target=self._process, args=(wav_path,)).start()

    def _process(self, wav_path: str) -> None:
        path = Path(wav_path)
        try:
            with open(path, "rb") as f:
                response = httpx.post(f"{self.server}/process", files={"file": f}, timeout=30)
            if response.status_code != 200:
                detail = ""
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    pass
                _safe_notify("Mutter — Error", "", detail or f"Server returned {response.status_code}")
                return
            self._notify(response.json())
        except (httpx.ConnectError, httpx.TimeoutException):
            _safe_notify("Mutter", "", "Server not running. Start with: mutter serve")
        except Exception as e:
            _safe_notify("Mutter — Error", "", str(e)[:200])
        finally:
            path.unlink(missing_ok=True)

    def _notify(self, result: dict) -> None:
        intent = result.get("intent", "unknown")
        if intent == "task":
            _safe_notify("Mutter — Task Created", "", result["description"])
        elif intent == "note":
            _safe_notify("Mutter — Note Saved", "", result["content"][:100])
        elif intent == "query":
            _safe_notify("Mutter — Answer", "", result["answer"][:200])

    def show_tasks(self, _) -> None:
        try:
            response = httpx.get(f"{self.server}/tasks")
            tasks = response.json()
            if tasks:
                msg = "\n".join(f"• {t['description']}" for t in tasks[:5])
            else:
                msg = "No tasks yet."
            rumps.alert("Recent Tasks", msg)
        except httpx.ConnectError:
            rumps.alert("Error", "Server not running.")

    def show_notes(self, _) -> None:
        try:
            response = httpx.get(f"{self.server}/notes")
            notes = response.json()
            if notes:
                msg = "\n".join(f"• {n['content'][:80]}" for n in notes[:5])
            else:
                msg = "No notes yet."
            rumps.alert("Recent Notes", msg)
        except httpx.ConnectError:
            rumps.alert("Error", "Server not running.")

    def show_status(self, _) -> None:
        try:
            response = httpx.get(f"{self.server}/health")
            _safe_notify("Mutter", "", f"Server: {response.json()['status']}")
        except httpx.ConnectError:
            _safe_notify("Mutter", "", "Server not running")


if __name__ == "__main__":
    MutterApp().run()
