import threading

import httpx
import rumps
from pynput import keyboard

from client.recorder import Recorder

SERVER = "http://127.0.0.1:7860"


class MutterApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Mutter", title="🎙")
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
        hotkeys = keyboard.GlobalHotKeys({
            "<cmd>+<shift>+m": self.toggle_record,
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
        with open(wav_path, "rb") as f:
            response = httpx.post(f"{SERVER}/process", files={"file": f})
        self._notify(response.json())

    def _notify(self, result: dict) -> None:
        intent = result.get("intent", "unknown")
        if intent == "task":
            rumps.notification("Mutter — Task Created", "", result["description"])
        elif intent == "note":
            rumps.notification("Mutter — Note Saved", "", result["content"][:100])
        elif intent == "query":
            rumps.notification("Mutter — Answer", "", result["answer"][:200])

    def show_tasks(self, _) -> None:
        try:
            response = httpx.get(f"{SERVER}/tasks")
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
            response = httpx.get(f"{SERVER}/notes")
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
            response = httpx.get(f"{SERVER}/health")
            rumps.notification("Mutter", "", f"Server: {response.json()['status']}")
        except httpx.ConnectError:
            rumps.notification("Mutter", "", "Server not running")


if __name__ == "__main__":
    MutterApp().run()
