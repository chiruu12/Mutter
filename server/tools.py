import json
from datetime import datetime, timedelta, timezone

from server.alarms import AlarmStore
from server.notes import NoteStore
from server.tasks import TaskStore

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a new task or reminder for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short, actionable task description.",
                    },
                    "due": {
                        "type": "string",
                        "description": "When it's due. Natural language like 'tomorrow morning', 'Friday 3pm', 'in 2 hours'. Null if no deadline.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "high = today/tomorrow, medium = this week, low = whenever.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm or timed reminder that fires after a delay.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What to remind the user about.",
                    },
                    "hours": {
                        "type": "string",
                        "description": "Hours from now as a number. Default 0.",
                    },
                    "minutes": {
                        "type": "string",
                        "description": "Minutes from now as a number. Default 0.",
                    },
                    "seconds": {
                        "type": "string",
                        "description": "Seconds from now as a number. Default 0.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "Get the user's current tasks. Use this to check what they have on their plate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_done": {
                        "type": "boolean",
                        "description": "Whether to include completed tasks.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark a task as done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID of the task to complete.",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search the user's saved notes by meaning. Use when the user references something they said before.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "How many results to return. Default 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note to the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note content to save.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_alarms",
            "description": "Get the user's pending alarms.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_alarm",
            "description": "Cancel a pending alarm by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alarm_id": {
                        "type": "integer",
                        "description": "ID of the alarm to cancel.",
                    },
                },
                "required": ["alarm_id"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(self, tasks: TaskStore, notes: NoteStore, alarms: AlarmStore) -> None:
        self.tasks = tasks
        self.notes = notes
        self.alarms = alarms

    def execute(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid arguments for {name}"})

        try:
            return self._dispatch(name, args)
        except Exception as e:
            return json.dumps({"error": f"{name} failed: {e}"})

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "create_task":
            desc = args.get("description")
            if not desc:
                return json.dumps({"error": "description is required"})
            task = self.tasks.add_task(
                description=desc,
                due=args.get("due"),
                priority=args.get("priority", "medium"),
            )
            return json.dumps({"id": task.id, "description": task.description, "due": task.due, "priority": task.priority})

        elif name == "set_alarm":
            desc = args.get("description")
            if not desc:
                return json.dumps({"error": "description is required"})
            try:
                hours = int(args.get("hours") or 0)
                minutes = int(args.get("minutes") or 0)
                seconds = int(args.get("seconds") or 0)
            except (TypeError, ValueError):
                return json.dumps({"error": "hours/minutes/seconds must be integers"})
            total_seconds = hours * 3600 + minutes * 60 + seconds
            if total_seconds <= 0:
                return json.dumps({"error": "alarm must be in the future (set hours, minutes, or seconds)"})
            now = datetime.now(timezone.utc).astimezone()
            fire_at = (now + timedelta(seconds=total_seconds)).isoformat()
            parts = []
            if hours:
                parts.append(f"{hours}h")
            if minutes:
                parts.append(f"{minutes}m")
            if seconds:
                parts.append(f"{seconds}s")
            label = "in " + " ".join(parts)
            try:
                alarm = self.alarms.add_alarm(description=desc, fire_at=fire_at, label=label)
            except ValueError as e:
                return json.dumps({"error": str(e)})
            return json.dumps({"id": alarm.id, "alarm": alarm.fire_at, "label": label, "description": desc})

        elif name == "list_tasks":
            task_list = self.tasks.list_tasks(include_done=args.get("include_done", False))
            return json.dumps([t.model_dump() for t in task_list])

        elif name == "complete_task":
            task_id = args.get("task_id")
            if task_id is None:
                return json.dumps({"error": "task_id is required"})
            success = self.tasks.complete_task(int(task_id))
            return json.dumps({"completed": success, "task_id": task_id})

        elif name == "search_notes":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "query is required"})
            results = self.notes.search(query, n_results=args.get("n_results", 5))
            return json.dumps([{"id": n.id, "content": n.content, "created_at": n.created_at} for n in results])

        elif name == "save_note":
            content = args.get("content", "")
            if not content:
                return json.dumps({"error": "content is required"})
            note = self.notes.store_raw(content)
            return json.dumps({"id": note.id, "content": note.content})

        elif name == "list_alarms":
            pending = self.alarms.list_pending()
            return json.dumps([a.model_dump() for a in pending])

        elif name == "cancel_alarm":
            alarm_id = args.get("alarm_id")
            if alarm_id is None:
                return json.dumps({"error": "alarm_id is required"})
            success = self.alarms.cancel_alarm(int(alarm_id))
            return json.dumps({"cancelled": success, "alarm_id": alarm_id})

        return json.dumps({"error": f"Unknown tool: {name}"})
