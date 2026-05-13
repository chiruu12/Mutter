import json
from datetime import datetime

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
            "description": "Set an alarm or timed reminder. Use this when the user wants to be reminded at a specific time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What to remind the user about.",
                    },
                    "alarm_time": {
                        "type": "string",
                        "description": "When to trigger the alarm. Natural language like 'in 30 minutes', 'at 3pm', 'tomorrow at 9am'.",
                    },
                },
                "required": ["description", "alarm_time"],
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
]


class ToolExecutor:
    def __init__(self, tasks: TaskStore, notes: NoteStore) -> None:
        self.tasks = tasks
        self.notes = notes

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
            task = self.tasks.add_task(
                description=args.get("description", "Untitled task"),
                due=args.get("due"),
                priority=args.get("priority", "medium"),
            )
            return json.dumps({"id": task.id, "description": task.description, "due": task.due, "priority": task.priority})

        elif name == "set_alarm":
            desc = args.get("description", "Alarm")
            alarm_time = args.get("alarm_time", "soon")
            task = self.tasks.add_task(
                description=f"[ALARM] {desc}",
                due=alarm_time,
                priority="high",
            )
            return json.dumps({"id": task.id, "alarm": alarm_time, "description": desc})

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

        return json.dumps({"error": f"Unknown tool: {name}"})
