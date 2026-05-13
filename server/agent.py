from datetime import datetime

from server.config import load_soul
from server.llm import LLMClient
from server.tools import TOOL_DEFINITIONS, ToolExecutor

AGENT_SYSTEM_PROMPT = """You are Mutter's task agent. You help the user manage tasks, set alarms, and organize their notes.

You have tools available. Use them to take action — don't just describe what you'd do.

Rules:
- When the user wants a task created, call create_task.
- When the user wants a reminder at a specific time, call set_alarm.
- When the user asks about something they said before, call search_notes.
- When the user says they're done with something, call list_tasks to find it, then complete_task.
- You can call multiple tools in sequence if needed.
- Be concise in your final response. Confirm what you did, nothing more.

Current time: {current_time}

{soul_context}"""


def _build_system_prompt() -> str:
    soul = load_soul()
    soul_section = f"User context:\n{soul}" if soul else ""
    return AGENT_SYSTEM_PROMPT.format(
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        soul_context=soul_section,
    )


def run_agent(llm: LLMClient, executor: ToolExecutor, user_message: str) -> str:
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_message},
    ]

    # max 10 tool-call rounds to prevent runaway loops
    for _ in range(10):
        response = llm.chat(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            agent="task_agent",
        )
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                result = executor.execute(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            return choice.message.content or "Done."

    return "Done."
