# Mutter

Local-first voice assistant. Menu bar app on Mac, CLI everywhere else.

## Architecture

Two-layer system:
1. **Client layer**: Menu bar app (rumps) or CLI that records audio and sends to server
2. **Server layer**: FastAPI that receives audio/text, runs Whisper, routes through LLM, stores to ChromaDB/SQLite

The LLM (LFM 2.5) runs externally via LM Studio (local) or Groq API (cloud). Mutter calls it through an OpenAI-compatible API. The task agent uses Groq function calling for tool use.

## Project Structure

```
mutter/
├── server/
│   ├── main.py              # FastAPI app, lifespan, endpoints
│   ├── router.py            # 3-way classification (TASK/NOTE/QUERY)
│   ├── tasks.py             # task extraction + SQLite CRUD
│   ├── notes.py             # note cleanup + ChromaDB storage
│   ├── query.py             # KB search + LLM answer synthesis
│   ├── agent.py             # tool-calling agent loop (Groq)
│   ├── tools.py             # tool definitions + executor
│   ├── llm.py               # LM Studio / Groq client, per-agent model selection
│   ├── whisper_client.py    # MLX Whisper / faster-whisper
│   └── config.py            # .env + models.yaml + soul.md loading
├── client/
│   ├── menubar.py           # rumps menu bar app
│   ├── recorder.py          # audio recording (sounddevice)
│   └── cli.py               # CLI commands (typer)
├── models.yaml              # per-agent model configuration
├── soul.md                  # user context for agent prompts
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── pyproject.toml
├── README.md
└── data/                    # gitignored, runtime data
    └── mutter.db
```

## Commands

```bash
# Native mode (Mac)
pip install -e ".[mac,dev]"
uvicorn server.main:app              # start FastAPI server
python -m client.menubar             # start menu bar app
mutter tasks                         # list tasks
mutter ask "..."                     # query KB
mutter agent "remind me at 3pm"      # talk to the agent

# Docker mode
docker-compose up                    # starts app + ChromaDB
```

## Agent System

The task agent (`server/agent.py`) uses Groq function calling with tools defined in `server/tools.py`:
- `create_task` — create a task with description, due date, priority
- `set_alarm` — set a timed reminder (stored as high-priority task)
- `list_tasks` — get current tasks
- `complete_task` — mark a task as done
- `search_notes` — semantic search over saved notes
- `save_note` — save a note to ChromaDB

The agent reads `soul.md` for user context and `models.yaml` for model selection.

## Configuration

- **`.env`** — environment variables (LLM provider, API keys, server settings)
- **`models.yaml`** — which model each agent uses (router, task_agent, note_cleanup, query)
- **`soul.md`** — user context injected into agent prompts

## Conventions

- Python 3.11+
- Type hints everywhere
- pydantic-settings for config (.env)
- OpenAI-compatible API client for LLM calls (works with LM Studio and Groq)
- No comments unless explaining a non-obvious WHY
- Keep modules small and focused — one responsibility per file
- All LLM prompts live in the module that uses them, not in a separate prompts file

## Key Design Decisions

- **Menu bar + server split**: the menu bar app is a thin client. All logic lives in the FastAPI server. This means the CLI and menu bar share the same backend.
- **LLM is external**: Mutter doesn't bundle or manage the LLM. It calls LM Studio or Groq via API. This keeps the codebase simple.
- **Per-agent model config**: `models.yaml` lets each agent use a different model/provider. The task agent uses Groq for function calling; the router can use a local model.
- **soul.md**: user context file injected into agent prompts. Edit it to change how the agent behaves.
- **ChromaDB in Docker always**: even in native mode, ChromaDB runs in Docker.
- **SQLite for tasks**: tasks are simple structured data. Alarms are tasks with a due time.
- **No web UI**: CLI + menu bar is the interface. A web dashboard is post-MVP.
