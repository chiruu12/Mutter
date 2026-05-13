# Mutter

Local-first voice assistant. Menu bar app on Mac, CLI everywhere else.

## Architecture

Two-layer system:
1. **Client layer**: Menu bar app (rumps) or CLI that records audio and sends to server
2. **Server layer**: FastAPI that receives audio/text, runs Whisper, routes through LLM, stores to ChromaDB/SQLite

The LLM (LFM 2.5) runs externally via LM Studio (local) or Groq API (cloud). Mutter calls it through an OpenAI-compatible API.

## Project Structure

```
mutter/
├── server/
│   ├── main.py              # FastAPI app, lifespan, endpoints
│   ├── router.py            # 3-way classification (TASK/NOTE/QUERY)
│   ├── tasks.py             # task extraction + SQLite CRUD
│   ├── notes.py             # note cleanup + ChromaDB storage
│   ├── query.py             # KB search + LLM answer synthesis
│   ├── llm.py               # LM Studio / Groq client abstraction
│   ├── whisper_client.py    # MLX Whisper wrapper
│   └── config.py            # .env loading via pydantic-settings
├── client/
│   ├── menubar.py           # rumps menu bar app
│   ├── recorder.py          # audio recording (sounddevice)
│   └── cli.py               # CLI commands (typer)
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
pip install -e ".[dev]"
python -m server.main              # start FastAPI server
python -m client.menubar           # start menu bar app
python -m client.cli tasks         # list tasks
python -m client.cli ask "..."     # query KB

# Docker mode
docker-compose up                  # starts app + ChromaDB
```

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
- **ChromaDB in Docker always**: even in native mode, ChromaDB runs in Docker. It's the one thing that benefits from containerization.
- **SQLite for tasks**: tasks are simple structured data. No need for a vector store.
- **No web UI**: CLI + menu bar is the interface. A web dashboard is post-MVP.
