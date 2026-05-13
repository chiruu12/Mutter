# Mutter

> You mutter things, it captures them.

Local-first voice assistant that lives in your Mac menu bar. Press a hotkey, talk, and it figures out the rest.

Mutter uses a three-way LLM router to classify your speech:
- **Task** — extracts details, stores with due dates
- **Note** — cleans up your words, saves to a searchable knowledge base
- **Question** — searches your saved notes and answers from what you've told it before

No commands. No wake words. You just talk naturally, and it does the right thing.

## How It Works

```
You speak → Whisper transcribes → LLM classifies → right thing happens
```

```
┌─────────────┐     ┌─────────┐     ┌──────────────────┐
│  Menu Bar   │────►│ Whisper  │────►│  Router (LLM)    │
│  (click /   │     │  (MLX)   │     │                  │
│   hotkey)   │     └─────────┘     │  TASK → SQLite   │
└─────────────┘                     │  NOTE → ChromaDB │
                                    │  QUERY → search  │
                                    └──────────────────┘
```

Everything runs locally. Your voice never leaves your machine.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for ChromaDB)
- [LM Studio](https://lmstudio.ai/) with LFM 2.5 loaded

### Install

```bash
git clone https://github.com/chiruu12/Mutter.git
cd Mutter

# Start ChromaDB
docker-compose up -d chromadb

# Install Mutter
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your LM Studio URL
```

### Run

```bash
# Start the server
mutter serve

# Start the menu bar app (Mac only)
mutter menubar

# Or use the CLI
mutter record          # record and process
mutter tasks           # list tasks
mutter notes           # list notes
mutter ask "what did I say about the pitch?"
```

### Docker (any platform)

```bash
cp .env.example .env
# Set LLM_PROVIDER=groq and add your GROQ_API_KEY
docker-compose up
```

## Configuration

All config lives in `.env`:

```bash
# LLM Backend
LLM_PROVIDER=local              # "local" (LM Studio) or "groq"
LM_STUDIO_URL=http://localhost:1234/v1
GROQ_API_KEY=                   # only needed if provider=groq

# ChromaDB
CHROMA_URL=http://localhost:8000

# Whisper
WHISPER_MODEL=base              # tiny, base, small, medium, large

# Hotkey (Mac menu bar)
HOTKEY=cmd+shift+m
```

## Stack

| Component | Tool |
|-----------|------|
| Menu bar | [rumps](https://github.com/jaredks/rumps) |
| Voice recording | sounddevice |
| Transcription | MLX Whisper (Mac) / faster-whisper (Docker) |
| LLM | LFM 2.5 via LM Studio or Groq |
| Knowledge base | ChromaDB |
| Task storage | SQLite |
| API server | FastAPI |
| CLI | Typer |

## Roadmap

- [ ] WhatsApp/Telegram task delivery
- [ ] Continuous listening mode
- [ ] Google Calendar awareness
- [ ] Email digest (morning inbox summary)
- [ ] Web dashboard
- [ ] Linux/Windows system tray support

## License

Apache 2.0
