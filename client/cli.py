import os
import platform
import sys

import httpx
import typer

app = typer.Typer(name="mutter")


def _server_url() -> str:
    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    port = os.environ.get("SERVER_PORT", "7860")
    return f"http://{host}:{port}"


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{_server_url()}{path}"
    try:
        response = getattr(httpx, method)(url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        typer.echo("Server not running. Start with: mutter serve", err=True)
        raise typer.Exit(1)
    except httpx.TimeoutException:
        typer.echo("Server timed out.", err=True)
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        typer.echo(f"Error: {detail or e}", err=True)
        raise typer.Exit(1)


def _display_result(result: dict) -> None:
    intent = result.get("intent", "unknown")
    if intent == "task":
        typer.echo(f"Task: {result['description']}")
        if result.get("due"):
            typer.echo(f"  Due: {result['due']}")
        typer.echo(f"  Priority: {result.get('priority', 'medium')}")
    elif intent == "note":
        typer.echo(f"Note saved: {result['content']}")
    elif intent == "query":
        typer.echo(f"Answer: {result['answer']}")
    else:
        typer.echo(result)


@app.command()
def record() -> None:
    from client.recorder import Recorder

    recorder = Recorder()
    typer.echo("Recording... press Enter to stop.")
    recorder.start()
    input()
    wav_path = recorder.stop_and_save()
    try:
        with open(wav_path, "rb") as f:
            result = _request("post", "/process", files={"file": f})
        _display_result(result)
    except typer.Exit:
        raise
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


@app.command()
def send(text: str) -> None:
    result = _request("post", "/process/text", json={"text": text})
    _display_result(result)


@app.command()
def tasks() -> None:
    task_list = _request("get", "/tasks")
    if not task_list:
        typer.echo("No tasks.")
        return
    for t in task_list:
        due = f" (due: {t['due']})" if t.get("due") else ""
        typer.echo(f"  [{t['id']}] {t['description']}{due} — {t['priority']}")


@app.command()
def notes() -> None:
    note_list = _request("get", "/notes")
    if not note_list:
        typer.echo("No notes.")
        return
    for n in note_list:
        typer.echo(f"  {n['content'][:80]}")


@app.command()
def ask(question: str) -> None:
    result = _request("post", "/query", json={"question": question})
    typer.echo(result["answer"])


@app.command()
def agent(message: str) -> None:
    result = _request("post", "/agent", json={"message": message})
    typer.echo(result["response"])


@app.command()
def serve() -> None:
    from pydantic_settings import BaseSettings

    class _Env(BaseSettings):
        model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
        llm_provider: str = "local"
        lm_studio_url: str = "http://localhost:1234/v1"
        groq_api_key: str = ""
        chroma_url: str = "http://localhost:8000"
        whisper_model: str = "base"
        server_host: str = "127.0.0.1"
        server_port: int = 7860

    env = _Env()

    chroma_ok = False
    try:
        r = httpx.get(f"{env.chroma_url}/api/v1/heartbeat", timeout=3)
        chroma_ok = r.status_code == 200
    except Exception:
        pass

    llm_label = f"local (LM Studio @ {env.lm_studio_url.replace('http://', '').rstrip('/v1')})"
    llm_ok = True
    if env.llm_provider == "local":
        try:
            httpx.get(f"{env.lm_studio_url}/models", timeout=3)
        except Exception:
            llm_ok = False
    elif env.llm_provider == "groq":
        llm_label = "groq"
        llm_ok = bool(env.groq_api_key)

    whisper_backend = "mlx" if platform.system() == "Darwin" else "faster-whisper"

    typer.echo("")
    typer.echo("Mutter v0.1.0")
    typer.echo(f"LLM: {llm_label}" + (" " if llm_ok else " (unreachable)"))
    typer.echo(f"ChromaDB: {'connected' if chroma_ok else 'disconnected'} ({env.chroma_url})")
    if not chroma_ok:
        typer.echo("  Start with: docker-compose up -d chromadb")
    typer.echo(f"Whisper: {whisper_backend} (model: {env.whisper_model})")
    typer.echo(f"Server: http://{env.server_host}:{env.server_port}")
    typer.echo("")

    import uvicorn

    uvicorn.run("server.main:app", host=env.server_host, port=env.server_port)


@app.command()
def status() -> None:
    result = _request("get", "/health")
    typer.echo(f"Server: {result['status']}")
    typer.echo(f"  ChromaDB: {result.get('chroma', 'unknown')}")
    typer.echo(f"  LLM: {result.get('llm', 'unknown')}")
