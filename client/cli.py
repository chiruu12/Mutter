import os
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
def status() -> None:
    result = _request("get", "/health")
    typer.echo(f"Server: {result['status']}")
    typer.echo(f"  ChromaDB: {result.get('chroma', 'unknown')}")
    typer.echo(f"  LLM: {result.get('llm', 'unknown')}")
