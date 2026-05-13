import os

import httpx
import typer

app = typer.Typer(name="mutter")


def _server_url() -> str:
    host = os.environ.get("SERVER_HOST", "127.0.0.1")
    port = os.environ.get("SERVER_PORT", "7860")
    return f"http://{host}:{port}"


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

    server = _server_url()
    recorder = Recorder()
    typer.echo("Recording... press Enter to stop.")
    recorder.start()
    input()
    wav_path = recorder.stop_and_save()
    with open(wav_path, "rb") as f:
        response = httpx.post(f"{server}/process", files={"file": f})
    _display_result(response.json())


@app.command()
def send(text: str) -> None:
    response = httpx.post(f"{_server_url()}/process/text", json={"text": text})
    _display_result(response.json())


@app.command()
def tasks() -> None:
    response = httpx.get(f"{_server_url()}/tasks")
    task_list = response.json()
    if not task_list:
        typer.echo("No tasks.")
        return
    for t in task_list:
        due = f" (due: {t['due']})" if t.get("due") else ""
        typer.echo(f"  [{t['id']}] {t['description']}{due} — {t['priority']}")


@app.command()
def notes() -> None:
    response = httpx.get(f"{_server_url()}/notes")
    note_list = response.json()
    if not note_list:
        typer.echo("No notes.")
        return
    for n in note_list:
        typer.echo(f"  {n['content'][:80]}")


@app.command()
def ask(question: str) -> None:
    response = httpx.post(f"{_server_url()}/query", json={"question": question})
    result = response.json()
    typer.echo(result["answer"])


@app.command()
def agent(message: str) -> None:
    response = httpx.post(f"{_server_url()}/agent", json={"message": message})
    result = response.json()
    typer.echo(result["response"])


@app.command()
def status() -> None:
    try:
        response = httpx.get(f"{_server_url()}/health")
        typer.echo(f"Server: {response.json()['status']}")
    except httpx.ConnectError:
        typer.echo("Server not running.")
