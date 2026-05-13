import typer
import httpx

app = typer.Typer(name="mutter")

SERVER = "http://127.0.0.1:7860"


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
    with open(wav_path, "rb") as f:
        response = httpx.post(f"{SERVER}/process", files={"file": f})
    _display_result(response.json())


@app.command()
def send(text: str) -> None:
    response = httpx.post(f"{SERVER}/process/text", json={"text": text})
    _display_result(response.json())


@app.command()
def tasks() -> None:
    response = httpx.get(f"{SERVER}/tasks")
    task_list = response.json()
    if not task_list:
        typer.echo("No tasks.")
        return
    for t in task_list:
        due = f" (due: {t['due']})" if t.get("due") else ""
        typer.echo(f"  [{t['id']}] {t['description']}{due} — {t['priority']}")


@app.command()
def notes() -> None:
    response = httpx.get(f"{SERVER}/notes")
    note_list = response.json()
    if not note_list:
        typer.echo("No notes.")
        return
    for n in note_list:
        typer.echo(f"  {n['content'][:80]}")


@app.command()
def ask(question: str) -> None:
    response = httpx.post(f"{SERVER}/query", json={"question": question})
    result = response.json()
    typer.echo(result["answer"])


@app.command()
def status() -> None:
    try:
        response = httpx.get(f"{SERVER}/health")
        typer.echo(f"Server: {response.json()['status']}")
    except httpx.ConnectError:
        typer.echo("Server not running.")
