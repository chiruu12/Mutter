import os
import platform
from importlib.metadata import version as pkg_version
from pathlib import Path

import click
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


def _format_time(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def _display_result(result: dict) -> None:
    pipeline = result.get("pipeline", {})
    intent = result.get("intent", "unknown")

    transcription = result.get("transcription", "")
    whisper_ms = pipeline.get("whisper_ms", 0)
    if transcription:
        if whisper_ms:
            click.secho(f"  Transcribed ({_format_time(whisper_ms)})", dim=True)
        typer.echo(f'  "{transcription}"')
        typer.echo("")

    router_ms = pipeline.get("router_ms", 0)
    click.secho(f"  {intent.upper()}", bold=True, nl=False)
    if router_ms:
        click.secho(f"  ({_format_time(router_ms)})", dim=True)
    else:
        typer.echo("")
    typer.echo("")

    if intent == "agent":
        for tc in result.get("tool_calls", []):
            name = tc["name"]
            tc_result = tc.get("result") or {}
            if "error" in tc_result:
                click.secho(f"  ✗ {tc_result['error']}", fg="red")
            else:
                summary = _format_tool_result(name, tc_result)
                click.secho(f"  ✓ {summary}", fg="green")
        response = result.get("response", "")
        if response:
            typer.echo("")
            typer.echo(f"  {response}")
    elif intent == "task":
        desc = result.get("description", "Unknown")
        click.secho(f"  ✓ Task created: {desc}", fg="green")
        details = []
        if result.get("due"):
            details.append(f"Due: {result['due']}")
        details.append(f"Priority: {result.get('priority', 'medium')}")
        click.secho(f"    {' · '.join(details)}", dim=True)
    elif intent == "note":
        click.secho("  ✓ Note saved", fg="green")
        content = result.get("content", "")
        if content:
            click.secho(f"    {content[:120]}", dim=True)
    elif intent == "query":
        typer.echo(f"  {result.get('answer', 'No answer')}")
        sources = result.get("sources", [])
        if sources:
            click.secho(f"    {len(sources)} sources", dim=True)

    total_ms = pipeline.get("total_ms", 0)
    if total_ms:
        typer.echo("")
        click.secho(f"  {_format_time(total_ms)} total", dim=True)


@app.command()
def dictate() -> None:
    from client.recorder import Recorder

    recorder = Recorder()
    typer.echo("Dictating... press Enter to stop.")
    recorder.start()
    input()
    wav_path = recorder.stop_and_save()
    try:
        with open(wav_path, "rb") as f:
            result = _request("post", "/transcribe", files={"file": f})
        text = result.get("text", "").strip()
        raw = result.get("raw", "").strip()
        if text:
            typer.echo(text)
            if raw and raw != text:
                click.secho(f"  (raw: {raw})", dim=True)
        else:
            typer.echo("No speech detected.")
    except typer.Exit:
        raise
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


@app.command()
def send(text: str) -> None:
    typer.echo("")
    result = _request("post", "/process/text", json={"text": text})
    _display_result(result)
    typer.echo("")


@app.command()
def tasks() -> None:
    task_list = _request("get", "/tasks")
    if not task_list:
        typer.echo("  No tasks.")
        return
    typer.echo("")
    click.secho(f"  Tasks ({len(task_list)})", bold=True)
    typer.echo("")
    for t in task_list:
        priority = t.get("priority", "medium").upper()
        pcolor = "red" if priority == "HIGH" else ("yellow" if priority == "MEDIUM" else "white")
        click.secho(f"  #{t['id']:<4}", dim=True, nl=False)
        typer.echo(f"  {t['description']}", nl=False)
        if t.get("due"):
            click.secho(f"  ({t['due']})", dim=True, nl=False)
        click.secho(f"  {priority}", fg=pcolor)
    typer.echo("")


@app.command()
def alarms() -> None:
    alarm_list = _request("get", "/alarms")
    if not alarm_list:
        typer.echo("  No pending alarms.")
        return
    typer.echo("")
    click.secho(f"  Alarms ({len(alarm_list)})", bold=True)
    typer.echo("")
    for a in alarm_list:
        click.secho(f"  #{a['id']:<4}", dim=True, nl=False)
        typer.echo(f"  {a['description']}", nl=False)
        label = a.get("label") or a.get("fire_at", "")
        click.secho(f"  ({label})", dim=True)
    typer.echo("")


@app.command(name="test-alarm")
def test_alarm(seconds: int = typer.Argument(60, help="Seconds from now to fire the alarm")) -> None:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc).astimezone()
    fire_at = (now + timedelta(seconds=seconds)).isoformat()
    label = f"in {seconds}s" if seconds < 120 else f"in {seconds // 60}m"
    result = _request("post", "/alarms", json={
        "description": "Test alarm",
        "fire_at": fire_at,
        "label": label,
    })
    typer.echo("")
    click.secho(f"  ✓ Alarm #{result['id']} set ({label})", fg="green")
    typer.echo("")


@app.command(name="cancel-alarm")
def cancel_alarm(alarm_id: int) -> None:
    _request("delete", f"/alarms/{alarm_id}")
    click.secho(f"✓ Alarm #{alarm_id} cancelled", fg="green")


@app.command()
def done(task_id: str = typer.Argument(..., help="Task ID or 'all'")) -> None:
    if task_id == "all":
        result = _request("post", "/tasks/done-all")
        count = result.get("completed", 0)
        click.secho(f"✓ {count} tasks completed", fg="green")
    else:
        _request("post", f"/tasks/{task_id}/done")
        click.secho(f"✓ Task #{task_id} completed", fg="green")


def _relative_time(iso_str: str) -> str:
    from datetime import datetime
    try:
        created = datetime.fromisoformat(iso_str)
        now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now()
        delta = now - created
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except (ValueError, TypeError):
        return ""


@app.command()
def notes() -> None:
    note_list = _request("get", "/notes")
    if not note_list:
        typer.echo("  No notes.")
        return
    typer.echo("")
    click.secho(f"  Notes ({len(note_list)})", bold=True)
    typer.echo("")
    for n in note_list:
        content = n["content"][:100]
        if len(n["content"]) > 100:
            content += "..."
        typer.echo(f"  - {content}")
        ts = _relative_time(n.get("created_at", ""))
        if ts:
            click.secho(f"    {ts}", dim=True)
    typer.echo("")


@app.command()
def ask(question: str) -> None:
    result = _request("post", "/query", json={"question": question})
    typer.echo("")
    typer.echo(f"  {result.get('answer', 'No answer')}")
    sources = result.get("sources", [])
    if sources:
        click.secho(f"    {len(sources)} sources", dim=True)
    typer.echo("")


def _format_tool_result(name: str, tc_result: dict) -> str:
    if name == "create_task":
        return f"Task #{tc_result.get('id', '?')} created"
    elif name == "set_alarm":
        return f"Alarm set for {tc_result.get('label') or tc_result.get('alarm', '?')}"
    elif name == "list_alarms":
        return f"{tc_result.get('count', 0)} alarms"
    elif name == "cancel_alarm":
        return f"Alarm #{tc_result.get('alarm_id', '?')} cancelled"
    elif name == "complete_task":
        return f"Task #{tc_result.get('task_id', '?')} completed"
    elif name == "search_notes":
        return f"Found {tc_result.get('count', 0)} notes"
    elif name == "save_note":
        return "Note saved"
    elif name == "list_tasks":
        return f"{tc_result.get('count', 0)} tasks"
    return "done"


@app.command()
def agent(message: str) -> None:
    typer.echo("")
    click.secho("  Agent", bold=True)
    typer.echo("")
    result = _request("post", "/agent", json={"message": message})

    for tc in result.get("tool_calls", []):
        name = tc["name"]
        tc_result = tc.get("result") or {}
        if "error" in tc_result:
            click.secho(f"  ✗ {name}: {tc_result['error']}", fg="red")
        elif tc_result.get("completed") is False:
            click.secho(f"  ✗ Task #{tc_result.get('task_id', '?')} not found", fg="red")
        elif tc_result.get("cancelled") is False:
            click.secho(f"  ✗ Alarm #{tc_result.get('alarm_id', '?')} not found", fg="red")
        else:
            summary = _format_tool_result(name, tc_result)
            click.secho(f"  ✓ {summary}", fg="green")

    response = result.get("response", "")
    if response:
        typer.echo("")
        typer.echo(f"  {response}")

    elapsed = result.get("elapsed_ms", 0)
    typer.echo("")
    click.secho(f"  {_format_time(elapsed)}", dim=True)


@app.command()
def digest() -> None:
    from datetime import datetime

    result = _request("get", "/digest")
    try:
        date = datetime.strptime(result.get("date", ""), "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        date = result.get("date", "Unknown")
    typer.echo("")
    click.secho(f"Daily Digest — {date}", bold=True)
    typer.echo("")
    click.secho("Summary:", bold=True)
    typer.echo(result.get("summary", "No summary available"))
    typer.echo("")
    click.secho("Pending Tasks:", bold=True)
    pending = result.get("pending_tasks", [])
    if pending:
        for t in pending:
            due = f" (due: {t.get('due')})" if t.get("due") else ""
            priority = t.get("priority", "medium").upper()
            typer.echo(f"  [ ] {t.get('description', '?')}{due} [{priority}]")
    else:
        typer.echo("  No pending tasks.")
    typer.echo("")
    typer.echo(f"Recent Notes: {result.get('notes_count', 0)} saved in last 24h")
    typer.echo("")


def _check_chroma(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/api/v2/heartbeat", timeout=3)
        if r.status_code == 200:
            return True
        r = httpx.get(f"{url}/api/v1/heartbeat", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _find_compose_file() -> Path | None:
    candidates = [
        Path.cwd() / "docker-compose.yml",
        Path(__file__).resolve().parent.parent / "docker-compose.yml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _start_chromadb() -> bool:
    import shutil
    import subprocess

    compose_file = _find_compose_file()
    if not compose_file:
        return False
    docker = shutil.which("docker")
    if not docker:
        return False
    try:
        subprocess.run(
            [docker, "compose", "-f", str(compose_file), "up", "-d", "chromadb"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        dc = shutil.which("docker-compose")
        if dc:
            subprocess.run(
                [dc, "-f", str(compose_file), "up", "-d", "chromadb"],
                check=True,
                capture_output=True,
            )
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False


@app.command()
def serve() -> None:
    from pydantic_settings import BaseSettings

    class _Env(BaseSettings):
        model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
        llm_provider: str = "local"
        lm_studio_url: str = "http://localhost:1234/v1"
        groq_api_key: str = ""
        chroma_url: str = "http://localhost:8000"
        whisper_model: str = "base"
        server_host: str = "127.0.0.1"
        server_port: int = 7860

    env = _Env()

    chroma_ok = _check_chroma(env.chroma_url)
    if not chroma_ok:
        typer.echo("ChromaDB not running, starting via Docker...")
        if _start_chromadb():
            for _ in range(15):
                import time
                time.sleep(1)
                if _check_chroma(env.chroma_url):
                    chroma_ok = True
                    break
            if chroma_ok:
                typer.echo("ChromaDB started.")
            else:
                typer.echo("ChromaDB container started but not yet reachable — it may need a moment.")
        else:
            typer.echo("Could not start ChromaDB. Is Docker running?")

    llm_label = f"local (LM Studio @ {env.lm_studio_url.replace('http://', '').rstrip('/v1')})"
    llm_ok = True
    if env.llm_provider == "local":
        try:
            r = httpx.get(f"{env.lm_studio_url}/models", timeout=3)
            llm_ok = r.status_code == 200
        except Exception:
            llm_ok = False
    elif env.llm_provider == "groq":
        llm_label = "groq"
        llm_ok = bool(env.groq_api_key)

    groq_ok = bool(env.groq_api_key)

    whisper_backend = "mlx" if platform.system() == "Darwin" else "faster-whisper"
    try:
        ver = pkg_version("mutter")
    except Exception:
        ver = "0.1.0"

    typer.echo("")
    typer.echo(f"Mutter v{ver}")
    typer.echo(f"LLM: {llm_label}" + ("" if llm_ok else " (not running — start LM Studio)"))
    typer.echo(f"Groq: {'configured' if groq_ok else 'missing GROQ_API_KEY in .env'}")
    if not groq_ok:
        typer.echo("  WARNING: agents use Groq — most features will fail without a key")
    typer.echo(f"ChromaDB: {'connected' if chroma_ok else 'disconnected'} ({env.chroma_url})")
    typer.echo(f"Whisper: {whisper_backend} (model: {env.whisper_model})")
    typer.echo(f"Server: http://{env.server_host}:{env.server_port}")
    typer.echo("")

    import uvicorn

    uvicorn.run("server.main:app", host=env.server_host, port=env.server_port)


@app.command()
def client() -> None:
    try:
        from client.menubar import MutterApp
    except ImportError:
        typer.echo("Menu bar app requires macOS extras: pip install -e '.[mac]'", err=True)
        raise typer.Exit(1)
    try:
        _request("get", "/health")
    except typer.Exit:
        typer.echo("Tip: start the server first with: mutter serve", err=True)
        raise typer.Exit(1)
    typer.echo("Starting menu bar app...")
    MutterApp().run()


@app.command()
def status() -> None:
    result = _request("get", "/health")
    typer.echo(f"Server: {result.get('status', 'unknown')}")
    typer.echo(f"  ChromaDB: {result.get('chroma', 'unknown')}")
    typer.echo(f"  LLM: {result.get('llm', 'unknown')}")


if __name__ == "__main__":
    app()
