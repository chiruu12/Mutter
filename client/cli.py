import os
import platform
from importlib.metadata import version as pkg_version
from pathlib import Path

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
    transcription = result.get("transcription", "")
    if transcription:
        typer.echo(f'  "{transcription}"')
        typer.echo("")

    intent = result.get("intent", "unknown")
    pipeline = result.get("pipeline", {})
    router_ms = pipeline.get("router_ms", 0)
    typer.echo(f"Routing... {typer.style(intent.upper(), bold=True)} ({router_ms}ms)")
    typer.echo("")

    if intent == "task":
        desc = result.get("description", "Unknown")
        typer.echo(typer.style(f"✓ Task created: {desc}", fg=typer.colors.GREEN))
        if result.get("due"):
            typer.echo(f"  Due: {result['due']}")
        typer.echo(f"  Priority: {result.get('priority', 'medium')}")
    elif intent == "note":
        typer.echo(typer.style("✓ Note saved", fg=typer.colors.GREEN))
        content = result.get("content", "")
        if content:
            typer.echo(f'  "{content[:120]}"')
    elif intent == "query":
        typer.echo(typer.style("Answer:", bold=True))
        typer.echo(f"  {result.get('answer', 'No answer')}")
        sources = result.get("sources", [])
        if sources:
            typer.echo(f"\n  Sources: {len(sources)} notes matched")

    total_ms = pipeline.get("total_ms", 0)
    if total_ms:
        typer.echo("")
        typer.echo(typer.style(f"Total: {total_ms}ms", dim=True))


@app.command()
def record() -> None:
    from client.recorder import Recorder

    recorder = Recorder()
    typer.echo("🎙 Recording... press Enter to stop.")
    recorder.start()
    input()
    wav_path = recorder.stop_and_save()
    try:
        typer.echo("Transcribing...", nl=False)
        with open(wav_path, "rb") as f:
            result = _request("post", "/process", files={"file": f})
        whisper_ms = result.get("pipeline", {}).get("whisper_ms", 0)
        typer.echo(f" done ({whisper_ms}ms)")
        _display_result(result)
    except typer.Exit:
        raise
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


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
                typer.echo(typer.style(f"  (raw: {raw})", dim=True))
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
    typer.echo("Processing...")
    typer.echo("")
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
    typer.echo(result.get("answer", "No answer"))


def _format_tool_result(name: str, tc_result: dict) -> str:
    if name == "create_task":
        return f"Task #{tc_result.get('id', '?')} created"
    elif name == "set_alarm":
        return f"Alarm set for {tc_result.get('alarm', '?')}"
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
    typer.echo("🤖 Processing...")
    typer.echo("")
    result = _request("post", "/agent", json={"message": message})

    for tc in result.get("tool_calls", []):
        name = tc["name"]
        args = tc.get("args", {})
        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
        typer.echo(f"→ {name}({args_str})")
        summary = _format_tool_result(name, tc.get("result", {}))
        typer.echo(typer.style(f"  ✓ {summary}", fg=typer.colors.GREEN))
        typer.echo("")

    rounds = result.get("rounds", 0)
    elapsed = result.get("elapsed_ms", 0)
    typer.echo(typer.style(f"Done in {rounds} rounds ({elapsed}ms)", dim=True))
    typer.echo("")

    response = result.get("response", "")
    if response:
        typer.echo(f'"{response}"')


@app.command()
def digest() -> None:
    from datetime import datetime

    result = _request("get", "/digest")
    try:
        date = datetime.strptime(result.get("date", ""), "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        date = result.get("date", "Unknown")
    typer.echo("")
    typer.echo(typer.style(f"Daily Digest — {date}", bold=True))
    typer.echo("")
    typer.echo(typer.style("Summary:", bold=True))
    typer.echo(result.get("summary", "No summary available"))
    typer.echo("")
    typer.echo(typer.style("Pending Tasks:", bold=True))
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
def status() -> None:
    result = _request("get", "/health")
    typer.echo(f"Server: {result.get('status', 'unknown')}")
    typer.echo(f"  ChromaDB: {result.get('chroma', 'unknown')}")
    typer.echo(f"  LLM: {result.get('llm', 'unknown')}")


if __name__ == "__main__":
    app()
