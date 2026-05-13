from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    llm_provider: str = "local"
    lm_studio_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "lfm-2.5"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    chroma_url: str = "http://localhost:8000"

    whisper_model: str = "base"

    hotkey: str = "cmd+shift+m"

    server_host: str = "127.0.0.1"
    server_port: int = 7860

    log_level: str = "INFO"

    @property
    def llm_base_url(self) -> str:
        if self.llm_provider == "local":
            return self.lm_studio_url
        return "https://api.groq.com/openai/v1"

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "local":
            return "lm-studio"
        return self.groq_api_key

    @property
    def llm_model(self) -> str:
        if self.llm_provider == "local":
            return self.lm_studio_model
        return self.groq_model


class ModelConfig:
    def __init__(self, provider: str, model: str, temperature: float) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature


class ModelsConfig:
    def __init__(self, path: Path | None = None) -> None:
        self.agents: dict[str, ModelConfig] = {}
        self.default = ModelConfig("local", "lfm-2.5", 0.3)
        config_path = path or _ROOT / "models.yaml"
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text())
            if data and "default" in data:
                d = data["default"]
                self.default = ModelConfig(d["provider"], d["model"], d.get("temperature", 0.3))
            if data and "agents" in data:
                for name, cfg in data["agents"].items():
                    self.agents[name] = ModelConfig(
                        cfg.get("provider", self.default.provider),
                        cfg.get("model", self.default.model),
                        cfg.get("temperature", self.default.temperature),
                    )

    def get(self, agent_name: str) -> ModelConfig:
        return self.agents.get(agent_name, self.default)


def load_soul(path: Path | None = None) -> str:
    soul_path = path or _ROOT / "soul.md"
    if soul_path.exists():
        return soul_path.read_text()
    return ""


def get_settings() -> Settings:
    return Settings()
