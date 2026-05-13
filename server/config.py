from pydantic_settings import BaseSettings


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


def get_settings() -> Settings:
    return Settings()
