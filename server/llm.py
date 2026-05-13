import json

from openai import OpenAI

from server.config import ModelConfig, ModelsConfig, Settings


class LLMClient:
    def __init__(self, settings: Settings, models: ModelsConfig | None = None) -> None:
        self._settings = settings
        self._models = models or ModelsConfig()
        self._clients: dict[str, OpenAI] = {}

    def _get_client(self, provider: str) -> OpenAI:
        if provider not in self._clients:
            if provider == "local":
                self._clients[provider] = OpenAI(
                    base_url=self._settings.lm_studio_url,
                    api_key="lm-studio",
                )
            else:
                self._clients[provider] = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self._settings.groq_api_key,
                )
        return self._clients[provider]

    def _resolve(self, agent: str | None) -> tuple[OpenAI, ModelConfig]:
        if agent:
            cfg = self._models.get(agent)
        else:
            cfg = ModelConfig(
                self._settings.llm_provider,
                self._settings.llm_model,
                0.3,
            )
        return self._get_client(cfg.provider), cfg

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        agent: str | None = None,
    ) -> str:
        client, cfg = self._resolve(agent)
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else cfg.temperature,
        )
        return response.choices[0].message.content

    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        agent: str | None = None,
    ) -> dict:
        client, cfg = self._resolve(agent)
        response = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else cfg.temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        agent: str | None = None,
    ) -> object:
        client, cfg = self._resolve(agent)
        kwargs = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        return client.chat.completions.create(**kwargs)
