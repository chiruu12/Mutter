import json
import logging

from openai import AuthenticationError, OpenAI

from server.config import ModelConfig, ModelsConfig, Settings

log = logging.getLogger("mutter.llm")


class LLMError(Exception):
    pass


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
                if not self._settings.groq_api_key:
                    raise LLMError(
                        "GROQ_API_KEY is not set. Add it to your .env file. "
                        "Get a free key at https://console.groq.com/keys"
                    )
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

    def _call(self, fn_name: str, agent: str | None, **kwargs):
        try:
            client, cfg = self._resolve(agent)
            return client.chat.completions.create(**kwargs)
        except AuthenticationError:
            provider = "groq" if agent else self._settings.llm_provider
            log.error("[llm] authentication failed for %s (agent=%s)", provider, agent)
            raise LLMError(
                f"Invalid API key for {provider}. Check GROQ_API_KEY in your .env file."
            )
        except Exception as e:
            if "Connection" in type(e).__name__ or "ConnectError" in str(type(e)):
                log.error("[llm] connection failed for agent=%s: %s", agent, e)
                raise LLMError(f"LLM provider unreachable: {e}")
            raise

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        agent: str | None = None,
    ) -> str:
        client, cfg = self._resolve(agent)
        response = self._call(
            "complete",
            agent,
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else cfg.temperature,
        )
        content = response.choices[0].message.content
        return content or ""

    def complete_json(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        agent: str | None = None,
    ) -> dict:
        client, cfg = self._resolve(agent)
        kwargs: dict = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system + "\nRespond ONLY with valid JSON."},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else cfg.temperature,
        }
        if cfg.provider != "local":
            kwargs["response_format"] = {"type": "json_object"}
        response = self._call("complete_json", agent, **kwargs)
        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned empty response")
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("[llm] failed to parse JSON: %s", text[:200])
            raise LLMError(f"LLM returned invalid JSON: {text[:100]}")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        agent: str | None = None,
    ) -> object:
        client, cfg = self._resolve(agent)
        kwargs: dict = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        return self._call("chat", agent, **kwargs)
