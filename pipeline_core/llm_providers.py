"""Provider adapters for interacting with different LLM backends."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Protocol

import logging
import requests

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Minimal client interface used by the pipeline."""

    provider_name: str

    def complete_json(self, prompt: str, schema: Dict[str, Any], timeout_s: float) -> str:
        """Return a JSON string for the given prompt following ``schema``."""

    def complete_text(self, prompt: str, timeout_s: float) -> str:
        """Return a plain text completion for the given prompt."""


@dataclass
class _OllamaConfig:
    base_url: str
    model_json: str
    model_text: str
    keep_alive: Optional[str]


class OllamaClient:
    provider_name = "ollama"

    def __init__(self, config: _OllamaConfig) -> None:
        self._config = config

    def _request(self, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        url = f"{self._config.base_url}/api/generate"
        response = requests.post(url, json=payload, timeout=timeout_s)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"response": response.text}

    def _extract_response(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        for key in ("response", "content", "message", "result", "data"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        return data.get("output", "") if isinstance(data.get("output"), str) else ""

    def complete_json(self, prompt: str, schema: Dict[str, Any], timeout_s: float) -> str:
        payload: Dict[str, Any] = {
            "model": self._config.model_json,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        if self._config.keep_alive:
            payload["keep_alive"] = self._config.keep_alive
        data = self._request(payload, timeout_s)
        return self._extract_response(data)

    def complete_text(self, prompt: str, timeout_s: float) -> str:
        payload: Dict[str, Any] = {
            "model": self._config.model_text,
            "prompt": prompt,
            "stream": False,
        }
        if self._config.keep_alive:
            payload["keep_alive"] = self._config.keep_alive
        data = self._request(payload, timeout_s)
        return self._extract_response(data)


class _OpenAICompatibleClient:
    provider_name = "openai"

    _CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        *,
        api_key: str,
        model_json: str,
        model_text: str,
        provider_name: str,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key for provider")
        self._api_key = api_key
        self._model_json = model_json
        self._model_text = model_text
        self.provider_name = provider_name
        self._base_url = (base_url or self._CHAT_ENDPOINT).rstrip("/")
        self._extra_headers = extra_headers or {}

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._extra_headers)
        return headers

    def _post(self, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        response = requests.post(self._base_url, headers=self._headers(), json=payload, timeout=timeout_s)
        response.raise_for_status()
        return response.json()

    def _extract_message(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return data.get("output", "") if isinstance(data.get("output"), str) else ""

    def _payload(self, prompt: str, *, model: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that only outputs JSON when requested."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "metadata", "schema": json_schema},
            }
        return payload

    def complete_json(self, prompt: str, schema: Dict[str, Any], timeout_s: float) -> str:
        payload = self._payload(prompt, model=self._model_json, json_schema=schema)
        data = self._post(payload, timeout_s)
        return self._extract_message(data)

    def complete_text(self, prompt: str, timeout_s: float) -> str:
        payload = self._payload(prompt, model=self._model_text)
        data = self._post(payload, timeout_s)
        return self._extract_message(data)


class GroqClient(_OpenAICompatibleClient):
    _CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, *, api_key: str, model_json: str, model_text: str) -> None:
        super().__init__(
            api_key=api_key,
            model_json=model_json,
            model_text=model_text,
            provider_name="groq",
            base_url=self._CHAT_ENDPOINT,
        )


class TogetherClient(_OpenAICompatibleClient):
    _CHAT_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, *, api_key: str, model_json: str, model_text: str) -> None:
        super().__init__(
            api_key=api_key,
            model_json=model_json,
            model_text=model_text,
            provider_name="together",
            base_url=self._CHAT_ENDPOINT,
        )


class OpenAIClient(_OpenAICompatibleClient):
    def __init__(self, *, api_key: str, model_json: str, model_text: str) -> None:
        super().__init__(
            api_key=api_key,
            model_json=model_json,
            model_text=model_text,
            provider_name="openai",
            base_url=self._CHAT_ENDPOINT,
        )


_DEFAULT_TEXT_MODELS = {
    "ollama": "qwen2.5:7b",
    "openai": "gpt-4o-mini",
    "groq": "mixtral-8x7b-32768",
    "together": "meta-llama/Llama-3-8b-chat-hf",
}

_DEFAULT_JSON_MODELS = {
    "ollama": "qwen2.5:7b",
    "openai": "gpt-4o-mini",
    "groq": "mixtral-8x7b-32768",
    "together": "meta-llama/Llama-3-8b-instruct",
}


def _resolve_models(provider: str, settings: Optional[Any]) -> tuple[str, str]:
    upper = provider.upper()
    json_key = f"PIPELINE_LLM_{upper}_MODEL_JSON"
    text_key = f"PIPELINE_LLM_{upper}_MODEL_TEXT"

    model_json = os.getenv(json_key)
    model_text = os.getenv(text_key)

    if settings is not None:
        try:
            llm_settings = getattr(settings, "llm", None)
            if not model_json:
                model_json = getattr(llm_settings, "model_json", None)
            if not model_text:
                model_text = getattr(llm_settings, "model_text", None)
        except Exception:
            pass

    fallback_json = os.getenv("PIPELINE_LLM_MODEL_JSON") or os.getenv("PIPELINE_LLM_MODEL")
    fallback_text = os.getenv("PIPELINE_LLM_MODEL_TEXT") or os.getenv("PIPELINE_LLM_MODEL")

    model_json = (model_json or fallback_json or _DEFAULT_JSON_MODELS.get(provider, ""))
    model_text = (model_text or fallback_text or _DEFAULT_TEXT_MODELS.get(provider, ""))

    return model_json, model_text


def get_llm_client(settings: Optional[Any] = None) -> LLMClient:
    provider = (os.getenv("PIPELINE_LLM_PROVIDER") or "ollama").strip().lower() or "ollama"

    model_json, model_text = _resolve_models(provider, settings)

    if provider == "ollama":
        base_url = (
            os.getenv("PIPELINE_LLM_ENDPOINT")
            or os.getenv("PIPELINE_LLM_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://127.0.0.1:11434"
        )
        keep_alive = os.getenv("PIPELINE_LLM_KEEP_ALIVE")
        config = _OllamaConfig(
            base_url=base_url.rstrip("/") or "http://127.0.0.1:11434",
            model_json=model_json,
            model_text=model_text,
            keep_alive=keep_alive,
        )
        return OllamaClient(config)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or ""
        return OpenAIClient(api_key=api_key, model_json=model_json, model_text=model_text)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY") or ""
        return GroqClient(api_key=api_key, model_json=model_json, model_text=model_text)

    if provider == "together":
        api_key = os.getenv("TOGETHER_API_KEY") or ""
        return TogetherClient(api_key=api_key, model_json=model_json, model_text=model_text)

    raise ValueError(f"Unsupported LLM provider: {provider}")
