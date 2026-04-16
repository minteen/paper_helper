from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from config import get_settings


class LLMTransport(Protocol):
    def chat_completion(self, payload: Dict[str, Any]) -> str:
        """Run a chat completion request and return text content."""


class OpenAICompatibleTransport:
    """Transport that talks to OpenAI-compatible chat completion APIs."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat_completion(self, payload: Dict[str, Any]) -> str:
        response = self._client.chat.completions.create(**payload)
        return response.choices[0].message.content or ""


@dataclass
class LLMResponse:
    content: str
    raw: Optional[Dict[str, Any]] = None


class LLMClient:
    def __init__(
        self,
        transport: Optional[LLMTransport] = None,
        *,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.QWEN_MODEL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.transport: LLMTransport = transport or OpenAICompatibleTransport(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
            model=self.model,
        )

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = self._request_with_retry(payload)
        return LLMResponse(content=content, raw={"payload": payload})

    def chat_json(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        response = self.chat(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return json.loads(response.content)

    def _build_payload(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _request_with_retry(self, payload: Dict[str, Any]) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.transport.chat_completion(payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_delay)
        assert last_error is not None
        raise last_error
