"""HTTP adapter for the single-rubric judge."""

from __future__ import annotations

import json
import time
import urllib.request


class JudgeRequestError(RuntimeError):
    """The judge endpoint did not return a usable response."""


class JudgeClient:
    def __init__(
        self, url: str, *, api_key: str = "", timeout: float = 180.0,
        retries: int = 3, response_format: str = "json_schema",
        max_tokens: int = 512,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("judge URL must use HTTP or HTTPS")
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        if response_format not in {"json_schema", "json_object", "off"}:
            raise ValueError("unsupported judge response format")
        self.response_format = response_format

    def __call__(self, model_name: str, prompt: str) -> str:
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rubric_verdict",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"winner": {"type": "string", "enum": ["A", "B", "TIE"]}},
                        "required": ["winner"],
                        "additionalProperties": False,
                    },
                },
            }
        elif self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.retries):
            request = urllib.request.Request(self.url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("judge returned non-text content")
                return content
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                if attempt + 1 == self.retries:
                    raise JudgeRequestError("judge request failed") from error
                time.sleep(min(2 ** attempt, 8))
        raise JudgeRequestError("judge request failed")
