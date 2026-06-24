"""Helpers for normalizing Ollama chat API responses."""

from __future__ import annotations

from typing import Any


def normalize_chat_response(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        if not data:
            raise ValueError("Empty Ollama response list")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected Ollama response type: {type(data)!r}")
    return data


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def extract_chat_content(data: Any) -> str:
    payload = normalize_chat_response(data)
    message = payload.get("message")
    if isinstance(message, list):
        parts = [normalize_message_content(item.get("content") if isinstance(item, dict) else item) for item in message]
        text = "".join(part for part in parts if part)
    elif isinstance(message, dict):
        text = normalize_message_content(message.get("content"))
    else:
        text = normalize_message_content(payload.get("content"))
    if not text.strip():
        raise ValueError("Empty Ollama chat content")
    return text
