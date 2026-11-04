"""DeepSeekClient 的 HTTP 协议与异常映射（使用 httpx.MockTransport，不联网）。"""

import httpx
import pytest

from llm import DeepSeekClient, DeepSeekError


def _make_client(handler):
    client = DeepSeekClient(
        api_key="test-key", base_url="https://example.invalid"
    )
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_chat_returns_raw_assistant_message_with_reasoning():
    def handler(request):
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "opaque",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": '{"expression": "1+1"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    client = _make_client(handler)
    try:
        message = client.chat(
            [{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "calculator"},
                }
            ],
        )
    finally:
        client.close()

    assert message["reasoning_content"] == "opaque"
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"]["name"] == "calculator"


def test_timeout_is_mapped_to_deepseek_error():
    def handler(request):
        raise httpx.TimeoutException("timed out", request=request)

    client = _make_client(handler)
    try:
        with pytest.raises(DeepSeekError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        client.close()

    assert exc_info.value.kind == "timeout"


def test_http_error_is_mapped_to_deepseek_error():
    def handler(request):
        return httpx.Response(500, text="server boom")

    client = _make_client(handler)
    try:
        with pytest.raises(DeepSeekError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        client.close()

    assert exc_info.value.kind == "http"
    assert "500" in str(exc_info.value)


def test_malformed_response_is_mapped_to_protocol_error():
    def handler(request):
        return httpx.Response(200, json={"choices": []})

    client = _make_client(handler)
    try:
        with pytest.raises(DeepSeekError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
    finally:
        client.close()

    assert exc_info.value.kind == "protocol"


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(DeepSeekError) as exc_info:
        DeepSeekClient()

    assert exc_info.value.kind == "config"
