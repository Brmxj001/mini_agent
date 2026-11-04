"""DeepSeek 原生 API 客户端（httpx 直连，无任何 Agent 框架）。"""

import os
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def _load_demo_env() -> None:
    """加载仓库内的 demo.env；已存在的环境变量优先，便于用户覆盖演示 Key。"""
    path = Path(__file__).with_name("demo.env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


_load_demo_env()


class DeepSeekError(Exception):
    """统一的模型调用异常。kind: config / timeout / connection / http / protocol。"""

    def __init__(self, message: str, kind: str = "api"):
        super().__init__(message)
        self.kind = kind


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise DeepSeekError("缺少 DEEPSEEK_API_KEY 环境变量", "config")
        self.base_url = (
            base_url or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> dict:
        """发送一轮请求，返回原始 assistant 消息（含 reasoning_content / tool_calls）。"""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise DeepSeekError(f"请求超时（{self.timeout:g}s）", "timeout") from exc
        except httpx.HTTPError as exc:
            raise DeepSeekError(f"网络错误：{exc}", "connection") from exc

        if response.status_code != 200:
            detail = response.text[:500].replace(self.api_key, "***")
            raise DeepSeekError(f"HTTP {response.status_code}：{detail}", "http")

        try:
            data = response.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise DeepSeekError(f"响应格式异常：{response.text[:300]}", "protocol") from exc

        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise DeepSeekError("响应中缺少合法的 assistant 消息", "protocol")
        return message

    def close(self) -> None:
        self._client.close()
