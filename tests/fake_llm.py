"""可脚本化的 Fake LLM，返回结构贴近 DeepSeek 原生 /chat/completions。"""

import copy
import json
from typing import Any


def assistant_text(text: str, *, reasoning_content: str = "") -> dict:
    return {
        "role": "assistant",
        "content": text,
        "reasoning_content": reasoning_content,
    }


def assistant_tools(
    calls: list[dict],
    *,
    text: str = "",
    reasoning_content: str = "",
) -> dict:
    """calls 每项支持 name / args / call_id / raw_arguments。"""
    tool_calls = []
    for index, call in enumerate(calls):
        arguments = call.get("raw_arguments")
        if arguments is None:
            arguments = json.dumps(call.get("args") or {}, ensure_ascii=False)
        tool_calls.append(
            {
                "index": index,
                "id": call.get("call_id") or call.get("id") or f"call_{index + 1}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": arguments,
                },
            }
        )
    return {
        "role": "assistant",
        "content": text,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    }


def assistant_tool(
    name: str,
    args: dict | None = None,
    *,
    call_id: str = "call_1",
    text: str = "",
    reasoning_content: str = "",
    raw_arguments: str | None = None,
) -> dict:
    return assistant_tools(
        [
            {
                "name": name,
                "args": args,
                "call_id": call_id,
                "raw_arguments": raw_arguments,
            }
        ],
        text=text,
        reasoning_content=reasoning_content,
    )


class FakeLLM:
    def __init__(
        self,
        script: list[Any],
        *,
        model: str = "fake-deepseek",
        repeat_last: bool = False,
    ):
        if not script:
            raise ValueError("FakeLLM 至少需要一个响应")
        self.model = model
        self.script = list(script)
        self.repeat_last = repeat_last
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
    ) -> dict:
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
                "max_tokens": max_tokens,
            }
        )
        index = len(self.calls) - 1
        if index >= len(self.script):
            if not self.repeat_last:
                raise AssertionError(
                    f"FakeLLM 脚本已耗尽：共 {len(self.script)} 个响应，"
                    f"第 {index + 1} 次调用"
                )
            index = len(self.script) - 1

        item = self.script[index]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            item = item(copy.deepcopy(messages), copy.deepcopy(tools))
        return copy.deepcopy(item)
