"""Trace 输出与 API 异常处理。"""

import pytest

from llm import DeepSeekError
from tests.fake_llm import assistant_text, assistant_tool


def test_trace_has_full_conversation_events(agent_factory, capsys):
    agent, _ = agent_factory([assistant_text("最终回答")])

    agent.run_turn("alice", "session_1", "你好")

    output = capsys.readouterr().out
    for token in [
        "[会话]",
        "[用户]",
        "[请求]",
        "[响应]",
        "[最终]",
        "Loop 1/8",
        "model=fake-deepseek",
        "最终回答",
    ]:
        assert token in output


def test_trace_records_tool_name_arguments_result_and_duration(
    agent_factory, capsys
):
    agent, _ = agent_factory(
        [
            assistant_tool("calculator", {"expression": "7*6"}, call_id="call_1"),
            assistant_text("42"),
        ]
    )

    agent.run_turn("alice", "session_1", "算 7*6")

    output = capsys.readouterr().out
    assert "[工具]" in output
    assert "calculator" in output
    assert "7*6" in output
    assert "7*6 = 42" in output
    assert "is_error=False" in output
    assert "ms" in output
    assert "Loop 1/8" in output


def test_trace_does_not_leak_reasoning_or_api_key(
    agent_factory, storage, capsys, monkeypatch
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-super-secret-key")
    reasoning = "SECRET_REASONING_XYZ"
    agent, _ = agent_factory(
        [
            assistant_tool(
                "calculator",
                {"expression": "1+1"},
                call_id="call_1",
                reasoning_content=reasoning,
            ),
            assistant_text("2", reasoning_content="FINAL_SECRET_REASONING"),
        ]
    )

    agent.run_turn("alice", "session_1", "算 1+1")

    output = capsys.readouterr().out
    assert reasoning not in output
    assert "FINAL_SECRET_REASONING" not in output
    assert "sk-super-secret-key" not in output

    session = storage.get_or_create_session("alice", "session_1")
    stored = storage.active_messages(session)
    assert any(m.data.get("reasoning_content") == reasoning for m in stored)


@pytest.mark.parametrize(
    "kind,message",
    [
        ("timeout", "请求超时（60s）"),
        ("http", "HTTP 500：server error"),
        ("connection", "网络错误：Connection refused"),
    ],
)
def test_api_errors_return_friendly_message_and_do_not_crash(
    agent_factory, capsys, kind, message
):
    agent, llm = agent_factory([DeepSeekError(message, kind)])

    result = agent.run_turn("alice", "session_1", "你好")

    assert result == "抱歉，本轮模型调用失败，请稍后再试。"
    assert len(llm.calls) == 1
    output = capsys.readouterr().out
    assert "API" in output
    assert kind in output
