"""Context、Memory 与历史压缩。"""

from context import SYSTEM_PROMPT, maybe_compact
from main import Agent
from storage import Storage
from tests.fake_llm import FakeLLM, assistant_text, assistant_tool
from tools import build_registry


def _assert_tool_pairs(messages):
    pending = set()
    for message in messages:
        data = message.data
        if data["role"] == "assistant":
            assert not pending
            pending = {call["id"] for call in data.get("tool_calls") or []}
        elif data["role"] == "tool":
            assert data["tool_call_id"] in pending
            pending.remove(data["tool_call_id"])
    assert not pending


def test_new_message_is_included_with_system_prompt(agent_factory):
    agent, llm = agent_factory([assistant_text("收到")])

    agent.run_turn("alice", "session_1", "新消息")

    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(SYSTEM_PROMPT)
    assert messages[-1] == {"role": "user", "content": "新消息"}


def test_pure_conversation_followup_contains_history(agent_factory):
    agent, llm = agent_factory(
        [assistant_text("第一轮回答"), assistant_text("第二轮回答")]
    )

    agent.run_turn("alice", "session_1", "第一轮问题")
    agent.run_turn("alice", "session_1", "第二轮问题")

    messages = llm.calls[1]["messages"]
    contents = [m.get("content") for m in messages]
    assert "第一轮问题" in contents
    assert "第一轮回答" in contents
    assert messages[-1] == {"role": "user", "content": "第二轮问题"}


def test_tool_followup_contains_tool_call_and_result(agent_factory):
    opaque_reasoning = "OPAQUE_REASONING_SENTINEL"
    agent, llm = agent_factory(
        [
            assistant_tool(
                "calculator",
                {"expression": "6*7"},
                call_id="call_42",
                reasoning_content=opaque_reasoning,
            ),
            assistant_text("42"),
            assistant_text("上一轮用了 calculator。"),
        ]
    )

    agent.run_turn("alice", "session_1", "算 6*7")
    agent.run_turn("alice", "session_1", "刚才用了什么工具？")

    messages = llm.calls[2]["messages"]
    assistant_calls = [
        m
        for m in messages
        if m["role"] == "assistant" and m.get("tool_calls")
    ]
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert assistant_calls
    assert assistant_calls[0]["tool_calls"][0]["id"] == "call_42"
    assert assistant_calls[0]["reasoning_content"] == opaque_reasoning
    assert tool_messages[0]["tool_call_id"] == "call_42"
    assert "6*7 = 42" in tool_messages[0]["content"]


def test_summary_is_loaded_into_system_prompt(storage, agent_factory):
    session = storage.get_or_create_session("alice", "session_1")
    storage.set_summary(session.id, "用户偏好：回答保持简洁。", 0)

    agent, llm = agent_factory([assistant_text("好的")])
    agent.run_turn("alice", "session_1", "你好")

    system_message = llm.calls[0]["messages"][0]
    assert system_message["role"] == "system"
    assert "用户偏好：回答保持简洁。" in system_message["content"]


def test_compaction_triggers_persists_and_keeps_recent(
    agent_factory, storage, db_path
):
    agent, llm = agent_factory(
        [
            assistant_text("第一轮回答"),
            assistant_text("第二轮回答"),
            assistant_text("第三轮回答"),
        ],
        compact_threshold=4,
        keep_recent=2,
    )

    agent.run_turn("alice", "session_1", "第一轮问题：暗号蓝色海豚")
    agent.run_turn("alice", "session_1", "第二轮问题")
    agent.run_turn("alice", "session_1", "第三轮问题")

    session = storage.get_or_create_session("alice", "session_1")
    assert session.summary
    assert "蓝色海豚" in session.summary
    assert session.summary_until_message_id > 0

    third_request = llm.calls[2]["messages"]
    system_content = third_request[0]["content"]
    assert "历史对话摘要" in system_content
    assert "蓝色海豚" in system_content
    assert any(
        m["role"] == "user" and m.get("content") == "第二轮问题"
        for m in third_request
    )
    assert third_request[-1]["content"] == "第三轮问题"
    assert not any(
        m["role"] == "user" and m.get("content") == "第一轮问题：暗号蓝色海豚"
        for m in third_request
    )

    reopened = Storage(db_path)
    restored = reopened.get_or_create_session("alice", "session_1")
    assert restored.summary == session.summary
    assert (
        restored.summary_until_message_id
        == session.summary_until_message_id
    )
    reopened.close()


def test_compaction_does_not_split_tool_calls_and_results(storage):
    session = storage.get_or_create_session("alice", "session_1")

    for index in range(1, 6):
        storage.append_message(
            session.id,
            "user",
            {"role": "user", "content": f"第 {index} 轮问题"},
        )
        calls = [
            {
                "index": 0,
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "1+1"}',
                },
            }
        ]
        if index == 3:
            calls.append(
                {
                    "index": 1,
                    "id": f"call_{index}_b",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "天气"}',
                    },
                }
            )
        assistant = {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"SECRET_REASONING_{index}",
            "tool_calls": calls,
        }
        tool_messages = [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": f"结果 {index}-{call['id']}",
            }
            for call in calls
        ]
        storage.append_exchange(session.id, assistant, tool_messages)
        storage.append_message(
            session.id,
            "assistant",
            {
                "role": "assistant",
                "content": f"第 {index} 轮回答",
                "reasoning_content": f"SECRET_FINAL_{index}",
            },
        )

    storage.append_message(
        session.id, "user", {"role": "user", "content": "当前问题"}
    )

    session = storage.get_session(session.id)
    info = maybe_compact(
        storage, session, compact_threshold=6, keep_recent=4
    )

    assert info is not None
    session = storage.get_session(session.id)
    active = storage.active_messages(session)
    assert active
    assert active[0].data["role"] == "user"
    assert isinstance(active[0].data["content"], str)
    assert active[-1].data["content"] == "当前问题"
    assert len(active) >= 4
    assert "第 1 轮问题" in session.summary
    assert "助手调用 calculator" in session.summary
    assert "SECRET_REASONING" not in session.summary
    assert all(m.id > session.summary_until_message_id for m in active)
    _assert_tool_pairs(active)


def test_compaction_not_triggered_below_threshold(storage):
    session = storage.get_or_create_session("alice", "session_1")
    storage.append_message(
        session.id, "user", {"role": "user", "content": "hi"}
    )
    storage.append_message(
        session.id, "assistant", {"role": "assistant", "content": "hello"}
    )
    session = storage.get_session(session.id)

    result = maybe_compact(
        storage, session, compact_threshold=5, keep_recent=2
    )

    assert result is None
    assert storage.get_session(session.id).summary == ""


def test_summary_is_reloaded_after_reopen(db_path):
    store = Storage(db_path)
    session = store.get_or_create_session("alice", "session_1")
    store.set_summary(session.id, "历史暗号：蓝色海豚", 0)
    store.close()

    reopened = Storage(db_path)
    llm = FakeLLM([assistant_text("蓝色海豚")])
    agent = Agent(reopened, llm, build_registry())

    result = agent.run_turn("alice", "session_1", "历史暗号是什么？")

    assert result == "蓝色海豚"
    assert "历史暗号：蓝色海豚" in llm.calls[0]["messages"][0]["content"]
    reopened.close()
