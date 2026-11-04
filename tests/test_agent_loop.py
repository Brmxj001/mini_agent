"""Agent Loop：普通对话、工具调用、并行/连续调用、最大轮次。"""

from tests.fake_llm import assistant_text, assistant_tool, assistant_tools


def test_plain_conversation_returns_text_and_persists_messages(agent_factory, storage):
    agent, llm = agent_factory([assistant_text("你好，我是测试助手。")])

    result = agent.run_turn("alice", "session_1", "你好")

    assert result == "你好，我是测试助手。"
    assert len(llm.calls) == 1
    request = llm.calls[0]
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][-1] == {"role": "user", "content": "你好"}

    session = storage.get_or_create_session("alice", "session_1")
    stored = storage.active_messages(session)
    assert [m.role for m in stored] == ["user", "assistant"]
    assert stored[-1].data["content"] == "你好，我是测试助手。"


def test_single_tool_call_executes_calculator_and_returns_final_answer(
    agent_factory, storage
):
    agent, llm = agent_factory(
        [
            assistant_tool(
                "calculator", {"expression": "(12+8)*3"}, call_id="call_calc"
            ),
            assistant_text("计算结果是 60。"),
        ]
    )

    result = agent.run_turn("alice", "session_1", "请算 (12+8)*3")

    assert result == "计算结果是 60。"
    assert len(llm.calls) == 2

    second_request = llm.calls[1]["messages"]
    tool_messages = [m for m in second_request if m["role"] == "tool"]
    assistant_tool_calls = [
        m
        for m in second_request
        if m["role"] == "assistant" and m.get("tool_calls")
    ]
    assert len(assistant_tool_calls) == 1
    assert assistant_tool_calls[0]["tool_calls"][0]["id"] == "call_calc"
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_calc"
    assert "= 60" in tool_messages[0]["content"]

    session = storage.get_or_create_session("alice", "session_1")
    stored = storage.active_messages(session)
    assert [m.role for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[-1].data["content"] == "计算结果是 60。"


def test_multiple_tool_calls_in_one_response(agent_factory):
    agent, llm = agent_factory(
        [
            assistant_tools(
                [
                    {
                        "name": "calculator",
                        "args": {"expression": "2**10"},
                        "call_id": "call_calc",
                    },
                    {
                        "name": "search",
                        "args": {"query": "DeepSeek"},
                        "call_id": "call_search",
                    },
                ],
                text="同时调用两个工具。",
            ),
            assistant_text("计算与搜索都完成了。"),
        ]
    )

    result = agent.run_turn("alice", "session_1", "算 2**10 并搜索 DeepSeek")

    assert result == "计算与搜索都完成了。"
    assert len(llm.calls) == 2
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [
        "call_calc",
        "call_search",
    ]
    assert "1024" in tool_messages[0]["content"]
    assert "mock://deepseek/api" in tool_messages[1]["content"]


def test_sequential_tool_calls_across_loop_iterations(agent_factory):
    agent, llm = agent_factory(
        [
            assistant_tool("calculator", {"expression": "1+1"}, call_id="call_1"),
            assistant_tool("search", {"query": "Agent Loop"}, call_id="call_2"),
            assistant_text("两步都完成了。"),
        ]
    )

    result = agent.run_turn("alice", "session_1", "先算 1+1，再搜索 Agent Loop")

    assert result == "两步都完成了。"
    assert len(llm.calls) == 3
    tool_messages = [m for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2"]
    assert "1+1 = 2" in tool_messages[0]["content"]
    assert "mock://agent/loop" in tool_messages[1]["content"]


def test_max_iterations_terminates_without_dangling_tool_calls(
    agent_factory, storage
):
    agent, llm = agent_factory(
        [assistant_tool("calculator", {"expression": "1+1"}, call_id="call_repeat")],
        llm_kwargs={"repeat_last": True},
        max_iterations=3,
    )

    result = agent.run_turn("alice", "session_1", "一直计算")

    assert len(llm.calls) == 3
    assert "最大循环次数（3）" in result

    session = storage.get_or_create_session("alice", "session_1")
    stored = storage.active_messages(session)
    assert sum(1 for m in stored if m.role == "assistant") == 2
    assert sum(1 for m in stored if m.role == "tool") == 2
    assert stored[-1].role == "tool"

    pending = set()
    for message in stored:
        if message.role == "assistant":
            assert not pending
            pending = {
                call["id"] for call in message.data.get("tool_calls") or []
            }
        elif message.role == "tool":
            assert message.data["tool_call_id"] in pending
            pending.remove(message.data["tool_call_id"])
    assert not pending
