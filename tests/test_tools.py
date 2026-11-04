"""Tool Registry 与三个内置工具。"""

import pytest

from tests.fake_llm import assistant_text, assistant_tool
from tools import Tool, ToolContext, ToolRegistry, build_registry


@pytest.fixture
def registry():
    return build_registry()


@pytest.fixture
def ctx(storage):
    session = storage.get_or_create_session("alice", "session_1")
    return ToolContext(storage=storage, user_id="alice", session_id=session.id)


def test_registry_registers_all_tools_and_exposes_schemas(registry):
    assert set(registry.names()) == {"calculator", "search", "todo"}

    specs = registry.specs()
    assert len(specs) == 3
    by_name = {spec["function"]["name"]: spec for spec in specs}
    assert set(by_name) == {"calculator", "search", "todo"}
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["parameters"]["type"] == "object"
    assert by_name["calculator"]["function"]["parameters"]["required"] == [
        "expression"
    ]
    assert by_name["search"]["function"]["parameters"]["required"] == ["query"]
    assert by_name["todo"]["function"]["parameters"]["required"] == ["action"]


def test_agent_passes_tool_schemas_to_llm(agent_factory):
    agent, llm = agent_factory([assistant_text("ok")])

    agent.run_turn("alice", "session_1", "你好")

    tools = llm.calls[0]["tools"]
    names = {tool["function"]["name"] for tool in tools}
    assert names == {"calculator", "search", "todo"}
    calculator = next(
        tool for tool in tools if tool["function"]["name"] == "calculator"
    )
    assert calculator["type"] == "function"
    assert calculator["function"]["parameters"]["required"] == ["expression"]


def test_registry_finds_and_executes_tool_by_name(registry, ctx):
    result = registry.execute("calculator", {"expression": "2+3"}, ctx)

    assert result.is_error is False
    assert result.content == "2+3 = 5"


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+3", "2+3 = 5"),
        ("(12+8)*3", "(12+8)*3 = 60"),
        ("2**10", "2**10 = 1024"),
        ("10/4", "10/4 = 2.5"),
        ("7*6", "7*6 = 42"),
    ],
)
def test_calculator_valid_expressions(registry, ctx, expression, expected):
    result = registry.execute("calculator", {"expression": expression}, ctx)

    assert result.is_error is False
    assert result.content == expected


@pytest.mark.parametrize(
    "expression",
    ["1/0", "__import__('os')", "(1+", "2**999", "", "abc"],
)
def test_calculator_rejects_invalid_expressions(registry, ctx, expression):
    result = registry.execute("calculator", {"expression": expression}, ctx)

    assert result.is_error is True
    assert result.content.startswith("[工具错误]")


def test_search_returns_mock_results(registry, ctx):
    result = registry.execute(
        "search", {"query": "Python asyncio", "top_k": 1}, ctx
    )

    assert result.is_error is False
    assert "Mock 搜索结果" in result.content
    assert "mock://python/asyncio" in result.content
    assert result.content.count("来源:") == 1


def test_search_returns_explicit_no_match_message(registry, ctx):
    result = registry.execute("search", {"query": "完全不存在的关键词xyz"}, ctx)

    assert result.is_error is False
    assert "未找到" in result.content


def test_todo_add_list_and_complete(registry, ctx, storage):
    added = registry.execute("todo", {"action": "add", "content": "写 README"}, ctx)
    assert added.is_error is False

    todos = storage.list_todos("alice", ctx.session_id)
    assert len(todos) == 1
    todo_id = todos[0].id
    assert todos[0].content == "写 README"
    assert todos[0].status == "pending"

    listed = registry.execute("todo", {"action": "list"}, ctx)
    assert listed.is_error is False
    assert f"#{todo_id} [ ] 写 README" in listed.content

    completed = registry.execute(
        "todo", {"action": "complete", "todo_id": todo_id}, ctx
    )
    assert completed.is_error is False
    assert "已完成待办" in completed.content

    completed_again = registry.execute(
        "todo", {"action": "complete", "todo_id": todo_id}, ctx
    )
    assert completed_again.is_error is False
    assert storage.list_todos("alice", ctx.session_id)[0].status == "completed"

    listed_again = registry.execute("todo", {"action": "list"}, ctx)
    assert f"#{todo_id} [x] 写 README" in listed_again.content


def test_todo_list_when_empty(registry, ctx):
    result = registry.execute("todo", {"action": "list"}, ctx)

    assert result.is_error is False
    assert result.content == "当前会话没有待办。"


def test_unknown_tool_returns_error_result(registry, ctx):
    result = registry.execute("weather", {"city": "Paris"}, ctx)

    assert result.is_error is True
    assert "不存在名为 weather" in result.content


@pytest.mark.parametrize("args", ["not-a-dict", [], 123])
def test_non_object_arguments_are_rejected(registry, ctx, args):
    result = registry.execute("calculator", args, ctx)

    assert result.is_error is True
    assert "JSON 对象" in result.content


@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("calculator", {}, "expression"),
        ("search", {}, "query"),
        ("todo", {"action": "complete"}, "todo_id"),
        ("todo", {"action": "unknown"}, "action"),
    ],
)
def test_missing_or_invalid_tool_arguments_are_rejected(
    registry, ctx, name, args, expected
):
    result = registry.execute(name, args, ctx)

    assert result.is_error is True
    assert expected in result.content


def test_registry_captures_arbitrary_tool_exception():
    def boom(args, ctx):
        raise RuntimeError("内部故障")

    registry = ToolRegistry(
        [
            Tool(
                name="boom",
                description="",
                input_schema={"type": "object"},
                execute=boom,
            )
        ]
    )

    result = registry.execute("boom", {}, None)

    assert result.is_error is True
    assert "RuntimeError" in result.content
    assert "内部故障" in result.content


@pytest.mark.parametrize(
    "tool_response,expected",
    [
        (
            assistant_tool(
                "calculator", {"expression": "1/0"}, call_id="call_div"
            ),
            "除数不能为 0",
        ),
        (
            assistant_tool("calculator", {}, call_id="call_missing"),
            "expression",
        ),
        (
            assistant_tool(
                "weather", {"city": "Paris"}, call_id="call_weather"
            ),
            "不存在名为 weather",
        ),
        (
            assistant_tool(
                "calculator",
                raw_arguments="{not-json",
                call_id="call_bad_json",
            ),
            "不是合法 JSON",
        ),
    ],
)
def test_agent_feeds_tool_errors_back_to_model(
    agent_factory, tool_response, expected
):
    agent, llm = agent_factory(
        [tool_response, assistant_text("工具执行失败，我已向用户说明。")]
    )

    result = agent.run_turn("alice", "session_1", "调用工具")

    assert result == "工具执行失败，我已向用户说明。"
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"].startswith("[工具错误]")
    assert expected in tool_messages[0]["content"]
