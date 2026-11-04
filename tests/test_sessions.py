"""Session 隔离与重启恢复。"""

from main import Agent
from storage import Storage
from tests.fake_llm import FakeLLM, assistant_text, assistant_tool
from tools import build_registry


def test_two_sessions_for_same_user_have_isolated_messages(agent_factory, storage):
    agent, _ = agent_factory([assistant_text("回答一"), assistant_text("回答二")])

    agent.run_turn("alice", "session_1", "问题一")
    agent.run_turn("alice", "session_2", "问题二")

    session_1 = storage.get_or_create_session("alice", "session_1")
    session_2 = storage.get_or_create_session("alice", "session_2")
    assert session_1.id != session_2.id

    contents_1 = [
        m.data.get("content") for m in storage.active_messages(session_1)
    ]
    contents_2 = [
        m.data.get("content") for m in storage.active_messages(session_2)
    ]
    assert "问题一" in contents_1 and "回答一" in contents_1
    assert "问题二" not in contents_1 and "回答二" not in contents_1
    assert "问题二" in contents_2 and "回答二" in contents_2
    assert "问题一" not in contents_2 and "回答一" not in contents_2


def test_two_sessions_for_same_user_have_isolated_todos(agent_factory, storage):
    agent_1, _ = agent_factory(
        [
            assistant_tool(
                "todo",
                {"action": "add", "content": "session_1 的待办"},
                call_id="add_1",
            ),
            assistant_text("已添加"),
        ]
    )
    agent_1.run_turn("alice", "session_1", "添加待办")

    agent_2, llm_2 = agent_factory(
        [
            assistant_tool("todo", {"action": "list"}, call_id="list_2"),
            assistant_text("查询完成"),
        ]
    )
    agent_2.run_turn("alice", "session_2", "查看待办")

    session_1 = storage.get_or_create_session("alice", "session_1")
    session_2 = storage.get_or_create_session("alice", "session_2")
    assert [t.content for t in storage.list_todos("alice", session_1.id)] == [
        "session_1 的待办"
    ]
    assert storage.list_todos("alice", session_2.id) == []

    tool_messages = [m for m in llm_2.calls[1]["messages"] if m["role"] == "tool"]
    assert "当前会话没有待办" in tool_messages[0]["content"]


def test_different_users_with_same_session_name_are_isolated(
    agent_factory, storage
):
    alice_agent, _ = agent_factory(
        [
            assistant_tool(
                "todo",
                {"action": "add", "content": "alice 的任务"},
                call_id="add_alice",
            ),
            assistant_text("已添加"),
        ]
    )
    alice_agent.run_turn("alice", "session_1", "添加待办")

    bob_agent, llm_bob = agent_factory(
        [
            assistant_tool("todo", {"action": "list"}, call_id="list_bob"),
            assistant_text("查询完成"),
        ]
    )
    bob_agent.run_turn("bob", "session_1", "查看待办")

    alice_session = storage.get_or_create_session("alice", "session_1")
    bob_session = storage.get_or_create_session("bob", "session_1")
    assert alice_session.id != bob_session.id
    assert [
        t.content for t in storage.list_todos("alice", alice_session.id)
    ] == ["alice 的任务"]
    assert storage.list_todos("bob", bob_session.id) == []

    alice_contents = [
        m.data.get("content")
        for m in storage.active_messages(alice_session)
    ]
    bob_contents = [
        m.data.get("content") for m in storage.active_messages(bob_session)
    ]
    assert "查看待办" not in alice_contents
    assert "添加待办" not in bob_contents
    assert "添加待办" not in [
        m.get("content") for m in llm_bob.calls[0]["messages"]
    ]


def test_reopening_database_restores_history_todos_and_conversation(db_path):
    store = Storage(db_path)
    first_agent = Agent(
        store, FakeLLM([assistant_text("记住了。")]), build_registry()
    )
    first_agent.run_turn("alice", "session_1", "请记住：明天写周报")

    session = store.get_or_create_session("alice", "session_1")
    session_id = session.id
    store.add_todo("alice", session_id, "写周报")
    store.close()

    reopened = Storage(db_path)
    restored = reopened.get_or_create_session("alice", "session_1")
    assert restored.id == session_id
    stored = reopened.active_messages(restored)
    assert any(
        m.data.get("content") == "请记住：明天写周报" for m in stored
    )
    assert [t.content for t in reopened.list_todos("alice", restored.id)] == [
        "写周报"
    ]

    second_llm = FakeLLM([assistant_text("你之前说要写周报。")])
    second_agent = Agent(reopened, second_llm, build_registry())
    result = second_agent.run_turn("alice", "session_1", "我之前说了什么？")

    assert result == "你之前说要写周报。"
    sent_messages = second_llm.calls[0]["messages"]
    assert any(
        m["role"] == "user" and m.get("content") == "请记住：明天写周报"
        for m in sent_messages
    )
    reopened.close()
