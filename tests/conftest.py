import pytest

from main import Agent
from storage import Storage
from tests.fake_llm import FakeLLM
from tools import build_registry


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "agent.db")


@pytest.fixture
def storage(db_path):
    store = Storage(db_path)
    yield store
    store.close()


@pytest.fixture
def agent_factory(storage):
    def make(script, *, llm_kwargs=None, **agent_kwargs):
        llm = FakeLLM(script, **(llm_kwargs or {}))
        agent = Agent(storage, llm, build_registry(), **agent_kwargs)
        return agent, llm

    return make
