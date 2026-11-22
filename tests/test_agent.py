"""End-to-end agent behavior: the demo path, tools, forget."""

from agent_memory.agent import MemoryAgent
from agent_memory.prompts import SYSTEM_PROMPT
from agent_memory.store import MemoryStore
from agent_memory.tools import build_tools


def test_demo_path_recall_across_restart(tmp_path):
    path = tmp_path / "memory.db"

    store = MemoryStore(path)
    agent = MemoryAgent(store, "user-42", session_id="session-1")
    agent.chat("Hi, my internet drops every evening around 7pm and the router reboots.")
    agent.chat("I prefer email over calls, and my timezone is America/Chicago.")
    remembered = agent.remembered()
    assert "internet drops" in remembered["facts"]["open_issue"]
    assert remembered["preferences"] == {
        "contact_channel": "email",
        "timezone": "America/Chicago",
    }
    store.close()

    # Restart: fresh instances, same store file. No re-explaining.
    store = MemoryStore(path)
    agent = MemoryAgent(store, "user-42", session_id="session-2")
    reply = agent.chat("Hi again — any update on my issue from yesterday?")
    assert "internet drops every evening" in reply
    assert "by email" in reply
    store.close()


def test_agent_forgets_on_request(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    agent = MemoryAgent(store, "u1", session_id="s1")
    agent.chat("My internet drops every evening around 7pm.")
    assert "open_issue" in agent.remembered()["facts"]
    agent.chat("Please forget that issue, it is fixed now.")
    assert "open_issue" not in agent.remembered()["facts"]
    store.close()


def test_tools_are_typed_and_bound(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    tools = build_tools(store)
    assert set(tools) == {"remember_fact", "recall_facts", "forget_fact"}
    for tool in tools.values():
        assert tool.description
        assert tool.input_schema["type"] == "object"
        assert "user_id" in tool.input_schema["properties"]
    tools["remember_fact"](user_id="u1", key="plan", value="premium")
    assert tools["recall_facts"](user_id="u1") == {"plan": "premium"}
    tools["forget_fact"](user_id="u1", key="plan")
    assert tools["recall_facts"](user_id="u1") == {}
    store.close()


def test_system_prompt_is_behavior_only():
    # The prompt tells the agent how to behave; mechanics (extraction
    # rules, budgets) live in code and must not leak into it.
    assert "remember_fact" in SYSTEM_PROMPT
    assert "recall_facts" in SYSTEM_PROMPT
    assert "token" not in SYSTEM_PROMPT.lower()
    assert "regex" not in SYSTEM_PROMPT.lower()


def test_session_turns_persist(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    agent = MemoryAgent(store, "u1", session_id="s1")
    agent.chat("hello there")
    store.close()

    store = MemoryStore(path)
    turns = store.get_turns("s1")
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["content"] == "hello there"
    store.close()
