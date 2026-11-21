"""Memory CRUD + persistence across instances (restarts)."""

import pytest

from agent_memory.store import MemoryStore


@pytest.fixture()
def store_path(tmp_path):
    return tmp_path / "memory.db"


@pytest.fixture()
def store(store_path):
    s = MemoryStore(store_path)
    yield s
    s.close()


def test_remember_and_recall(store):
    store.remember_fact("u1", "open_issue", "router reboots at 7pm")
    assert store.recall_facts("u1") == {"open_issue": "router reboots at 7pm"}
    assert store.recall_facts("u1", "open_issue") == {"open_issue": "router reboots at 7pm"}
    assert store.recall_facts("u1", "missing") == {}


def test_remember_overwrites(store):
    store.remember_fact("u1", "plan", "basic")
    store.remember_fact("u1", "plan", "premium")
    assert store.recall_facts("u1") == {"plan": "premium"}


def test_forget_fact(store):
    store.remember_fact("u1", "open_issue", "x")
    assert store.forget_fact("u1", "open_issue") is True
    assert store.recall_facts("u1") == {}
    assert store.forget_fact("u1", "open_issue") is False


def test_facts_are_per_user(store):
    store.remember_fact("u1", "open_issue", "a")
    store.remember_fact("u2", "open_issue", "b")
    assert store.recall_facts("u1") == {"open_issue": "a"}
    assert store.recall_facts("u2") == {"open_issue": "b"}


def test_preferences_round_trip(store):
    store.set_preference("u1", "contact_channel", "email")
    store.set_preference("u1", "timezone", "America/Chicago")
    assert store.get_preferences("u1") == {
        "contact_channel": "email",
        "timezone": "America/Chicago",
    }


def test_persistence_across_instances(store_path):
    first = MemoryStore(store_path)
    first.remember_fact("u1", "open_issue", "router reboots at 7pm")
    first.set_preference("u1", "contact_channel", "email")
    first.add_turn("s1", "u1", "user", "hello")
    first.close()

    # A new instance over the same file = a process restart.
    second = MemoryStore(store_path)
    assert second.recall_facts("u1") == {"open_issue": "router reboots at 7pm"}
    assert second.get_preferences("u1") == {"contact_channel": "email"}
    assert second.get_turns("s1") == [{"role": "user", "content": "hello"}]
    second.close()
