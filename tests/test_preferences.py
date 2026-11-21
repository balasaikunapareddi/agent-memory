"""Preference extraction rules + application in later sessions."""

from agent_memory.agent import MemoryAgent
from agent_memory.preferences import extract_preferences
from agent_memory.store import MemoryStore


def test_extract_contact_channel():
    assert extract_preferences("I prefer email over calls") == {"contact_channel": "email"}
    assert extract_preferences("please, I prefer texts instead of calls") == {
        "contact_channel": "sms"
    }
    assert extract_preferences("I prefer phone over email") == {"contact_channel": "phone"}


def test_extract_timezone():
    assert extract_preferences("my timezone is America/Chicago") == {
        "timezone": "America/Chicago"
    }


def test_extract_name_and_email():
    assert extract_preferences("call me Sam") == {"preferred_name": "Sam"}
    assert extract_preferences("my email is sam@example.com") == {
        "email": "sam@example.com"
    }


def test_extract_multiple_from_one_message():
    found = extract_preferences(
        "I prefer email over calls, and my timezone is America/Chicago."
    )
    assert found == {"contact_channel": "email", "timezone": "America/Chicago"}


def test_no_preference_in_plain_message():
    assert extract_preferences("the router is blinking red again") == {}


def test_preferences_applied_in_later_session(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    agent = MemoryAgent(store, "u1", session_id="s1")
    agent.chat("My internet drops every evening around 7pm.")
    agent.chat("I prefer email over calls, and my timezone is America/Chicago.")
    store.close()

    # Restart: new store + agent instances, same file.
    store = MemoryStore(path)
    agent = MemoryAgent(store, "u1", session_id="s2")
    reply = agent.chat("Any update on my issue?")
    assert "by email" in reply
    assert "America/Chicago" in reply
    store.close()
