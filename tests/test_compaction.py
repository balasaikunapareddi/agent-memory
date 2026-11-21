"""Compaction trigger, fact preservation, recent turns kept."""

from agent_memory.agent import MemoryAgent
from agent_memory.compaction import Compactor, estimate_tokens, turns_tokens
from agent_memory.store import MemoryStore


def _turns(n, words=20):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": " ".join(["word"] * words)}
        for i in range(n)
    ]


def test_no_compaction_under_budget():
    compactor = Compactor(budget_tokens=10_000)
    turns = _turns(4)
    new_turns, event = compactor.maybe_compact(turns, {}, {})
    assert event is None
    assert new_turns == turns


def test_compaction_triggers_over_budget():
    compactor = Compactor(budget_tokens=50, keep_recent=2)
    turns = _turns(10)
    new_turns, event = compactor.maybe_compact(turns, {}, {})
    assert event is not None
    assert event.turns_before == 10
    assert event.turns_after == 3  # 1 summary + 2 recent
    assert event.tokens_after < event.tokens_before
    assert new_turns[0]["role"] == "summary"
    assert new_turns[-2:] == turns[-2:]  # recent turns kept verbatim


def test_summary_preserves_facts_and_preferences():
    compactor = Compactor(budget_tokens=50, keep_recent=2)
    turns = _turns(10)
    facts = {"open_issue": "router reboots at 7pm"}
    prefs = {"contact_channel": "email", "timezone": "America/Chicago"}
    _, event = compactor.maybe_compact(turns, facts, prefs)
    assert event is not None
    assert "router reboots at 7pm" in event.summary
    assert "email" in event.summary
    assert "America/Chicago" in event.summary


def test_agent_compacts_and_still_recalls(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    agent = MemoryAgent(store, "u1", session_id="s1", token_budget=120)
    agent.chat("My internet drops every evening around 7pm and the router reboots.")
    for i in range(12):
        agent.chat(f"Detail number {i}: the lights blink red for about a minute each time.")
    assert agent.compaction_events, "compaction should have fired"
    assert turns_tokens(agent.turns) < turns_tokens(
        store.get_turns("s1")
    ), "compacted context must be smaller than the raw history"
    reply = agent.chat("What do you remember about my issue?")
    assert "internet drops every evening" in reply
    store.close()


def test_estimate_tokens_is_deterministic():
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("") == 1
