"""One-command demo: memory across a restart, then compaction.

Session 1 — the user reports an issue and states two preferences.
Restart   — a brand-new agent instance over the same store file.
Session 2 — the agent recalls the issue and applies the preferences
            without being re-told, then a long conversation blows
            the token budget, compaction folds the old turns into a
            summary, and the agent still answers from memory.

Run:  agent-memory-demo        (or: python -m agent_memory.demo)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .agent import MemoryAgent
from .store import MemoryStore


def _say(agent: MemoryAgent, user: str) -> str:
    print(f"  user : {user}")
    reply = agent.chat(user)
    print(f"  agent: {reply}")
    return reply


def main() -> None:
    store_path = Path(tempfile.mkdtemp()) / "memory.db"
    user_id = "user-42"

    print("=" * 68)
    print("SESSION 1 — first contact")
    print("=" * 68)
    store = MemoryStore(store_path)
    agent = MemoryAgent(store, user_id, session_id="session-1")
    _say(agent, "Hi, my internet drops every evening around 7pm and the router reboots.")
    _say(agent, "I prefer email over calls, and my timezone is America/Chicago.")
    print(f"\n  remembered after session 1: {agent.remembered()}")

    print()
    print("=" * 68)
    print("RESTART — new agent instance, same store file")
    print("=" * 68)
    store.close()
    store = MemoryStore(store_path)
    agent = MemoryAgent(store, user_id, session_id="session-2", token_budget=300)
    print("  (the user does not re-explain anything below)")
    _say(agent, "Hi again — any update on my issue from yesterday?")

    print()
    print("=" * 68)
    print("LONG CONVERSATION — filling the context window")
    print("=" * 68)
    fillers = [
        "The lights on the router blink red for about a minute.",
        "It happens on weekdays more than weekends, I think.",
        "My neighbor on the same street has the same problem.",
        "I already tried unplugging it for thirty seconds.",
        "The firmware page says everything is up to date.",
        "Streaming is fine in the morning, it is only evenings.",
        "Speed test shows 300 down when it works.",
        "The drop lasts maybe two or three minutes each time.",
        "My work laptop loses the VPN connection when it happens.",
        "It started about two weeks ago, roughly.",
    ]
    for line in fillers:
        _say(agent, line)

    for event in agent.compaction_events:
        print(
            f"\n  [compaction] {event.turns_before} turns / {event.tokens_before} tokens "
            f"-> {event.turns_after} turns / {event.tokens_after} tokens"
        )
        print(f"  [compaction] summary kept: {event.summary.splitlines()[0]}")

    print()
    print("=" * 68)
    print("AFTER COMPACTION — memory still intact")
    print("=" * 68)
    _say(agent, "So, what do you remember about my issue?")
    print(f"\n  final memory: {agent.remembered()}")
    store.close()
    print("\nDemo complete: recalled across a restart, compacted without forgetting.")


if __name__ == "__main__":
    main()
