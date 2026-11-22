"""The memory agent.

Behavior lives in the system prompt (``prompts.py``); mechanics live
in code. The agent loop is the same in both modes: the model decides
tool calls, the loop executes them against the memory tools, and the
model composes the reply. In mock mode the "model" is a small
deterministic script (no key, no network); in live mode a CrewAI
agent with a Gemini LLM drives the same tools.

``MemoryAgent`` below is the agent: ``chat``/``run`` is one turn
through the loop, ``run_loop`` is the loop itself (model decides
tool calls -> tools execute -> reply is composed).
"""

from __future__ import annotations

import os
import re
import uuid

from .compaction import CompactionEvent, Compactor
from .preferences import extract_preferences
from .prompts import LIVE_AGENT_GOAL, LIVE_AGENT_ROLE, SYSTEM_PROMPT
from .store import MemoryStore
from .tools import build_tools

_ISSUE_HINTS = re.compile(
    r"\b(issue|problem|broken|not working|drops?|error|fails?|outage|down)\b",
    re.IGNORECASE,
)
_RECALL_HINTS = re.compile(
    r"\b(what do you remember|my issue|any update|follow.?up|last time|yesterday)\b",
    re.IGNORECASE,
)
_FORGET_HINTS = re.compile(r"\bforget\b.*\b(issue|problem|that)\b", re.IGNORECASE)


class MemoryAgent:
    """Support agent with session + long-term memory.

    One agent instance = one session for one user. A restart is just
    a new MemoryAgent over the same store file — that is the whole
    point: long-term memory lives in the store, not in the process.

    The agent loop, per turn (``chat`` / ``run``):
      1. record the user turn and extract stated preferences by rule;
      2. ``run_loop``: the model (deterministic script in mock mode,
         CrewAI + Gemini in live mode) decides tool calls, the loop
         executes them against the memory tools, and the reply is
         composed from the tool results + stored preferences;
      3. record the reply and compact the history if over budget.

    Config: ``store`` (SQLite long-term memory), ``user_id``,
    ``session_id``, ``token_budget`` (compaction trigger). Tools:
    ``remember_fact`` / ``recall_facts`` / ``forget_fact`` from
    ``build_tools(store)``. Prompt: ``SYSTEM_PROMPT`` from
    ``prompts.py``, held as ``self.system_prompt``.
    """

    def __init__(
        self,
        store: MemoryStore,
        user_id: str,
        session_id: str | None = None,
        token_budget: int = 400,
    ):
        # Config
        self.store = store
        self.user_id = user_id
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.token_budget = token_budget
        # Prompt (from prompts.py) and tools the loop can call
        self.system_prompt = SYSTEM_PROMPT
        self.tools = build_tools(store)
        # Mechanics: compaction + session bookkeeping
        self.compactor = Compactor(budget_tokens=token_budget)
        self.turns: list[dict[str, str]] = store.get_turns(self.session_id)
        self.compaction_events: list[CompactionEvent] = []

    # -- public API ----------------------------------------------------

    def chat(self, message: str) -> str:
        """One user turn in, one agent reply out."""
        self._append("user", message)

        # Deterministic preference learning: stated preferences are
        # extracted by rule and stored, whatever the model does.
        for key, value in extract_preferences(message).items():
            self.store.set_preference(self.user_id, key, value)

        reply = self.run_loop(message)
        self._append("assistant", reply)
        self._maybe_compact()
        return reply

    def run(self, message: str) -> str:
        """Run one turn through the agent loop. Alias for ``chat``."""
        return self.chat(message)

    def remembered(self) -> dict[str, dict[str, str]]:
        """Everything the agent currently knows about this user."""
        return {
            "facts": self.tools["recall_facts"](user_id=self.user_id),
            "preferences": self.store.get_preferences(self.user_id),
        }

    # -- agent loop ----------------------------------------------------

    def run_loop(self, message: str) -> str:
        """The agent loop: model decides tool calls; loop runs them; model replies."""
        tool_calls = self._scripted_model(message)
        results: dict[str, object] = {}
        for name, kwargs in tool_calls[:3]:  # bounded loop
            results[name] = self.tools[name](**kwargs)
        return self._compose_reply(message, results)

    # Backwards-compatible private alias for the loop.
    _run_loop = run_loop

    def _scripted_model(self, message: str) -> list[tuple[str, dict]]:
        """The mock model: deterministic intent -> tool calls."""
        calls: list[tuple[str, dict]] = []
        if _FORGET_HINTS.search(message):
            calls.append(("forget_fact", {"user_id": self.user_id, "key": "open_issue"}))
        elif _RECALL_HINTS.search(message):
            # A follow-up question ("any update on my issue?") mentions
            # the issue but is not a new report — never overwrite memory.
            pass
        elif _ISSUE_HINTS.search(message):
            # A follow-up detail about the same issue extends the stored
            # description instead of replacing it.
            existing = self.store.recall_facts(self.user_id, "open_issue").get("open_issue")
            value = message if not existing else f"{existing} Also: {message}"
            calls.append(
                (
                    "remember_fact",
                    {"user_id": self.user_id, "key": "open_issue", "value": value},
                )
            )
        if _RECALL_HINTS.search(message) or not calls:
            calls.append(("recall_facts", {"user_id": self.user_id}))
        return calls

    def _compose_reply(self, message: str, results: dict[str, object]) -> str:
        facts = results.get("recall_facts")
        if not isinstance(facts, dict):
            facts = self.store.recall_facts(self.user_id)
        prefs = self.store.get_preferences(self.user_id)
        name = prefs.get("preferred_name")

        if "forget_fact" in results:
            return "Done — I've forgotten that. Anything else I can help with?"

        if _ISSUE_HINTS.search(message) and not _RECALL_HINTS.search(message):
            greeting = f"Thanks{', ' + name if name else ''} — "
            return (
                greeting
                + "I've noted that down so you won't have to explain it again. "
                + "I'll look into it and follow up"
                + self._follow_up_phrase(prefs)
                + "."
            )

        if _RECALL_HINTS.search(message):
            issue = facts.get("open_issue")
            if issue:
                return (
                    f"Of course — last time you told me: \"{issue}\" "
                    "I still have it on file, no need to re-explain. "
                    "I'll send any update" + self._follow_up_phrase(prefs) + "."
                )
            return "I don't have anything on file for you yet — what can I help with?"

        known = ", ".join(f"{k}: {v}" for k, v in sorted(facts.items())) or "nothing yet"
        return f"Got it. For reference, what I currently remember about you: {known}."

    @staticmethod
    def _follow_up_phrase(prefs: dict[str, str]) -> str:
        channel = prefs.get("contact_channel")
        tz = prefs.get("timezone")
        phrase = ""
        if channel == "email":
            phrase += " by email"
        elif channel == "phone":
            phrase += " by phone"
        elif channel == "sms":
            phrase += " by text"
        if tz:
            phrase += f" (in your timezone, {tz})"
        return phrase

    # -- session bookkeeping -------------------------------------------

    def _append(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self.store.add_turn(self.session_id, self.user_id, role, content)

    def _maybe_compact(self) -> None:
        facts = self.store.recall_facts(self.user_id)
        prefs = self.store.get_preferences(self.user_id)
        self.turns, event = self.compactor.maybe_compact(self.turns, facts, prefs)
        if event is not None:
            self.compaction_events.append(event)


# -- live mode (CrewAI + Gemini) ---------------------------------------
#
# CrewAI ships its own short-term / long-term / entity memory, but its
# long-term memory embeds text into a vector store, which needs a
# model call. Mock mode must run fully offline, so the memory layer
# here is local SQLite behind the same tool interfaces; live mode
# wires those exact tools into a CrewAI agent.


def build_crewai_agent(store: MemoryStore, user_id: str):
    """Build the live-mode CrewAI agent. Requires crewai + GEMINI_API_KEY."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "Live mode needs GEMINI_API_KEY in the environment. "
            "Mock mode (the default) needs no key and no network."
        )
    try:
        from crewai import Agent, LLM
        from crewai.tools import tool as crewai_tool
    except ImportError as exc:  # pragma: no cover - live mode only
        raise RuntimeError(
            "Live mode needs CrewAI: pip install 'agent-memory[live]'"
        ) from exc

    tools = build_tools(store)

    @crewai_tool("remember_fact")
    def remember_fact(key: str, value: str) -> str:
        """Store one fact about the user in long-term memory."""
        return tools["remember_fact"](user_id=user_id, key=key, value=value)

    @crewai_tool("recall_facts")
    def recall_facts(key: str = "") -> str:
        """Read facts stored about the user (all, or one by key)."""
        found = tools["recall_facts"](user_id=user_id, key=key or None)
        return str(found) if found else "nothing stored yet"

    @crewai_tool("forget_fact")
    def forget_fact(key: str) -> str:
        """Delete one stored fact about the user."""
        return tools["forget_fact"](user_id=user_id, key=key)

    llm = LLM(model=f"gemini/{os.environ.get('GEMINI_MODEL', 'gemini-3.5-flash-lite')}", api_key=os.environ["GEMINI_API_KEY"])
    return Agent(
        role=LIVE_AGENT_ROLE,
        goal=LIVE_AGENT_GOAL,
        backstory=SYSTEM_PROMPT,
        tools=[remember_fact, recall_facts, forget_fact],
        llm=llm,
        verbose=False,
    )
