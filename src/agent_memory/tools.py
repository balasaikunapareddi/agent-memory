"""Typed memory tools.

The agent never touches the store directly — it calls these three
tools. Each tool carries a name, a description, and a JSON-schema
input definition, the same shape a CrewAI (or any framework) tool
expects. In live mode these exact definitions are wrapped as CrewAI
tools; in mock mode the scripted model emits calls to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .store import MemoryStore


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., Any]

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


def build_tools(store: MemoryStore) -> dict[str, Tool]:
    """Create the three memory tools bound to a store."""

    def remember_fact(user_id: str, key: str, value: str) -> str:
        store.remember_fact(user_id, key, value)
        return f"remembered {key} for {user_id}"

    def recall_facts(user_id: str, key: str | None = None) -> dict[str, str]:
        return store.recall_facts(user_id, key)

    def forget_fact(user_id: str, key: str) -> str:
        deleted = store.forget_fact(user_id, key)
        return f"forgot {key}" if deleted else f"nothing stored under {key}"

    user_id_schema = {"type": "string", "description": "The user's id."}
    tools = [
        Tool(
            name="remember_fact",
            description="Store one fact about the user in long-term memory, "
            "overwriting any previous value for the same key.",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": user_id_schema,
                    "key": {"type": "string", "description": "Short fact key, e.g. 'open_issue'."},
                    "value": {"type": "string", "description": "The fact to remember."},
                },
                "required": ["user_id", "key", "value"],
            },
            func=remember_fact,
        ),
        Tool(
            name="recall_facts",
            description="Read facts stored about the user. Pass a key for one "
            "fact, or omit it to get everything remembered.",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": user_id_schema,
                    "key": {"type": "string", "description": "Optional fact key."},
                },
                "required": ["user_id"],
            },
            func=recall_facts,
        ),
        Tool(
            name="forget_fact",
            description="Delete one stored fact about the user.",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": user_id_schema,
                    "key": {"type": "string", "description": "The fact key to delete."},
                },
                "required": ["user_id", "key"],
            },
            func=forget_fact,
        ),
    ]
    return {tool.name: tool for tool in tools}
