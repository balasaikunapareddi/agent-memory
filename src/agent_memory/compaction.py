"""Context compaction.

A conversation has a token budget. When the running history exceeds
it, the oldest turns are folded into one compact summary turn and the
recent turns are kept verbatim. The summary is deterministic in mock
mode: it is built from the structured facts and preferences already
in the store plus the topics of the folded turns, so nothing the
agent was told to remember is lost when the window fills.
"""

from __future__ import annotations

from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token). Deterministic, offline."""
    return max(1, len(text) // 4)


def turns_tokens(turns: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(turn["content"]) for turn in turns)


@dataclass
class CompactionEvent:
    turns_before: int
    turns_after: int
    tokens_before: int
    tokens_after: int
    summary: str


class Compactor:
    def __init__(self, budget_tokens: int = 400, keep_recent: int = 4):
        self.budget_tokens = budget_tokens
        self.keep_recent = keep_recent

    def maybe_compact(
        self,
        turns: list[dict[str, str]],
        facts: dict[str, str],
        preferences: dict[str, str],
    ) -> tuple[list[dict[str, str]], CompactionEvent | None]:
        """Fold old turns into a summary when the budget is exceeded."""
        before = turns_tokens(turns)
        if before <= self.budget_tokens or len(turns) <= self.keep_recent:
            return turns, None

        old, recent = turns[: -self.keep_recent], turns[-self.keep_recent :]
        summary = self._summarize(old, facts, preferences)
        new_turns = [{"role": "summary", "content": summary}, *recent]
        event = CompactionEvent(
            turns_before=len(turns),
            turns_after=len(new_turns),
            tokens_before=before,
            tokens_after=turns_tokens(new_turns),
            summary=summary,
        )
        return new_turns, event

    @staticmethod
    def _summarize(
        old_turns: list[dict[str, str]],
        facts: dict[str, str],
        preferences: dict[str, str],
    ) -> str:
        user_topics = [t["content"] for t in old_turns if t["role"] == "user"]
        parts = [f"[compacted summary of {len(old_turns)} earlier turns]"]
        if facts:
            parts.append(
                "Remembered facts: "
                + "; ".join(f"{k} = {v}" for k, v in sorted(facts.items()))
            )
        if preferences:
            parts.append(
                "User preferences: "
                + "; ".join(f"{k} = {v}" for k, v in sorted(preferences.items()))
            )
        if user_topics:
            parts.append("Earlier user messages covered: " + " | ".join(user_topics))
        return "\n".join(parts)
