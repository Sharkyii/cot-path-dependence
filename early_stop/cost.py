"""Cost accounting (project plan §10, §14).

Tracks token counts and call counts by category (main generation, forced
extraction, difficulty sampling) so the "for free" story in the writeup is
backed by honest numbers, not asserted. Every forced_extract() call and every
main-generation call should feed a CostTracker -- signals differ in
per-decision overhead, and that overhead is exactly what the two-panel cost
figure (project plan §10) reports.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    prompt_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    completion_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, category: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls[category] += 1
        self.prompt_tokens[category] += prompt_tokens
        self.completion_tokens[category] += completion_tokens

    def total_tokens(self, category: str | None = None) -> int:
        if category is not None:
            return self.prompt_tokens[category] + self.completion_tokens[category]
        return sum(self.prompt_tokens.values()) + sum(self.completion_tokens.values())

    def summary(self) -> str:
        lines = []
        for cat in sorted(self.calls):
            lines.append(
                f"{cat}: calls={self.calls[cat]} "
                f"prompt_tok={self.prompt_tokens[cat]} "
                f"completion_tok={self.completion_tokens[cat]} "
                f"total_tok={self.total_tokens(cat)}"
            )
        lines.append(f"GRAND TOTAL tokens: {self.total_tokens()}")
        return "\n".join(lines)
