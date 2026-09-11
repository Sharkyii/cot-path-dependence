"""Blank-line step segmentation + pilot-gate statistics (project plan §5).

This is the first thing to validate on the real model before writing any
more pipeline code: distilled reasoning models sometimes emit dense,
blank-line-free blocks, in which case this whole step-level early-exit
design breaks. `segmentation_stats` is what the pilot script calls to
decide go/no-go per domain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def split_into_steps(text: str) -> list[str]:
    """Split a reasoning trace into steps on blank-line boundaries.

    Leading/trailing whitespace is stripped per step; empty steps (e.g. from
    3+ consecutive newlines) are dropped rather than kept as degenerate
    zero-length steps, since a zero-length "step" is not a real candidate
    stop boundary.
    """
    raw_parts = _BLANK_LINE_RE.split(text)
    return [p.strip() for p in raw_parts if p.strip()]


def word_count(s: str) -> int:
    """Whitespace-token proxy for step length. Use a real tokenizer's
    token count once the model/stack is available (see project plan §8) --
    this is only for offline pilot-script debugging without a model.
    """
    return len(s.split())


@dataclass
class SegmentationStats:
    domain: str
    n_traces: int
    steps_per_trace: list[int]
    step_word_lengths: list[int]  # flattened across all traces/steps
    n_traces_lt_3_steps: int

    @property
    def mean_steps_per_trace(self) -> float:
        if not self.steps_per_trace:
            return 0.0
        return sum(self.steps_per_trace) / len(self.steps_per_trace)

    @property
    def frac_traces_lt_3_steps(self) -> float:
        if self.n_traces == 0:
            return 0.0
        return self.n_traces_lt_3_steps / self.n_traces

    @property
    def mean_step_word_length(self) -> float:
        if not self.step_word_lengths:
            return 0.0
        return sum(self.step_word_lengths) / len(self.step_word_lengths)

    def passes_gate(self, min_mean_steps: float = 4.0, max_frac_lt_3: float = 0.30) -> bool:
        """Decision rule for the pilot gate (project plan §5): need enough
        steps/problem for a non-coarse early-exit curve, and segmentation
        must not be degenerate on most traces. Thresholds are a starting
        point -- tune against what the actual step-length distribution
        looks like, don't just trust these numbers blindly.
        """
        return (
            self.mean_steps_per_trace >= min_mean_steps
            and self.frac_traces_lt_3_steps <= max_frac_lt_3
        )

    def summary(self) -> str:
        return (
            f"[{self.domain}] n_traces={self.n_traces} "
            f"mean_steps/trace={self.mean_steps_per_trace:.2f} "
            f"frac_traces<3_steps={self.frac_traces_lt_3_steps:.2%} "
            f"mean_step_word_len={self.mean_step_word_length:.1f} "
            f"gate={'PASS' if self.passes_gate() else 'FAIL'}"
        )


def compute_segmentation_stats(domain: str, traces: list[str]) -> SegmentationStats:
    steps_per_trace = []
    step_word_lengths = []
    n_lt_3 = 0
    for trace in traces:
        steps = split_into_steps(trace)
        steps_per_trace.append(len(steps))
        if len(steps) < 3:
            n_lt_3 += 1
        step_word_lengths.extend(word_count(s) for s in steps)
    return SegmentationStats(
        domain=domain,
        n_traces=len(traces),
        steps_per_trace=steps_per_trace,
        step_word_lengths=step_word_lengths,
        n_traces_lt_3_steps=n_lt_3,
    )


def step_prefix(steps: list[str], num_steps: int, join_with: str = "\n\n") -> str:
    """Reconstruct the trace text truncated to the first `num_steps` steps.
    Used both for the naive-truncation baseline and as the input to the
    forced-extraction suffix at a candidate stop boundary.
    """
    return join_with.join(steps[:num_steps])
