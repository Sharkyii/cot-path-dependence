"""Sufficiency-Gated Early Stopping (SG-ES) -- Phase 4 of the low-compute
plan (2026-09-11).

Pure post-hoc logic over a trajectory's already-scored step-wise prefixes
(early_stop.candidate_b_pipeline.TrajectoryPrefixes, produced by Phase 1's
existing pipeline -- generate ONE full natural trace per problem, score
phi(p_t) at every step boundary via forced extraction, same as Candidate B
already does). No new generation is needed to compute any of the three
Phase 4 conditions from that one trace:

  1. "Full generation" baseline: the LAST prefix's answer (equivalent to
     letting the trace run to completion).
  2. "Forced-extraction" baseline: the answer at a FIXED fraction of the
     trace (e.g. 50%), regardless of confidence -- the naive baseline
     every training-free early-stopping paper compares against.
  3. SG-ES: stop the first time the answer has been STABLE for
     `stability_window` consecutive scored steps AND confidence at the
     stopping step is above `confidence_threshold`.

The confidence gate is not arbitrary -- it is directly motivated by this
project's own finding (CANDIDATE_B_PREREGISTRATION.md, 2026-09-10/11
amendments): plain answer-stability (L1 matching) has a measurable
decisiveness blind spot that a confidence check (L2) resolved in the
exploratory follow-up. SG-ES is that fix, turned into an actual stopping
rule instead of a post-hoc observation.
"""
from __future__ import annotations

from dataclasses import dataclass

from early_stop.path_dependence import Prefix


@dataclass(frozen=True)
class StoppingDecision:
    answer: str | None
    stopped_at_step: int
    total_steps: int
    triggered: bool  # False if the rule never fired and we fell through to the full trace

    @property
    def fraction_of_trace_used(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.stopped_at_step / self.total_steps


def full_generation_answer(prefixes: list[Prefix]) -> StoppingDecision:
    """Baseline 1: let it run to the end."""
    if not prefixes:
        return StoppingDecision(None, 0, 0, triggered=False)
    last = prefixes[-1]
    return StoppingDecision(last.state.answer, last.step_index, last.total_steps, triggered=True)


def forced_extraction_baseline(prefixes: list[Prefix], fraction: float = 0.5) -> StoppingDecision:
    """Baseline 2: stop at a fixed fraction of the trace regardless of
    confidence -- the naive comparison point SG-ES needs to beat."""
    if not prefixes:
        return StoppingDecision(None, 0, 0, triggered=False)
    target = min(prefixes, key=lambda p: abs(p.normalized_position - fraction))
    return StoppingDecision(target.state.answer, target.step_index, target.total_steps, triggered=True)


def sg_es_decision(
    prefixes: list[Prefix],
    confidence_threshold: float = 0.70,
    stability_window: int = 2,
) -> StoppingDecision:
    """SG-ES: first step where the answer has been identical for
    `stability_window` consecutive scored prefixes AND the confidence at
    that step clears `confidence_threshold`. Falls through to the full
    trace's answer (triggered=False) if the condition is never met --
    SG-ES should never be WORSE than full generation's accuracy by
    construction, only save tokens when it can.
    """
    if not prefixes:
        return StoppingDecision(None, 0, 0, triggered=False)
    for i in range(stability_window - 1, len(prefixes)):
        window = prefixes[i - stability_window + 1: i + 1]
        answers = {p.state.answer for p in window}
        min_confidence = min(p.state.confidence for p in window)
        if len(answers) == 1 and min_confidence >= confidence_threshold:
            stopped = prefixes[i]
            return StoppingDecision(
                stopped.state.answer, stopped.step_index, stopped.total_steps, triggered=True,
            )
    last = prefixes[-1]
    return StoppingDecision(last.state.answer, last.step_index, last.total_steps, triggered=False)
