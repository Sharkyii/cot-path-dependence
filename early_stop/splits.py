"""Problem-disjoint split manager.

Difficulty-stratification sampling, dev-calibration, and test MUST NOT share
problems -- otherwise thresholds calibrated on a problem leak into the
accuracy numbers we report for it later. This module is the single place
that decides the split for a given (domain, problem_id), so no downstream
code can accidentally re-derive an overlapping split.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SPLIT_NAMES = ("pilot", "difficulty", "dev", "test")


@dataclass
class SplitConfig:
    """Fractions must sum to <= 1.0; remainder is unused ('pool').

    pilot_frac defaults to 0.0 (opt-in): pilot-gate problems (project plan
    §5) are burned and must never resurface in difficulty/dev/test, so a
    pilot run should construct SplitConfig with an explicit pilot_frac
    carved out of the other fractions, not rely on a nonzero default here.
    """

    pilot_frac: float = 0.0
    difficulty_frac: float = 0.20
    dev_frac: float = 0.15
    test_frac: float = 0.65
    seed: int = 0

    def __post_init__(self) -> None:
        total = self.pilot_frac + self.difficulty_frac + self.dev_frac + self.test_frac
        if total > 1.0 + 1e-9:
            raise ValueError(f"Split fractions sum to {total} > 1.0")


def _stable_hash_unit(key: str, seed: int) -> float:
    """Deterministic, uniform-ish float in [0, 1) for a (key, seed) pair.

    Using a hash instead of a seeded RNG over a shuffled list means the
    split assignment for a given problem is stable even if the dataset
    loader later adds/removes/reorders other problems.
    """
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def assign_split(problem_id: str, config: SplitConfig) -> str:
    """Return 'pilot', 'difficulty', 'dev', 'test', or 'pool' (unused)."""
    u = _stable_hash_unit(problem_id, config.seed)
    if u < config.pilot_frac:
        return "pilot"
    u -= config.pilot_frac
    if u < config.difficulty_frac:
        return "difficulty"
    u -= config.difficulty_frac
    if u < config.dev_frac:
        return "dev"
    u -= config.dev_frac
    if u < config.test_frac:
        return "test"
    return "pool"


@dataclass
class SplitManager:
    """Assigns and records splits per-domain so assignments can be audited
    and re-loaded (important: assignment is deterministic given the same
    config, but we persist it anyway so a config change doesn't silently
    reshuffle splits after data has already been collected).
    """

    config: SplitConfig
    record_path: Path | None = None
    _cache: dict[str, dict[str, str]] = field(default_factory=dict)  # domain -> problem_id -> split

    def split_for(self, domain: str, problem_id: str) -> str:
        domain_cache = self._cache.setdefault(domain, {})
        if problem_id not in domain_cache:
            domain_cache[problem_id] = assign_split(problem_id, self.config)
        return domain_cache[problem_id]

    def filter_split(self, domain: str, problem_ids: list[str], split: str) -> list[str]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}, expected one of {SPLIT_NAMES}")
        return [pid for pid in problem_ids if self.split_for(domain, pid) == split]

    def save(self, path: Path | None = None) -> None:
        path = path or self.record_path
        if path is None:
            raise ValueError("No path given and no record_path set on SplitManager")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "pilot_frac": self.config.pilot_frac,
                "difficulty_frac": self.config.difficulty_frac,
                "dev_frac": self.config.dev_frac,
                "test_frac": self.config.test_frac,
                "seed": self.config.seed,
            },
            "assignments": self._cache,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "SplitManager":
        payload = json.loads(path.read_text())
        cfg = SplitConfig(**payload["config"])
        mgr = cls(config=cfg, record_path=path)
        mgr._cache = payload["assignments"]
        return mgr


def assert_disjoint(*problem_id_lists: list[str]) -> None:
    """Sanity check: no problem_id appears in more than one of the given
    lists. Raises AssertionError naming the offending id if violated.

    This checks actual list membership, not re-derived split assignment --
    assign_split(pid) is a pure function of pid, so re-deriving it can never
    detect a caller accidentally putting the same id in two hand-built
    lists (e.g. a bug that copies a dev id into a test list). Call this
    after building any split-derived subsets, not just in tests -- a
    silent leak here invalidates the whole study.
    """
    seen: dict[str, int] = {}
    for list_idx, ids in enumerate(problem_id_lists):
        for pid in ids:
            if pid in seen and seen[pid] != list_idx:
                raise AssertionError(
                    f"Problem {pid!r} appears in both list #{seen[pid]} and "
                    f"list #{list_idx} -- split leakage detected"
                )
            seen[pid] = list_idx
