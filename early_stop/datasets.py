"""Dataset loaders: MATH-500, GSM8K, CommonsenseQA/StrategyQA.

Each loader returns a list of Problem records with a stable, domain-prefixed
problem_id so SplitManager assignments stay valid across reruns even if the
underlying HF dataset iteration order changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from early_stop.parsers import normalize_math_answer, parse_gsm8k_gold


@dataclass
class Problem:
    problem_id: str  # e.g. "math500:12"
    domain: str  # "math500" | "gsm8k" | "commonsenseqa" | "strategyqa"
    question: str
    gold_answer: str  # normalized, comparable directly against parser output
    level: str | None = None  # dataset-provided difficulty label, if any
    choices: dict[str, str] | None = None  # for MC: {"A": "...", ...}
    meta: dict[str, Any] = field(default_factory=dict)


def _require_datasets():
    try:
        import datasets  # noqa: F401
        return datasets
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load benchmark data. "
            "Install with: pip install datasets"
        ) from e


def load_math500() -> list[Problem]:
    """MATH-500 (HuggingFaceH4/MATH-500). Provides a 'level' field (1-5) we
    use as a cheap secondary difficulty proxy (see project plan §6).
    """
    ds_lib = _require_datasets()
    ds = ds_lib.load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for i, row in enumerate(ds):
        gold = normalize_math_answer(row["answer"])
        problems.append(
            Problem(
                problem_id=f"math500:{i}",
                domain="math500",
                question=row["problem"],
                gold_answer=gold,
                level=str(row.get("level")) if row.get("level") is not None else None,
                meta={"subject": row.get("subject")},
            )
        )
    return problems


def load_math500_levels_1_to_3() -> list[Problem]:
    """Primary-domain filter per project plan §4 (levels 1-3 only, so the
    model has room to overthink -- level 4-5 problems are frequently missed
    even with full CoT, which confounds an early-stopping accuracy signal
    with a difficulty-ceiling effect).
    """
    return [p for p in load_math500() if p.level in {"1", "2", "3"}]


def load_gsm8k(split: str = "test") -> list[Problem]:
    """GSM8K (openai/gsm8k, 'main' config). Used only as a "little to save
    here" contrast domain per project plan §4 -- NOT the primary eval.
    """
    ds_lib = _require_datasets()
    ds = ds_lib.load_dataset("openai/gsm8k", "main", split=split)
    problems = []
    for i, row in enumerate(ds):
        gold = parse_gsm8k_gold(row["answer"])
        problems.append(
            Problem(
                problem_id=f"gsm8k:{split}:{i}",
                domain="gsm8k",
                question=row["question"],
                gold_answer=gold,
            )
        )
    return problems


def load_commonsenseqa(split: str = "validation") -> list[Problem]:
    """CommonsenseQA (tau/commonsense_qa). Test split has no public labels,
    so validation is used as our commonsense eval pool.
    """
    ds_lib = _require_datasets()
    ds = ds_lib.load_dataset("tau/commonsense_qa", split=split)
    problems = []
    for i, row in enumerate(ds):
        choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
        gold = row["answerKey"]
        if not gold:
            continue  # a handful of rows ship without a gold label; skip rather than silently mis-grade
        problems.append(
            Problem(
                problem_id=f"commonsenseqa:{split}:{i}",
                domain="commonsenseqa",
                question=row["question"],
                gold_answer=gold,
                choices=choices,
            )
        )
    return problems


def load_strategyqa(split: str = "test") -> list[Problem]:
    """StrategyQA (ChilleD/StrategyQA). NOTE: the canonical wics/strategy-qa
    mirror uses a legacy HF "dataset script" format that recent `datasets`
    versions refuse to load; ChilleD/StrategyQA is a script-free parquet
    mirror with the same fields and public boolean labels on both splits.
    """
    ds_lib = _require_datasets()
    ds = ds_lib.load_dataset("ChilleD/StrategyQA", split=split)
    problems = []
    for i, row in enumerate(ds):
        gold = "yes" if row["answer"] else "no"
        problems.append(
            Problem(
                problem_id=f"strategyqa:{split}:{i}",
                domain="strategyqa",
                question=row["question"],
                gold_answer=gold,
                meta={"qid": row.get("qid"), "facts": row.get("facts")},
            )
        )
    return problems


DOMAIN_LOADERS = {
    "math500": load_math500_levels_1_to_3,
    "math500_all": load_math500,
    "gsm8k": load_gsm8k,
    "commonsenseqa": load_commonsenseqa,
    "strategyqa": load_strategyqa,
}


def load_domain(domain: str) -> list[Problem]:
    if domain not in DOMAIN_LOADERS:
        raise ValueError(f"Unknown domain {domain!r}, expected one of {list(DOMAIN_LOADERS)}")
    return DOMAIN_LOADERS[domain]()
