"""Answer extraction from model outputs.

Every parser here is used identically at calibration and test time, and
identically on full-CoT and early-stopped/forced-extraction traces. Do not
special-case any of these paths -- asymmetric parsing would bias the
early-stopping comparison in the eval harness's favor or against it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class ParseResult:
    raw_text: str
    answer: str | None  # normalized answer string, or None if extraction failed
    method: str  # which extraction path fired, for failure-rate breakdowns


# ---------------------------------------------------------------------------
# Math (MATH-500, GSM8K)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{")
_LAST_NUMBER_RE = re.compile(
    r"-?\d[\d,]*\.?\d*(?:/\d+)?"
)
_GSM8K_HASH_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")


def _find_matching_brace(text: str, open_idx: int) -> int | None:
    """Given index of the '{' right after \\boxed, return index of matching '}'."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_boxed(text: str) -> str | None:
    """Extract the content of the LAST \\boxed{...} in text, brace-matched.

    Last occurrence is used because models often restate the answer in a
    final \\boxed{} after earlier scratch-work boxes.
    """
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    open_idx = last.end() - 1  # index of the '{'
    close_idx = _find_matching_brace(text, open_idx)
    if close_idx is None:
        return None
    return text[open_idx + 1 : close_idx].strip()


def normalize_math_answer(s: str) -> str:
    """Normalize a math answer string for equality comparison.

    Handles: whitespace, $ delimiters, trailing punctuation, \\text{},
    \\!  thin-space, commas in numbers, \\dfrac/\\frac -> a/b, percent signs,
    leading + signs.
    """
    if s is None:
        return ""
    s = s.strip()
    s = s.strip("$")
    s = s.strip()
    # strip \text{...} wrapper
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # remove thin-space and spacing commands
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    # normalize fractions: \frac{a}{b} or \dfrac{a}{b} -> a/b
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = s.replace(" ", "")
    s = s.rstrip(".")
    s = s.replace(",", "")
    if s.startswith("+"):
        s = s[1:]
    s = s.replace("\\%", "").replace("%", "")
    # try numeric/fraction canonicalization
    try:
        if "/" in s:
            frac = Fraction(s)
        else:
            frac = Fraction(s)
        # keep integers as integers, fractions as fractions
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"
    except (ValueError, ZeroDivisionError):
        pass
    # A bare number with a trailing unit word (e.g. "2000calories" from a
    # model that writes \boxed{2000 calories}) is not the same as a
    # different answer -- strip a pure-alphabetic suffix and retry, but
    # only when the remainder is a clean number, so this never touches
    # genuinely symbolic answers (sqrt{51}, 3\pi/2, x^2+1, ...), which
    # won't match a leading-digit pattern in the first place. Found via
    # the Candidate B full pilot: this exact pattern (`d1_prob3`) inflated
    # a Path Dependence Index measurement with no real divergence behind it.
    m = re.match(r"^(-?\d+(?:\.\d+)?)[a-zA-Z]+$", s)
    if m:
        try:
            frac = Fraction(m.group(1))
            if frac.denominator == 1:
                return str(frac.numerator)
            return f"{frac.numerator}/{frac.denominator}"
        except (ValueError, ZeroDivisionError):
            pass
    return s.lower()


def parse_math_answer(text: str) -> ParseResult:
    """Extract answer from a MATH-style trace. Prefers \\boxed{}, falls back
    to the last number in the text (never a hidden fallback that changes
    results silently -- the `method` field records which path fired so
    extraction-failure/fallback rates are reportable).
    """
    boxed = extract_boxed(text)
    if boxed is not None:
        return ParseResult(raw_text=text, answer=normalize_math_answer(boxed), method="boxed")
    nums = _LAST_NUMBER_RE.findall(text)
    if nums:
        return ParseResult(raw_text=text, answer=normalize_math_answer(nums[-1]), method="last_number")
    return ParseResult(raw_text=text, answer=None, method="failed")


def parse_gsm8k_gold(text: str) -> str:
    """Extract the gold answer from a GSM8K reference solution (uses the
    dataset's own #### marker -- only for gold labels, not model output).
    """
    m = _GSM8K_HASH_RE.search(text)
    if not m:
        raise ValueError(f"No #### marker found in GSM8K gold solution: {text!r}")
    return normalize_math_answer(m.group(1))


def parse_gsm8k_answer(text: str) -> ParseResult:
    """Extract answer from a model trace for a GSM8K problem.

    GSM8K traces from reasoning models still often emit \\boxed{}; fall back
    to last-number if not present, same as MATH.
    """
    return parse_math_answer(text)


def math_answers_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return a == b


# ---------------------------------------------------------------------------
# Commonsense (StrategyQA yes/no, CommonsenseQA A-E)
# ---------------------------------------------------------------------------

_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_MC_LETTER_RE = re.compile(r"\b\(?([A-E])\)?\b")


def parse_strategyqa_answer(text: str) -> ParseResult:
    """Extract yes/no from a StrategyQA trace. Uses the LAST yes/no token
    mentioned, on the assumption a forced 'final answer' suffix pulls the
    real answer to the end of the text.
    """
    matches = _YES_NO_RE.findall(text)
    if not matches:
        return ParseResult(raw_text=text, answer=None, method="failed")
    return ParseResult(raw_text=text, answer=matches[-1].lower(), method="yes_no")


def parse_commonsenseqa_answer(text: str, num_choices: int = 5) -> ParseResult:
    """Extract an (A)-(E) choice letter, taking the last match in the text."""
    valid_letters = set("ABCDE"[:num_choices])
    matches = [m for m in _MC_LETTER_RE.findall(text) if m in valid_letters]
    if not matches:
        return ParseResult(raw_text=text, answer=None, method="failed")
    return ParseResult(raw_text=text, answer=matches[-1], method="mc_letter")
