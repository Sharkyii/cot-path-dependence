from early_stop.parsers import (
    extract_boxed,
    normalize_math_answer,
    parse_commonsenseqa_answer,
    parse_gsm8k_gold,
    parse_math_answer,
    parse_strategyqa_answer,
)


def test_extract_boxed_simple():
    assert extract_boxed(r"The answer is \boxed{42}.") == "42"


def test_extract_boxed_nested_braces():
    text = r"So we get \boxed{\frac{1}{2}}"
    assert extract_boxed(text) == r"\frac{1}{2}"


def test_extract_boxed_takes_last_occurrence():
    text = r"First I thought \boxed{5} but actually \boxed{7}"
    assert extract_boxed(text) == "7"


def test_extract_boxed_missing():
    assert extract_boxed("no boxed answer here") is None


def test_normalize_math_answer_fraction_forms():
    assert normalize_math_answer(r"\frac{1}{2}") == "1/2"
    assert normalize_math_answer(r"\dfrac{3}{4}") == "3/4"
    # 0.5 and 1/2 are the same number -- Fraction canonicalizes exact
    # decimals, so these SHOULD normalize identically for grading purposes.
    assert normalize_math_answer("0.5") == normalize_math_answer("1/2") == "1/2"


def test_normalize_math_answer_integer_equivalences():
    assert normalize_math_answer("42") == normalize_math_answer("+42")
    assert normalize_math_answer("1,000") == normalize_math_answer("1000")
    assert normalize_math_answer("$5$") == "5"


def test_normalize_math_answer_text_wrapper():
    assert normalize_math_answer(r"\text{5}") == "5"


def test_parse_math_answer_realistic_r1_trace():
    # Shape modeled on real R1-Distill-Qwen output: scratch work with an
    # early \boxed{} in reasoning, then a final answer after "</think>".
    trace = (
        "Let me compute this step by step.\n\n"
        r"First guess \boxed{10}, but let me verify..."
        "\n\nActually recomputing: 3 * 4 = 12, so the answer is "
        r"\boxed{12}."
    )
    result = parse_math_answer(trace)
    assert result.answer == "12"
    assert result.method == "boxed"


def test_parse_math_answer_fallback_to_last_number():
    trace = "I worked through it and got 17 as the final result."
    result = parse_math_answer(trace)
    assert result.answer == "17"
    assert result.method == "last_number"


def test_parse_math_answer_extraction_failure():
    trace = "I am not sure how to proceed with this problem."
    result = parse_math_answer(trace)
    assert result.answer is None
    assert result.method == "failed"


def test_parse_gsm8k_gold():
    gold_solution = "Natalia sold 48 clips.\n#### 48"
    assert parse_gsm8k_gold(gold_solution) == "48"


def test_parse_gsm8k_gold_missing_marker_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_gsm8k_gold("no marker here")


def test_parse_strategyqa_answer_last_mention_wins():
    trace = "Hmm, maybe no? Actually wait, reconsidering: yes, that's right."
    result = parse_strategyqa_answer(trace)
    assert result.answer == "yes"


def test_parse_strategyqa_answer_failure():
    result = parse_strategyqa_answer("I cannot determine this.")
    assert result.answer is None


def test_parse_commonsenseqa_answer():
    trace = "Between B and D, I think the answer is (D) because it fits best."
    result = parse_commonsenseqa_answer(trace, num_choices=5)
    assert result.answer == "D"


def test_parse_commonsenseqa_answer_out_of_range_letter_ignored():
    # num_choices=3 means only A/B/C are valid; a stray "E" should not match
    trace = "Hmm E doesn't sound right, actually the answer is A."
    result = parse_commonsenseqa_answer(trace, num_choices=3)
    assert result.answer == "A"
