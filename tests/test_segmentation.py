from early_stop.segmentation import (
    compute_segmentation_stats,
    split_into_steps,
    step_prefix,
    word_count,
)

# Modeled on real DeepSeek-R1-Distill output shape: blank-line-delimited
# reasoning steps, ending in a boxed answer.
GOOD_TRACE = """Let me think about this problem carefully.

First, I need to identify what's being asked. We have a quadratic equation.

Setting up the equation: x^2 - 5x + 6 = 0.

Factoring: (x-2)(x-3) = 0.

So x = 2 or x = 3. The answer is \\boxed{2, 3}."""

# Modeled on the failure mode the pilot gate is designed to catch: dense,
# blank-line-free reasoning with no step boundaries at all.
DENSE_TRACE = (
    "Let me think about this. First I need to identify what's being asked. "
    "We have a quadratic equation. Setting up: x^2 - 5x + 6 = 0. "
    "Factoring: (x-2)(x-3) = 0. So x = 2 or x = 3. The answer is \\boxed{2, 3}."
)


def test_split_into_steps_basic():
    steps = split_into_steps(GOOD_TRACE)
    assert len(steps) == 5
    assert steps[0].startswith("Let me think")
    assert steps[-1].endswith("boxed{2, 3}.")


def test_split_into_steps_collapses_extra_blank_lines():
    text = "step one\n\n\n\nstep two"
    steps = split_into_steps(text)
    assert steps == ["step one", "step two"]


def test_split_into_steps_dense_trace_yields_one_step():
    steps = split_into_steps(DENSE_TRACE)
    assert len(steps) == 1


def test_split_into_steps_strips_whitespace():
    text = "  step one  \n\n  step two  "
    steps = split_into_steps(text)
    assert steps == ["step one", "step two"]


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0


def test_step_prefix_reconstruction():
    steps = split_into_steps(GOOD_TRACE)
    prefix = step_prefix(steps, 2)
    assert prefix == steps[0] + "\n\n" + steps[1]


def test_step_prefix_full_length_reconstructs_all_steps_joined():
    steps = split_into_steps(GOOD_TRACE)
    full = step_prefix(steps, len(steps))
    assert full == "\n\n".join(steps)


def test_segmentation_stats_good_domain_passes_gate():
    traces = [GOOD_TRACE] * 10
    stats = compute_segmentation_stats("math500_pilot", traces)
    assert stats.n_traces == 10
    assert stats.mean_steps_per_trace == 5.0
    assert stats.frac_traces_lt_3_steps == 0.0
    assert stats.passes_gate() is True


def test_segmentation_stats_dense_domain_fails_gate():
    traces = [DENSE_TRACE] * 10
    stats = compute_segmentation_stats("some_dense_model_domain", traces)
    assert stats.mean_steps_per_trace == 1.0
    assert stats.frac_traces_lt_3_steps == 1.0
    assert stats.passes_gate() is False


def test_segmentation_stats_mixed_domain():
    traces = [GOOD_TRACE] * 7 + [DENSE_TRACE] * 3
    stats = compute_segmentation_stats("mixed", traces)
    assert stats.n_traces == 10
    assert stats.n_traces_lt_3_steps == 3
    assert stats.frac_traces_lt_3_steps == 0.3


def test_segmentation_stats_summary_is_string():
    stats = compute_segmentation_stats("d", [GOOD_TRACE])
    assert "d" in stats.summary()
    assert "PASS" in stats.summary() or "FAIL" in stats.summary()


def test_segmentation_stats_empty_traces():
    stats = compute_segmentation_stats("empty", [])
    assert stats.mean_steps_per_trace == 0.0
    assert stats.frac_traces_lt_3_steps == 0.0
    assert stats.passes_gate() is False
