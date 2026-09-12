"""Tests for hardware-fit logic.

These are pure arithmetic -- no GPU, no model download -- but they encode
the rule that prevented the first pilot run from finishing: if the weights
don't fit alongside a real KV-cache budget, pick a smaller precision rather
than letting device_map='auto' silently offload to CPU at ~2 tok/s.
"""
from early_stop.backend import (
    DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B,
    GPUInfo,
    HFBackend,
    estimate_weight_gb,
    recommend_precision,
)


def _gpu(total_gb: float, n: int = 1, bf16: bool = True) -> GPUInfo:
    per = total_gb / n
    return GPUInfo(
        available=True,
        n_devices=n,
        names=["test"] * n,
        total_memory_gb=[per] * n,
        capability=(8, 0) if bf16 else (7, 5),
        supports_bf16=bf16,
    )


def test_weight_estimates_shrink_with_precision():
    assert estimate_weight_gb(7.0, "fp16") > estimate_weight_gb(7.0, "8bit")
    assert estimate_weight_gb(7.0, "8bit") > estimate_weight_gb(7.0, "4bit")


def test_7b_fp16_technically_fits_t4_but_leaves_no_kv_room():
    # The precise shape of the observed failure. Weights are ~14.2 GiB on a
    # 14.56 GiB card, so they *nominally* fit -- which is exactly why
    # device_map='auto' didn't complain. What's left (~0.4 GiB) cannot hold
    # a KV cache, and the run died trying to allocate 1.02 GiB with 122 MiB
    # free. So the weights-only check must pass while the headroom-aware
    # recommendation still rejects fp16.
    t4_usable = 14.56
    weights = estimate_weight_gb(DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, "fp16")
    assert weights < t4_usable, "weights alone do fit -- that's the trap"
    assert t4_usable - weights < 1.02, "and the remainder cannot hold the KV cache"
    assert recommend_precision(DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, _gpu(t4_usable, bf16=False)) != "fp16"


def test_single_t4_7b_falls_back_to_quantization():
    # A 16GB Turing card cannot hold 7B in 16-bit AND a KV cache.
    assert recommend_precision(DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, _gpu(15.0, bf16=False)) in {"8bit", "4bit"}


def test_turing_never_gets_bf16():
    # T4 is SM 7.5 -- bf16 is emulated and slow, so it must never be chosen
    # even when the memory arithmetic alone would allow it.
    assert recommend_precision(1.5, _gpu(15.0, bf16=False)) == "fp16"


def test_ampere_prefers_bf16_when_it_fits():
    assert recommend_precision(DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, _gpu(40.0, bf16=True)) == "bf16"


def test_dual_t4_fits_7b_in_fp16():
    # Kaggle "T4 x2" = ~29 GiB total, enough for 15.2 GiB of weights plus
    # KV headroom -- this is why T4 x2 avoids the quantization confound.
    assert recommend_precision(DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, _gpu(29.0, n=2, bf16=False)) == "fp16"


def test_small_model_fits_anywhere():
    assert recommend_precision(1.5, _gpu(8.0, bf16=False)) == "fp16"


def test_headroom_is_actually_reserved():
    # Weights that fit only by consuming every byte must be rejected: the
    # first run OOMed trying to allocate 1.02 GiB of KV cache with ~120 MiB
    # free, which is precisely this case.
    weights = estimate_weight_gb(DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, "fp16")
    just_barely = _gpu(weights + 0.5, bf16=False)
    assert recommend_precision(DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, just_barely, kv_cache_headroom_gb=3.0) != "fp16"


def _bare_backend(temperature: float, top_p: float | None) -> HFBackend:
    """Builds an HFBackend instance WITHOUT running __init__ (no torch/
    transformers import, no model load) -- just enough state set directly
    to test _sampling_kwargs()'s pure logic. See that method's docstring
    (2026-09-13, REVISION_PLAN.md Tier 1.3 rerun): it must include top_p
    in the kwargs it returns only when top_p was actually set, so every
    caller that never passes top_p keeps getting the model's own default
    behavior unchanged."""
    backend = object.__new__(HFBackend)
    backend.temperature = temperature
    backend.top_p = top_p
    return backend


def test_sampling_kwargs_omits_top_p_when_unset():
    # The default (top_p=None) -- every existing caller of HFBackend that
    # doesn't pass top_p must see IDENTICAL generate() kwargs to before
    # this parameter was added, or their results would silently stop
    # being comparable to earlier runs.
    kwargs = _bare_backend(temperature=0.6, top_p=None)._sampling_kwargs()
    assert kwargs == {"do_sample": True, "temperature": 0.6}
    assert "top_p" not in kwargs


def test_sampling_kwargs_includes_top_p_when_set():
    # The fix this test guards: modal_audit2a/2b_toppinned.py pass
    # top_p=0.95 to match the primary sample's real configuration, after
    # the first audit run silently tested the wrong one because this
    # parameter didn't exist yet.
    kwargs = _bare_backend(temperature=0.6, top_p=0.95)._sampling_kwargs()
    assert kwargs == {"do_sample": True, "temperature": 0.6, "top_p": 0.95}
