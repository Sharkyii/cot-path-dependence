"""Model backend with hardware-aware loading.

Exists as its own module because getting this wrong is silent and expensive:
an earlier pilot attempt on a Kaggle T4 hardcoded bfloat16 and
device_map="auto", which (a) hit bf16 emulation on Turing and (b) silently
offloaded layers to CPU when the 7B didn't fit in 14.56 GiB -- producing
1.9 tok/s instead of an error. A pilot at 1.9 tok/s never finishes; a pilot
that refuses to load tells you immediately to change plan. This module
prefers loud failure over slow success.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("torch is required for GPU backends. pip install torch transformers") from e


@dataclass
class GPUInfo:
    available: bool
    n_devices: int
    names: list[str]
    total_memory_gb: list[float]
    capability: tuple[int, int] | None  # (major, minor) of device 0
    supports_bf16: bool

    @property
    def total_vram_gb(self) -> float:
        return sum(self.total_memory_gb)

    def summary(self) -> str:
        if not self.available:
            return "No CUDA GPU detected (CPU only)"
        parts = [
            f"{n} ({m:.1f} GiB)" for n, m in zip(self.names, self.total_memory_gb)
        ]
        return (
            f"{self.n_devices} GPU(s): {', '.join(parts)} | "
            f"compute capability {self.capability} | "
            f"bf16 {'supported' if self.supports_bf16 else 'NOT supported (use fp16)'}"
        )


def probe_gpu() -> GPUInfo:
    torch = _torch()
    if not torch.cuda.is_available():
        return GPUInfo(False, 0, [], [], None, False)
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    mems = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(n)]
    cap = torch.cuda.get_device_capability(0)
    # bf16 needs Ampere (SM 8.0) or newer. Turing (T4, SM 7.5) and older
    # will "work" via emulation but at a large speed penalty.
    supports_bf16 = cap[0] >= 8
    return GPUInfo(True, n, names, mems, cap, supports_bf16)


# Bytes per parameter by precision. 4bit is >0.5 because NF4 stores
# per-block scales alongside the packed weights.
_BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "8bit": 1.0, "4bit": 0.55}

_BYTES_PER_GIB = 1024**3

# DeepSeek-R1-Distill-Qwen-7B is built on Qwen2.5-Math-7B, which is really
# 7.62B params -- not 7.0B. The difference is not cosmetic: at fp16 it is
# the difference between "14.0 GiB, fits a T4 with room" and "14.2 GiB,
# fits with ~0.4 GiB to spare and then OOMs on the first KV allocation."
DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B = 7.62
DEEPSEEK_R1_DISTILL_QWEN_1_5B_PARAMS_B = 1.78


def estimate_weight_gb(param_count_b: float, precision: str) -> float:
    """Weight footprint in GiB (1024^3 bytes), matching how torch and
    nvidia-smi report memory."""
    if precision not in _BYTES_PER_PARAM:
        raise ValueError(f"Unknown precision {precision!r}")
    return param_count_b * 1e9 * _BYTES_PER_PARAM[precision] / _BYTES_PER_GIB


def recommend_precision(
    param_count_b: float,
    gpu: GPUInfo,
    kv_cache_headroom_gb: float = 3.0,
) -> str:
    """Pick the highest-fidelity precision whose weights leave enough room
    for KV cache on the available VRAM.

    kv_cache_headroom_gb defaults to 3.0 because long reasoning traces
    (2k-4k tokens) need real KV space -- the failed T4 run OOMed trying to
    allocate 1.02 GiB of cache with only ~120 MiB free. Budgeting zero
    headroom is what produced that.
    """
    if not gpu.available:
        return "fp32"  # CPU; caller should refuse to run the real pilot anyway
    usable = gpu.total_vram_gb - kv_cache_headroom_gb
    for precision in ("bf16", "fp16", "8bit", "4bit"):
        if precision == "bf16" and not gpu.supports_bf16:
            continue
        if estimate_weight_gb(param_count_b, precision) <= usable:
            return precision
    return "4bit"


class HFBackend:
    """transformers ModelBackend, loaded once and reused for both full-trace
    generation and forced extraction.

    Key differences from a naive from_pretrained call:
      * dtype is chosen for the actual GPU (fp16 on Turing, bf16 on Ampere+)
      * CPU offload is DISABLED by default -- a model that doesn't fit
        raises instead of silently running ~10x slower
      * optional 4-bit/8-bit quantization to fit big models on small cards
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        precision: str = "auto",
        param_count_b: float = 7.0,
        allow_cpu_offload: bool = False,
        max_memory: dict | None = None,
    ):
        torch = _torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Reduces fragmentation-driven OOM; must be set before CUDA allocs.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.torch = torch
        self.temperature = temperature
        self.gpu = probe_gpu()
        print(f"[backend] {self.gpu.summary()}")

        if precision == "auto":
            precision = recommend_precision(param_count_b, self.gpu)
            print(f"[backend] auto-selected precision: {precision}")
        self.precision = precision

        est = estimate_weight_gb(param_count_b, precision)
        if self.gpu.available:
            print(
                f"[backend] ~{est:.1f} GiB weights @ {precision} vs "
                f"{self.gpu.total_vram_gb:.1f} GiB VRAM "
                f"({self.gpu.total_vram_gb - est:.1f} GiB left for KV cache)"
            )
            if est > self.gpu.total_vram_gb and not allow_cpu_offload:
                raise RuntimeError(
                    f"Model needs ~{est:.1f} GiB at {precision} but GPU has only "
                    f"{self.gpu.total_vram_gb:.1f} GiB. Refusing to load: device_map='auto' "
                    f"would silently offload to CPU and run at ~2 tok/s, which never finishes. "
                    f"Use precision='4bit', a smaller model, or a bigger GPU. "
                    f"Pass allow_cpu_offload=True only if you truly want the slow path."
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs: dict = {"device_map": "auto"}

        if precision in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ImportError("pip install bitsandbytes accelerate for quantized loading") from e
            if precision == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    # compute dtype must match what the GPU can actually do fast
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.gpu.supports_bf16 else torch.float16,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if precision == "bf16" else torch.float16

        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        if not allow_cpu_offload:
            # Empty cpu budget => accelerate errors instead of spilling layers.
            kwargs.setdefault("max_memory", self._default_max_memory())

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self._warn_if_offloaded()

    def _default_max_memory(self) -> dict:
        """Give every GPU its full memory and the CPU zero, so accelerate
        raises rather than silently offloading."""
        if not self.gpu.available:
            return {}
        mm: dict = {i: f"{int(self.gpu.total_memory_gb[i])}GiB" for i in range(self.gpu.n_devices)}
        mm["cpu"] = "0GiB"
        return mm

    def _warn_if_offloaded(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        if not device_map:
            return
        offloaded = [k for k, v in device_map.items() if v in ("cpu", "disk")]
        if offloaded:
            print(
                f"[backend] WARNING: {len(offloaded)} module(s) offloaded to CPU/disk. "
                f"Expect ~10x slowdown. First few: {offloaded[:5]}"
            )

    @property
    def device(self):
        return self.model.device

    def _generate_from_ids(self, input_ids, max_new_tokens: int) -> str:
        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _chat_input_ids(self, prompt: str):
        """apply_chat_template's return type is not stable across
        transformers versions -- some return a raw tensor, others a
        BatchEncoding/dict. Passing a BatchEncoding straight into
        model.generate() fails deep inside generate() with a confusing
        AttributeError ('shape') from BatchEncoding.__getattr__, since
        generate() expects a plain tensor. This normalizes either return
        shape to a plain 2D tensor before use. Found the hard way: this
        exact bug was already fixed in the transcribed Kaggle scripts
        (_chat_input_ids there) but not here -- surfaced on Modal with
        transformers 5.17.0, a newer version than earlier Kaggle runs used.
        """
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        if isinstance(enc, dict) or hasattr(enc, "input_ids"):
            enc = enc["input_ids"]
        if enc.dim() == 1:
            enc = enc.unsqueeze(0)
        return enc.to(self.model.device)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        input_ids = self._chat_input_ids(prompt)
        return self._generate_from_ids(input_ids, max_new_tokens)

    def continue_generate(self, prefix_text: str, max_new_tokens: int) -> str:
        """Continue raw text (NOT re-wrapped in a fresh chat template) --
        used for forced extraction, where prefix_text is already the model's
        own prior completion plus the forced-answer suffix.
        """
        input_ids = self.tokenizer(prefix_text, return_tensors="pt").input_ids.to(self.model.device)
        return self._generate_from_ids(input_ids, max_new_tokens)

    def free_vram(self) -> None:
        """Call between problems if fragmentation becomes an issue."""
        if self.gpu.available:
            self.torch.cuda.empty_cache()

    def forced_extract_with_scores(
        self, prefix_text: str, suffix: str, max_new_tokens: int = 48,
    ) -> "ScoredCompletion":
        """Like continue_generate(), but ALSO captures the model's own
        probability distribution at the FIRST generated token (right after
        the primed suffix, e.g. immediately after "\\boxed{") to compute
        Candidate B's phi(p_t): local token confidence and entropy.

        output_scores=True is SAFE here ONLY because max_new_tokens is
        small (~48). This project already learned the hard way that the
        same flag on a full ~2000-4000 token trace retains a
        [1, vocab_size] logits tensor per generated token and OOMs (the
        pilot's first failed Kaggle run, ~0.6-1.2 GiB at trace length).
        Never enable output_scores on generate()/continue_generate() at
        full trace length -- this method exists specifically so that
        stays true, by keeping the scored call short and separate.

        Confidence/entropy are computed on the TEMPERATURE-SCALED
        distribution actually used for sampling (what `scores` returns
        under do_sample=True), not raw pre-temperature logits -- this
        matches the rest of the pipeline's fixed T=0.6 protocol, so a
        confidence number here means the same thing it would if read
        directly off the sampling step that produced the token.
        """
        input_ids = self.tokenizer(prefix_text, return_tensors="pt").input_ids.to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        new_tokens = out.sequences[0][input_ids.shape[1]:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if len(out.scores) == 0 or new_tokens.shape[0] == 0:
            raise RuntimeError("forced_extract_with_scores: generated zero tokens, cannot score")

        first_step_logits = out.scores[0][0]  # scores[step][batch] -> (vocab,)
        probs = self.torch.softmax(first_step_logits.float(), dim=-1)
        first_token_id = int(new_tokens[0].item())
        confidence = float(probs[first_token_id].item())

        p = probs.detach().cpu().numpy()
        nz = p[p > 0]
        entropy_bits = float(-(nz * np.log2(nz)).sum())

        return ScoredCompletion(
            completion_text=completion,
            full_text=prefix_text + completion,
            top_token_confidence=confidence,
            entropy_bits=entropy_bits,
        )

    def extract_hidden_state(self, prefix_text: str, layer: int = -1) -> list[float]:
        """Single forward pass (NO generation) returning the residual-stream
        activation at the LAST token of prefix_text, at the given layer
        (default: final layer). For the mechanistic probe (Phase 3, low-
        compute plan) -- is "will this prefix go on to state a decisive
        answer or trail off empty" linearly decodable from hidden state at
        the exact matched checkpoint used for branching?

        Cheap: no autoregressive generation, one pass over prefix_text's
        existing tokens. Must be called on the SAME prefix_text used for
        branch_naturally() for that pair/side, in the SAME run -- prefix
        text is not persisted anywhere after a run ends (see
        CANDIDATE_B_PREREGISTRATION.md 2026-09-11 amendment), so this
        cannot be run retroactively on already-collected results.
        """
        input_ids = self.tokenizer(prefix_text, return_tensors="pt").input_ids.to(self.model.device)
        with self.torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                output_hidden_states=True,
            )
        # hidden_states: tuple of (n_layers + 1) tensors, each [batch, seq, hidden]
        # index 0 is the embedding layer's output, -1 is the final layer.
        h = out.hidden_states[layer][0, -1, :]
        return h.detach().float().cpu().numpy().tolist()


@dataclass(frozen=True)
class ScoredCompletion:
    """Result of forced_extract_with_scores() -- a short forced-extraction
    completion plus the model's own confidence/entropy at its first
    generated token. Feeds directly into
    early_stop.path_dependence.ObservableState (via the answer parsed from
    completion_text/full_text, elsewhere)."""

    completion_text: str
    full_text: str  # prefix_text + completion_text, for grading (matches forced_extract_and_grade's convention)
    top_token_confidence: float  # LOCAL TOKEN confidence -- see ObservableState's own docstring caveat
    entropy_bits: float


class MockScoredBackend:
    """Deterministic stand-in for HFBackend.forced_extract_with_scores(),
    for testing Candidate B's orchestration pipeline without a GPU. Mirrors
    early_stop.extraction.MockBackend's canned-completion-by-substring
    design, extended with per-call confidence/entropy.
    """

    def __init__(
        self,
        canned: dict[str, tuple[str, float, float]] | None = None,
        default: tuple[str, float, float] = (" 0}.", 0.5, 1.0),
    ):
        # canned: {substring_to_match_in_prefix: (completion_text, confidence, entropy_bits)}
        self.canned = canned or {}
        self.default = default
        self.calls: list[str] = []
        # also usable as a plain ModelBackend (unscored) for natural
        # branching via continue_generate-style calls
        self.continue_canned: dict[str, str] = {}
        self.continue_default = " the answer is 0."

    def forced_extract_with_scores(self, prefix_text: str, suffix: str, max_new_tokens: int = 48) -> ScoredCompletion:
        self.calls.append(prefix_text + suffix)
        prompt = prefix_text + suffix
        completion, confidence, entropy = self.default
        for key, val in self.canned.items():
            if key in prompt:
                completion, confidence, entropy = val
                break
        return ScoredCompletion(
            completion_text=completion,
            full_text=prompt + completion,
            top_token_confidence=confidence,
            entropy_bits=entropy,
        )

    def continue_generate(self, prefix_text: str, max_new_tokens: int) -> str:
        """Unscored natural continuation, for branch_naturally() -- matches
        HFBackend.continue_generate's signature so both satisfy the same
        minimal protocol Candidate B's orchestration code depends on."""
        self.calls.append(prefix_text)
        for key, val in self.continue_canned.items():
            if key in prefix_text:
                return val
        return self.continue_default

    def extract_hidden_state(self, prefix_text: str, layer: int = -1) -> list[float]:
        """Deterministic stand-in for HFBackend.extract_hidden_state(), for
        testing the probe pipeline without a GPU: a fixed-length vector
        derived from a hash of prefix_text, so different prefixes get
        different (but reproducible) vectors."""
        import hashlib
        self.calls.append(prefix_text)
        digest = hashlib.sha256(prefix_text.encode()).digest()
        return [b / 255.0 for b in digest[:16]]

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Chat-templated full-trace generation, for
        generate_diverse_trajectories() -- same signature as
        HFBackend.generate(). Delegates to a simple canned/default lookup
        like continue_generate; kept separate so callers testing "did this
        go through the chat-template path vs raw continuation" can tell
        which one fired by checking self.calls."""
        self.calls.append(f"[chat]{prompt}")
        for key, val in self.continue_canned.items():
            if key in prompt:
                return val
        return self.continue_default


class ForcedExtractionBackendAdapter:
    """Adapts HFBackend.continue_generate to the ModelBackend protocol so
    forced_extract() can call it uniformly, while full-trace generation
    still goes through the chat-templated HFBackend.generate().
    """

    def __init__(self, hf_backend: HFBackend):
        self._backend = hf_backend

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return self._backend.continue_generate(prompt, max_new_tokens)
