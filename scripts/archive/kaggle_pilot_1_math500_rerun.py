# ============================================================
# NOTEBOOK 1 of 3 -- MATH-500 RE-RUN (PRIORITY 1, BLOCKING)
#
# Purpose: the 2026-08-27 pilot capped 35% of MATH-500 traces at
# MAX_NEW_TOKENS=2000, so "acc@full" was itself truncated. This re-run
# changes ONLY the token limit (2000 -> 4000), holding N=40 and the
# SAME 40 problems fixed (deterministic first-N slice), to isolate
# whether the token cap -- not real overthinking -- produced the
# original +7.5pp "truncation beats full trace" effect.
#
# Reading: see PILOT_PREREGISTRATION.md, "pre-committed reading of the
# 4000-token re-run" (dated 2026-08-28, written BEFORE this was run).
# The script prints an explicit RE-RUN BRANCH READING block (A/B/C) at
# the end -- read that, don't eyeball the curves yourself.
#
# Do NOT change N_PROBLEMS/PROBLEM_START/PROBLEM_END in this file: doing
# so would confound the token-cap question with a different problem mix.
# ============================================================

# ============================================================
# PILOT GATE -- DeepSeek-R1-Distill-Qwen-7B
# Paste as ONE Kaggle cell. Requires: Settings -> Accelerator -> GPU T4 x2
#                                     Settings -> Internet -> ON
#                                     (or mount the model, see below)
#
# Implements PILOT_PREREGISTRATION.md. Fixes accumulated across runs:
#   1. NO output_scores -- it retained a [1, 152064] logits tensor per
#      generated token (~0.6-1.2 GB at 2k tokens) and caused the OOM.
#      The pilot gate does not need logprobs; signals come later.
#   2. Model actually fits: auto 4-bit (NF4) on one 16GB card, or fp16
#      sharded if 2 GPUs are present. Refuses to silently CPU-offload,
#      which is what produced 1.9 tok/s.
#   3. Uses the chat template -- R1-Distill expects <|User|>...<|Assistant|>.
#      Raw tokenizer() input is off-distribution and changes the very
#      segmentation behavior this pilot measures.
#   4. temperature 0.6 + do_sample (DeepSeek explicitly warns that greedy
#      decoding sends R1 models into repetition loops).
#   5. Answer normalization no longer strips ALL '}' characters, which
#      mangled \frac{1}{2} -> \frac{1{2.
#   6. Forced extraction primes an open \boxed{ and grades the FULL text
#      (prompt+suffix+completion), not the completion alone.
#   7. transformers 5.0: apply_chat_template returns BatchEncoding not a
#      tensor (this raised on every problem -> 0 traces); torch_dtype
#      renamed to dtype; temperature/top_p ignored as loose kwargs, so
#      they now go through an explicit GenerationConfig that is asserted
#      AND empirically verified (two draws must differ).
#   8. Errors are never swallowed -- the first failure prints its full
#      traceback and stops, instead of 40 silent failures.
# ============================================================

# '|| true' so a dead network here does not abort the cell -- Kaggle already
# ships transformers/datasets/accelerate, and bitsandbytes is only needed on
# the 4-bit path (not used when 2 GPUs give enough VRAM for fp16).
!pip -q install -U datasets accelerate bitsandbytes sentencepiece 2>/dev/null || true

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# The 7B is ~15 GB of weights. The default HF cache is /root/.cache, which on
# Kaggle sits on a small root disk and can fill mid-download. /kaggle/temp is
# scratch that does not count toward the 20 GB output quota.
# CAVEAT: /kaggle/temp is WIPED between sessions, so each new session
# re-downloads 15 GB and therefore needs Internet ON. To avoid that entirely,
# mount the model as a Kaggle Model input (see _find_local_model below).
if os.path.isdir("/kaggle"):
    os.makedirs("/kaggle/temp/hf", exist_ok=True)
    os.environ["HF_HOME"] = "/kaggle/temp/hf"
    print("HF cache -> /kaggle/temp/hf (scratch, not persisted)")

import re, json, time
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Prefer a locally MOUNTED copy if one exists: Add Input -> Models -> search
# "DeepSeek-R1-Distill-Qwen-7B". A mount needs no download and no Internet,
# and survives session restarts. Falls back to the hub otherwise.
def _find_local_model():
    import glob
    for pat in ("/kaggle/input/**/config.json", "/kaggle/working/**/config.json"):
        for cfg in glob.glob(pat, recursive=True):
            d = os.path.dirname(cfg)
            if glob.glob(os.path.join(d, "*.safetensors")) or \
               glob.glob(os.path.join(d, "*.safetensors.index.json")):
                return d
    return None

_local = _find_local_model()
MODEL = _local or MODEL_REPO
print(f"Model source: {'LOCAL MOUNT ' + MODEL if _local else 'HuggingFace hub (needs Internet)'}")
N_PROBLEMS = 40         # see banner above -- purpose-specific for this notebook

# For the MAIN STUDY (after the pilot re-run passes), the primary pool is too
# small at N=40/238 -- see PILOT_RESULTS.md, "sample size" finding. Widen by
# raising N_PROBLEMS (up to len(filtered dataset)) and, if splitting work
# across two sessions/accounts to cut wall-clock time, set PROBLEM_START/END
# below so each covers a disjoint index range of the SAME deterministic
# ordering -- outputs concatenate cleanly afterward with zero overlap, no
# need to touch SplitManager or coordinate between sessions live.
PROBLEM_START = 0      # see banner above
PROBLEM_END = None     # exclusive; None = up to N_PROBLEMS
# 4000, not 2000: at 2000 the FIRST pilot run truncated 35% of MATH-500
# traces mid-reasoning, so the f=1.0 "full trace" baseline was itself capped
# and acc@full was a lower bound. PROPOSAL §14 anticipated 2k-4k traces.
MAX_NEW_TOKENS = 4000
TEMPERATURE = 0.6           # DeepSeek recommends 0.5-0.7; NOT 0.0
OUTPUT_DIR = "pilot_results"
TRUNCATION_FRACTIONS = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00]

# PILOT_PREREGISTRATION.md §3b -- the gate is read at ONE pre-specified
# fraction. Reading whichever fraction looks best inflates spurious passes
# from ~22% to ~71%. Do not change these after seeing results.
GATE_FRACTION = 0.60
HEADROOM_THRESHOLD = 0.10

# Run ONE domain per session. ~1.3 h each at N=40; doing all three in one
# cell is ~4 h and Kaggle sessions disconnect. Start with MATH-500: it is the
# primary domain, and if it fails the segmentation gate the design changes
# and the other two are moot (PILOT_PREREGISTRATION.md §5).
DOMAINS = ["MATH-500"]   # fixed -- this notebook is the MATH-500 diagnostic re-run only

os.makedirs(OUTPUT_DIR, exist_ok=True)
_range_tag = f"_{PROBLEM_START}-{PROBLEM_END or N_PROBLEMS}" if (PROBLEM_START, PROBLEM_END) != (0, None) else ""
tag = "_".join(DOMAINS).replace("-", "").replace(" ", "") + _range_tag

# ------------------------------------------------------------
# HARDWARE-AWARE LOADING
# ------------------------------------------------------------
n_gpu = torch.cuda.device_count()
total_vram = sum(
    torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(n_gpu)
)
cap = torch.cuda.get_device_capability(0)
supports_bf16 = cap[0] >= 8          # Ampere+. T4 is SM 7.5 -> fp16 only.
print(f"GPUs: {n_gpu} | total VRAM: {total_vram:.1f} GiB | capability: {cap} "
      f"| bf16: {supports_bf16}")

# DeepSeek-R1-Distill-Qwen-7B is really 7.62B params (Qwen2.5-Math-7B base),
# = ~14.2 GiB at fp16. On a 14.56 GiB T4 the WEIGHTS FIT -- which is exactly
# why device_map='auto' didn't complain last time -- but that leaves ~0.4 GiB,
# and the run then died allocating 1.02 GiB of KV cache. So the check must
# require weights + KV headroom, not just weights.
FP16_WEIGHTS_GB = 7.62 * 1e9 * 2 / 1024**3     # ~14.2 GiB
KV_HEADROOM_GB = 3.0
use_4bit = total_vram < (FP16_WEIGHTS_GB + KV_HEADROOM_GB)

kwargs = {"device_map": "auto"}
if use_4bit:
    print(f"-> 4-bit NF4 (fp16 weights would need ~{FP16_WEIGHTS_GB + KV_HEADROOM_GB:.1f} GiB)")
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if supports_bf16 else torch.float16,
    )
else:
    print("-> fp16 (fits without quantization)")
    # transformers 5.0 renamed torch_dtype -> dtype (old name warns, may break)
    _dt = torch.bfloat16 if supports_bf16 else torch.float16
    try:
        import transformers as _tf
        _major = int(_tf.__version__.split(".")[0])
    except Exception:
        _major = 4
    kwargs["dtype" if _major >= 5 else "torch_dtype"] = _dt
    # Zero CPU budget: make accelerate RAISE instead of silently offloading
    # to CPU at ~2 tok/s.
    mm = {i: f"{int(torch.cuda.get_device_properties(i).total_memory/1024**3)}GiB"
          for i in range(n_gpu)}
    mm["cpu"] = "0GiB"
    kwargs["max_memory"] = mm

print("\nLoading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
except OSError as e:
    if "name resolution" in str(e) or "Can't load the configuration" in str(e):
        raise SystemExit(
            "\n" + "!" * 70 +
            "\nCANNOT REACH HUGGINGFACE -- this is a NETWORK problem, not a code bug."
            "\n"
            "\n  Kaggle resets the Internet toggle on every new session."
            "\n  Fix: Notebook -> Settings (right panel) -> Internet -> ON"
            "\n       (requires a phone-verified Kaggle account)"
            "\n"
            "\n  Better long-term fix, works OFFLINE and survives restarts:"
            "\n    Add Input -> Models -> search 'DeepSeek-R1-Distill-Qwen-7B'"
            "\n    -> Add. It mounts under /kaggle/input and this script will"
            "\n    auto-detect it, with no 15 GB download at all."
            "\n" + "!" * 70
        ) from e
    raise
model.eval()

offloaded = [k for k, v in (model.hf_device_map or {}).items() if v in ("cpu", "disk")]
if offloaded:
    print(f"WARNING: {len(offloaded)} modules on CPU/disk -- expect ~10x slowdown.")
else:
    print("Model fully on GPU.")

# ------------------------------------------------------------
# GENERATION  (no output_scores -- that was the OOM)
#
# transformers 5.0 warns "generation flags are not valid and may be
# ignored: ['temperature','top_p']" when they are passed as loose kwargs.
# Temperature is a PRE-REGISTERED protocol parameter (PILOT_PREREG §1) --
# silently sampling at some other value would put the whole run off
# protocol. So build an explicit GenerationConfig and assert it took.
# ------------------------------------------------------------
from transformers import GenerationConfig

GEN_CFG = GenerationConfig(
    do_sample=True,
    temperature=TEMPERATURE,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
)

_eff = GEN_CFG.to_dict()
print(f"\nEffective sampling config: do_sample={_eff.get('do_sample')} "
      f"temperature={_eff.get('temperature')} top_p={_eff.get('top_p')}")
assert _eff.get("do_sample") is True, "sampling disabled -- greedy decoding degenerates on R1"
assert abs(_eff.get("temperature", -1) - TEMPERATURE) < 1e-9, \
    f"temperature is {_eff.get('temperature')}, expected {TEMPERATURE} (PILOT_PREREG §1)"
print("Sampling params verified against pre-registered protocol.\n")
def _chat_input_ids(user_prompt):
    """apply_chat_template's return type varies across transformers versions:
    some return a bare tensor, others a BatchEncoding/dict. Normalize to a
    2-D LongTensor on the model device. Getting this wrong makes EVERY
    generation raise, which is what produced 0 traces on the first run."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(enc, dict) or hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc.dim() == 1:
        enc = enc.unsqueeze(0)
    return enc.to(model.device)


def generate_chat(user_prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Full trace from a user turn, using the model's chat template."""
    ids = _chat_input_ids(user_prompt)
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            generation_config=GEN_CFG,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    gen = out[0][ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    elapsed = time.time() - start
    return {"text": text, "tokens": len(gen), "seconds": elapsed,
            "tokens_per_second": len(gen) / max(elapsed, 1e-9)}


def continue_raw(prefix_text, max_new_tokens=48):
    """Continue raw text WITHOUT re-wrapping in a new chat turn -- used for
    forced extraction, where prefix_text is already the model's own partial
    output plus the forced-answer suffix."""
    ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            generation_config=GEN_CFG,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

# ------------------------------------------------------------
# SELF-TEST: is sampling ACTUALLY happening?
#
# The assert above only proves the config object holds T=0.6 -- not that
# generate() honours it. transformers 5.0 warned these flags "may be
# ignored", so verify empirically: two sampled generations of the same
# prompt should differ. Identical output every time means we silently
# collapsed to greedy, which per PILOT_PREREG §1 is off-protocol AND
# degenerates into repetition loops on R1 models.
# ------------------------------------------------------------
_probe = "Compute 17 * 23. Reason step by step."
_a = generate_chat(_probe, max_new_tokens=40)["text"]
_b = generate_chat(_probe, max_new_tokens=40)["text"]
if _a == _b:
    print("!" * 70)
    print("WARNING: two sampled generations were IDENTICAL.")
    print("Sampling may have collapsed to greedy despite temperature=%.2f." % TEMPERATURE)
    print("This is off-protocol (PILOT_PREREG §1). Investigate before trusting results.")
    print("!" * 70)
else:
    print(f"Sampling verified: two draws differ (T={TEMPERATURE}).")
del _a, _b, _probe

# ------------------------------------------------------------
# SEGMENTATION
# ------------------------------------------------------------
def segment_trace(text):
    return [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]

# ------------------------------------------------------------
# ANSWER EXTRACTION
# ------------------------------------------------------------
def _find_matching_brace(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None

def extract_boxed(text):
    """Brace-matched, LAST \\boxed{...} in the text."""
    ms = list(re.finditer(r"\\boxed\{", text))
    if not ms:
        return None
    open_idx = ms[-1].end() - 1
    close_idx = _find_matching_brace(text, open_idx)
    if close_idx is None:
        return text[open_idx + 1:].strip()   # unterminated (truncated) box
    return text[open_idx + 1:close_idx].strip()

def normalize_answer(a):
    """NOTE: does NOT strip all '}' -- the original did, turning
    \\frac{1}{2} into \\frac{1{2."""
    if a is None:
        return None
    s = str(a).strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace(" ", "").replace(",", "").rstrip(".").lstrip("+")
    s = s.replace("\\%", "").replace("%", "")
    try:
        from fractions import Fraction
        f = Fraction(s)
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    except Exception:
        return s.lower()

def extract_answer(text):
    b = extract_boxed(text)
    if b is not None:
        return normalize_answer(b)
    for pat in [r"final answer\s*(?:is|:)\s*([^\n]+)", r"answer\s*(?:is|:)\s*([^\n]+)"]:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return normalize_answer(m[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*(?:/\d+)?", text)
    return normalize_answer(nums[-1]) if nums else None

def answers_match(pred, gold, name="MATH-500"):
    if pred is None or gold is None:
        return False
    if name == "CommonsenseQA":
        return str(pred).strip().upper() == str(gold).strip().upper()
    return normalize_answer(pred) == normalize_answer(gold)

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
def make_prompt(problem, name="MATH-500"):
    if name == "CommonsenseQA":
        return (f"{problem}\n\nPlease reason step by step, then give your "
                f"final answer as a single letter.")
    return (f"{problem}\n\nPlease reason step by step, and put your final "
            f"answer within \\boxed{{}}.")


# The forced-answer suffix must match the domain's answer format. For math it
# primes an open \boxed{ ; for multiple choice it primes an open paren.
FORCED_SUFFIXES = {
    "MATH-500": "\n\nTherefore, the final answer is \\boxed{",
    "GSM8K": "\n\nTherefore, the final answer is \\boxed{",
    "CommonsenseQA": "\n\nTherefore, the final answer is (",
}


def extract_for(name, text, n_choices=5):
    """Domain-dispatched extraction. MC answers are single letters, so the
    math \boxed/last-number parser would mis-read them."""
    if name == "CommonsenseQA":
        valid = set("ABCDE"[:n_choices])
        ms = [m for m in re.findall(r"\b\(?([A-E])\)?\b", text) if m in valid]
        return ms[-1] if ms else None
    return extract_answer(text)

ds_map = {}
def _slice(ds, start, end, n_default):
    """Deterministic index range into a stable-order HF dataset. Two
    sessions using the SAME dataset + filter + (start,end) pair always get
    disjoint, non-overlapping, order-matched problem sets -- this is what
    makes splitting across accounts safe without any live coordination."""
    lo = max(0, start)
    hi = min(len(ds), end if end is not None else n_default)
    if lo >= hi:
        raise SystemExit(f"Empty slice [{lo}:{hi}] into a dataset of {len(ds)} -- check PROBLEM_START/END")
    return ds.select(range(lo, hi))

if "MATH-500" in DOMAINS:
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_ds = math_ds.filter(lambda r: str(r["level"]) in {"1", "2", "3"})
    ds_map["MATH-500"] = _slice(math_ds, PROBLEM_START, PROBLEM_END, N_PROBLEMS)
if "GSM8K" in DOMAINS:
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    ds_map["GSM8K"] = _slice(gsm, PROBLEM_START, PROBLEM_END, N_PROBLEMS)
if "CommonsenseQA" in DOMAINS:
    cqa = load_dataset("tau/commonsense_qa", split="validation")
    cqa = cqa.filter(lambda r: bool(r["answerKey"]))
    ds_map["CommonsenseQA"] = _slice(cqa, PROBLEM_START, PROBLEM_END, N_PROBLEMS)
if not ds_map:
    raise SystemExit(f"DOMAINS={DOMAINS} matched nothing")
for k, v in ds_map.items():
    print(f"Loaded {k}: {len(v)}")

def get_qa(name, item):
    if name == "MATH-500":
        return item["problem"], normalize_answer(item["answer"])
    if name == "CommonsenseQA":
        ch = dict(zip(item["choices"]["label"], item["choices"]["text"]))
        q = item["question"] + "\n" + "\n".join(f"({k}) {v}" for k, v in sorted(ch.items()))
        return q, item["answerKey"].strip().upper()
    m = re.search(r"####\s*(.*)", item["answer"])
    return item["question"], normalize_answer(m.group(1) if m else item["answer"])

# ------------------------------------------------------------
# ------------------------------------------------------------
# 1) MAIN LOOP -- generation AND truncation together, per problem
#
# These used to be two sequential phases (generate all 40 traces, THEN
# truncate all 40). Merged into one pass so that:
#   * each problem's COMPLETE result is printed the moment it exists, so a
#     notebook deactivation at problem 35/40 leaves 35 usable results in the
#     log instead of nothing;
#   * each trace's text is dropped from RAM as soon as its truncations are
#     done, so memory stays flat no matter how large N or MAX_NEW_TOKENS get.
#     Only small numeric records accumulate (~200 bytes/problem).
# ------------------------------------------------------------
def _fmt_gold(g):
    g = str(g)
    return g if len(g) <= 12 else g[:11] + "\u2026"

all_results, rows = [], []
_trace_path = f"{OUTPUT_DIR}/traces_{tag}.jsonl"
_trace_fh = open(_trace_path, "w")

print("\n" + "=" * 78)
print("MAIN LOOP -- results print per problem; safe to lose the session after any line")
print("=" * 78)
print("  'c' column = correct at each fraction, in order: "
      + " ".join(f"{f:g}" for f in TRUNCATION_FRACTIONS))
print("  CAP = trace hit MAX_NEW_TOKENS (its 'full trace' is itself truncated)\n")

for name, ds in ds_map.items():
    print(f"--- {name} ---")
    for _local_i, item in enumerate(ds):
        i = _local_i + PROBLEM_START   # global index -- safe to merge across sessions
        problem, gold = get_qa(name, item)
        try:
            tr = generate_chat(make_prompt(problem, name))
            steps = segment_trace(tr["text"])
            step_lens = [len(tokenizer.encode(s)) for s in steps]
            capped = tr["tokens"] >= MAX_NEW_TOKENS

            # --- truncations for THIS problem, before moving on ---
            base = make_prompt(problem, name)
            marks = []
            for frac in TRUNCATION_FRACTIONS:
                k = max(1, int(np.ceil(len(steps) * frac)))
                partial = "\n\n".join(steps[:k])
                try:
                    # Prime the answer delimiter and grade the FULL accumulated
                    # text: the opening brace lives in the prompt, so grading
                    # the completion alone breaks non-numeric answers.
                    prefix = base + "\n\n" + partial + FORCED_SUFFIXES[name]
                    completion = continue_raw(prefix, max_new_tokens=48)
                    pred = extract_for(name, prefix + completion)
                    parseable = pred is not None
                except Exception as e:
                    print(f"    extraction error at frac={frac}: {e}")
                    pred, parseable = None, False
                ok = answers_match(pred, gold, name)
                marks.append("1" if ok else "0")
                rows.append({
                    "dataset": name, "index": i, "fraction": frac,
                    "steps_total": len(steps), "steps_kept": k,
                    "tokens_kept": len(tokenizer.encode(partial)),
                    "parseable": parseable, "correct": ok,
                    "gold": gold, "prediction": pred,
                })

            # --- one compact, self-contained line per problem ---
            print(f"[{name}] idx{i:>4} ({_local_i+1:>3}/{len(ds)}) | {tr['tokens']:>4}tok {len(steps):>3}st "
                  f"{'CAP' if capped else '   '} | {tr['tokens_per_second']:>4.1f}t/s | "
                  f"c {' '.join(marks)} | gold={_fmt_gold(gold)}", flush=True)

            # keep only small numeric fields in RAM
            all_results.append({
                "dataset": name, "index": i, "tokens": tr["tokens"],
                "seconds": tr["seconds"], "tokens_per_second": tr["tokens_per_second"],
                "num_steps": len(steps), "capped": bool(capped),
                "mean_step_tokens": float(np.mean(step_lens)) if step_lens else 0.0,
                "median_step_tokens": float(np.median(step_lens)) if step_lens else 0.0,
                "max_step_tokens": max(step_lens) if step_lens else 0,
            })
            # full text goes to disk, never accumulates in memory
            _trace_fh.write(json.dumps({
                "dataset": name, "index": i, "problem": problem,
                "gold_answer": gold, "full_text": tr["text"],
                "tokens": tr["tokens"], "num_steps": len(steps), "capped": bool(capped),
            }) + "\n")
            _trace_fh.flush()
            del tr, steps, step_lens

        except Exception:
            import traceback
            print("\n" + "!" * 70)
            print("GENERATION FAILED -- full traceback below. Results printed above")
            print("this line are still valid and can be used.")
            print("!" * 70)
            traceback.print_exc()
            raise
        torch.cuda.empty_cache()

_trace_fh.close()
print(f"\nGenerated {len(all_results)} traces.")
if not all_results:
    raise SystemExit("No traces were generated -- see the traceback above.")

seg_df = pd.DataFrame(all_results)[
    ["dataset", "index", "tokens", "num_steps", "mean_step_tokens",
     "median_step_tokens", "tokens_per_second"]
]
seg_df.to_csv(f"{OUTPUT_DIR}/segmentation_{tag}.csv", index=False)
trunc_df = pd.DataFrame(rows)
trunc_df.to_csv(f"{OUTPUT_DIR}/truncation_{tag}.csv", index=False)
trunc_df.to_csv(f"{OUTPUT_DIR}/truncation_{tag}.csv", index=False)

curve = (trunc_df.groupby(["dataset", "fraction"])
         .agg(accuracy=("correct", "mean"), parse_rate=("parseable", "mean"),
              avg_tokens=("tokens_kept", "mean"), avg_steps=("steps_kept", "mean"))
         .reset_index())
print("\n" + "=" * 70 + "\nACCURACY VS TRACE RETAINED (marginal)\n" + "=" * 70)
print(curve.to_string(index=False))

# ------------------------------------------------------------
# PAIRED ANALYSIS  (this is what the gate is actually read from)
#
# Every problem is measured at EVERY fraction, so the comparison is paired.
# Marginal accuracies above throw that pairing away and give a CI roughly
# 3x wider (+/-20pp vs +/-8pp at N=40). The information is in which
# problems FLIPPED and in which direction:
#   gained = wrong at full, right when truncated  <- direct overthinking evidence
#   lost   = right at full, wrong when truncated
# ------------------------------------------------------------
def paired_bootstrap_ci(diffs, reps=5000, seed=0):
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    bs = rng.choice(d, size=(reps, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

print("\n" + "=" * 70 + "\nPAIRED DIFFERENCE vs FULL TRACE\n" + "=" * 70)
print(f"delta = accuracy@full - accuracy@fraction  "
      f"(lower is better; gate reads f={GATE_FRACTION}, threshold {HEADROOM_THRESHOLD})\n")
paired_rows = []
for name in trunc_df["dataset"].unique():
    sub = trunc_df[trunc_df["dataset"] == name]
    full = sub[sub["fraction"] == 1.0].set_index("index")["correct"]
    print(f"{name}")
    print(f"  {'frac':>5} {'delta':>8} {'95% CI':>18} {'gained':>7} {'lost':>6} {'same':>6}")
    for f in sorted(sub["fraction"].unique()):
        at_f = sub[sub["fraction"] == f].set_index("index")["correct"]
        common = full.index.intersection(at_f.index)
        fu, tr = full.loc[common].astype(int), at_f.loc[common].astype(int)
        diffs = (fu - tr).values
        lo, hi = paired_bootstrap_ci(diffs)
        gained = int(((fu == 0) & (tr == 1)).sum())
        lost = int(((fu == 1) & (tr == 0)).sum())
        same = int((fu == tr).sum())
        print(f"  {f:>5.2f} {diffs.mean():>8.3f} [{lo:>7.3f},{hi:>7.3f}] "
              f"{gained:>7} {lost:>6} {same:>6}")
        paired_rows.append({"dataset": name, "fraction": f, "delta": diffs.mean(),
                            "ci_low": lo, "ci_high": hi, "gained": gained,
                            "lost": lost, "same": same, "n": len(common)})
    print()

paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(f"{OUTPUT_DIR}/paired_{tag}.csv", index=False)
total_gained = paired_df["gained"].sum()
print(f">>> Problems where truncation HELPED (across all fractions): {total_gained}")
print("    Non-zero = direct evidence of overthinking = the paper's premise.")

# ------------------------------------------------------------
# RE-RUN BRANCH READING (MATH-500 only) -- PILOT_PREREGISTRATION.md,
# "pre-committed reading of the 4000-token re-run" addendum, 2026-08-28.
#
# Named explicitly so the branch isn't picked after the fact by eyeballing
# two separate tables. The marginal effect (acc@full vs acc@0.75) and the
# per-problem discordance count are read SEPARATELY and printed separately,
# because branch C's whole point is that they can move independently: the
# marginal curve can flatten while the per-problem evidence still holds.
# ------------------------------------------------------------
if "MATH-500" in curve["dataset"].unique():
    _ORIG = {  # from the 2000-token pilot (PILOT_RESULTS.md), for comparison
        "acc_full": 0.800, "acc_075": 0.875, "gained_075": 3, "lost_075": 0,
        "cap_rate": 0.35,
    }
    _c = curve[curve["dataset"] == "MATH-500"].set_index("fraction")
    _p = paired_df[paired_df["dataset"] == "MATH-500"].set_index("fraction")
    _seg = seg_df[seg_df["dataset"] == "MATH-500"]
    _new_cap = (_seg["tokens"] >= MAX_NEW_TOKENS).mean()

    if 1.00 in _c.index and 0.75 in _c.index:
        new_full, new_075 = _c.loc[1.00, "accuracy"], _c.loc[0.75, "accuracy"]
        gap = new_075 - new_full   # positive = f=0.75 still beats full trace
        g075 = int(_p.loc[0.75, "gained"]) if 0.75 in _p.index else None
        l075 = int(_p.loc[0.75, "lost"]) if 0.75 in _p.index else None

        print("\n" + "#" * 78)
        print("RE-RUN BRANCH READING -- MATH-500 vs the 2000-token pilot")
        print("#" * 78)
        print(f"  cap rate:  was {_ORIG['cap_rate']:.0%} of traces  ->  now {_new_cap:.0%}"
              f"  (MAX_NEW_TOKENS={MAX_NEW_TOKENS})")
        print(f"  acc@full:  was {_ORIG['acc_full']:.1%}  ->  now {new_full:.1%}")
        print(f"  acc@0.75:  was {_ORIG['acc_075']:.1%}  ->  now {new_075:.1%}")
        print(f"  gap (acc@0.75 - acc@full): {gap:+.1%}"
              f"   (marginal-curve reading, NOT the per-problem reading below)")
        print()
        print(f"  per-problem discordance AT f=0.75 SPECIFICALLY (not the")
        print(f"  all-fractions total above -- this is branch-C's support):")
        print(f"    was: {_ORIG['gained_075']} gained / {_ORIG['lost_075']} lost")
        if g075 is not None:
            print(f"    now: {g075} gained / {l075} lost")

        _EPS = 1e-9  # accuracy values are ratios of small integers (N=40); a
        # boundary case like gap==0.05 exactly can land at 0.049999... in
        # float subtraction and silently misclassify without this tolerance
        if gap >= 0.05 - _EPS:
            branch = "A -- effect holds"
            action = "Proceed to signal work (PILOT_RESULTS.md §6)."
        elif gap >= 0.02 - _EPS:
            branch = "B -- effect shrinks but survives"
            action = ("Real but small. The sample-size problem (prereg SS2) is now "
                       "ACUTE: a gap this size vs a +/-6.7pp test-set CI is unresolvable. "
                       "Pool-widening becomes required, not optional groundwork.")
        else:
            branch = "C -- effect vanishes (or reverses) in the marginal curve"
            action = ("Pre-committed: this is NOT a failure and NOT license to search "
                       "other fractions for a surviving gap. Check the per-problem "
                       "discordance printed above -- if gained_075 > 0 still holds, "
                       "reframe to 'truncation preserves accuracy at lower cost' and "
                       "report that. Do not fish.")
        print(f"\n  ==> BRANCH: {branch}")
        print(f"  ==> ACTION: {action}")
    else:
        print("\n(Re-run branch reading skipped: need both f=0.75 and f=1.00 in this run.)")

# ------------------------------------------------------------
# 3) PLOTS + GATE SUMMARY
# ------------------------------------------------------------
import matplotlib.pyplot as plt
for name in curve["dataset"].unique():
    s = curve[curve["dataset"] == name]
    plt.figure(figsize=(8, 5))
    plt.plot(s["fraction"], s["accuracy"], marker="o", label="Accuracy")
    plt.plot(s["fraction"], s["parse_rate"], marker="s", label="Parse rate")
    plt.xlabel("Fraction of CoT steps retained"); plt.ylabel("Rate")
    plt.title(f"{name}: accuracy vs CoT retained"); plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3); plt.legend(); plt.show()

print("\n" + "=" * 70 + "\nPILOT GATE SUMMARY\n" + "=" * 70)
for name in seg_df["dataset"].unique():
    s = seg_df[seg_df["dataset"] == name]
    c = curve[curve["dataset"] == name].set_index("fraction")
    mean_steps = s["num_steps"].mean()
    med_steps = s["num_steps"].median()
    frac_lt3 = (s["num_steps"] < 3).mean()
    # median is PRIMARY; mean is diagnostic only (outlier-sensitive)
    seg_pass = (med_steps >= 4.0) and (frac_lt3 <= 0.30)
    print(f"\n{name}\n" + "-" * 40)
    print(f"  median steps/prob  : {med_steps:.1f}   <- PRIMARY")
    print(f"  mean steps/problem : {mean_steps:.2f}   (diagnostic; if it "
          f"disagrees with median, median rules)")
    q = s["num_steps"].quantile([0, .25, .5, .75, 1.0]).values
    print(f"  step distribution  : min={q[0]:.0f} p25={q[1]:.0f} "
          f"med={q[2]:.0f} p75={q[3]:.0f} max={q[4]:.0f}")
    print(f"  frac traces <3 steps: {frac_lt3:.0%}")
    print(f"  mean speed         : {s['tokens_per_second'].mean():.1f} tok/s")
    print(f"  mean trace tokens  : {s['tokens'].mean():.0f}")
    _capped = (s["tokens"] >= MAX_NEW_TOKENS).mean()
    print(f"  traces AT token cap: {_capped:.0%}"
          f"   {'<-- WARNING: full-trace baseline is itself truncated' if _capped > 0.10 else ''}")
    for f in [0.25, 0.50, 0.75, 1.00]:
        if f in c.index:
            print(f"  accuracy @{int(f*100):>3}%     : {c.loc[f,'accuracy']:.0%}"
                  f"   (parse {c.loc[f,'parse_rate']:.0%})")

    # ---- pre-registered gates (PILOT_PREREGISTRATION.md §3) ----
    p = paired_df[paired_df["dataset"] == name].set_index("fraction")
    acc_full = c.loc[1.00, "accuracy"] if 1.00 in c.index else float("nan")

    # 3b: read at the ONE pre-specified fraction. Scanning all fractions and
    # taking the best inflates spurious passes from ~22% to ~71%.
    if GATE_FRACTION in p.index:
        d = p.loc[GATE_FRACTION, "delta"]
        lo, hi = p.loc[GATE_FRACTION, "ci_low"], p.loc[GATE_FRACTION, "ci_high"]
        if hi <= HEADROOM_THRESHOLD:
            headroom = "PASS"
        elif lo > HEADROOM_THRESHOLD:
            headroom = "FAIL"
        else:
            headroom = "INCONCLUSIVE"
    else:
        d = lo = hi = float("nan")
        headroom = "FAIL"

    not_degenerate = 0.30 <= acc_full <= 0.95
    parse_full = c.loc[1.00, "parse_rate"] if 1.00 in c.index else float("nan")
    parse_low = c.loc[0.25, "parse_rate"] if 0.25 in c.index else float("nan")
    parse_ok = (parse_low >= 0.80) and ((parse_full - parse_low) <= 0.15)

    print(f"  [3a] segmentation  : {'PASS' if seg_pass else 'FAIL'}"
          f"   (median={med_steps:.1f} need>=4, <3steps={frac_lt3:.0%} need<=30%)")
    print(f"  [3b] headroom@{GATE_FRACTION:.2f}  : {headroom}"
          f"   (delta={d:+.3f} CI[{lo:+.3f},{hi:+.3f}] vs threshold {HEADROOM_THRESHOLD:.2f})")
    print(f"  [3b] not degenerate: {'PASS' if not_degenerate else 'FAIL'}   (acc@full={acc_full:.0%}, need 30-95%)")
    print(f"  [3c] extraction    : {'PASS' if parse_ok else 'FAIL'}   (parse@25%={parse_low:.0%}, drop={parse_full-parse_low:+.0%})")

    if headroom == "INCONCLUSIVE":
        verdict = "ESCALATE to N=100 (pre-registered, this domain only)"
    elif seg_pass and headroom == "PASS" and not_degenerate and parse_ok:
        verdict = "COMMIT"
    else:
        verdict = "DROP"
    print(f"  ==> DOMAIN VERDICT : {verdict}")

print("\n" + "=" * 70)
print("Read [3c] FIRST: if extraction failed, every accuracy number below")
print("that fraction is an artifact and the other gates are meaningless.")
print("GSM8K is EXPECTED to fail [3b] -- it is the negative control.")
print("Non-gate fractions are DESCRIPTIVE ONLY -- do not re-read the gate")
print("at a different fraction because the number looks better there.")
print("=" * 70)
# ------------------------------------------------------------
# RECONSTRUCTION BLOCK
#
# Everything needed to recompute every number above, printed as text so it
# survives the notebook being deactivated. The correctness matrix is the key
# artifact: 1 char per (problem, fraction), so the whole paired analysis --
# deltas, bootstrap CIs, gained/lost -- can be rebuilt from these lines alone.
# Deliberately does NOT print trace text (megabytes); that stays on disk.
# ------------------------------------------------------------
print("\n\n" + "=" * 78)
print("RECONSTRUCTION BLOCK -- copy everything between the BEGIN/END markers")
print("=" * 78)
print(">>>>> BEGIN")
print(f"# model={MODEL_REPO} T={TEMPERATURE} top_p=0.95 "
      f"max_new_tokens={MAX_NEW_TOKENS} N={N_PROBLEMS}")
print(f"# fractions={[float(f) for f in TRUNCATION_FRACTIONS]}")
print(f"# gate_fraction={GATE_FRACTION} headroom_threshold={HEADROOM_THRESHOLD}")
print("# cols: dataset,idx,tokens,steps,capped,correct_bits(in fraction order)")
_piv = trunc_df.pivot_table(index=["dataset", "index"], columns="fraction",
                            values="correct", aggfunc="first")
_meta = pd.DataFrame(all_results).set_index(["dataset", "index"])
for key, r in _piv.iterrows():
    bits = "".join("1" if bool(r[f]) else "0" for f in TRUNCATION_FRACTIONS)
    m = _meta.loc[key]
    print(f"{key[0]},{key[1]},{int(m['tokens'])},{int(m['num_steps'])},"
          f"{int(bool(m['capped']))},{bits}")
print(">>>>> END")

print("\n" + "=" * 78)
print("PAIRED SUMMARY (recomputable from the block above)")
print("=" * 78)
print(paired_df.to_string(index=False))

print(f"\nAlso saved to {OUTPUT_DIR}/ (lost if the session is not saved -- the")
print("printed block above is the durable copy).")
