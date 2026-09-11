# ============================================================
# CANDIDATE B FULL PILOT -- difficulty stratum 3 of 3, sub-split 1of3 of 3
# Paste as ONE Kaggle cell. Requires: Settings -> Accelerator -> GPU T4 x2
#                                     Settings -> Internet -> ON (or mount model)
#
# Implements CANDIDATE_B_PREREGISTRATION.md's design for a SLICE of ONE
# MATH-500 difficulty stratum (level 3, problems 0-1 of 6).
# Split 3 ways per stratum (9 notebooks total across all 3 strata) so each
# session stays well under Kaggle's per-session runtime cap (~4h estimated
# vs ~12h for the whole stratum in one notebook -- see DESIGN_CANDIDATE_B.md
# and the session-cap discussion before this split was made).
#
# problem_id is "d3_prob{pi}" with pi in [0,2) -- the SAME
# global numbering the undivided level-3 script would have used, so this
# sub-notebook's results concatenate cleanly with the other 2 sub-splits of
# this stratum with zero id collision (same pattern as the micro-pilot's
# PROBLEM_START/PROBLEM_END split, just applied within one stratum here).
#
# Reuses every hard-won fix from this project's earlier pilot scripts:
#   - hardware-aware precision (fp16 vs 4-bit), never silently CPU-offloads
#   - BatchEncoding-safe chat template handling (transformers 5.0)
#   - explicit GenerationConfig + EMPIRICAL sampling verification
#   - model-mount auto-detection (skips the 15GB download if mounted)
#   - per-item printing so a session death loses nothing already run
#   - output_scores=True used ONLY on short (~48 token) forced-extraction
#     calls, NEVER on full traces -- that combination OOMed in this
#     project's very first pilot attempt
#   - method=="boxed" required for phi(p_t) answers, NOT the last-number
#     fallback -- an incidental digit in the problem text (e.g. "x+5=12")
#     can get scraped as a fake "answer" by that fallback when the primed
#     box fails to close, silently manufacturing a spurious matched pair.
#   - stratified_sample_pairs stratifies by BOTH confidence AND problem_id
#     (fix for the imbalance found in micro-pilot run #2).
# ============================================================

!pip -q install -U datasets accelerate bitsandbytes sentencepiece scipy 2>/dev/null || true

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if os.path.isdir("/kaggle"):
    os.makedirs("/kaggle/temp/hf", exist_ok=True)
    os.environ["HF_HOME"] = "/kaggle/temp/hf"
    print("HF cache -> /kaggle/temp/hf (scratch, not persisted)")

import re, json, time, math as _math
import numpy as np
import pandas as pd
import torch
from fractions import Fraction
from dataclasses import dataclass
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon
from scipy.stats import combine_pvalues
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

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

# ------------------------------------------------------------
# FULL PILOT SIZE -- CANDIDATE_B_PREREGISTRATION.md SS1-3, this stratum SLICE only
# ------------------------------------------------------------
DIFFICULTY_LEVEL = "3"    # MATH-500's own level field -- this stratum
N_PROBLEMS_THIS_LEVEL_TOTAL = 6            # SS3: 6 problems per difficulty level, ACROSS all 3 sub-notebooks
SUB_START = 0                    # this sub-notebook's problem slice within the stratum
SUB_END = 2                        # exclusive
N_PROBLEMS_THIS_SUB = SUB_END - SUB_START
K_TRAJECTORIES_PER_PROBLEM = 5
N_BRANCHES_PER_PAIR = 40                   # SS4.9 power-corrected value
TOTAL_MAX_PAIRS_TO_BRANCH_THIS_LEVEL = 8   # SS3: 8 pairs per stratum, ACROSS all 3 sub-notebooks
# this sub-notebook's share of the stratum's pair budget, proportional to
# its slice of the stratum's problems -- same pattern as the micro-pilot's
# PROBLEM_START/END split (slight rounding overage across sub-notebooks is
# expected and fine, same as before).
MAX_PAIRS_TO_BRANCH = max(1, _math.ceil(TOTAL_MAX_PAIRS_TO_BRANCH_THIS_LEVEL * N_PROBLEMS_THIS_SUB / N_PROBLEMS_THIS_LEVEL_TOTAL))
MAX_NEW_TOKENS_TRACE = 2000
MAX_NEW_TOKENS_SCORE = 48
MAX_NEW_TOKENS_BRANCH = 1500
TEMPERATURE = 0.6
CONFIDENCE_TOL = 0.10
ENTROPY_TOL = 0.15
POSITION_TOL = 0.15
MIN_PARSE_RATE = 0.80
OUTPUT_DIR = f"candidate_b_fullpilot_level{DIFFICULTY_LEVEL}_{SUB_START}-{SUB_END}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# HARDWARE-AWARE LOADING
# ------------------------------------------------------------
n_gpu = torch.cuda.device_count()
total_vram = sum(torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(n_gpu))
cap = torch.cuda.get_device_capability(0)
supports_bf16 = cap[0] >= 8
print(f"GPUs: {n_gpu} | total VRAM: {total_vram:.1f} GiB | capability: {cap} | bf16: {supports_bf16}")

FP16_WEIGHTS_GB = 7.62 * 1e9 * 2 / 1024**3
KV_HEADROOM_GB = 3.0
use_4bit = total_vram < (FP16_WEIGHTS_GB + KV_HEADROOM_GB)

kwargs = {"device_map": "auto"}
if use_4bit:
    print(f"-> 4-bit NF4 (fp16 would need ~{FP16_WEIGHTS_GB + KV_HEADROOM_GB:.1f} GiB)")
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if supports_bf16 else torch.float16,
    )
else:
    print("-> fp16 (fits without quantization)")
    import transformers as _tf
    _major = int(_tf.__version__.split(".")[0])
    kwargs["dtype" if _major >= 5 else "torch_dtype"] = torch.bfloat16 if supports_bf16 else torch.float16
    mm = {i: f"{int(torch.cuda.get_device_properties(i).total_memory/1024**3)}GiB" for i in range(n_gpu)}
    mm["cpu"] = "0GiB"
    kwargs["max_memory"] = mm

print("\nLoading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
except OSError as e:
    if "name resolution" in str(e) or "Can't load the configuration" in str(e):
        raise SystemExit(
            "\nCANNOT REACH HUGGINGFACE. Settings -> Internet -> ON, or mount the "
            "model as a Kaggle Model input (Add Input -> Models -> search "
            "'DeepSeek-R1-Distill-Qwen-7B')."
        ) from e
    raise
model.eval()
offloaded = [k for k, v in (model.hf_device_map or {}).items() if v in ("cpu", "disk")]
print(f"WARNING: {len(offloaded)} modules on CPU/disk" if offloaded else "Model fully on GPU.")

GEN_CFG = GenerationConfig(do_sample=True, temperature=TEMPERATURE, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
_eff = GEN_CFG.to_dict()
assert _eff.get("do_sample") is True
assert abs(_eff.get("temperature", -1) - TEMPERATURE) < 1e-9
print(f"Effective sampling config: do_sample=True temperature={TEMPERATURE} top_p=0.95")

def _chat_input_ids(user_prompt):
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(enc, dict) or hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc.dim() == 1:
        enc = enc.unsqueeze(0)
    return enc.to(model.device)

def generate_chat(user_prompt, max_new_tokens):
    ids = _chat_input_ids(user_prompt)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=GEN_CFG,
                             max_new_tokens=max_new_tokens, use_cache=True)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def continue_raw(prefix_text, max_new_tokens):
    ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=GEN_CFG,
                             max_new_tokens=max_new_tokens, use_cache=True)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def forced_extract_with_scores(prefix_text, suffix, max_new_tokens):
    prompt = prefix_text + suffix
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=GEN_CFG,
                             max_new_tokens=max_new_tokens, use_cache=True,
                             output_scores=True, return_dict_in_generate=True)
    new_tokens = out.sequences[0][ids.shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if len(out.scores) == 0 or new_tokens.shape[0] == 0:
        return prompt, completion, 0.0, 0.0
    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
    first_id = int(new_tokens[0].item())
    confidence = float(probs[first_id].item())
    p = probs.detach().cpu().numpy()
    nz = p[p > 0]
    entropy_bits = float(-(nz * np.log2(nz)).sum())
    return prompt, completion, confidence, entropy_bits

print("Sampling verified (structure only; see full pilot scripts for the two-draws-differ empirical check).\n")

# ------------------------------------------------------------
# PARSING (matches early_stop/parsers.py, tested there)
# ------------------------------------------------------------
def _find_matching_brace(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return i
    return None

def extract_boxed(text):
    ms = list(re.finditer(r"\\boxed\{", text))
    if not ms: return None
    open_idx = ms[-1].end() - 1
    close_idx = _find_matching_brace(text, open_idx)
    if close_idx is None: return None
    return text[open_idx + 1:close_idx].strip()

def normalize_answer(a):
    if a is None: return None
    s = str(a).strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace("\\left", "").replace("\\right", "")
    s = s.replace(" ", "").replace(",", "").rstrip(".").lstrip("+").replace("\\%", "").replace("%", "")
    try:
        f = Fraction(s)
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    except Exception:
        return s.lower()

def parse_math_answer(text):
    b = extract_boxed(text)
    if b is not None:
        return normalize_answer(b), "boxed"
    nums = re.findall(r"-?\d[\d,]*\.?\d*(?:/\d+)?", text)
    if nums:
        return normalize_answer(nums[-1]), "last_number"
    return None, "failed"

def split_into_steps(text):
    return [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]

@dataclass
class ObservableState:
    answer: str
    confidence: float
    entropy: float

@dataclass
class Prefix:
    problem_id: str
    trajectory_id: str
    step_index: int
    total_steps: int
    state: ObservableState
    @property
    def normalized_position(self):
        return self.step_index / self.total_steps

def matches(a, b, level, confidence_tol=CONFIDENCE_TOL, entropy_tol=ENTROPY_TOL):
    if a.answer != b.answer: return False
    if level == "L1": return True
    if abs(a.confidence - b.confidence) > confidence_tol: return False
    if level == "L2": return True
    if abs(a.entropy - b.entropy) > entropy_tol: return False
    return True

def stratified_sample_pairs(pairs, budget, confidence_bounds=(0.0, 0.5, 0.85, 1.01), seed=0):
    """Confidence AND problem_id stratified sampling -- see
    DESIGN_CANDIDATE_B.md SS4.6/4.8 for the rationale."""
    if budget <= 0 or not pairs:
        return []
    n_strata = len(confidence_bounds) - 1
    buckets = [[] for _ in range(n_strata)]
    for a, b in pairs:
        mean_conf = (a.state.confidence + b.state.confidence) / 2.0
        for s_idx in range(n_strata):
            lo, hi = confidence_bounds[s_idx], confidence_bounds[s_idx + 1]
            if lo <= mean_conf < hi:
                buckets[s_idx].append((a, b))
                break

    rng = np.random.default_rng(seed)

    def _sample_evenly_by_problem(items, take_budget):
        if take_budget <= 0 or not items:
            return []
        by_problem = {}
        for pair in items:
            by_problem.setdefault(pair[0].problem_id, []).append(pair)
        for group in by_problem.values():
            rng.shuffle(group)
        groups = list(by_problem.values())
        out = []
        remaining = take_budget
        active_groups = [g for g in groups if g]
        while remaining > 0 and active_groups:
            share = max(1, remaining // len(active_groups))
            next_active = []
            for g in active_groups:
                take = min(share, len(g), remaining)
                out.extend(g[:take])
                del g[:take]
                remaining -= take
                if g and remaining > 0:
                    next_active.append(g)
                if remaining <= 0:
                    break
            active_groups = next_active
        return out

    for bucket in buckets:
        rng.shuffle(bucket)

    selected = []
    remaining = budget
    active = [i for i in range(n_strata) if buckets[i]]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        next_active = []
        for i in active:
            take = min(share, len(buckets[i]), remaining)
            selected.extend(_sample_evenly_by_problem(buckets[i], take))
            buckets[i] = buckets[i][take:]
            remaining -= take
            if buckets[i] and remaining > 0:
                next_active.append(i)
            if remaining <= 0:
                break
        active = next_active
    return selected


def find_matched_pairs(prefixes, level, position_tol=POSITION_TOL):
    by_problem = {}
    for p in prefixes:
        by_problem.setdefault(p.problem_id, []).append(p)
    pairs = []
    for group in by_problem.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.trajectory_id == b.trajectory_id: continue
                if position_tol is not None and abs(a.normalized_position - b.normalized_position) > position_tol:
                    continue
                if matches(a.state, b.state, level):
                    pairs.append((a, b))
    return pairs

def answer_distribution(answers):
    counts = {}
    for a in answers:
        if a is None: continue
        counts[a] = counts.get(a, 0) + 1
    total = sum(counts.values())
    if total == 0: raise ValueError("no parseable answers")
    return {k: v / total for k, v in counts.items()}

def pdi(dist_a, dist_b):
    keys = sorted(set(dist_a) | set(dist_b))
    pv = np.array([dist_a.get(k, 0.0) for k in keys])
    qv = np.array([dist_b.get(k, 0.0) for k in keys])
    d = jensenshannon(pv, qv, base=2)
    return float(d ** 2) if not np.isnan(d) else 0.0

def noise_floor_distribution(answers, reps=500, seed=0):
    rng = np.random.default_rng(seed)
    n = len(answers)
    if n < 4: return np.array([0.0])
    out = np.empty(reps)
    for i in range(reps):
        idx = rng.permutation(n)
        half = n // 2
        out[i] = pdi(answer_distribution([answers[k] for k in idx[:half]]),
                     answer_distribution([answers[k] for k in idx[half:]]))
    return out

def pair_permutation_test(answers_a, answers_b, reps=2000, seed=0):
    observed = pdi(answer_distribution(answers_a), answer_distribution(answers_b))
    pooled = answers_a + answers_b
    n_a = len(answers_a)
    rng = np.random.default_rng(seed)
    null = np.empty(reps)
    for i in range(reps):
        idx = rng.permutation(len(pooled))
        null[i] = pdi(answer_distribution([pooled[k] for k in idx[:n_a]]),
                      answer_distribution([pooled[k] for k in idx[n_a:]]))
    p = float((null >= observed).sum() + 1) / (reps + 1)
    return {"observed_pdi": observed, "null_mean": float(null.mean()),
            "effect_size": observed - float(null.mean()), "p_value": p,
            "n_a": n_a, "n_b": len(answers_b)}

# ------------------------------------------------------------
# DATA -- this stratum's problems 0-1 of 6 (MATH-500 level == DIFFICULTY_LEVEL)
# ------------------------------------------------------------
math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
math_ds = math_ds.filter(lambda r: str(r["level"]) == DIFFICULTY_LEVEL)
if len(math_ds) < N_PROBLEMS_THIS_LEVEL_TOTAL:
    raise SystemExit(f"Only {len(math_ds)} MATH-500 problems at level {DIFFICULTY_LEVEL}, "
                      f"need {N_PROBLEMS_THIS_LEVEL_TOTAL}")
_all_level_problems = math_ds.select(range(N_PROBLEMS_THIS_LEVEL_TOTAL))  # same 6, full stratum
problems = _all_level_problems.select(range(SUB_START, SUB_END))          # this sub-notebook's slice
print(f"Loaded {len(problems)} MATH-500 level-{DIFFICULTY_LEVEL} problems "
      f"(global indices {SUB_START}-{SUB_END-1}) for this sub-notebook\n")

CANDIDATE_B_SUFFIX = "\n\nTherefore, the final answer is \\boxed{"

def make_prompt(problem_text):
    return f"{problem_text}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
errors = []
all_prefixes = []
n_prefix_calls = 0
n_prefix_unparseable = 0
n_prefix_rejected_fallback = 0
trace_steps_cache = {}

print("=" * 78)
print(f"PHASE 1: diverse trajectories + phi(p_t), level {DIFFICULTY_LEVEL} problems {SUB_START}-{SUB_END-1}")
print("=" * 78)

for _local_pi, item in enumerate(problems):
    pi = _local_pi + SUB_START   # GLOBAL index within this stratum (0-5) -- must match what the
    # undivided level-3 script (or the other 2 sub-notebooks of this same
    # stratum) would have assigned, so results concatenate without collision.
    problem_id = f"d{DIFFICULTY_LEVEL}_prob{pi}"
    base_prompt = make_prompt(item["problem"])
    print(f"\n--- {problem_id} ---")
    for ti in range(K_TRAJECTORIES_PER_PROBLEM):
        traj_id = f"traj_{ti}"
        try:
            t0 = time.time()
            trace = generate_chat(base_prompt, MAX_NEW_TOKENS_TRACE)
            steps = split_into_steps(trace)
            total_steps = len(steps)
            trace_steps_cache[(problem_id, traj_id)] = steps
            n_scored = 0
            for step_index in range(1, total_steps + 1):
                prefix_text = "\n\n".join(steps[:step_index])
                full_prefix = base_prompt + "\n\n" + prefix_text
                prompt, completion, confidence, entropy = forced_extract_with_scores(
                    full_prefix, CANDIDATE_B_SUFFIX, MAX_NEW_TOKENS_SCORE
                )
                n_prefix_calls += 1
                answer, method = parse_math_answer(prompt + completion)
                if answer is None:
                    n_prefix_unparseable += 1
                    continue
                if method != "boxed":
                    n_prefix_rejected_fallback += 1
                    continue
                all_prefixes.append(Prefix(
                    problem_id=problem_id, trajectory_id=traj_id,
                    step_index=step_index, total_steps=total_steps,
                    state=ObservableState(answer=answer, confidence=confidence, entropy=entropy),
                ))
                n_scored += 1
            print(f"  {traj_id}: {total_steps} steps, {n_scored} scored, {time.time()-t0:.1f}s")
        except Exception as e:
            errors.append(f"{problem_id} {traj_id}: {type(e).__name__}: {e}")
            print(f"  {traj_id}: ERROR: {e}")
        torch.cuda.empty_cache()

print(f"\nTotal scored prefixes: {len(all_prefixes)}")
print(f"phi(p_t) calls: {n_prefix_calls} | unparseable: {n_prefix_unparseable} "
      f"({n_prefix_unparseable/max(n_prefix_calls,1):.0%}) | "
      f"rejected (last_number fallback, not a genuine boxed answer): {n_prefix_rejected_fallback} "
      f"({n_prefix_rejected_fallback/max(n_prefix_calls,1):.0%})")

# ------------------------------------------------------------
# PHASE 2: matching
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("PHASE 2: L1/L2/L3 matched pairs (different trajectory, same problem)")
print("=" * 78)
pairs_by_level = {}
for level in ("L1", "L2", "L3"):
    pairs = find_matched_pairs(all_prefixes, level)
    pairs_by_level[level] = pairs
    print(f"  {level}: {len(pairs)} matched pairs")

# ------------------------------------------------------------
# PHASE 3: natural branching + PDI on this sub-notebook's pair budget
# ------------------------------------------------------------
print("\n" + "=" * 78)
print(f"PHASE 3: natural branching (N={N_BRANCHES_PER_PAIR}/prefix, NO forced suffix), "
      f"capped at {MAX_PAIRS_TO_BRANCH} pairs (this sub-notebook)")
print("=" * 78)

def branch_naturally(base_prompt, prefix_step_text, n, max_new_tokens):
    full_prefix = base_prompt + "\n\n" + prefix_step_text
    outs = []
    for _ in range(n):
        completion = continue_raw(full_prefix, max_new_tokens)
        answer, _method = parse_math_answer(full_prefix + completion)
        outs.append(answer)
    return outs

pair_results = []
noise_diagnostics = []
branch_errors = 0
budget = MAX_PAIRS_TO_BRANCH
for level in ("L1", "L2", "L3"):
    if budget <= 0:
        break
    level_seed = {"L1": 1, "L2": 2, "L3": 3}[level]
    sampled_pairs = stratified_sample_pairs(pairs_by_level[level], budget, seed=level_seed)
    for a, b in sampled_pairs:
        try:
            _local_idx = int(a.problem_id.split("_prob")[1]) - SUB_START  # global -> local (this slice)
            problem_row = problems[_local_idx]
            base_prompt = make_prompt(problem_row["problem"])
            steps_a = trace_steps_cache[(a.problem_id, a.trajectory_id)]
            steps_b = trace_steps_cache[(b.problem_id, b.trajectory_id)]
            prefix_text_a = "\n\n".join(steps_a[:a.step_index])
            prefix_text_b = "\n\n".join(steps_b[:b.step_index])
            branches_a = branch_naturally(base_prompt, prefix_text_a, N_BRANCHES_PER_PAIR, MAX_NEW_TOKENS_BRANCH)
            branches_b = branch_naturally(base_prompt, prefix_text_b, N_BRANCHES_PER_PAIR, MAX_NEW_TOKENS_BRANCH)
            parsed_a = [x for x in branches_a if x is not None]
            parsed_b = [x for x in branches_b if x is not None]
            rate_a, rate_b = len(parsed_a) / len(branches_a), len(parsed_b) / len(branches_b)
            if rate_a < MIN_PARSE_RATE or rate_b < MIN_PARSE_RATE:
                print(f"  [{level}] {a.problem_id} pair skipped: parse rate a={rate_a:.0%} b={rate_b:.0%} (gate={MIN_PARSE_RATE:.0%})")
                continue
            result = pair_permutation_test(parsed_a, parsed_b)
            result["level"] = level
            result["problem_id"] = a.problem_id
            result["difficulty"] = DIFFICULTY_LEVEL
            # Raw branch outcomes -- saved so the actual answer breakdown
            # (not just PDI/p-value) is recoverable after the Kaggle session
            # ends. Added for consistency across the whole dataset (this
            # slice is being rerun specifically to backfill this detail).
            result["branches_a"] = parsed_a
            result["branches_b"] = parsed_b
            pair_results.append(result)
            noise_diagnostics.append(noise_floor_distribution(parsed_a + parsed_b).mean())
            dist_a = answer_distribution(parsed_a)
            dist_b = answer_distribution(parsed_b)
            print(f"  [{level}] {a.problem_id}: PDI={result['observed_pdi']:.3f} "
                  f"null={result['null_mean']:.3f} p={result['p_value']:.3f} "
                  f"(n_a={result['n_a']} n_b={result['n_b']})")
            print(f"      side A answers: {dict(sorted(dist_a.items(), key=lambda x: -x[1]))}")
            print(f"      side B answers: {dict(sorted(dist_b.items(), key=lambda x: -x[1]))}")
            budget -= 1
        except Exception as e:
            branch_errors += 1
            print(f"  [{level}] branching ERROR: {type(e).__name__}: {e}")
        if budget <= 0:
            break
        torch.cuda.empty_cache()

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------
print("\n\n" + "#" * 78)
print(f"FULL PILOT REPORT -- DIFFICULTY LEVEL {DIFFICULTY_LEVEL}, PROBLEMS {SUB_START}-{SUB_END-1}")
print("#" * 78)

print(f"\n1. MATCHED PAIRS FOUND:")
for level in ("L1", "L2", "L3"):
    print(f"   {level}: {len(pairs_by_level[level])}")

print(f"\n2. PIPELINE HEALTH:")
print(f"   trajectory/scoring errors: {len(errors)}")
for e in errors[:5]:
    print(f"     - {e}")
print(f"   branching errors: {branch_errors}")
print(f"   pairs successfully branched + tested: {len(pair_results)}")

print(f"\n3. PDI vs NOISE FLOOR:")
if pair_results:
    df = pd.DataFrame(pair_results)
    print(df[["level", "problem_id", "observed_pdi", "null_mean", "effect_size", "p_value"]].to_string(index=False))
    if len(pair_results) >= 2:
        stat, combined_p = combine_pvalues([r["p_value"] for r in pair_results], method="fisher")
        print(f"\n   Combined (Fisher's method, {len(pair_results)} independent pairs, THIS SUB-NOTEBOOK ONLY): p={combined_p:.4f}")
        print(f"   Mean effect size (observed - null): {df['effect_size'].mean():.3f}")
    print(f"   Mean per-pair diagnostic noise floor: {np.mean(noise_diagnostics):.3f}")
else:
    print("   No pairs were successfully branched -- see pipeline health above.")

print(f"\n4. DATA QUALITY:")
print(f"   phi(p_t) unparseable rate: {n_prefix_unparseable/max(n_prefix_calls,1):.0%}")
print(f"   phi(p_t) rejected (last_number fallback, not genuine boxed): "
      f"{n_prefix_rejected_fallback/max(n_prefix_calls,1):.0%}")

with open(f"{OUTPUT_DIR}/pair_results.json", "w") as f:
    json.dump(pair_results, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR}/pair_results.json")
print(f"\nThis is level {DIFFICULTY_LEVEL}, problems {SUB_START}-{SUB_END-1} of 6 "
      f"(CANDIDATE_B_PREREGISTRATION.md). Combine with the other 8 sub-notebooks'")
print("pair_results.json for the overall Fisher's-method verdict across all 24 pairs -- see SS5.")
