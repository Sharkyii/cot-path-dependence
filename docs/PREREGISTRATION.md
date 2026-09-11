# Candidate B (Path Dependence) — Full-Scale Pilot Pre-Registration

**Status: written BEFORE the full-scale pilot is run. Do not edit after
seeing results.** Amendments go in a dated "Amendments" section at the
bottom, with reasons — same discipline as PILOT_PREREGISTRATION.md.

This implements DESIGN_CANDIDATE_B.md's Step 4. Every parameter below is
either inherited unchanged from the micro-pilot (Steps 1-3, all validated
on real hardware) or was determined by an explicit, already-completed
analysis (the power simulation, §4.9) rather than being picked freely
after seeing results — that is the whole point of writing it now.

---

## 0. The hypothesis being tested

**H1**: two prefixes from different reasoning trajectories on the same
problem, matched to the same observable state φ(p) = (answer, local token
confidence, entropy) at the same normalized depth, can still diverge in
their future-answer distribution by more than pure sampling noise
predicts.

**H0** (null): matched prefixes' future-answer distributions differ only
by sampling noise — i.e., φ(p) is a sufficient statistic for the future,
and every training-free early-stopping method built only on current-state
features is not leaving predictive information on the table by ignoring
history.

Unit of analysis: one matched pair (two prefixes, N branches generated
from each). Test statistic: Path Dependence Index (PDI, Jensen-Shannon
divergence between the two branch sets' answer distributions), compared
against a per-pair permutation null built from that pair's own pooled
data (`pair_permutation_test`). K independent pair-level p-values combined
via Fisher's method (`combine_pair_results`).

---

## 1. Parameters carried over unchanged from the micro-pilot

These were validated end-to-end on the real model (DeepSeek-R1-Distill-
Qwen-7B, 2×T4 fp16) across all 8 micro-pilot problems with 0% totally
unparseable and ~0.4% rejected extractions — no reason to change them.

| Parameter | Value |
|---|---|
| temperature | 0.6 |
| top_p | 0.95 |
| max_new_tokens (trace) | 2000 |
| max_new_tokens (forced extraction / φ scoring) | 48 |
| max_new_tokens (branch continuation) | 1500 |
| forced-extraction suffix | `"\n\nTherefore, the final answer is \boxed{"` |
| `confidence_tol` (L2) | 0.10 |
| `entropy_tol` (L3) | 0.15 |
| `position_tol` (depth-confound guard) | 0.15 |
| `min_parse_rate` gate (per branch set) | 0.80 |
| dataset | MATH-500, levels 1-3 |

---

## 2. Parameters changed from the micro-pilot defaults (this is the point of Step 4)

| Parameter | Micro-pilot value | Full-pilot value | Why |
|---|---|---|---|
| `N_BRANCHES_PER_PAIR` | 15 | **40** | §4.9 power analysis (reconfirmed at full precision, using the actual project statistical code): at 15 branches/pair, power to detect a plausible small effect (delta=0.15 in future-answer probability) is 3-9% at ANY pair count from 13-150. At 40, the earlier fast-approximation stage estimated ~89% power at n_pairs=21. This is the single most important parameter in this whole document — it is what makes the pilot capable of detecting a real-but-modest effect at all. |
| Target matched pairs to branch | 20-21 (informal) | **24** (pre-registered) | Not 150. §4.9: pair count barely moves power once branches/pair is fixed at 40 (7% -> 9% -> 3% power at delta=0.15 across n_pairs=21/50/150 — noise in a coarse simulation, not a real trend). 24 = 8 pairs × 3 problem-difficulty strata (see §3), keeps GPU cost proportional to what actually buys power. |
| Pair sampling | confidence-stratified only | **confidence- AND problem_id-stratified** | §4.8: fixed bug where one problem's abundance of matched pairs could still starve other problems within a confidence bucket even with confidence stratification working correctly (observed directly in micro-pilot notebook 1/3: 8/8 branched pairs came from only 2 of 3 problems). Implemented in `stratified_sample_pairs()`, regression-tested. |
| Problem pool size | 8 | **18** | Sized for pair-supply headroom (need ~24 pairs found and spread across problems/strata, not just 24 to exist at all) and for H4 difficulty coverage (§3) — not for raw branch volume, since branches/pair (not problem count) is the power lever. |

## 3. Difficulty stratification (H4, carried from the original proposal, now made concrete)

18 problems, drawn evenly across MATH-500 levels 1-3 (6 each), so that if
path dependence turns out to correlate with problem difficulty — plausible,
since harder problems have longer, more divergent reasoning paths — the
pilot can see that rather than average it away. The 24-pair branching
budget is likewise split 8/8/8 across the three difficulty strata via
`stratified_sample_pairs()` calls run independently per stratum (not one
pooled call), so a difficulty stratum with fewer matched pairs can't be
crowded out by another the way individual problems could within a
confidence bucket.

## 4. Matching levels tested

All three (L1: answer only, L2: +confidence, L3: +confidence+entropy) are
run and reported, exactly as in the micro-pilot. No level is dropped —
each answers a different, still-open question: L1 is the most permissive
match (closest to what a pure answer-stability early-stopping method
sees), L3 the strictest (closest to what a method using confidence+entropy
sees). A finding that only appears at L1 and vanishes by L3 is itself
informative (state richness closing the path-dependence gap), not a
reason to have skipped L1.

## 5. Decision rule (fixed before seeing full-pilot data)

Applied independently **per difficulty stratum** (6 problems, 8 pairs each,
40 branches/pair), then combined across strata via Fisher's method for an
overall verdict — this two-level structure is decided now specifically so
that a stratum-level result cannot be cherry-picked after the fact.

- **Reject H0 (path dependence detected)**: combined Fisher's p < 0.05 at
  any matching level (L1, L2, or L3), with the specific level and stratum
  reported. Multiple-comparisons note: 3 levels × 1 combined test each =
  3 tests; report the raw p-values and flag if only the least-strict
  (L1) reaches significance while L2/L3 do not — that pattern would mean
  richer state resolves most of the gap, which is a real, reportable
  finding on its own, not a failure to replicate.
- **Fail to reject (no detected path dependence at this scale)**: all
  combined p-values ≥ 0.05. Given §4.9's confirmed power curve (~89%
  power at delta=0.15, n_pairs=21, 40 branches — this design's regime),
  a null result here is meaningfully more informative than the
  micro-pilot's null (which had ~3-9% power and was uninformative by
  design). Report as "no evidence of path dependence beyond sampling
  noise at N_BRANCHES_PER_PAIR=40," not as proof of state sufficiency —
  absence of evidence at 89% power for THIS effect size does not rule out
  a smaller one.
- **Effect size reporting, regardless of significance**: mean(observed
  PDI − null mean) across all branched pairs, per level and per
  stratum, reported alongside p-values. This is what actually goes in
  the paper's headline number — the p-value gates whether the framing is
  "detected" or "not detected," the effect size is what readers compare
  against.

## 6. Cost accounting

24 pairs × 40 branches/side × 2 sides = 1920 branch generations, plus
Phase 1 (18 problems × 5 trajectories × ~2000 max tokens) and Phase 2
(φ scoring at every step, ~48 tokens each — cheap). Per §4.9's cost
table, this is the "power-corrected" plan: ~26 GPU-hours for the
branching phase at this project's observed generation rate, vs. ~69
GPU-hours for the originally-planned 150×15 design that §4.9 showed
would have had only 3-9% power to detect the effect size of interest.

## 7. What would make this pilot itself inconclusive (pre-committed, not decided after the fact)

- Any stratum where `min_parse_rate` (0.80) causes more than 25% of
  attempted pairs to be dropped from the analysis — report separately as
  a data-quality flag, do not silently fold into the effect-size average.
- If problem-level or confidence-level stratification still cannot fill
  its allotted share in a given stratum (e.g., a difficulty level simply
  doesn't produce enough matched pairs even at 18 problems) — report the
  actual achieved N per stratum rather than treating the 8-pairs-per-
  stratum target as guaranteed.

---

## Amendments

**2026-09-10 — scope narrowed to L1 matching only; N=21 not 24.**

Root cause: Phase 3's per-chunk branching loop (in `modal_candidate_b_full_pilot.py`
and all 9 `kaggle_candidate_b_fullpilot_level*_*of3.py` scripts) used a single
`budget` counter shared across all three matching levels:

```python
budget = max_pairs
for match_level in ("L1", "L2", "L3"):
    if budget <= 0:
        break
    ...
    budget -= 1
```

Once L1 sampling filled `budget` (e.g. 3 pairs), the loop broke before ever
sampling L2 or L3 — despite Phase 2 finding thousands of matched pairs at
both stricter levels in every chunk. This ran identically across all 9
Kaggle sub-notebooks and both Modal slices, so **all 21 collected pairs are
L1 (answer-only matching); zero pairs exist at L2 (+confidence) or L3
(+entropy)**. Discovered only after full data collection completed, when
computing the combined Fisher's-method verdict per §5.

Decision (made with the user after weighing a ~$5-25 / <1hr-few-hr
supplemental Modal run to backfill L2/L3): **stop here, do not collect
L2/L3 data.** Per §7's actual-achieved-N principle, this pilot's results
are reported as an **L1-matching-only result**, not the originally-planned
three-way L1/L2/L3 comparison. The question of whether richer phi(p_t)
state (confidence, entropy) resolves path dependence beyond answer-only
matching is left untested by this pilot and should be flagged as future
work, not answered by omission.

Achieved N: 21 pairs (7 per difficulty stratum), vs. the pre-registered
target of 24 (8 per stratum) — 3 short because `level3_4-6`'s original
attempt (before a separate token-budget fix, see below) yielded 0 matched
pairs and was not re-expanded back to 3 pairs after the rerun succeeded
with only 1.

**Also amended: `MAX_NEW_TOKENS_TRACE` 2000→3000, `MAX_NEW_TOKENS_SCORE`
48→64** (in `modal_candidate_b_full_pilot.py` only, mid-collection) after
`level3_4-6`'s first attempt produced 0/500 parseable forced-extractions —
diagnostic traces showed the model still mid-reasoning ("Wait, perhaps
2220" cut off, no `\boxed{}`) at the 2000-token ceiling for that chunk's
two problems. The bump fixed convergence (47 scored prefixes on rerun,
1 pair successfully branched) but was not retroactively applied to the
Kaggle track or Modal's earlier successful chunks, which completed fine
under the original budget.

**Combined result across the 21 L1 pairs**: Fisher's method p=0.513 (not
significant) — no evidence of path dependence at this scale, using the
§5 fail-to-reject framing. 3/21 pairs individually significant at p<0.05
(`d2_prob1` p=0.0005 PDI=0.373; `d1_prob3` p=0.0025 PDI=0.123; `d1_prob1`
p=0.021 PDI=0.061 — note `d1_prob1` and `d2_prob1` each had a SECOND
sampled pair at the same problem/level that did NOT replicate: `d1_prob1`'s
other pair was PDI=0.0023, p=0.807; `d2_prob1`'s other pair was
PDI=0.0655, p=0.058, just above the 0.05 threshold); 16/21 pairs show
exactly PDI=0.000, p=1.000 (branches
converged identically regardless of path). Mean effect size (observed −
null) across all 21 pairs: 0.026. Per §5, report as "no evidence of path
dependence beyond sampling noise at N_BRANCHES_PER_PAIR=40 under L1
(answer-only) matching," with the 3 significant pairs noted as
problem-specific exceptions worth mechanistic follow-up, not as
contradicting the null verdict.

**2026-09-10 — post-hoc targeted L2/L3 follow-up on the two real anomalies
(exact mechanism check, not part of the pre-registered decision rule).**

Of the 3 nominally-significant L1 pairs above, inspection of the raw
branch answers showed `d1_prob3` was a normalizer artifact (`'2000'` vs
`'2000calories'` — same numeric answer, units word leaking into the box)
rather than real divergence, so it was excluded from follow-up. The other
two (`d2_prob1`, `d1_prob1`) were genuine: not disagreement about the
answer, but about whether the model closes a decisive `\boxed{}` at all
vs. trailing into an empty box (`d2_prob1`: 72.5% clean "evelyn" vs. 85%
empty on the two sides; `d1_prob1`: 75% clean "-50" vs. 25% empty on one
side).

Rather than spend budget re-running the full 9-chunk L2/L3 sweep, ran a
minimal targeted test: for exactly these two problems, fixed the Phase 3
budget bug (see amendment above — each matching level now gets its own
independent `max_pairs`, via a new `match_levels` param on `run_chunk`),
regenerated Phase 1 for just that one problem each (trajectory text
wasn't preserved from the original Kaggle runs), and branched 1 pair at
L2 and 1 pair at L3 for each (skipping L1, already covered above). Cost:
~$3 total, ~45 min wall clock, both problems run in parallel.

**Result: the decisiveness gap fully closes under both L2 and L3
matching.**

| problem | level | side A | side B | PDI | p |
|---|---|---|---|---|---|
| d1_prob1 | L1 (original) | 75% "-50" | 95% "-50" | 0.061 | 0.021 |
| d1_prob1 | L2 (confidence-matched) | 40/40 "-50" | 40/40 "-50" | 0.000 | 1.000 |
| d1_prob1 | L3 (+entropy-matched) | 40/40 "-50" | 40/40 "-50" | 0.000 | 1.000 |
| d2_prob1 | L1 (original) | 72.5% "evelyn" | 15% "evelyn" | 0.373 | 0.0005 |
| d2_prob1 | L2 (confidence-matched) | 40/40 "evelyn" | 40/40 "evelyn" | 0.000 | 1.000 |
| d2_prob1 | L3 (+entropy-matched) | 40/40 "evelyn" | 40/40 "evelyn" | 0.000 | 1.000 |

This is the paper's strongest result: **answer-only (L1) early-stopping
has a real, measurable decisiveness blind spot, and it is fully patched
by matching on confidence alone (L2) — entropy (L3) adds nothing further
in either case tested.** Framed correctly, this upgrades the paper from
"here is an assumption check with two unexplained exceptions" to "here is
exactly which state variable resolves the exception, and it's the cheap
one." Caveat for the write-up: this is a targeted, post-hoc, n=2
mechanism check (not a pre-registered, independently-powered test of L2
vs. L3 across the full design) — report it as exactly that, alongside the
pre-registered L1-only headline result above, not as a replacement for
the untested general L2/L3 comparison across all 21 pairs.

**2026-09-11 — self-critique of the above, applied before write-up (no
new data collected; this corrects the framing, not the numbers).**

Re-reviewing §7's framing with a skeptical read surfaced three issues
that materially weaken the "strongest result" claim above as originally
stated. Recorded here rather than silently tightened in the prose, per
this section's own stated purpose.

1. **Multiple-comparisons correction was never applied to the 3
   nominally-significant L1 pairs.** With 21 independent tests at
   α=0.05, ~1 false positive is expected under the true global null
   (21 × 0.05 ≈ 1.05) — so 3 nominal hits is barely above chance on its
   own. Applying Bonferroni (α/21 ≈ 0.00238): `d2_prob1` (p=0.0005)
   survives; `d1_prob3` (p=0.0025, and separately already known to be a
   normalizer artifact) does not; **`d1_prob1` (p=0.021) does not
   survive and should not have been reported as a confirmed second
   anomaly** — it sits squarely in the range chance alone would produce
   across 21 tests. Corrected count: **1 statistically defensible L1
   anomaly (`d2_prob1`), not 2.**

2. **The n=2 L2/L3 "fix" has a real regression-to-the-mean /
   base-rate problem — and there is DIRECT evidence of this already in
   the L1 data itself, not just a hypothetical.** 16/21 (~76%) of the
   original L1 pairs showed PDI=0.000 exactly. More tellingly: both
   `d1_prob1` and `d2_prob1` were each independently sampled *twice* at
   L1 (two different matched pairs, same problem, same matching level,
   different trajectory draws) — and in both cases, **the second draw
   did not replicate the first**: `d1_prob1`'s second pair was PDI=0.0023,
   p=0.807 (nowhere near the first pair's p=0.021); `d2_prob1`'s second
   pair was PDI=0.0655, p=0.058 (not significant, vs. the first pair's
   p=0.0005). So even holding the matching level fixed at L1, redrawing
   the same problem does not reliably reproduce the "anomaly." Against
   that backdrop, getting null on 2 fresh L2/L3 draws is close to what
   you'd expect from this problem's own noise floor, independent of
   whether confidence-matching does anything causally at all.

3. **A more powerful, already-built test was never used, and the data
   to run it isn't saved.** `early_stop/path_dependence.py`'s own
   `state_distance()` docstring states that regressing PDI against
   *continuous* confidence/entropy distance across all matched pairs is
   "the stronger and correct way" to test whether richer state resolves
   path dependence — discrete L1/L2/L3 bucket comparisons are
   acknowledged in the library's own documentation to be the weaker
   method. `state_distance()` was never called in any full-pilot script
   (Kaggle or Modal), and `pair_results.json` never saved the
   prefix-level `ObservableState` (confidence, entropy, position) for
   either side of a pair — only branch outcomes. This means the
   continuous analysis cannot even be run retroactively on the existing
   21 pairs.

**Decision**: do not spend further budget patching this with more
discrete-bucket replication (which would also risk pseudoreplication —
`K_TRAJECTORIES_PER_PROBLEM=5` means multiple sampled pairs per problem
likely reuse the same underlying trajectories, the same bug class an
external review already caught once in this project's Step 2). Instead:

- Report the L2/L3 follow-up in the paper as a **suggestive, hypothesis-
  generating pilot observation** (n=2, post-hoc, not independently
  powered), not as a confirmed mechanism — pair it explicitly with the
  base-rate caveat in point 2 above.
- Report `d1_prob1`'s original L1 result as **not surviving
  multiple-comparisons correction**; `d2_prob1` remains the one
  genuinely defensible L1 anomaly.
- List the missing `ObservableState` logging and the never-invoked
  continuous `state_distance`-vs-PDI regression as **concrete, scoped
  future work** in the paper's limitations/future-work section — this is
  the principled next experiment, not a quick patch, and should be
  scoped as its own follow-up study rather than retrofitted under
  budget pressure.
- The pre-registered, properly-powered **21-pair L1 null result (Fisher
  p=0.513) remains the paper's actual headline finding** and is
  unaffected by any of the above — this self-critique only concerns how
  the secondary anomaly-and-follow-up story should be framed, not the
  main result's validity.

**2026-09-11 (later) — low-compute follow-on plan executed: 45-pair
scale-up, continuous regression finally run, hidden-state probe attempted,
SG-ES built and evaluated. Budget: ~$11 total across two Modal accounts
(smoke tests + real runs), against a $15-20 ceiling.**

Per the future-work list in the amendment above, instrumented
`modal_candidate_b_full_pilot.py` to persist per-pair `ObservableState`
for both sides (`side_a_state`, `side_b_state`, `state_distance`) and,
separately, a hidden-state activation vector at the matched checkpoint
(`HFBackend.extract_hidden_state()`, one extra forward pass per side, new
`--capture-hidden-states` flag) — the state-persistence gap identified in
point 3 above. Smoke-tested both additions on a throwaway 1-pair run
before committing budget; caught nothing broken, but did observe a
transient CUDA OOM warning during hidden-state extraction (non-fatal,
torch's allocator recovered both times it occurred across ~11 real runs)
worth monitoring at larger scale in future work.

**Execution**: 9 parallel Modal chunks (`level{1,2,3}_{0-2,2-4,4-6}`),
5 pairs/chunk, `n_branches=10` (not the pre-registered 40 — a deliberate
budget tradeoff, see below), L1 matching, hidden-state capture on. All 9
completed with exit code 0; 3 chunks hit a transient Modal
"App create rate limit exceeded" error on a fresh account during
same-second parallel launch (before any GPU work started, so no budget
lost) and were relaunched successfully with brief staggering.

**Why N=10 branches, not 40, for this round**: the pre-registered N=40 was
set by the original power analysis (§4.9) to detect a 15-point
future-answer-probability shift at ~89% power for a *fixed, small* number
of pairs. This round's goal was different — breadth (many pairs, for the
continuous regression's pooled-data test and for a proper scale-up check)
under a hard $15-20 budget ceiling, not per-pair power on a handful of
pairs. This is a real, acknowledged tradeoff: individual pairs in this
45-pair set are lower-power than the primary sample, and should not be
read as equally strong evidence pair-for-pair. Aggregate patterns
(fraction exactly-zero, combined Fisher p, and especially the continuous
regression, which uses graded distance rather than a significance
threshold per pair) are the right way to use this data, not scanning it
for new individually-significant pairs.

**Results** (full numbers in PAPER_DRAFT.md §4.5-§4.6; summarized here for
the record):

- **45/45 new pairs: 0 nominally significant** (Fisher-combined p≈1.0,
  40/45 exact-zero PDI). Per-difficulty: level 1 15/15 exact-zero, level 2
  14/15, level 3 11/15 (mean PDI 0.035 — more variance on harder problems,
  as hypothesized in §3.6/§4.9, but not remotely significant).
- **Pooled 21+45=66 pairs**: same 3 nominal hits as before (no new ones),
  Fisher-combined p=0.9999996. `d2_prob1` (p=0.0005) survives Bonferroni
  at the pooled n (threshold 0.05/66≈0.00076) by a thin margin; given
  points 1-2 of the previous amendment plus this round's null scale-up, we
  do not read this survival as evidence of a real effect — recorded
  honestly rather than omitted because it's inconvenient for the framing.
- **Continuous regression (the "stronger and correct" test per point 3
  above, run for the first time)**: `regress_pdi_vs_state_distance()` on
  the 45 pairs with saved state → slope=−0.0086, intercept=0.0214,
  r=−0.1096, p=0.4738, std_err=0.0118. Null, and directionally opposite a
  path-dependence hypothesis (slightly *less* divergence at *more* distant
  states, though far too small/noisy to read as a real negative
  relationship either). This is the single piece of evidence we weight
  most heavily in the paper's write-up, since it's exactly the test the
  project's own prior self-critique called for.
- **Hidden-state decisiveness probe**: could not be trained.
  `train_decisiveness_probe.py` on all 90 examples (45 pairs × 2 sides)
  found zero "trails off empty" labels — every branch set at N=10 produced
  a non-empty answer, so the label has no variance. Not a bug (verified:
  grepped all 45 pairs' `side_a/b_answer_counts` directly, confirmed no
  empty-string keys anywhere in this round's data). Reads as mildly
  consistent with the null: the decisiveness gap large enough to detect at
  N=40 (original `d2_prob1`/`d1_prob1`) may not recur reliably at N=10,
  consistent with treating it as sparse-sample noise rather than a robust
  phenomenon.
- **SG-ES (Sufficiency-Gated Early Stopping) vs. baselines**: built
  `early_stop/sg_es.py` (pure post-hoc logic, 9 unit tests) and
  `scripts/modal_candidate_b_phase4_sges.py` (live generation + grading).
  Real run: 24 MATH-500 problems sampled (levels 1-4), 19 usable (5
  dropped to 0 scored prefixes, same forced-extraction failure mode as
  elsewhere in this project). All three conditions (full generation,
  forced-extraction@50%, SG-ES) tied on accuracy (0.842). SG-ES used
  *more* of the trace on average than the naive 50%-cutoff baseline
  (81.3% vs. 68.7%) despite triggering on 18/19 problems — i.e. the
  confidence+stability gate did not pay for itself against the simplest
  possible baseline, on either metric.

**Reframing decision** (made with the user, recorded here rather than
silently changed in the prose): given the convergence above — pooled
null, continuous-regression null, degenerate/consistent-with-null probe
result, and SG-ES failing to beat a trivial baseline — the paper's framing
shifted from "assumption check with a partially-explained exception" to
leading with the null as the finding: current observable state behaves as
a sufficient statistic within this study's scope, which is itself a
useful result for a field that has been assuming this without testing it.
Language claiming this as *proof* or a general *validation of the
Markovian assumption* was deliberately avoided in the paper text — a
well-powered null across one model, one domain, and predominantly one
matching level is evidence for sufficiency, not proof of it, and the
paper says so explicitly (see PAPER_DRAFT.md §5.5).
