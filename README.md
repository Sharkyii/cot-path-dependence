# Does checking the current answer alone tell you enough to stop early?

When a large language model works through a math problem step by step, a lot of people are building tools that try to stop it early, the moment it looks like it has already decided the answer. This saves time and money. Most of these tools check one thing: has the current answer stopped changing? A few also check confidence. We tested whether checking the answer alone is actually enough, and along the way we proposed two specific explanations for the cases where it isn't, then tested both explanations directly. Neither one survived the test cleanly. We think that's the most useful part of this project, and we're leading with it rather than burying it.

## The short version

We took a reasoning model (DeepSeek-R1-Distill-Qwen-7B), gave it math problems, and found many pairs of moments where two *different* reasoning attempts had landed on the exact same answer at the exact same point in their reasoning, matched on the answer alone. Same answer, different history, sometimes different confidence.

Then we let each of those moments continue naturally, many times over, and checked what happened next.

**Across 66 tested pairs and 17 distinct problems, checking the answer alone was enough almost everywhere.** 14 of 17 problems agreed every single time, no matter which history got them there. A handful of pairs also happened to match on confidence too, not by design, just because both sides had already fully committed, and every one of those showed perfect agreement as well. That part of the finding is solid and we stand behind it fully.

Divergence showed up in exactly 3 problems. We noticed two candidate patterns in that divergence and, being suspicious of our own first read, we went and ran direct follow-up tests on both before writing this up as a finished result. **Both follow-ups complicated the story instead of confirming it.**

- One pattern looked like the model simply not committing to an answer at all in certain histories. Our follow-up ruled out one boring explanation (running out of token budget) under the setting we tested, but we made a real mistake, our own test accidentally ran under a different generation setting than the one that actually produced the effect, so it doesn't settle the question either way. Open, not resolved. Full story below.
- The other pattern looked cleaner: in one problem, the side of a matched pair with lower confidence tended to give a wrong answer later, while the higher-confidence side didn't. We designed a direct follow-up test on fresh data from the same problem to check whether confidence really explained it. It didn't hold up: divergence still happened even when confidence *was* matched, and when it happened without confidence matching, it went in the wrong direction as often as the right one. The underlying divergence is real and keeps showing up, but our explanation for it doesn't survive scrutiny.

So the honest finding is narrower than either of the two more exciting stories we tried to tell along the way: **the answer alone is enough almost everywhere, real exceptions exist in a small number of problems, and we do not yet know what explains those exceptions.** We'd rather report that than the more flattering, less true version.

## A picture is worth it

<p align="center">
  <img src="paper/figures/fig6_per_problem_pdi.png" width="75%">
</p>

Every bar is one of the 17 problems we tested. Fourteen of them (grey) never showed any disagreement at all, no matter how many times or how many ways we sampled them. Three (orange) are where all the signal lives. As the text above explains, follow-up testing complicated the story behind all three of them, not just two.

<p align="center">
  <img src="paper/figures/fig4_continuous_regression.png" width="55%">
</p>

Each dot is one tested pair. The x-axis is how differently two moments "felt" (confidence and uncertainty combined). The y-axis is how much their eventual answers actually diverged, corrected for a statistical bias in the raw measurement (see below). The diamond points sit at true zero distance, meaning both sides matched on confidence too, not by design, just because both had already fully committed. Every single one of those diamonds shows zero divergence. That is the most direct evidence in this whole project: when the observable state is genuinely, fully matched, the future is not up for grabs.

<p align="center">
  <img src="paper/figures/fig7_d3prob5_confidence_error.png" width="80%">
</p>

This is the original pattern in the one problem where confidence looked like the explanation, before the follow-up test complicated it (see below). In 3 of these 4 original pairs, the lower-confidence side (left dot) is the one that later gave a wrong answer (right dot). We're keeping this figure because it's honest history, this is what we saw first and what motivated the follow-up test, not because we still think it's the full story.

## What actually happened, told straight

Early on, we combined all 66 pairs' individual test results using a standard statistical tool (Fisher's method) and got a strong "no effect" verdict. That tool turned out to be the wrong one for this data: 56 of the 66 pairs were cases where both sides agreed 100% of the time, which makes that pair's individual test mathematically return "no signal detected" no matter what. Averaging those in with the pairs that did show disagreement quietly buried the real signal. Once we fixed that, a real effect showed up, concentrated in exactly 3 problems out of 17.

We then designed and ran direct follow-up tests on the two mechanisms we proposed for that effect, rather than stop at the first read. Both follow-ups are described in full in `docs/PREREGISTRATION.md`, dated the night they ran. Short version: for two of the three problems, we couldn't rule out that the model simply needed more tokens than we gave it, and our own follow-up test (meant to check this) ran under the wrong generation settings by mistake, so it's still an open question. For the third problem, where confidence looked like a clean explanation, a follow-up on fresh trajectories from the same problem showed divergence still happening even when confidence *was* matched, and going the "wrong" direction just as often as the "right" one when it wasn't. The divergence itself is real and repeatable. Our explanation for it isn't.

## What's actually in this repository

```
early_stop/     the core library: generation, scoring, and the statistical tests
scripts/        the scripts that ran the experiments (Kaggle, Modal) and the
                analysis that turns raw results into every number in the paper
tests/          the test suite (pytest, no GPU required)
results/        the raw, real output from every experiment, as JSON
paper/          the paper itself: LaTeX source, and every figure as a plain PNG
docs/           the pre-registration document (what we planned to do, and every
                honest amendment we made once real data showed up)
```

Nothing here is a toy example. Every number in the paper traces back to a file in `results/`, and `scripts/final_analysis.py` reproduces all of them in one run.

## Reproducing this

```bash
pip install -r requirements.txt
pytest tests/                        # runs entirely on CPU, no GPU or API keys needed
python scripts/audit_smoke_test.py   # same, a free end-to-end check of the audit pipeline
python scripts/final_analysis.py     # reproduces every statistic in the paper from results/
```

The GPU-heavy generation scripts in `scripts/` are written for Kaggle's free T4 GPUs or Modal's paid A10 GPUs (see the comments at the top of each script for exact usage and cost). They need `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and the `HuggingFaceH4/MATH-500` dataset, both public and free to download.

## Being honest about the limits

This result is scoped to one model and one kind of math problem. It is evidence that this specific finding holds here, not a universal law about all reasoning models everywhere. The paper spends real space on this, including a whole section on why this model being *distilled* (trained to imitate another model's outputs, not trained directly) might be exactly why most checkpoints turned out sufficient in the first place, and what that predicts about non-distilled models. See `docs/PREREGISTRATION.md` for the full, dated history of every course correction along the way, including results we initially thought were solid and later walked back or flagged as open once we dug into the data ourselves. We would rather show that process than hide it.

## Paper

The full paper is in `paper/latex/` (compile `main.tex` with the NeurIPS style file included, or open it in Overleaf). It covers the statistics in more depth than this README, states plainly which findings are load-bearing and which are open questions, and includes a section on why this sufficiency might hold and where it's likely to break.

arXiv link: coming soon.

## Citation

```
Sneh Kansagara. "On the Sufficiency of Observable State for Chain-of-Thought Early Stopping." 2026.
```

## License

MIT for the code. See `LICENSE`.
