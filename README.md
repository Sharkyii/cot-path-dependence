# Does checking the current answer alone tell you enough to stop early?

When a large language model works through a math problem step by step, a lot of people are building tools that try to stop it early, the moment it looks like it has already decided the answer. This saves time and money. Most of these tools check one thing: has the current answer stopped changing? A few also check confidence. But almost none of them ask whether confidence is actually doing anything once the answer has stabilized, or whether checking the answer alone was already good enough.

We tested that directly.

## The short version

We took a reasoning model (DeepSeek-R1-Distill-Qwen-7B), gave it math problems, and found many pairs of moments where two *different* reasoning attempts had landed on the exact same answer at the exact same point in their reasoning, but with different confidence levels, since we only matched on the answer itself. Same answer, different confidence, different history.

Then we let each of those moments continue naturally, many times over, and checked what happened next.

**Across 66 tested pairs and 17 distinct problems, checking the answer alone was enough almost everywhere.** 14 of 17 problems agreed every single time, no matter which history got them there. But in one problem, a clean and repeatable pattern showed up: whichever side of a matched pair had **lower confidence**, even though both sides currently stated the same answer, was the side that went on to give a wrong answer 10 to 30% of the time. The higher-confidence side almost never did.

That is the actual finding. Not "history secretly matters." Something much more useful for anyone building a stopping rule: **the answer alone can lie about how safe a checkpoint is, and confidence, a number these tools already compute, catches it.**

A second pattern, in two other problems, looked at first like the same kind of thing (history predicting whether the model states an answer at all, or trails off). We're not confident that one is real. It only shows up under one of our two generation settings, and it might just be that those two problems needed more tokens than we gave them per branch. We say so plainly below rather than quietly picking the more exciting result.

## A picture is worth it

<p align="center">
  <img src="paper/figures/fig7_d3prob5_confidence_error.png" width="80%">
</p>

This is the clearest look at the real finding, one problem ("find the smallest value of a" for a cubic equation) where confidence, not history, explains what happens next. In 3 of its 4 tested pairs, whichever side had lower confidence at the matched checkpoint (left dot) is exactly the side that went on to give a wrong answer some of the time (right dot). The higher-confidence side never did. Confidence isn't part of what "answer-only" matching checks, so this is a real blind spot in that kind of rule, not an artifact of how we set up the test.

<p align="center">
  <img src="paper/figures/fig6_per_problem_pdi.png" width="75%">
</p>

Every bar is one of the 17 problems we tested. Fourteen of them (grey) never showed any disagreement at all, no matter how many times or how many ways we sampled them. Three (orange) are where all the signal lives, and as explained above, only one of those three is a result we currently stand behind without reservation.

<p align="center">
  <img src="paper/figures/fig4_continuous_regression.png" width="55%">
</p>

Each dot is one tested pair. The x-axis is how differently two moments "felt" (confidence and uncertainty combined). The y-axis is how much their eventual answers actually diverged, corrected for a statistical bias in the raw measurement (see below). The diamond points sit at true zero distance, meaning both sides matched on confidence too, not by design, just because both had already fully committed. Every single one of those diamonds shows zero divergence. That is the most direct evidence in this whole project: when the observable state is genuinely, fully matched, the future is not up for grabs.

## What actually happened, told straight

Early on, we combined all 66 pairs' individual test results using a standard statistical tool (Fisher's method) and got a strong "no effect" verdict. That tool turned out to be the wrong one for this data: 56 of the 66 pairs were cases where both sides agreed 100% of the time, which makes that pair's individual test mathematically return "no signal detected" no matter what. Averaging those in with the pairs that did show disagreement quietly buried the real signal.

Once we fixed that, a real effect showed up, concentrated in exactly 3 problems out of 17. But fixing that bug also meant looking harder at where the effect actually was, and that's when we found something we want to be upfront about: two of the three anomalous problems have a specific vulnerability. They're both long-output problems (a 100-term sum, a free-text answer), and our branches were capped at 1500 tokens. It's entirely possible some of those "the model trailed off without answering" cases were really "the model needed more tokens and we cut it off." We designed a test for this (generate longer, see if the gap closes) and it's either running or already run by the time you're reading this, see `docs/PREREGISTRATION.md` for the dated result.

The third problem, the confidence one shown above, has no such vulnerability. It has zero empty answers on either side in all four tested pairs, so there's nothing to censor. That's why we lead with it.

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
