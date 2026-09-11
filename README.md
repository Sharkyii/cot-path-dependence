# Does a reasoning model's history matter, or just its current answer?

When a large language model works through a math problem step by step, a lot of people are building tools that try to stop it early, the moment it looks like it has already decided the answer. This saves time and money. But all of those tools quietly assume something: that the model's *current* answer, how confident it sounds, and how "sure" its next word is, tells you everything you need to know. The path the model took to get there, they assume, doesn't matter anymore.

Nobody had actually tested that assumption. This project does.

## The short version

We took a reasoning model (DeepSeek-R1-Distill-Qwen-7B), gave it math problems, and found many pairs of moments where two *different* reasoning attempts had landed on the exact same answer, with the exact same confidence, at the exact same point in their reasoning. Same answer, same confidence, different history.

Then we let each of those moments continue naturally, many times over, and checked: do they keep agreeing, or does their shared history start pulling them apart?

**Across 66 tested pairs, they kept agreeing.** The model's current state really does seem to be "enough." History barely adds anything on top of it.

That is a useful thing to know if you are building or using an early-stopping tool: you are probably safe to trust the current answer without needing to remember how the model got there.

## A picture is worth it

<p align="center">
  <img src="paper/figures/fig4_continuous_regression.png" width="55%">
</p>

Each dot is one tested pair. The x-axis is how differently two moments "felt" (confidence and uncertainty combined). The y-axis is how much their eventual answers actually diverged. If history mattered, you'd expect the dots to trend upward as you move right. They don't. The line is flat, and the shaded band (our uncertainty about that line) comfortably includes "no relationship at all."

<p align="center">
  <img src="paper/figures/fig5_sges_comparison.png" width="80%">
</p>

We also tried building a smarter stopping rule that pays attention to confidence, not just the answer. It got the same accuracy as a much dumber rule that just stops halfway through, and used *more* compute doing it. Another point in favor of "the simple thing is already enough."

## What's actually in this repository

```
early_stop/     the core library: generation, scoring, and the statistical tests
scripts/        the scripts that actually ran the experiments on Kaggle and Modal
tests/          the test suite (pytest, no GPU required)
results/        the raw, real output from every experiment, as JSON
paper/          the paper itself: LaTeX source, and every figure as a plain PNG
docs/           the pre-registration document (what we planned to do, and every
                honest amendment we made once real data showed up)
```

Nothing here is a toy example. Every number in the paper traces back to a file in `results/`.

## Reproducing this

```bash
pip install -r requirements.txt
pytest tests/          # runs entirely on CPU, no GPU or API keys needed
```

The GPU-heavy generation scripts in `scripts/` are written for Kaggle's free T4 GPUs or Modal's paid A10 GPUs (see the comments at the top of each script for exact usage and cost). They need `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and the `HuggingFaceH4/MATH-500` dataset, both public and free to download.

## Being honest about the limits

This result is scoped to one model and one kind of math problem. It is evidence that this specific assumption holds here, not a universal law about all reasoning models everywhere. The paper spends real space on this (see `docs/PREREGISTRATION.md` for the full, dated history of every course correction along the way, including two results we initially thought were real and later walked back after digging into the data). We would rather show that process than hide it.

## Paper

The full paper is in `paper/latex/` (compile `main.tex` with the NeurIPS style file included, or open it in Overleaf). It covers the statistics in more depth than this README, along with a discussion of *why* this sufficiency might hold and where it's likely to break.

arXiv link: coming soon.

## Citation

```
Sneh Kansagara. "On the Sufficiency of Observable State for Chain-of-Thought Early Stopping." 2026.
```

## License

MIT for the code. See `LICENSE`.
