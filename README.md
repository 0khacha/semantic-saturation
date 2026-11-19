# Semantic Saturation Benchmark (SSB)

> **Does adding more textual context improve a language model's understanding? Is there a
> point where more context starts to make it less reliable?**

SSB holds the **question and the gold evidence fixed** and changes only what surrounds
them. Then it measures how answer quality *and reliability* change: calibration,
abstention, and behaviour under conflicting claims.

**Status: v0.1.** The dataset, runner, metrics and report are complete and tested. The
results section below is written by `ssb report` from the actual run.

## Method

**Context size.** N ∈ {1, 5, 20, 100} paragraphs, each 108–132 tokens (Qwen2.5
tokenizer), up to about 11.7k prompt tokens. A context never holds less than its own
evidence: a multi-hop question at N = 1 sees its two gold paragraphs alone, so every
question appears at every N. The exact token count of every prompt is logged.

**Axes.** One axis varies at a time, relative to a base context of gold plus random
off-topic filler.

| Axis | What changes | Levels |
|---|---|---|
| Dilution | Only N (gold fixed at one paragraph per fact, so the gold ratio is 1/N) | N |
| Semantic noise | Filler becomes *hard negatives*: same entities and relations, most similar to the question by embedding (all-MiniLM-L6-v2), never containing the answer | random vs hard |
| Redundancy | Each gold fact appears k times, as distinct paraphrases | k ∈ {1, 5} |
| Contradiction | One paragraph asserts a different value for the asked fact | before / after the gold |
| Position | Gold placed at relative depth d | d ∈ {0, 0.5, 1} |

**Data (contamination-free).** A seeded fictional world of 200 entities (people, towns,
companies, inventions) with 560 facts. Every queried value is unique, so a string search
proves where an answer can and cannot appear. There are 100 questions: 34 single-hop,
33 multi-hop (two facts in different paragraphs), and 33 unanswerable (the fact exists in
the world but is never written anywhere). Every question also runs closed-book, to catch
answers that come from memory.

**Files.**
- `data/ssb_v0.1.jsonl`: 2,173 samples, one per line. `paragraphs` is the ordered list of
  paragraph ids in the context.
- `data/ssb_v0.1.paragraphs.jsonl`: the text of each distinct paragraph (1,347).
- `data/ssb_v0.1.manifest.json`: counts, token totals, SHA-256 hashes, and the full
  config.

`ssb.build.load_dataset(path)` joins the samples and paragraphs and adds `context` and
`prompt` to each sample.

**Guarantees, checked at build time and in `tests/`:**
- In every answerable sample, the gold paragraph is present and byte-identical to its
  base rendering.
- The answer appears in gold paragraphs only.
- No unanswerable sample contains the hidden value or any paragraph stating the hidden
  fact.
- Paragraph counts and token counts match the condition.
- A contradiction paragraph states the same fact with a different value, on the correct
  side of the gold.
- Depth placement is exact.

**Models.** Qwen2.5-1.5B-Instruct (fp16) and Qwen2.5-7B-Instruct (4-bit NF4 via
bitsandbytes, or AWQ when vLLM is installed). If the measured runtime of the 7B goes over
budget, Qwen2.5-3B-Instruct (fp16) replaces it. Decoding is greedy with at most 32 new
tokens, one seed. The fixed prompt asks for a short answer, or exactly `UNANSWERABLE`
when the context lacks one.

**Metrics.** Each is computed per model, axis and N, with 95% bootstrap CIs over
questions (1,000 resamples).
- Correctness: normalised exact match, token F1, and a lenient "answer mentioned"
  rate.
- Abstention: the hallucination rate on unanswerable questions, and the
  false-abstention rate on answerable ones.
- Contradiction: the share of answers giving the gold, the conflicting claim, or
  something else.
- Calibration: confidence is the mean probability of the answer tokens, with ECE
  (10 bins) and Brier score.
- Closed-book gap.

From the accuracy curves: the saturation point **N\***, the degradation onset **N†**
(bootstrap, Holm-corrected), and the **Semantic Saturation Score** (normalised area under
accuracy vs log N).

Hypotheses are pre-registered in [HYPOTHESES.md](HYPOTHESES.md).

## Reproduce

**Locally (CPU is enough for everything except running the models):**

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.11 or 3.12
pip install -e ".[build,analysis,dev]"
ssb build --config configs/main.yaml                # ~1 min; rebuilds data/ bit-for-bit
pytest                                              # 68 tests, ~1.5 min
```

**On a Kaggle or Colab T4.** In Kaggle, turn on Internet and choose *GPU T4*. Use
*Save & Run All*, so the run continues in the background and `results/` is kept as
output.

```bash
git clone <this repo> semantic-saturation && cd semantic-saturation
pip install -q -e ".[run,analysis]"
ssb run --config configs/smoke.yaml && ssb report --config configs/smoke.yaml   # < 15 min
ssb estimate --config configs/main.yaml           # prompts, tokens, prior GPU-hours
ssb run --config configs/main.yaml --time-limit-hours 11
ssb report --config configs/main.yaml             # metrics, figures, README results
```

`ssb run` can always be resumed. Every finished batch is fsynced to `results/cache/`,
so after a crash or a timeout you rerun the same command and it skips the finished
work. Before the bulk of the work it times one real batch per N and prints the
model's estimated GPU-hours. Then:
- if that estimate is over `run.max_gpu_hours`, it switches the 7B to the 3B
  fallback;
- if there is no fallback, it stops and prints how many questions would fit (it cuts
  questions, never axes).

Outputs:
- `results/<run>/generations/<model>.jsonl` (raw answers and token log-probs);
- `scores.parquet` (per answer) and `metrics.parquet` (every metric with CIs);
- `leaderboard.csv`, `hypotheses.json`, `REPORT.md`, and the figures in
  `figures/` (PNG and PDF).

<!-- results:start -->
*Results appear here after `ssb report`.*
<!-- results:end -->

## Roadmap (not in v0.1)

- Real data: HotpotQA, SQuAD 2.0 and Natural Questions, with Wikipedia distractors.
- Larger N (200–500); more models (Llama-3.1-8B, Mistral-7B); API models with caching.
- Finer axes: redundancy k ∈ {1, 2, 5, 10}, exact copies vs paraphrases; an independent
  gold-ratio sweep; contradiction count (minority/majority) and source-credibility cues;
  seven depth levels.
- Aggregation tasks (counting and comparison across paragraphs).
- Multiple seeds and order shuffling, to measure consistency.
- LLM-as-judge with a human agreement check.
- Mixed-effects logistic regression across factors.
- Encoder baseline: DeBERTa-v3 fine-tuned on SQuAD 2.0, with a sliding window.
- Paper draft in LaTeX.

## License

MIT
