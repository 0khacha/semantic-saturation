# Semantic Saturation Benchmark (SSB)

> **Does adding more textual context improve a language model's understanding? Is there a
> point where more context starts to make it less reliable?**

SSB holds the **question and the gold evidence fixed** and changes only what surrounds
them. Then it measures how answer quality *and reliability* change: calibration,
abstention, and behaviour under conflicting claims.

**Status: v0.1, under construction.** The dataset and its checks are done. The model
runner, metrics and report come next. The results section below is filled in by
`ssb report`.

## Method

**Context size.** N ∈ {1, 5, 20, 100} paragraphs, each 108–132 tokens (Qwen2.5
tokenizer). The exact token count of every prompt is logged.

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
- `data/ssb_v0.1.jsonl`: 2,007 samples, one per line. `paragraphs` is the ordered list of
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

Hypotheses are pre-registered in [HYPOTHESES.md](HYPOTHESES.md).

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.11 or 3.12
pip install -e ".[build,dev]"
ssb build --config configs/main.yaml                # ~2 min on CPU; rebuilds data/ bit-for-bit
pytest
```

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
