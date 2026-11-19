# Semantic Saturation Benchmark (SSB)

> **Does adding more textual context improve an NLP model's understanding? Is there a
> point where more context starts to make it less reliable?**

SSB keeps the **signal paragraph fixed** and adds paragraphs around it. It then measures
whether classic NLP models stay correct *and* well calibrated. There are no large
language models and no text generation: only TF-IDF, BM25 and BERT-family encoders, on
three kinds of "understanding".

**Status: v0.1 in progress.** The dataset builder and its checks are done. Model runs and
the report come next. The results section is written by `ssb report` from the actual
run.

(An earlier LLM-based version of this benchmark is preserved on the
[`llm-v0.1`](../../tree/llm-v0.1) branch.)

## Method

| Task | Data | Signal | Contradiction | Models |
|---|---|---|---|---|
| A. Topic classification | AG News (4 classes) | one test article, unedited | an article from another class | TF-IDF + logistic regression; BERT fine-tuned on AG News, with `truncate` (first 512 tokens) and `chunk-mean` (average logits over 512-token chunks) |
| B. Passage retrieval | SQuAD 2.0 dev, answerable | the question's source paragraph | gold copy with the answer entity swapped (spaCy NER) | TF-IDF, BM25, all-MiniLM-L6-v2 |
| C. Extractive QA | SQuAD 2.0 dev, answerable + unanswerable | the source paragraph | same NER-swapped copy | distilbert-base-cased-distilled-squad, deepset/roberta-base-squad2 (sliding window 384/128) |

**Context size.** N ∈ {1, 5, 20, 100} paragraphs of about 120 tokens (108–132,
bert-base-uncased), up to about 12k tokens. A context never holds less than its own
evidence: five copies at N = 1 are five paragraphs. This way every example appears at
every N, and curves always compare the same examples. Every sample's token count is
stored.

**Axes.** One varies at a time; the base context is the signal plus random filler.

| Axis | What changes | Levels |
|---|---|---|
| Dilution | Only N (one signal paragraph, so the signal ratio is 1/N) | N |
| Semantic noise | Filler becomes hard negatives: the paragraphs most similar to the query (all-mpnet-base-v2, not an evaluated model) that don't contain the answer | random vs hard |
| Redundancy | The signal appears k times | k ∈ {1, 5}; exact copies vs back-translated paraphrases (MarianMT en→fr→en) |
| Contradiction | One conflicting paragraph next to the signal | before / after |
| Position | Where the signal sits | depth 0, 0.5, 1 |

Random filler is Wikipedia: SQuAD 2.0 *train* paragraphs, whose 442 articles share no
title with the 35 dev articles.

**Guarantees, checked at build time and in `tests/`:**
- The signal is present and byte-identical to the source data in every sample.
- Paragraph counts, token counts and depths match the condition.
- In B and C, the answer appears only in signal copies, never in filler.
- An NER-swapped copy no longer contains the answer but does contain its replacement
  entity.
- A cross-class article has a different label.
- Every paraphrase keeps the answer span (B/C), or keeps the label according to a
  separate AG News classifier (A); the ones that don't are dropped.

**Hypotheses** are pre-registered in [HYPOTHESES.md](HYPOTHESES.md).

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate        # Python 3.11–3.13
pip install -e ".[analysis,dev]"
ssb build --config configs/smoke.yaml                 # 20 examples per task, CPU
pytest                                                # dataset invariants and unit tests
```

<!-- results:start -->
*Results appear here after `ssb report`.*
<!-- results:end -->

## Limitations

- English only.
- The perturbations are controlled but partly synthetic: entity swaps and
  back-translations.
- Contexts reach at most about 12k tokens.
- Encoder models only.
- Hard negatives depend on the selector encoder.
- AG News signals are shorter than the 120-token filler, because the signal is never
  edited.

## Roadmap

- More datasets (IMDB, HotpotQA).
- Longformer/BigBird for long inputs.
- More N values.
- Multilingual (French, Arabic, Tamazight).
- Multiple seeds.

## License

MIT
