# Semantic Saturation Benchmark (SSB)

> **Does adding more textual context improve an NLP model's understanding? Is there a
> point where more context starts to make it less reliable?**

SSB keeps the **signal paragraph fixed** and adds paragraphs around it. It then measures
whether classic NLP models stay correct *and* well calibrated. There are no large
language models and no text generation: only TF-IDF, BM25 and BERT-family encoders, on
three kinds of "understanding".

**Status: v0.1 in progress.** The dataset builder, model runner, analysis and report are
done and pass an end-to-end smoke run on CPU. The full run (300 examples per task) is
next, on a T4. The results section is written by `ssb report` from the actual run.

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
pytest                                                # dataset invariants, metrics, resume

# smoke test (CPU): 20 base examples per task
ssb build  --config configs/smoke.yaml
ssb run    --config configs/smoke.yaml
ssb report --config configs/smoke.yaml

# full v0.1 (one T4): print the runtime estimate, then run
ssb run    --config configs/main.yaml --estimate
ssb run    --config configs/main.yaml
ssb report --config configs/main.yaml                 # fills the results section below
```

The committed `data/ssb_v0.1.jsonl` (SHA-256 hashes in its manifest) is the canonical
dataset. `ssb build --config configs/main.yaml` regenerates it, but back-translation and
embeddings can differ slightly between GPU and CPU builds, so runs read the committed
file. `ssb run` can always be resumed: predictions are appended and fsynced after each
batch, and a rerun skips finished samples. On Kaggle, `kaggle/run.py` runs build, run and
report on a T4 (`kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4`).

<!-- results:start -->
*Results appear here after `ssb report`.*
<!-- results:end -->

## Related work

- **Lost in the Middle** (Liu et al., TACL 2024, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)):
  LLM accuracy on multi-document QA and key-value retrieval is U-shaped in the position
  of the relevant passage, and gains saturate long before retriever recall does.
- **LLMs Can Be Easily Distracted by Irrelevant Context** (Shi et al., ICML 2023,
  [arXiv:2302.00093](https://arxiv.org/abs/2302.00093)): GSM-IC adds irrelevant sentences
  to arithmetic problems, and accuracy drops sharply.
- **Adversarial Examples for Evaluating Reading Comprehension Systems** (Jia & Liang,
  EMNLP 2017, [arXiv:1707.07328](https://arxiv.org/abs/1707.07328)): one inserted
  distracting sentence cuts SQuAD F1 from 75% to 36% on average across 16 models.
- **BEIR** (Thakur et al., NeurIPS 2021 Datasets & Benchmarks,
  [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)): across 18 retrieval datasets,
  BM25 is a robust zero-shot baseline that dense retrievers often fail to beat.

**What SSB adds:** quantity, relevance, redundancy, conflict and position as
**independent controlled axes**, across three task types (classification, retrieval,
QA). It also measures **reliability** (calibration, abstention, conflict behaviour),
not just accuracy.

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
