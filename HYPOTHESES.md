# Pre-registered hypotheses (SSB v0.1)

Written and committed before the full run. The commit that adds this file is the
timestamp: no model had been evaluated on the v0.1 dataset when it was written. Smoke runs
(20 examples per task) only check the pipeline. Their numbers are not used to accept or
reject anything below.

## Design facts the tests rely on

- **Tasks:**
  - A: AG News topic classification (4 classes).
  - B: SQuAD 2.0 passage retrieval (answerable questions).
  - C: SQuAD 2.0 extractive QA (half answerable, half unanswerable).
- **Size:** 300 base examples per task.
- **Context size:** N ∈ {1, 5, 20, 100} is a size level. The context holds
  max(N, evidence) paragraphs, so N = 1 with k = 5 copies is five paragraphs, and every
  example appears at every level of every axis. Position starts at N = 5.
- **Paragraph length:** filler, SQuAD paraphrases and NER-swapped copies are 108–132
  bert-base-uncased tokens. Signals are never edited: AG News articles keep their natural
  length (median about 48 tokens), and SQuAD signals are dev paragraphs already in the
  window.
- **Base context:** the signal plus random filler, which is SQuAD *train* Wikipedia
  paragraphs from articles disjoint from SQuAD dev. Each axis changes one thing:
  - *noise*: hard-negative filler, the paragraphs most similar to the query by
    all-mpnet-base-v2 (not an evaluated model) that do not contain the answer. For A,
    these are the Wikipedia paragraphs most similar to the article.
  - *redundancy*: the signal appears k = 5 times, either as exact copies or as
    back-translated paraphrases (en→fr→en). A paraphrase is kept only if it keeps the
    label (checked by a separate AG News classifier) or contains the answer span (B/C).
  - *contradiction*: one conflicting paragraph, adjacent to the signal at the start of
    the context, before or after it. In A it's an article from a different class; in
    B/C, a copy of the gold paragraph with the answer entity replaced by another entity
    of the same spaCy type.
  - *position*: the signal at relative depth d ∈ {0, 0.5, 1}.

## Measurement

- **Primary score:**
  - A: accuracy.
  - B: Recall@1 (any copy of the gold counts).
  - C: exact match on answerable questions, and no-answer accuracy on unanswerable ones.
    distilbert-squad is scored on answerable questions only.
- **Secondary scores:** macro-F1 (A), Recall@5 and MRR (B), token F1 (C).
- **Confidence:** the max softmax probability (A), or the pipeline answer score (C).
  ECE uses 10 equal-width bins; Brier is reported for A.
- **Uncertainty:** 95% CIs from 1,000 bootstrap resamples **over base examples**, paired
  across conditions. One-sided tests use the bootstrap distribution of the paired
  difference. Holm correction is applied within each hypothesis where several
  comparisons are made.
- **N\* / N† / SSS:**
  - N\* is the smallest N whose gain over the previous N is below 1 point.
  - N† is the smallest N after the curve's peak that is significantly below the peak
    (Holm, p < 0.05).
  - SSS is the trapezoid area under score against log10(N), divided by the log range.
    Position curves span N = 5…100; the other curves span 1…100.

## Hypotheses

**H1: All models degrade as N grows, and hard-negative noise hurts more than random noise.**
(a) For every model, score(N=1) − score(N=100) > 0 on the dilution axis (one-sided,
Holm over models). (b) For every model, score(random, N=100) − score(hard, N=100) > 0
(one-sided, Holm over models). Supported if both hold for all models. Partly supported if
they hold for most models.

**H2: Bag-of-words models show no position bias, and truncating transformers collapse
past the first 512 tokens.**
(a) For tfidf-lr, tfidf and bm25 at N = 100, both CIs, for score(d=0) − score(d=0.5) and
score(d=0) − score(d=1), lie inside ±5 points. (b) For bert-truncate at N = 100, accuracy
at d = 0.5 and at d = 1 is each below 50% and significantly below accuracy at d = 0.

**H3: Chunk-mean fixes position bias but still degrades from dilution.**
For bert-chunk-mean: (a) at N = 100, the CIs of score(d=0) − score(d=0.5) and of
score(d=0) − score(d=1) lie inside ±5 points; (b) score(N=1) − score(N=100) > 0 on
dilution (one-sided, p < 0.05).

**H4: Lexical retrievers beat the dense encoder on hard negatives; the dense encoder wins
on random noise.**
At N = 100: R@1(bm25) − R@1(minilm) > 0 and R@1(tfidf) − R@1(minilm) > 0 on hard
negatives, and R@1(minilm) − R@1(bm25) > 0 and R@1(minilm) − R@1(tfidf) > 0 on random
filler (one-sided, Holm over the four comparisons). Each half is judged separately.

**H5: Redundancy (k = 5) slows degradation. Exact copies help TF-IDF more, and
paraphrases help transformers more.**
Gain(mode) = score(k=5, mode, N=100) − score(k=1, N=100).
- (a) Max over modes of Gain > 0, one-sided, for a majority of models.
- (b) Gain(copy) − Gain(paraphrase) > 0 for tfidf-lr and for tfidf.
- (c) Gain(paraphrase) − Gain(copy) > 0 for bert-truncate, bert-chunk-mean, minilm,
  distilbert-squad and roberta-squad2.

(b) and (c) are judged by the sign of the point estimate with its CI, and by
significance (Holm within each part).

**H6: Under contradiction, QA models pick the conflicting answer about half the time, and
position affects which one wins.**
For each QA model, among answers that give either the true or the swapped entity:
(a) the conflict share lies in [0.35, 0.65]; (b) the conflict share with the conflict
before the gold minus the share with it after is ≠ 0 (two-sided bootstrap, p < 0.05).

**H7: Confidence stays high while accuracy drops (calibration worsens with N).**
Overconfidence = mean confidence − score. For every classification and QA model,
overconfidence(N=100) − overconfidence(N=1) > 0 on the dilution axis (CI excludes 0).
Retrieval models produce no probability and are excluded.

**H8: As N grows, unanswerable questions get more false answers.**
For roberta-squad2, no-answer accuracy(N=1) − no-answer accuracy(N=100) > 0 on both the
dilution and the noise axes (one-sided, p < 0.05 each).

## What counts as exploratory

Everything not listed above. That includes per-class and per-entity-type breakdowns,
secondary metrics, the number of QA windows, and anything added after this file was
committed.

## Known limits of this design

- The signal is fixed and N = 1 is the cleanest input, so most curves can only go
  down. "Does more context help?" is answered by the redundancy axis. The other axes
  measure how much extra context hurts.
- Hard negatives depend on the selector encoder (mpnet). A different selector would pick
  different paragraphs.
- 300 examples per task give a CI half-width of about ±5.5 points near 50% accuracy.
  Smaller effects may not reach significance. Effect sizes with CIs are reported either
  way.
