# Pre-registered hypotheses (SSB v0.1)

Written and committed before the full run. The commit that adds this file is the
timestamp: no model output on `data/ssb_v0.1.jsonl` existed when it was written. The
smoke run (10 questions) only checks the pipeline. Its numbers are not used to accept or
reject anything below.

## Design facts the tests rely on

- **Data:** `data/ssb_v0.1.jsonl`, 100 questions (34 single-hop, 33 multi-hop,
  33 unanswerable) about a fictional world, 2,007 samples. N ∈ {1, 5, 20, 100}
  paragraphs of 108–132 Qwen2.5 tokens each.
- **Base context (dilution axis):** the gold paragraph(s) plus random off-topic filler.
  The gold paragraph's text is identical in every condition. Every other axis changes
  one thing relative to this base:
  - *noise*: hard-negative filler instead of off-topic filler (same N, same gold);
  - *redundancy*: each gold fact appears k = 5 times as distinct paraphrases;
  - *contradiction*: one paragraph states a different value for the target fact, at the
    one-third or two-thirds mark, with the gold at the other mark;
  - *position*: gold placed as a block at relative depth d ∈ {0, 0.5, 1}.
- **Models:** Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct (4-bit). If the 7B runtime
  estimate exceeds the budget, Qwen2.5-3B-Instruct replaces it, and every "larger model"
  statement below refers to that model instead. Greedy decoding, one seed.

## Measurement

- **Correct (answerable):** the normalised answer exactly matches one of the gold
  aliases (lower-cased, punctuation and articles removed). Token F1 is a secondary
  measure.
- **Abstained:** the normalised answer is exactly `unanswerable`.
- **Accuracy curves** use answerable questions only (single + multi hop). Unanswerable
  questions feed the hallucination rate (answered when it should have abstained).
- **Confidence:** the mean probability of the generated answer tokens. ECE uses 10
  equal-width bins; Brier is computed on the same confidence.
- **Closed-book exclusion:** for each model, any question it answers correctly with no
  context is dropped from that model's primary analysis and reported separately. As a
  sensitivity check, every result is also reported with those questions included.
- **Uncertainty:** 95% CIs from 1,000 bootstrap resamples **over questions** (paired
  across conditions when two conditions are compared). One-sided tests use the bootstrap
  distribution of the paired difference.
- **N\* (saturation point):** the smallest N in {5, 20, 100} where accuracy minus the
  accuracy at the previous N is below 1 point. It is "not reached" if no step qualifies.
- **N† (degradation onset):** the smallest N where accuracy is significantly below the
  curve's peak (one-sided paired bootstrap, p < 0.05, Holm-corrected over the N values
  compared with the peak). It is "> 100" if there is none.
- **SSS:** the area under accuracy against log10(N) from N = 1 to N = 100 (trapezoid
  rule), divided by log10(100). It ranges over [0, 1], one value per model and axis.

## Hypotheses

Each test runs separately per model unless stated otherwise.

**H1: Accuracy plateaus, then declines as N grows, most with hard-negative noise.**
Supported if, for the hard-negative curve, (a) accuracy at N = 5 is not significantly
below N = 1 (the plateau), (b) N† exists (a significant decline), and (c) its drop from
peak to N = 100 is larger than the dilution curve's (paired bootstrap CI of the
difference excludes 0). Partly supported if (b) holds but (a) or (c) fails.

**H2: Hard negatives hurt more than random noise at equal N.**
For N ∈ {5, 20, 100}: accuracy(random) − accuracy(hard negative) > 0, one-sided paired
test, Holm-corrected over the 3 sizes. Supported if significant at N = 100 and the point
estimate is positive at N = 20 and 100.

**H3: Gold at depth 0.5 underperforms depths 0 and 1, and the gap widens with N.**
Gap(N) = mean(accuracy at d=0, accuracy at d=1) − accuracy at d=0.5. Supported if
Gap(100) > 0 (one-sided, p < 0.05) and Gap(100) − Gap(20) > 0 (bootstrap CI excludes 0).
Partly supported if only the first holds.

**H4: Redundancy (k = 5) protects against dilution at large N.**
accuracy(redundancy, k=5) − accuracy(dilution, k=1) > 0 at N = 100 (one-sided paired,
p < 0.05). We also report N = 20. (At N = 5 a single-hop redundancy context is all gold,
so N = 5 is descriptive only.)

**H5: Under contradiction, models favour the claim that appears last.**
Among contradiction samples where the model gives either the gold or the conflicting
answer, the share choosing the later claim (the gold when the conflict comes *before*,
the conflict when it comes *after*) is > 0.5. Pooled over N; bootstrap CI over
questions must exclude 0.5. We report each N separately as well.

**H6: Calibration degrades faster than accuracy (models stay confident while getting worse).**
Overconfidence(N) = mean confidence − accuracy, on answerable samples. Supported if
Overconfidence(100) − Overconfidence(1) > 0 (CI excludes 0) on both the dilution and the
hard-negative axes. We also report ECE(100) − ECE(1).

**H7: The larger model has a later degradation onset N†.**
Compare N† per axis (dilution, noise, redundancy, position at d=0.5, contradiction
gold-rate), with "> 100" ranked after 100. Supported if the larger model's N† is later on
a majority of axes and earlier on none.

## What counts as exploratory

Everything not listed above. That includes per-task-type breakdowns (single vs multi hop),
false-abstention rates, the closed-book gap, token-F1 results, and any condition added
after this file was committed. Exploratory results are labelled as such in the README.

## Known limits of this design

- The gold evidence is fixed and only the surroundings change. More context therefore
  never adds information, except on the redundancy axis. "Does more context help?" is
  answered here only by redundancy. The other axes measure how much extra context hurts.
- 67 answerable questions per condition: a CI half-width of roughly ±10–12 points near
  50% accuracy. Small effects will not reach significance. Effect sizes with CIs are
  reported whether or not they are significant.
