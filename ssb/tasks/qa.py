"""Task C: extractive QA over a multi-paragraph context with a sliding window
(max_seq_len=384, doc_stride=128). The number of windows is logged.

transformers 5 removed the extractive `question-answering` pipeline, so the decoding is
done here, with the same rules the v4 pipeline used:
- per window, softmax over the context tokens plus CLS (question and padding masked);
- null score = p_start(CLS) * p_end(CLS), the minimum over windows;
- best span per window with start <= end < start + max_answer_len, CLS excluded;
- the best span over all windows wins; with handle_impossible_answer the null answer
  competes with it on the same probability scale;
- spans are widened to whole words of the context.
"""

from __future__ import annotations

import numpy as np

from ..build import context_text
from ..config import ModelConfig
from ..metrics import exact_match, qa_outcome, token_f1

MAX_ANSWER_LEN = 30


def best_span(start_logits: np.ndarray, end_logits: np.ndarray, context_mask: np.ndarray,
              cls_index: int, max_answer_len: int = MAX_ANSWER_LEN) -> tuple[int, int, float, float]:
    """Best (start, end, score) in one window, plus the window's null score.

    context_mask is 1 on context tokens. Probabilities are a softmax over context tokens
    and CLS, as in the transformers v4 question-answering pipeline."""
    keep = context_mask.astype(bool).copy()
    keep[cls_index] = True
    s = np.where(keep, start_logits, -10000.0)
    e = np.where(keep, end_logits, -10000.0)
    s = np.exp(s - s.max())
    s /= s.sum()
    e = np.exp(e - e.max())
    e /= e.sum()
    null = float(s[cls_index] * e[cls_index])
    s[cls_index] = e[cls_index] = 0.0
    outer = np.tril(np.triu(np.outer(s, e)), max_answer_len - 1)
    i, j = np.unravel_index(int(outer.argmax()), outer.shape)
    return int(i), int(j), float(outer[i, j]), null


class QAModel:
    def __init__(self, m: ModelConfig, device: str, max_seq_len: int, doc_stride: int):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.m = m
        self.device = device
        self.max_seq_len, self.doc_stride = max_seq_len, doc_stride
        self.tok = AutoTokenizer.from_pretrained(m.model_id)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            m.model_id, dtype=torch.float16 if device == "cuda" else torch.float32)
        self.model.to(device).eval()
        self.pad_right = self.tok.padding_side == "right"

    def _encode(self, question: str, context: str):
        first, second = ((question, context) if self.pad_right else (context, question))
        return self.tok(first, second,
                        truncation="only_second" if self.pad_right else "only_first",
                        max_length=self.max_seq_len, stride=self.doc_stride,
                        return_overflowing_tokens=True, return_offsets_mapping=True,
                        padding="max_length")

    def windows(self, question: str, context: str) -> int:
        return len(self._encode(question, context)["input_ids"])

    def _answer(self, question: str, context: str) -> tuple[str, float, int]:
        import torch

        enc = self._encode(question, context)
        n = len(enc["input_ids"])
        ctx_seq = 1 if self.pad_right else 0
        names = [k for k in ("input_ids", "attention_mask", "token_type_ids")
                 if k in enc and k in self.tok.model_input_names]
        starts, ends = [], []
        bs = max(1, self.m.batch_size) * 8          # windows per forward pass
        with torch.inference_mode():
            for b in range(0, n, bs):
                batch = {k: torch.tensor(enc[k][b:b + bs], device=self.device) for k in names}
                out = self.model(**batch)
                starts.append(out.start_logits.float().cpu().numpy())
                ends.append(out.end_logits.float().cpu().numpy())
        start_all, end_all = np.concatenate(starts), np.concatenate(ends)

        best = (-1.0, 0, 0, 0)                       # score, window, start token, end token
        min_null = 1.0
        for w in range(n):
            seq_ids = enc.sequence_ids(w)
            ctx_mask = np.array([1 if sid == ctx_seq else 0 for sid in seq_ids])
            ids = enc["input_ids"][w]
            cls = ids.index(self.tok.cls_token_id) if self.tok.cls_token_id in ids else 0
            i, j, score, null = best_span(start_all[w], end_all[w], ctx_mask, cls)
            min_null = min(min_null, null)
            if score > best[0]:
                best = (score, w, i, j)

        score, w, i, j = best
        if self.m.handles_unanswerable and min_null >= score:
            return "", min_null, n
        c0, c1 = self._word_chars(enc, w, i, j, ctx_seq)
        return context[c0:c1], score, n

    def _word_chars(self, enc, w: int, i: int, j: int, ctx_seq: int) -> tuple[int, int]:
        """Character span of tokens i..j, widened to whole words (align_to_words)."""
        offsets = enc["offset_mapping"][w]
        try:
            wi, wj = enc.token_to_word(w, i), enc.token_to_word(w, j)
            if wi is not None and wj is not None:
                c0 = enc.word_to_chars(w, wi, sequence_index=ctx_seq)[0]
                c1 = enc.word_to_chars(w, wj, sequence_index=ctx_seq)[1]
                return c0, c1
        except (TypeError, ValueError):
            pass
        return offsets[i][0], offsets[j][1]

    def predict(self, samples: list[dict]) -> list[dict]:
        out = []
        for s in samples:
            question, context = s["question"], context_text(s["context"])
            text, conf, n_windows = self._answer(question, context)
            text = text.strip()
            res = {"answer": text, "confidence": float(conf), "abstained": text == "",
                   "windows": n_windows}
            if s["answerable"]:
                res["em"] = float(exact_match(text, s["answers"]))
                res["f1"] = token_f1(text, s["answers"])
                if s["conflict_id"]:
                    res["outcome"] = qa_outcome(text, s["answers"], s["conflict_value"])
            else:
                res["noans_correct"] = float(text == "")
            out.append(res)
        return out
