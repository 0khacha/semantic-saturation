"""Task C: extractive QA over a multi-paragraph context, with the Hugging Face pipeline's
sliding window (max_seq_len=384, doc_stride=128). The number of windows is logged."""

from __future__ import annotations

from ..build import context_text
from ..config import ModelConfig
from ..metrics import exact_match, qa_outcome, token_f1


class QAModel:
    def __init__(self, m: ModelConfig, device: str, max_seq_len: int, doc_stride: int):
        from transformers import AutoTokenizer, pipeline

        self.m = m
        self.max_seq_len, self.doc_stride = max_seq_len, doc_stride
        self.tok = AutoTokenizer.from_pretrained(m.model_id)
        self.pipe = pipeline("question-answering", model=m.model_id, tokenizer=self.tok,
                             device=0 if device == "cuda" else -1,
                             dtype="float16" if device == "cuda" else None)

    def windows(self, question: str, context: str) -> int:
        enc = self.tok(question, context, truncation="only_second",
                       max_length=self.max_seq_len, stride=self.doc_stride,
                       return_overflowing_tokens=True)
        return len(enc["input_ids"])

    def predict(self, samples: list[dict]) -> list[dict]:
        inputs = [{"question": s["question"], "context": context_text(s["context"])}
                  for s in samples]
        preds = self.pipe(inputs, batch_size=self.m.batch_size, max_seq_len=self.max_seq_len,
                          doc_stride=self.doc_stride, max_answer_len=30,
                          handle_impossible_answer=self.m.handles_unanswerable)
        if isinstance(preds, dict):
            preds = [preds]
        out = []
        for s, x, p in zip(samples, inputs, preds, strict=True):
            text = (p.get("answer") or "").strip()
            res = {"answer": text, "confidence": float(p.get("score", 0.0)),
                   "abstained": text == "", "windows": self.windows(x["question"], x["context"])}
            if s["answerable"]:
                res["em"] = float(exact_match(text, s["answers"]))
                res["f1"] = token_f1(text, s["answers"])
                if s["conflict_id"]:
                    res["outcome"] = qa_outcome(text, s["answers"], s["conflict_value"])
            else:
                res["noans_correct"] = float(text == "")
            out.append(res)
        return out
