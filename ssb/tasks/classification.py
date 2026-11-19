"""Task A: AG News topic classification over a multi-paragraph context.

Models:
- TF-IDF + logistic regression, trained once on clean single-article AG News train data
  (cached to disk).
- A BERT fine-tuned on AG News, reading long inputs two ways: `truncate` (first 512
  tokens) or `chunk_mean` (512-token chunks, logits averaged).
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from ..build import context_text
from ..config import ModelConfig


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _result(probs: np.ndarray, **extra) -> dict:
    return {"probs": [round(float(p), 6) for p in probs], "pred": int(np.argmax(probs)),
            "confidence": float(np.max(probs)), **extra}


class TfidfLR:
    def __init__(self, m: ModelConfig, cache_dir: str, train_texts, train_labels, seed: int):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        key = hashlib.sha256(f"{len(train_texts)}|{seed}|v1".encode()).hexdigest()[:12]
        path = Path(cache_dir) / f"tfidf_lr_{key}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                self.vec, self.clf = pickle.load(fh)
        else:
            self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                       max_features=300_000)
            X = self.vec.fit_transform(train_texts)
            self.clf = LogisticRegression(C=4.0, max_iter=2000, random_state=seed)
            self.clf.fit(X, train_labels)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh:
                pickle.dump((self.vec, self.clf), fh)

    def predict(self, samples: list[dict]) -> list[dict]:
        probs = self.clf.predict_proba(self.vec.transform(
            [context_text(s["context"]) for s in samples]))
        return [_result(p) for p in probs]


class TransformerClassifier:
    def __init__(self, m: ModelConfig, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device, self.mode = torch, device, m.long_input
        self.batch_size = m.batch_size
        self.tok = AutoTokenizer.from_pretrained(m.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(m.model_id).to(device)
        if device == "cuda":
            self.model.half()
        self.model.eval()
        self.max_len = min(512, self.tok.model_max_length)

    def _logits(self, ids_list: list[list[int]]) -> np.ndarray:
        """Logits for sequences of token ids (special tokens already included)."""
        torch = self.torch
        out = []
        order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
        for i in range(0, len(order), self.batch_size):
            idx = order[i:i + self.batch_size]
            batch = [ids_list[j] for j in idx]
            width = max(len(b) for b in batch)
            pad = self.tok.pad_token_id
            ids = torch.tensor([b + [pad] * (width - len(b)) for b in batch], device=self.device)
            mask = (torch.arange(width, device=self.device)[None, :]
                    < torch.tensor([len(b) for b in batch], device=self.device)[:, None])
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=mask.long()).logits
            out += list(zip(idx, logits.float().cpu().numpy(), strict=True))
        res = np.zeros((len(ids_list), out[0][1].shape[0])) if out else np.zeros((0, 4))
        for j, lg in out:
            res[j] = lg
        return res

    def predict(self, samples: list[dict]) -> list[dict]:
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        body = self.max_len - 2
        seqs, owner = [], []
        for k, s in enumerate(samples):
            ids = self.tok(context_text(s["context"]), add_special_tokens=False).input_ids
            chunks = [ids[:body]] if self.mode == "truncate" else \
                [ids[i:i + body] for i in range(0, max(len(ids), 1), body)]
            for c in chunks:
                seqs.append([cls] + c + [sep])
                owner.append(k)
        logits = self._logits(seqs)
        owner = np.array(owner)
        out = []
        for k in range(len(samples)):
            mean = logits[owner == k].mean(axis=0)
            out.append(_result(_softmax(mean), n_chunks=int((owner == k).sum())))
        return out
