"""Paraphrases by back-translation (en -> fr -> en) with MarianMT.

Beam search with several returned sequences in each direction gives up to
`fr_returns * en_returns` distinct English candidates per paragraph. Callers filter
them (answer span still present, label kept, length window) and keep the first k - 1.
Results are cached on disk keyed by the source text and settings, so a rebuild never
translates twice.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


class BackTranslator:
    def __init__(self, models: list[str], cache_path: str | Path, device: str = "cpu",
                 fr_returns: int = 3, en_returns: int = 3, batch_size: int = 8):
        self.models = models
        self.device = device
        self.fr_returns, self.en_returns = fr_returns, en_returns
        self.batch_size = batch_size
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[str]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["paraphrases"]
        self._loaded = None

    def _key(self, text: str) -> str:
        blob = json.dumps([self.models, self.fr_returns, self.en_returns, text])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _load(self):
        if self._loaded is None:
            from transformers import MarianMTModel, MarianTokenizer

            self._loaded = [(MarianTokenizer.from_pretrained(m),
                             MarianMTModel.from_pretrained(m).to(self.device).eval())
                            for m in self.models]
        return self._loaded

    def _translate(self, pair, texts: list[str], returns: int) -> list[list[str]]:
        import torch

        tok, model = pair
        out: list[list[str]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(self.device)
            with torch.inference_mode():
                gen = model.generate(**enc, num_beams=max(4, returns + 2),
                                     num_return_sequences=returns, max_new_tokens=400,
                                     do_sample=False)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            out += [dec[j * returns:(j + 1) * returns] for j in range(len(batch))]
        return out

    def paraphrases(self, texts: list[str]) -> dict[str, list[str]]:
        """Distinct candidates per source text (source itself excluded), in beam order."""
        todo = [t for t in dict.fromkeys(texts) if self._key(t) not in self._cache]
        if todo:
            en_fr, fr_en = self._load()
            french = self._translate(en_fr, todo, self.fr_returns)
            flat = [f for fs in french for f in fs]
            back = self._translate(fr_en, flat, self.en_returns)
            records = []
            for j, src in enumerate(todo):
                cands = [e for k in range(self.fr_returns)
                         for e in back[j * self.fr_returns + k]]
                seen, keep = {_norm(src)}, []
                for c in cands:
                    c = c.strip()
                    if c and _norm(c) not in seen:
                        seen.add(_norm(c))
                        keep.append(c)
                self._cache[self._key(src)] = keep
                records.append({"key": self._key(src), "source": src, "paraphrases": keep})
            with self.cache_path.open("a", encoding="utf-8", newline="\n") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return {t: self._cache[self._key(t)] for t in texts}
