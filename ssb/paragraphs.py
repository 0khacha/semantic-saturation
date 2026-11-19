"""Token counting and paragraph length normalisation (~120 tokens, +/-10%).

All counts use one reference tokenizer (bert-base-uncased WordPiece by default), so
"paragraphs" and "tokens" stay proportional across tasks. Normalisation never edits
words: it keeps a run of whole consecutive sentences whose token count lands in the
window, or rejects the paragraph.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Split after . ! ? (optionally followed by a closing quote/bracket) when the next
# character starts a new sentence. Avoids splitting "U.S." or "e.g. the".
_SENT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREV = re.compile(r"\b(?:[A-Z]\.){1,3}$"
                     r"|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|e\.g|i\.e|No|Co|Inc|Ltd)\.$")


class TokenCounter:
    def __init__(self, name: str):
        from tokenizers import Tokenizer

        self.name = name
        self._tok = Tokenizer.from_pretrained(name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence split that keeps common abbreviations intact."""
    parts, buf = [], ""
    for piece in _SENT.split(text.strip()):
        buf = f"{buf} {piece}".strip() if buf else piece
        if not _ABBREV.search(buf):
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts


def normalize(text: str, counter: TokenCounter, lo: int, hi: int) -> tuple[str, int] | None:
    """A window of whole consecutive sentences with lo <= tokens <= hi, or None.

    From the earliest possible starting sentence, take the *longest* window that still
    fits, so a paragraph already inside the window comes back unchanged. Deterministic.
    """
    if lo <= (n := counter.count(text)) <= hi:
        return text, n
    sents = split_sentences(text)
    for start in range(len(sents)):
        best = None
        for end in range(start, len(sents)):
            cand = " ".join(sents[start:end + 1])
            n = counter.count(cand)
            if n > hi:
                break
            if n >= lo:
                best = (cand, n)
        if best:
            return best
    return None


def text_id(prefix: str, text: str) -> str:
    """Stable id from content, e.g. 'w:3f9a1c2b7d'."""
    return f"{prefix}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


@dataclass
class Paragraph:
    id: str
    text: str
    tokens: int
    source: str  # "agnews" | "squad_dev" | "squad_train" | "paraphrase" | "conflict"
    label: int | None = None  # AG News class of news paragraphs
    title: str | None = None  # Wikipedia article title for SQuAD paragraphs
    origin: str | None = None  # id of the original SQuAD dev paragraph it was cut from

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> Paragraph:
        return cls(**d)


def contains_answer(text: str, answer: str) -> bool:
    """Case-insensitive containment on word boundaries (answers can be numbers)."""
    a = re.escape(answer.strip())
    return bool(a) and re.search(rf"(?<!\w){a}(?!\w)", text, re.I) is not None
