"""Model backends behind one interface, plus an append-only disk cache.

    backend.generate(prompts) -> [{"text", "answer_logprobs", "prompt_tokens", "gen_tokens"}]

`answer_logprobs` are the log-probabilities of the generated tokens (end-of-turn token
excluded), taken from the raw logits. Decoding is greedy with every sampling knob
neutralised, because Qwen's generation_config otherwise applies repetition_penalty=1.05.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .config import ModelConfig


def chat_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


class Backend:
    name = "base"

    def __init__(self, model_id: str, cfg: ModelConfig, max_new_tokens: int):
        self.model_id = model_id
        self.cfg = cfg
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: list[str]) -> list[dict]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "model_id": self.model_id,
                "quantization": self.cfg.quantization}


# --------------------------------------------------------------------------- HF


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str, cfg: ModelConfig, max_new_tokens: int):
        super().__init__(model_id, cfg, max_new_tokens)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cuda = torch.cuda.is_available()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs: dict = {"dtype": torch.float16 if self.cuda else torch.float32}
        if cfg.quantization == "4bit":
            if not self.cuda:
                raise RuntimeError("4-bit quantization needs a CUDA GPU")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        if self.cuda:
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa",
                                                          **kwargs)
        self.model.eval()
        self.device = self.model.device
        if self.cuda:
            _memory_efficient_attention_only()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.stop_ids = {i for i in eos + [self.tok.eos_token_id] if i is not None}

    def render(self, prompt: str) -> str:
        return self.tok.apply_chat_template(chat_messages(prompt), tokenize=False,
                                            add_generation_prompt=True)

    def _attention_ctx(self):
        """On CUDA, forbid the "math" SDPA kernel: it builds the full N x N attention
        matrix (6+ GiB at 12k tokens). A context that cannot use a fused kernel then
        fails loudly instead of silently exhausting memory."""
        import contextlib

        if not self.cuda:
            return contextlib.nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.CUDNN_ATTENTION])

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tok(self.render(prompt), add_special_tokens=False).input_ids)

    def generate(self, prompts: list[str]) -> list[dict]:
        """Generate; on CUDA out-of-memory, retry the batch as two halves."""
        try:
            return self._generate(prompts)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(prompts) == 1:
                raise
            mid = len(prompts) // 2
            print(f"[{self.cfg.name}] out of memory on a batch of {len(prompts)}; splitting")
            return self.generate(prompts[:mid]) + self.generate(prompts[mid:])

    def _generate(self, prompts: list[str]) -> list[dict]:
        torch = self.torch
        # Pad to a multiple of 64 so the padding mask stays aligned for the
        # memory-efficient attention kernel (unaligned masks fall back to "math").
        enc = self.tok([self.render(p) for p in prompts], return_tensors="pt", padding=True,
                       pad_to_multiple_of=64 if len(prompts) > 1 else None,
                       add_special_tokens=False).to(self.device)
        with torch.inference_mode(), self._attention_ctx():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                output_logits=True, return_dict_in_generate=True,
                pad_token_id=self.tok.pad_token_id)
        new = out.sequences[:, enc.input_ids.shape[1]:].tolist()
        logps = torch.stack(out.logits, dim=1).float().log_softmax(-1)  # B x T x V
        results = []
        for b, ids in enumerate(new):
            kept, lps = [], []
            for t, tid in enumerate(ids):
                if tid in self.stop_ids or tid == self.tok.pad_token_id:
                    break
                kept.append(tid)
                lps.append(float(logps[b, t, tid]))
            results.append({"text": self.tok.decode(kept, skip_special_tokens=True),
                            "answer_logprobs": lps,
                            "prompt_tokens": int(enc.attention_mask[b].sum()),
                            "gen_tokens": len(kept)})
        del out, logps, enc
        if self.cuda:
            torch.cuda.empty_cache()
        return results


def _memory_efficient_attention_only() -> None:
    """Stop transformers from passing enable_gqa=True to SDPA.

    Without a padding mask (one prompt per batch) transformers asks SDPA for native
    grouped-query attention, which GPUs without flash attention (such as the T4) can
    only run with the "math" kernel. Repeating the key/value heads instead costs a few
    MB per layer and lets the memory-efficient kernel run.
    """
    from transformers.integrations import sdpa_attention

    sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


# --------------------------------------------------------------------------- vLLM


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model_id: str, cfg: ModelConfig, max_new_tokens: int):
        super().__init__(model_id, cfg, max_new_tokens)
        from vllm import LLM, SamplingParams

        quant = None
        if cfg.quantization == "4bit":
            quant = "awq" if re.search(r"awq", model_id, re.I) else "bitsandbytes"
        self.llm = LLM(model=model_id, quantization=quant, dtype="float16",
                       max_model_len=cfg.max_model_len, gpu_memory_utilization=0.90, seed=0,
                       generation_config="vllm")  # ignore the model's sampling defaults
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                                     repetition_penalty=1.0, max_tokens=max_new_tokens,
                                     logprobs=0, seed=0)
        self.quant = quant

    def render(self, prompt: str) -> str:
        return self.tok.apply_chat_template(chat_messages(prompt), tokenize=False,
                                            add_generation_prompt=True)

    def generate(self, prompts: list[str]) -> list[dict]:
        outs = self.llm.generate([self.render(p) for p in prompts], self.params, use_tqdm=False)
        results = []
        for o in outs:
            c = o.outputs[0]
            lps = [float(step[tid].logprob) for tid, step in zip(c.token_ids, c.logprobs or [],
                                                                  strict=False)
                   if tid not in (self.tok.eos_token_id,) and step and tid in step]
            results.append({"text": c.text, "answer_logprobs": lps,
                            "prompt_tokens": len(o.prompt_token_ids),
                            "gen_tokens": len(c.token_ids)})
        return results

    def info(self) -> dict:
        return {**super().info(), "vllm_quantization": self.quant}


# --------------------------------------------------------------------------- fake


class FakeBackend(Backend):
    """Deterministic stand-in for tests: answers with the first year in the context."""

    name = "fake"

    def generate(self, prompts: list[str]) -> list[dict]:
        out = []
        for p in prompts:
            ctx = p.split("Question:")[0]
            m = re.search(r"\b1[6-9]\d\d\b", ctx)
            text = m.group(0) if m else "UNANSWERABLE"
            out.append({"text": text, "answer_logprobs": [-0.2] * 2,
                        "prompt_tokens": len(p) // 4, "gen_tokens": 2})
        return out


# --------------------------------------------------------------------------- factory


def vllm_available() -> bool:
    try:
        import torch
        import vllm  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def resolve_backend(cfg: ModelConfig) -> str:
    if cfg.backend != "auto":
        return cfg.backend
    return "vllm" if vllm_available() else "hf"


def model_id_for(cfg: ModelConfig, backend: str) -> str:
    if backend == "vllm" and cfg.vllm_model_id:
        return cfg.vllm_model_id
    return cfg.model_id


def load_backend(cfg: ModelConfig, max_new_tokens: int) -> Backend:
    backend = resolve_backend(cfg)
    classes = {"hf": HFBackend, "vllm": VLLMBackend, "fake": FakeBackend}
    if backend == "vllm":
        try:
            return VLLMBackend(model_id_for(cfg, "vllm"), cfg, max_new_tokens)
        except Exception as e:  # noqa: BLE001 - any vLLM start-up failure falls back to HF
            if cfg.backend == "vllm":
                raise
            print(f"[{cfg.name}] vLLM failed to start ({type(e).__name__}: {e}); using HF")
            backend = "hf"
    return classes[backend](model_id_for(cfg, backend), cfg, max_new_tokens)


# --------------------------------------------------------------------------- cache


class GenerationCache:
    """Append-only JSONL keyed by a hash of everything that determines the output.

    Each batch is flushed and fsynced, so a crash loses at most the batch in flight.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # a torn last line after a crash
                        continue
                    self._data[rec["key"]] = rec

    @staticmethod
    def key(model_id: str, backend: str, quantization: str | None, max_new_tokens: int,
            prompt: str) -> str:
        blob = json.dumps([model_id, backend, quantization, max_new_tokens, prompt])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def __len__(self) -> int:
        return len(self._data)

    def add_many(self, records: list[dict]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                self._data[r["key"]] = r
            fh.flush()
            os.fsync(fh.fileno())
