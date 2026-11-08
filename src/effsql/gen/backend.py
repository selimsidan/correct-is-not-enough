"""Generation backend: the only abstraction in the codebase.

`Backend.generate(batch, params)` takes chat messages and per-request sampling parameters and
returns, for each request, its list of completions. `VLLMBackend` is the real implementation; it
imports vLLM lazily so CPU commands never load vLLM or torch. Tests use a fake backend.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class Completion:
    text: str
    finish_reason: str | None
    n_tokens: int

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenParams:
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = None

    def record(self) -> dict[str, Any]:
        return asdict(self)


def stable_seed(*parts: Any) -> int:
    """Per-request seed from stable identifiers, independent of batching and process."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class Backend(Protocol):
    model_id: str
    model_revision: str | None

    def count_tokens(self, messages: list[dict[str, str]]) -> int: ...

    def generate(self, batch: list[list[dict[str, str]]], params: list[GenParams]) -> list[list[Completion]]: ...


class VLLMBackend:
    """vLLM offline engine with prefix caching. Chat templates are applied with the model's tokenizer."""

    def __init__(self, model_cfg: dict[str, Any], seed: int = 0):
        from vllm import LLM  # GPU-only import

        self.model_id = model_cfg["id"]
        self.model_revision = self._resolve_revision(self.model_id, model_cfg.get("revision"))
        self.llm = LLM(
            model=self.model_id,
            revision=self.model_revision or model_cfg.get("revision"),
            dtype=model_cfg.get("dtype", "auto"),
            max_model_len=model_cfg.get("max_model_len"),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.9),
            enable_prefix_caching=model_cfg.get("enable_prefix_caching", True),
            seed=seed,
        )
        self.tok = self.llm.get_tokenizer()
        import vllm

        self.vllm_version = vllm.__version__

    @staticmethod
    def _resolve_revision(model_id: str, revision: str | None) -> str | None:
        """Exact Hugging Face commit, so every record names the weights it came from."""
        try:
            from huggingface_hub import model_info

            return model_info(model_id, revision=revision).sha
        except Exception as e:  # noqa: BLE001 (offline or rate limited: record what we know)
            log.warning("could not resolve model revision: %s", e)
            return revision

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return len(self.tok.encode(self._prompt(messages), add_special_tokens=False))

    def generate(self, batch: list[list[dict[str, str]]], params: list[GenParams]) -> list[list[Completion]]:
        from vllm import SamplingParams

        prompts = [self._prompt(m) for m in batch]
        sps = [SamplingParams(n=p.n, temperature=p.temperature, top_p=p.top_p, max_tokens=p.max_tokens,
                              seed=p.seed) for p in params]
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        return [[Completion(o.text, o.finish_reason, len(o.token_ids)) for o in out.outputs] for out in outs]


def make_backend(gen_cfg: dict[str, Any]) -> Backend:
    return VLLMBackend(gen_cfg["model"], seed=gen_cfg.get("seed", 0))
