"""CPU inference for Qwen2.5-0.5B-Instruct with an on-disk generation cache.

Generation on CPU is the slow part of this study, so every raw model output is
cached under ``results/raw/`` keyed by a hash of (model, template, record,
generation parameters).  Re-running a study, adding a template, or fixing the
post-processor therefore costs nothing for work already done.

The cache key deliberately includes the rendered prompt text, so editing a
template's wording invalidates only that template's entries.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .data import ClinicalRecord
from .prompts import PromptTemplate

__all__ = [
    "DEFAULT_MODEL_ID",
    "GenerationConfig",
    "Generation",
    "ModelUnavailableError",
    "QwenRunner",
    "cache_key",
]

#: Default model: small enough for CPU inference, instruction-tuned.
DEFAULT_MODEL_ID: Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"


class ModelUnavailableError(RuntimeError):
    """Raised when the model cannot be loaded, for example with no network."""


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters.

    Attributes:
        temperature: Sampling temperature; ignored when ``do_sample`` is False.
        top_p: Nucleus sampling mass; ignored when ``do_sample`` is False.
        do_sample: ``False`` gives greedy decoding, which makes the study
            reproducible without relying on seeded sampling.
        seed: Seed set before each batch, so sampled runs are reproducible.
        batch_size: Prompts per generate call.
        max_new_tokens: Fallback budget when a template does not set one.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 20260321
    batch_size: int = 4
    max_new_tokens: int | None = None

    def for_template(self, template: PromptTemplate) -> dict[str, Any]:
        """Build keyword arguments for ``model.generate`` for one template.

        ``logits_to_keep=1`` matters more than it looks on a memory-constrained
        machine. Without it the prefill pass materialises logits for *every*
        prompt position: a batch of 5 prompts of 1,500 tokens against Qwen2.5's
        151,936-token vocabulary is a 4.6 GB float32 tensor, which pushes a
        laptop into swap and slows generation by more than an order of
        magnitude. Only the final position is needed to sample the next token.
        """
        budget = self.max_new_tokens or template.max_new_tokens
        kwargs: dict[str, Any] = {
            "max_new_tokens": budget,
            "do_sample": self.do_sample,
            "logits_to_keep": 1,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        return kwargs


@dataclass(frozen=True)
class Generation:
    """One raw model output.

    Attributes:
        record_id: Record the output belongs to.
        template: Template name used.
        text: Raw generated continuation, with the prompt removed.
        n_prompt_tokens: Prompt length in tokens.
        n_new_tokens: Generated length in tokens.
        seconds: Wall-clock generation time attributed to this record.
        from_cache: Whether the output was read from disk.
    """

    record_id: str
    template: str
    text: str
    n_prompt_tokens: int = 0
    n_new_tokens: int = 0
    seconds: float = 0.0
    from_cache: bool = False


def cache_key(
    model_id: str,
    template: PromptTemplate,
    record: ClinicalRecord,
    config: GenerationConfig,
    prompt_text: str,
) -> str:
    """Return the cache key for one (model, template, record, params) combination.

    Args:
        model_id: Hugging Face model identifier.
        template: Template being run.
        record: Record being coded.
        config: Decoding parameters.
        prompt_text: The fully rendered prompt, so wording changes invalidate.

    Returns:
        A 16-character hex digest.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "template": template.name,
            "record": record.record_id,
            "max_new_tokens": config.max_new_tokens or template.max_new_tokens,
            "do_sample": config.do_sample,
            "temperature": config.temperature if config.do_sample else None,
            "top_p": config.top_p if config.do_sample else None,
            "seed": config.seed if config.do_sample else None,
            "prompt_sha": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class QwenRunner:
    """Runs prompt templates over records on CPU, with caching.

    The tokenizer is loaded on first use because cache keys depend on the
    rendered chat prompt; the model weights are loaded lazily and only when a
    record is not already cached, so a fully cached study never loads them.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Path | None = None,
        config: GenerationConfig | None = None,
        device: str = "cpu",
        threads: int | None = None,
    ) -> None:
        """Initialise the runner.

        Args:
            model_id: Hugging Face model identifier.
            cache_dir: Directory for raw output JSON; defaults to ``results/raw``.
            config: Decoding parameters; defaults to greedy decoding.
            device: Torch device. The study is specified and reported on
                ``"cpu"``; other values are offered only for convenience and are
                not part of the documented method.
            threads: Intra-op thread count, or ``None`` to leave torch's default.
        """
        self.model_id = model_id
        self.config = config or GenerationConfig()
        self.device = device
        self.threads = threads
        repo_root = Path(__file__).resolve().parent.parent
        self.cache_dir = cache_dir or repo_root / "results" / "raw"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    # -- model loading ----------------------------------------------------
    def _ensure_tokenizer(self) -> Any:
        """Load just the tokenizer, which is cheap.

        Prompts cannot be rendered into the model's chat format without it, and
        cache keys depend on the rendered prompt, so the tokenizer is needed to
        decide whether the model has to be loaded at all.

        Returns:
            The tokenizer.

        Raises:
            ModelUnavailableError: If the tokenizer cannot be loaded.
        """
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Left padding is required so that a batch's generated tokens all
            # start at the end of their own prompt.
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as error:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                f"could not load the tokenizer for {self.model_id}: "
                f"{type(error).__name__}: {error}"
            ) from error
        self._tokenizer = tokenizer
        return tokenizer

    def _ensure_model(self) -> tuple[Any, Any]:
        """Load the tokenizer and model if needed.

        Returns:
            The ``(tokenizer, model)`` pair.

        Raises:
            ModelUnavailableError: If loading fails for any reason.
        """
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        tokenizer = self._ensure_tokenizer()
        try:
            import torch
            from transformers import AutoModelForCausalLM

            if self.threads:
                torch.set_num_threads(max(1, self.threads))
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch.float32, low_cpu_mem_usage=True
            )
            model.eval()
            if self.device != "cpu":
                model.to(self.device)
        except Exception as error:  # pragma: no cover - environment dependent
            raise ModelUnavailableError(
                f"could not load {self.model_id}: {type(error).__name__}: {error}"
            ) from error
        self._model = model
        return tokenizer, model

    def is_available(self) -> bool:
        """Whether the model can be loaded in this environment."""
        try:
            self._ensure_model()
        except ModelUnavailableError:
            return False
        return True

    # -- caching ----------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        """Return the on-disk path for a cache key."""
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        """Read a cached entry, returning ``None`` when absent or corrupt."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        """Write a cache entry, ignoring write failures."""
        try:
            self._cache_path(key).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:  # pragma: no cover - disk dependent
            pass

    # -- generation -------------------------------------------------------
    def _render_prompt(self, template: PromptTemplate, record: ClinicalRecord) -> str:
        """Render a record into the model's chat format.

        Falls back to a plain concatenation when the tokenizer has no chat
        template, so prompts can be inspected without the model present.
        """
        messages = template.render_messages(record.note)
        if self._tokenizer is None:
            return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return str(
            self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    def run(
        self,
        template: PromptTemplate,
        records: Sequence[ClinicalRecord],
        progress: bool = True,
    ) -> list[Generation]:
        """Run one template over records, using and filling the cache.

        Args:
            template: Template to run.
            records: Records to code.
            progress: Whether to print a one-line progress indicator.

        Returns:
            One :class:`Generation` per record, in input order.

        Raises:
            ModelUnavailableError: If uncached records remain and the model
                cannot be loaded.
        """
        results: dict[str, Generation] = {}
        pending: list[ClinicalRecord] = []

        # Cache keys depend on the rendered prompt, which depends on the
        # tokenizer's chat template, so the tokenizer is loaded up front. The
        # model weights are loaded only if something is actually uncached.
        self._ensure_tokenizer()
        for record in records:
            prompt = self._render_prompt(template, record)
            cached = self._read_cache(
                cache_key(self.model_id, template, record, self.config, prompt)
            )
            if cached is not None:
                results[record.record_id] = Generation(
                    record_id=record.record_id,
                    template=template.name,
                    text=str(cached.get("text", "")),
                    n_prompt_tokens=int(cached.get("n_prompt_tokens", 0)),
                    n_new_tokens=int(cached.get("n_new_tokens", 0)),
                    seconds=float(cached.get("seconds", 0.0)),
                    from_cache=True,
                )
            else:
                pending.append(record)

        if pending:
            for generation in self._generate_batched(template, pending, progress):
                results[generation.record_id] = generation

        return [results[record.record_id] for record in records]

    def _generate_batched(
        self,
        template: PromptTemplate,
        records: Sequence[ClinicalRecord],
        progress: bool,
    ) -> list[Generation]:
        """Generate for uncached records in batches."""
        if not records:
            return []
        import torch

        tokenizer, model = self._ensure_model()
        generate_kwargs = self.config.for_template(template)
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        outputs: list[Generation] = []
        batch_size = max(1, self.config.batch_size)

        for start in range(0, len(records), batch_size):
            batch = list(records[start : start + batch_size])
            prompts = [self._render_prompt(template, record) for record in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            if self.device != "cpu":
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
            began = time.perf_counter()
            torch.manual_seed(self.config.seed)
            with torch.no_grad():
                generated = model.generate(**encoded, **generate_kwargs)
            elapsed = time.perf_counter() - began
            prompt_length = int(encoded["input_ids"].shape[1])
            per_record = elapsed / len(batch)

            for offset, record in enumerate(batch):
                new_tokens = generated[offset][prompt_length:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                n_prompt = int(encoded["attention_mask"][offset].sum().item())
                generation = Generation(
                    record_id=record.record_id,
                    template=template.name,
                    text=text,
                    n_prompt_tokens=n_prompt,
                    n_new_tokens=int((new_tokens != tokenizer.pad_token_id).sum().item()),
                    seconds=per_record,
                    from_cache=False,
                )
                outputs.append(generation)
                self._write_cache(
                    cache_key(
                        self.model_id, template, record, self.config, prompts[offset]
                    ),
                    {
                        "model": self.model_id,
                        "template": template.name,
                        "record_id": record.record_id,
                        "config": asdict(self.config),
                        "prompt": prompts[offset],
                        "text": text,
                        "n_prompt_tokens": generation.n_prompt_tokens,
                        "n_new_tokens": generation.n_new_tokens,
                        "seconds": per_record,
                    },
                )
            if progress:
                done = min(start + batch_size, len(records))
                print(
                    f"  [{template.name}] generated {done}/{len(records)} "
                    f"({elapsed:.1f}s for {len(batch)}, prompt {prompt_length} tok)",
                    flush=True,
                )
        return outputs
