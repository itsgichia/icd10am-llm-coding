"""Loading, filtering and seeded splitting of the evaluation records.

Source dataset: ``starmpcc/Asclepius-Synthetic-Clinical-Notes`` on the Hugging
Face Hub.  The notes are synthetic discharge summaries, so no identifiable
patient data enters this repository.

Selection rule, which is fully determined by ``ScanConfig`` and the seed:

1. Read the first ``scan_rows`` rows of the ``train`` split in dataset order.
2. Deduplicate on the note text; the dataset repeats each note across its eight
   downstream tasks.
3. Keep notes whose stated diagnosis section yields exactly one weak label (see
   :mod:`icdprompt.labeling`).
4. Keep notes whose redacted text length falls inside
   ``[min_chars, max_chars]``, so prompts fit a small model's context without
   being too short to reason about.
5. Shuffle the eligible pool with ``random.Random(seed)`` and take the first
   ``dev_n`` records as the development split and the next ``test_n`` as the
   held-out test split.  The two are disjoint by construction.

Nothing here hardcodes a record identifier.  Record ids are the first twelve hex
digits of the SHA-256 of the note text, which makes them stable across runs and
independent of dataset row order.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .labeling import LabelOutcome, redact_diagnosis_sections, weak_label

__all__ = [
    "DATASET_ID",
    "ClinicalRecord",
    "ScanConfig",
    "build_splits",
    "load_cached_split",
    "load_splits",
    "write_jsonl",
]

#: Hugging Face dataset identifier for the synthetic clinical notes.
DATASET_ID: Final[str] = "starmpcc/Asclepius-Synthetic-Clinical-Notes"


@dataclass(frozen=True)
class ScanConfig:
    """Parameters that fully determine which records are selected.

    Attributes:
        seed: Seed for the shuffle that assigns records to splits.
        dev_n: Number of development records.
        test_n: Number of held-out test records.
        scan_rows: How many dataset rows to read before filtering.
        min_chars: Minimum length of the redacted note.
        max_chars: Maximum length of the redacted note.
        split: Dataset split to read.
    """

    seed: int = 20260321
    dev_n: int = 10
    test_n: int = 10
    scan_rows: int = 40_000
    min_chars: int = 600
    max_chars: int = 2_000
    split: str = "train"

    def fingerprint(self) -> str:
        """Short hash of the configuration, for cache invalidation."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class ClinicalRecord:
    """One evaluation record.

    Attributes:
        record_id: Stable twelve-character hash of the source note.
        split: ``"dev"`` or ``"test"``.
        note: Hospital-course text with stated diagnosis sections redacted.
            This is what the model sees.
        label_code: Weak three-character ICD-10-AM category label.
        label_title: Plain-English title of that category.
        label_evidence: The stated diagnosis text the label was derived from,
            retained so every label can be checked by hand.
        label_matched_text: The exact substring the labelling rule matched.
        n_chars: Length of ``note``.
    """

    record_id: str
    split: str
    note: str
    label_code: str
    label_title: str
    label_evidence: str
    label_matched_text: str
    n_chars: int

    @property
    def label_chapter(self) -> str:
        """Chapter letter of the weak label."""
        return self.label_code[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClinicalRecord:
        """Rebuild a record from its serialised form."""
        fields = {key: payload[key] for key in cls.__dataclass_fields__}
        return cls(**fields)


def record_id_for(note: str) -> str:
    """Return the stable record id for a note."""
    return hashlib.sha256(note.encode("utf-8")).hexdigest()[:12]


def write_jsonl(records: Sequence[ClinicalRecord], path: Path) -> Path:
    """Write records to ``path`` as JSON Lines.

    Args:
        records: Records to serialise.
        path: Destination file; parent directories are created.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_cached_split(path: Path) -> list[ClinicalRecord]:
    """Read a cached split from a JSON Lines file.

    Args:
        path: File written by :func:`write_jsonl`.

    Returns:
        The records in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    with path.open("r", encoding="utf-8") as handle:
        return [ClinicalRecord.from_dict(json.loads(line)) for line in handle if line.strip()]


def _iter_unique_notes(config: ScanConfig) -> Iterator[str]:
    """Yield unique note texts from the dataset, in dataset order.

    Raises:
        RuntimeError: If ``datasets`` is unavailable or the download fails.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the 'datasets' package is required to build splits; "
            "install requirements.txt or use the cached JSONL files in data/"
        ) from error

    try:
        dataset = load_dataset(DATASET_ID, split=config.split)
    except Exception as error:  # pragma: no cover - network dependent
        raise RuntimeError(f"could not load {DATASET_ID}: {error}") from error

    seen: set[str] = set()
    limit = min(config.scan_rows, len(dataset))
    for row in dataset.select(range(limit)):
        note = str(row["note"])
        if note in seen:
            continue
        seen.add(note)
        yield note


def build_splits(
    config: ScanConfig | None = None,
    data_dir: Path | None = None,
    write_cache: bool = True,
) -> dict[str, list[ClinicalRecord]]:
    """Build the dev and test splits from the source dataset.

    Args:
        config: Selection parameters; defaults to :class:`ScanConfig`.
        data_dir: Directory for the JSONL cache; defaults to ``data/``.
        write_cache: Whether to write the splits and the label audit file.

    Returns:
        Mapping with keys ``"dev"`` and ``"test"``.

    Raises:
        RuntimeError: If the dataset cannot be loaded, or if too few eligible
            records were found for the requested split sizes.
    """
    config = config or ScanConfig()
    data_dir = data_dir or Path(__file__).resolve().parent.parent / "data"

    eligible: list[tuple[str, str, LabelOutcome]] = []
    for note in _iter_unique_notes(config):
        outcome = weak_label(note)
        if not outcome.is_labelled:
            continue
        redacted = redact_diagnosis_sections(note)
        if not config.min_chars <= len(redacted) <= config.max_chars:
            continue
        eligible.append((record_id_for(note), redacted, outcome))

    needed = config.dev_n + config.test_n
    if len(eligible) < needed:
        raise RuntimeError(
            f"only {len(eligible)} eligible records found but {needed} requested; "
            "raise ScanConfig.scan_rows or lower the split sizes"
        )

    # Deduplicate on record id, then shuffle deterministically.
    unique: dict[str, tuple[str, str, LabelOutcome]] = {item[0]: item for item in eligible}
    pool = sorted(unique.values(), key=lambda item: item[0])
    random.Random(config.seed).shuffle(pool)

    splits: dict[str, list[ClinicalRecord]] = {"dev": [], "test": []}
    for index, (record_id, redacted, outcome) in enumerate(pool[:needed]):
        split = "dev" if index < config.dev_n else "test"
        splits[split].append(
            ClinicalRecord(
                record_id=record_id,
                split=split,
                note=redacted,
                label_code=str(outcome.code),
                label_title=str(outcome.title),
                label_evidence=outcome.evidence,
                label_matched_text=outcome.matched_text,
                n_chars=len(redacted),
            )
        )

    if write_cache:
        write_jsonl(splits["dev"], data_dir / "dev.jsonl")
        write_jsonl(splits["test"], data_dir / "test.jsonl")
        _write_label_audit(splits, data_dir / "labels.jsonl")
        (data_dir / "scan_config.json").write_text(
            json.dumps({**asdict(config), "eligible_pool": len(pool)}, indent=2) + "\n",
            encoding="utf-8",
        )
    return splits


def _write_label_audit(splits: dict[str, list[ClinicalRecord]], path: Path) -> Path:
    """Write a compact, hand-checkable file of every weak label and its evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for split in ("dev", "test"):
            for record in splits[split]:
                handle.write(
                    json.dumps(
                        {
                            "record_id": record.record_id,
                            "split": record.split,
                            "label_code": record.label_code,
                            "label_title": record.label_title,
                            "matched_text": record.label_matched_text,
                            "stated_diagnosis": record.label_evidence,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return path


def load_splits(
    config: ScanConfig | None = None,
    data_dir: Path | None = None,
    allow_download: bool = True,
) -> dict[str, list[ClinicalRecord]]:
    """Load the splits, preferring the local cache so re-runs work offline.

    Args:
        config: Selection parameters; defaults to :class:`ScanConfig`.
        data_dir: Directory holding ``dev.jsonl`` and ``test.jsonl``.
        allow_download: Whether to fall back to building from the Hub.

    Returns:
        Mapping with keys ``"dev"`` and ``"test"``.

    Raises:
        RuntimeError: If no cache exists and downloading is disabled or fails.
    """
    config = config or ScanConfig()
    data_dir = data_dir or Path(__file__).resolve().parent.parent / "data"
    dev_path, test_path = data_dir / "dev.jsonl", data_dir / "test.jsonl"

    if dev_path.exists() and test_path.exists():
        cached = {"dev": load_cached_split(dev_path), "test": load_cached_split(test_path)}
        if _cache_matches(config, data_dir, cached):
            return cached
        if not allow_download:
            raise RuntimeError(
                f"cached splits in {data_dir} were built with a different configuration "
                "and cannot be rebuilt while downloading is disabled; delete them or "
                "pass the configuration they were built with"
            )
        # Silently returning a cache built under different settings would make
        # the recorded configuration a lie, so rebuild instead.
        return build_splits(config=config, data_dir=data_dir)

    if not allow_download:
        raise RuntimeError(f"no cached splits in {data_dir} and downloading is disabled")
    return build_splits(config=config, data_dir=data_dir)


def _cache_matches(
    config: ScanConfig, data_dir: Path, cached: dict[str, list[ClinicalRecord]]
) -> bool:
    """Whether a cached split was built with ``config``.

    Args:
        config: Requested selection parameters.
        data_dir: Directory holding the cache and ``scan_config.json``.
        cached: The splits read from disk.

    Returns:
        ``True`` when the recorded configuration and the split sizes both match.
    """
    if len(cached["dev"]) != config.dev_n or len(cached["test"]) != config.test_n:
        return False
    record_path = data_dir / "scan_config.json"
    if not record_path.exists():
        return False
    try:
        recorded = json.loads(record_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return all(recorded.get(key) == value for key, value in asdict(config).items())
