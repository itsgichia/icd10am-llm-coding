"""Tests for record serialisation and the split cache.

These run offline: they exercise the cache layer with hand-built records rather
than downloading the dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from icdprompt.data import (
    ClinicalRecord,
    ScanConfig,
    load_cached_split,
    load_splits,
    record_id_for,
    write_jsonl,
)


def make_record(index: int, split: str) -> ClinicalRecord:
    """Build a record with a stable, distinct id."""
    note = f"Hospital course number {index} for the {split} split."
    return ClinicalRecord(
        record_id=record_id_for(note),
        split=split,
        note=note,
        label_code="J44",
        label_title="Chronic obstructive pulmonary disease",
        label_evidence="COPD",
        label_matched_text="COPD",
        n_chars=len(note),
    )


def seed_cache(data_dir: Path, config: ScanConfig) -> None:
    """Write a cache consistent with ``config``."""
    write_jsonl([make_record(i, "dev") for i in range(config.dev_n)], data_dir / "dev.jsonl")
    write_jsonl([make_record(100 + i, "test") for i in range(config.test_n)], data_dir / "test.jsonl")
    (data_dir / "scan_config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")


class TestRecordIds:
    """Record ids must be content-derived and stable."""

    def test_the_same_note_always_gets_the_same_id(self) -> None:
        assert record_id_for("a note") == record_id_for("a note")

    def test_different_notes_get_different_ids(self) -> None:
        assert record_id_for("a note") != record_id_for("another note")

    def test_ids_are_twelve_hex_characters(self) -> None:
        identifier = record_id_for("a note")
        assert len(identifier) == 12
        int(identifier, 16)


class TestSerialisation:
    """JSONL round-tripping of records."""

    def test_round_trip_preserves_every_field(self, tmp_path: Path) -> None:
        records = [make_record(0, "dev"), make_record(1, "dev")]
        path = write_jsonl(records, tmp_path / "dev.jsonl")
        assert load_cached_split(path) == records

    def test_chapter_letter_is_derived_from_the_label(self) -> None:
        assert make_record(0, "dev").label_chapter == "J"

    def test_missing_cache_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cached_split(tmp_path / "absent.jsonl")


class TestSplitCache:
    """The cache must never be used under a configuration it was not built with."""

    def test_a_matching_cache_is_used_without_downloading(self, tmp_path: Path) -> None:
        config = ScanConfig(dev_n=3, test_n=5)
        seed_cache(tmp_path, config)
        splits = load_splits(config=config, data_dir=tmp_path, allow_download=False)
        assert len(splits["dev"]) == 3
        assert len(splits["test"]) == 5

    def test_splits_are_disjoint(self, tmp_path: Path) -> None:
        config = ScanConfig(dev_n=3, test_n=5)
        seed_cache(tmp_path, config)
        splits = load_splits(config=config, data_dir=tmp_path, allow_download=False)
        dev_ids = {record.record_id for record in splits["dev"]}
        test_ids = {record.record_id for record in splits["test"]}
        assert dev_ids.isdisjoint(test_ids)

    def test_a_cache_of_the_wrong_size_is_refused(self, tmp_path: Path) -> None:
        """Silently returning 3 records for a request of 8 would misreport the study."""
        seed_cache(tmp_path, ScanConfig(dev_n=3, test_n=5))
        with pytest.raises(RuntimeError, match="different configuration"):
            load_splits(
                config=ScanConfig(dev_n=8, test_n=5), data_dir=tmp_path, allow_download=False
            )

    def test_a_cache_built_with_other_settings_is_refused(self, tmp_path: Path) -> None:
        """Split sizes can match while the selection rule differs."""
        seed_cache(tmp_path, ScanConfig(dev_n=3, test_n=5, seed=1))
        with pytest.raises(RuntimeError, match="different configuration"):
            load_splits(
                config=ScanConfig(dev_n=3, test_n=5, seed=2),
                data_dir=tmp_path,
                allow_download=False,
            )

    def test_a_cache_without_its_configuration_record_is_refused(self, tmp_path: Path) -> None:
        config = ScanConfig(dev_n=3, test_n=5)
        seed_cache(tmp_path, config)
        (tmp_path / "scan_config.json").unlink()
        with pytest.raises(RuntimeError, match="different configuration"):
            load_splits(config=config, data_dir=tmp_path, allow_download=False)

    def test_an_absent_cache_is_reported_when_offline(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="no cached splits"):
            load_splits(config=ScanConfig(), data_dir=tmp_path, allow_download=False)


class TestScanConfig:
    """The configuration fingerprint."""

    def test_identical_configurations_share_a_fingerprint(self) -> None:
        assert ScanConfig().fingerprint() == ScanConfig().fingerprint()

    def test_a_different_seed_changes_the_fingerprint(self) -> None:
        assert ScanConfig(seed=1).fingerprint() != ScanConfig(seed=2).fingerprint()
