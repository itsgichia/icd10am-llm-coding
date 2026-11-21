"""Tests for the metrics and the contamination detector.

Metrics are checked on hand-built fixtures whose correct values can be computed
by hand, so a regression in the aggregation shows up as a specific wrong number
rather than as a plausible-looking table.
"""

from __future__ import annotations

import pytest

from icdprompt.data import ClinicalRecord
from icdprompt.evaluate import (
    build_predictions,
    chapter_confusion,
    evaluate_predictions,
    flag_contamination,
)
from icdprompt.infer import Generation
from icdprompt.prompts import get_template


def make_record(record_id: str, code: str, split: str = "test") -> ClinicalRecord:
    """Build a minimal record carrying a weak label."""
    return ClinicalRecord(
        record_id=record_id,
        split=split,
        note=f"Hospital course for {record_id}.",
        label_code=code,
        label_title="fixture",
        label_evidence="fixture diagnosis",
        label_matched_text="fixture",
        n_chars=32,
    )


def make_generation(record_id: str, text: str, template: str = "t") -> Generation:
    """Build a minimal raw generation."""
    return Generation(record_id=record_id, template=template, text=text, n_new_tokens=4)


class TestBuildPredictions:
    """Turning raw outputs into predictions."""

    def test_predictions_follow_record_order(self) -> None:
        template = get_template("zero_shot_minimal")
        records = [make_record("a", "J44"), make_record("b", "C79")]
        generations = [make_generation("a", "J44"), make_generation("b", "nothing here")]
        predictions = build_predictions(template, records, generations)

        assert [p.record_id for p in predictions] == ["a", "b"]
        assert predictions[0].is_correct
        assert predictions[0].status == "VALID"
        assert not predictions[1].is_correct
        assert predictions[1].status == "INVALID"
        assert predictions[1].predicted_code is None

    def test_length_mismatch_is_rejected(self) -> None:
        template = get_template("zero_shot_minimal")
        with pytest.raises(ValueError, match="must be aligned"):
            build_predictions(template, [make_record("a", "J44")], [])

    def test_misaligned_ids_are_rejected(self) -> None:
        """Silently mismatched pairs would corrupt every metric, so they raise."""
        template = get_template("zero_shot_minimal")
        with pytest.raises(ValueError, match="order must be preserved"):
            build_predictions(
                template, [make_record("a", "J44")], [make_generation("b", "J44")]
            )

    def test_chain_of_thought_extraction_takes_the_conclusion(self) -> None:
        template = get_template("cot_then_extract")
        records = [make_record("a", "J44")]
        text = "The patient may have pneumonia, chapter J.\nCODE: J44"
        predictions = build_predictions(template, records, [make_generation("a", text)])
        assert predictions[0].predicted_code == "J44"
        assert predictions[0].is_correct

    def test_newlines_are_flattened_in_the_stored_raw_output(self) -> None:
        template = get_template("zero_shot_minimal")
        predictions = build_predictions(
            template, [make_record("a", "J44")], [make_generation("a", "line one\nline two J44")]
        )
        assert "\n" not in predictions[0].raw_output


class TestMetricsOnFixtures:
    """Aggregate metrics with hand-computable values."""

    @pytest.fixture
    def fixture_predictions(self) -> list:
        """Four records: one exact hit, one chapter-only hit, one wrong, one unparsed."""
        template = get_template("zero_shot_minimal")
        records = [
            make_record("r1", "J44"),
            make_record("r2", "J18"),
            make_record("r3", "C79"),
            make_record("r4", "N17"),
        ]
        generations = [
            make_generation("r1", "J44"),
            make_generation("r2", "J44"),
            make_generation("r3", "I21"),
            make_generation("r4", "I cannot say."),
        ]
        return build_predictions(template, records, generations)

    def test_accuracy_counts_only_exact_matches(self, fixture_predictions: list) -> None:
        metrics = evaluate_predictions(
            fixture_predictions, get_template("zero_shot_minimal"), "test"
        )
        assert metrics.n == 4
        assert metrics.n_correct == 1
        assert metrics.accuracy == pytest.approx(0.25)

    def test_unparsed_output_lowers_format_validity_not_only_accuracy(
        self, fixture_predictions: list
    ) -> None:
        metrics = evaluate_predictions(
            fixture_predictions, get_template("zero_shot_minimal"), "test"
        )
        assert metrics.format_validity == pytest.approx(0.75)

    def test_chapter_agreement_is_looser_than_exact_match(
        self, fixture_predictions: list
    ) -> None:
        """r1 and r2 are both chapter J; r3 and r4 are not."""
        metrics = evaluate_predictions(
            fixture_predictions, get_template("zero_shot_minimal"), "test"
        )
        assert metrics.chapter_agreement == pytest.approx(0.5)
        assert metrics.chapter_agreement > metrics.accuracy

    def test_code_diversity_and_modal_code(self, fixture_predictions: list) -> None:
        """Three valid outputs, two distinct codes, J44 given twice of four records."""
        metrics = evaluate_predictions(
            fixture_predictions, get_template("zero_shot_minimal"), "test"
        )
        assert metrics.n_unique_codes == 2
        assert metrics.code_diversity == pytest.approx(0.5)
        assert metrics.modal_code == "J44"
        assert metrics.modal_share == pytest.approx(0.5)

    def test_collapse_onto_one_code_gives_minimal_diversity(self) -> None:
        template = get_template("zero_shot_minimal")
        records = [make_record(f"r{i}", code) for i, code in enumerate(["J44", "C79", "N17"])]
        generations = [make_generation(r.record_id, "J44") for r in records]
        metrics = evaluate_predictions(
            build_predictions(template, records, generations), template, "test"
        )
        assert metrics.code_diversity == pytest.approx(1 / 3)
        assert metrics.modal_share == pytest.approx(1.0)

    def test_mean_new_tokens_is_reported_when_generations_are_supplied(self) -> None:
        template = get_template("zero_shot_minimal")
        records = [make_record("a", "J44")]
        generations = [Generation("a", "t", "J44", n_new_tokens=7)]
        metrics = evaluate_predictions(
            build_predictions(template, records, generations), template, "test", generations
        )
        assert metrics.mean_new_tokens == pytest.approx(7.0)

    def test_empty_prediction_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty prediction set"):
            evaluate_predictions([], get_template("zero_shot_minimal"), "test")

    def test_ambiguous_outputs_are_counted(self) -> None:
        template = get_template("zero_shot_minimal")
        records = [make_record("a", "J44"), make_record("b", "C79")]
        generations = [make_generation("a", "J44 or J18"), make_generation("b", "C79")]
        metrics = evaluate_predictions(
            build_predictions(template, records, generations), template, "test"
        )
        assert metrics.n_ambiguous == 1


class TestContaminationDetector:
    """The honest finding of the study depends on this detector being right."""

    def test_a_correct_demonstrated_code_is_flagged(self) -> None:
        """few_shot_three demonstrates J44, so a correct J44 may be a copy."""
        template = get_template("few_shot_three")
        assert "J44" in template.example_codes
        records = [make_record("a", "J44")]
        predictions = build_predictions(template, records, [make_generation("a", "J44")])

        assert predictions[0].is_correct
        assert predictions[0].in_prompt_examples
        assert predictions[0].label_in_prompt_examples
        assert predictions[0].is_contaminated_correct
        assert flag_contamination(predictions) == predictions

    def test_a_correct_undemonstrated_code_is_not_flagged(self) -> None:
        """few_shot_three does not demonstrate K35, so a correct K35 is genuine."""
        template = get_template("few_shot_three")
        assert "K35" not in template.example_codes
        records = [make_record("a", "K35")]
        predictions = build_predictions(template, records, [make_generation("a", "K35")])

        assert predictions[0].is_correct
        assert not predictions[0].in_prompt_examples
        assert not predictions[0].is_contaminated_correct
        assert flag_contamination(predictions) == []

    def test_a_wrong_demonstrated_code_is_not_flagged_as_contaminated_correct(self) -> None:
        """Echoing J44 for an N17 record is an error, not inflated accuracy."""
        template = get_template("few_shot_three")
        records = [make_record("a", "N17")]
        predictions = build_predictions(template, records, [make_generation("a", "J44")])

        assert not predictions[0].is_correct
        assert predictions[0].in_prompt_examples
        assert not predictions[0].is_contaminated_correct
        assert flag_contamination(predictions) == []

    def test_zero_shot_predictions_can_never_be_contaminated(self) -> None:
        template = get_template("zero_shot_minimal")
        records = [make_record("a", "J44")]
        predictions = build_predictions(template, records, [make_generation("a", "J44")])
        assert predictions[0].is_correct
        assert not predictions[0].in_prompt_examples
        assert flag_contamination(predictions) == []

    def test_contamination_share_and_clean_accuracy(self) -> None:
        """Both correct answers are demonstrated codes, so clean accuracy is zero."""
        template = get_template("few_shot_three")
        records = [
            make_record("r1", "J44"),  # demonstrated, answered correctly
            make_record("r2", "I21"),  # demonstrated, answered correctly
            make_record("r3", "K35"),  # not demonstrated, answered wrongly
            make_record("r4", "C22"),  # not demonstrated, answered wrongly
        ]
        generations = [
            make_generation("r1", "J44"),
            make_generation("r2", "I21"),
            make_generation("r3", "J44"),
            make_generation("r4", "I21"),
        ]
        predictions = build_predictions(template, records, generations)
        metrics = evaluate_predictions(predictions, template, "test")

        assert metrics.accuracy == pytest.approx(0.5)
        assert metrics.contaminated_correct == 2
        assert metrics.contamination_share == pytest.approx(1.0)
        assert metrics.n_clean == 2
        assert metrics.clean_accuracy == pytest.approx(0.0)

    def test_clean_accuracy_equals_accuracy_without_demonstrated_labels(self) -> None:
        template = get_template("few_shot_three")
        records = [make_record("r1", "K35"), make_record("r2", "C22")]
        generations = [make_generation("r1", "K35"), make_generation("r2", "J18")]
        metrics = evaluate_predictions(
            build_predictions(template, records, generations), template, "test"
        )
        assert metrics.n_clean == 2
        assert metrics.clean_accuracy == pytest.approx(metrics.accuracy)
        assert metrics.contaminated_correct == 0

    def test_echo_rate_measures_copying_even_when_nothing_is_correct(self) -> None:
        """The failure mode this study found: every answer is a demonstrated code."""
        template = get_template("few_shot_three")
        records = [make_record("r1", "K35"), make_record("r2", "C22"), make_record("r3", "N17")]
        generations = [
            make_generation("r1", "J44"),
            make_generation("r2", "I21"),
            make_generation("r3", "N10"),
        ]
        metrics = evaluate_predictions(
            build_predictions(template, records, generations), template, "test"
        )
        assert metrics.n_correct == 0
        assert metrics.contamination_share == pytest.approx(0.0)
        assert metrics.example_echo_rate == pytest.approx(1.0)

    def test_echo_rate_ignores_unparsed_outputs(self) -> None:
        template = get_template("few_shot_three")
        records = [make_record("r1", "K35"), make_record("r2", "C22")]
        generations = [make_generation("r1", "J44"), make_generation("r2", "no code")]
        metrics = evaluate_predictions(
            build_predictions(template, records, generations), template, "test"
        )
        assert metrics.format_validity == pytest.approx(0.5)
        assert metrics.example_echo_rate == pytest.approx(1.0)

    def test_echo_rate_is_zero_for_a_zero_shot_template(self) -> None:
        template = get_template("zero_shot_minimal")
        records = [make_record("r1", "J44")]
        metrics = evaluate_predictions(
            build_predictions(template, records, [make_generation("r1", "J44")]),
            template,
            "test",
        )
        assert metrics.example_echo_rate == pytest.approx(0.0)

    def test_contamination_share_is_zero_when_nothing_is_correct(self) -> None:
        template = get_template("few_shot_three")
        records = [make_record("r1", "K35")]
        metrics = evaluate_predictions(
            build_predictions(template, records, [make_generation("r1", "no code")]),
            template,
            "test",
        )
        assert metrics.n_correct == 0
        assert metrics.contamination_share == pytest.approx(0.0)


class TestChapterConfusion:
    """The cross-tabulation behind the confusion figure."""

    def test_counts_are_tabulated_by_letter(self) -> None:
        template = get_template("zero_shot_minimal")
        records = [make_record("r1", "J44"), make_record("r2", "J18"), make_record("r3", "C79")]
        generations = [
            make_generation("r1", "J44"),
            make_generation("r2", "I21"),
            make_generation("r3", "C34"),
        ]
        table = chapter_confusion(build_predictions(template, records, generations))
        assert table == {"J": {"J": 1, "I": 1}, "C": {"C": 1}}

    def test_unextractable_predictions_are_counted_under_a_question_mark(self) -> None:
        template = get_template("zero_shot_minimal")
        records = [make_record("r1", "J44")]
        table = chapter_confusion(
            build_predictions(template, records, [make_generation("r1", "no idea")])
        )
        assert table == {"J": {"?": 1}}
