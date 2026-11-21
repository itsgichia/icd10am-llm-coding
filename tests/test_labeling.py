"""Tests for the weak-labelling rules and the diagnosis redaction.

These labels are the study's ground truth, so the guards that keep a bad label
out matter as much as the rules that produce a good one.
"""

from __future__ import annotations

import pytest

from icdprompt.labeling import (
    LABEL_RULES,
    count_diagnosis_items,
    find_diagnosis_section,
    label_rules_by_code,
    redact_diagnosis_sections,
    weak_label,
)
from icdprompt.postprocess import is_valid_category

NOTE = """Discharge Summary:

Patient Name: [Redacted]
Age: 68

Hospital Course:
The patient was admitted with worsening breathlessness and treated with
nebulised bronchodilators and oral corticosteroids.

Discharge Diagnosis: Infective exacerbation of chronic obstructive pulmonary disease

Follow-up: respiratory clinic in four weeks.
"""


class TestRuleTableIntegrity:
    """Properties the whole rule table must satisfy."""

    def test_every_rule_code_is_a_valid_category(self) -> None:
        for rule in LABEL_RULES:
            assert is_valid_category(rule.code), rule.code

    def test_codes_are_unique(self) -> None:
        codes = [rule.code for rule in LABEL_RULES]
        assert len(codes) == len(set(codes))
        assert len(label_rules_by_code()) == len(LABEL_RULES)

    def test_every_rule_has_a_title(self) -> None:
        for rule in LABEL_RULES:
            assert rule.title.strip()

    def test_the_table_spans_several_chapters(self) -> None:
        assert len({rule.code[0] for rule in LABEL_RULES}) >= 10

    def test_case_insensitive_prose_does_not_trigger_abbreviation_rules(self) -> None:
        """A lowercase English word must not be read as a clinical abbreviation."""
        for text in ["all of the results were normal", "the patient aids her mother"]:
            assert weak_label(f"Diagnosis: {text}").code is None


class TestSectionLocation:
    """Finding and redacting the stated diagnosis."""

    def test_diagnosis_body_is_found(self) -> None:
        assert "chronic obstructive pulmonary disease" in find_diagnosis_section(NOTE)

    def test_following_section_is_not_swallowed(self) -> None:
        assert "respiratory clinic" not in find_diagnosis_section(NOTE)

    @pytest.mark.parametrize("body", ["N/A", "None", "nil", "Not applicable", "See above"])
    def test_uninformative_bodies_are_ignored(self, body: str) -> None:
        assert find_diagnosis_section(f"Diagnosis: {body}\n") == ""

    def test_a_note_without_a_diagnosis_heading_yields_nothing(self) -> None:
        assert find_diagnosis_section("Hospital Course: the patient improved.") == ""

    def test_duplicate_bodies_are_not_repeated(self) -> None:
        note = "Diagnosis: Acute appendicitis\n\nFinal Diagnosis: Acute appendicitis\n"
        assert find_diagnosis_section(note) == "Acute appendicitis"


class TestRedaction:
    """Redaction is what stops the label leaking into the prompt."""

    def test_the_diagnosis_text_is_removed(self) -> None:
        redacted = redact_diagnosis_sections(NOTE)
        assert "chronic obstructive pulmonary disease" not in redacted
        assert "withheld" in redacted

    def test_the_heading_is_kept_so_the_note_still_reads_naturally(self) -> None:
        assert "Discharge Diagnosis:" in redact_diagnosis_sections(NOTE)

    def test_the_hospital_course_survives(self) -> None:
        redacted = redact_diagnosis_sections(NOTE)
        assert "nebulised bronchodilators" in redacted
        assert "worsening breathlessness" in redacted

    def test_every_diagnosis_section_is_redacted(self) -> None:
        note = "Admitting Diagnosis: Sepsis\n\nDischarge Diagnosis: Septic shock\n"
        redacted = redact_diagnosis_sections(note)
        assert "Sepsis" not in redacted
        assert "Septic shock" not in redacted

    def test_a_note_with_no_diagnosis_section_is_unchanged_apart_from_stripping(self) -> None:
        note = "Hospital Course: the patient improved."
        assert redact_diagnosis_sections(note) == note


class TestWeakLabel:
    """End-to-end labelling of a note."""

    def test_a_single_condition_is_labelled(self) -> None:
        outcome = weak_label(NOTE)
        assert outcome.code == "J44"
        assert outcome.is_labelled
        assert outcome.reason == "single rule matched"

    def test_the_evidence_is_retained_for_hand_checking(self) -> None:
        outcome = weak_label(NOTE)
        assert "chronic obstructive pulmonary disease" in outcome.evidence
        assert outcome.matched_text

    @pytest.mark.parametrize(
        "diagnosis, expected",
        [
            ("Acute appendicitis", "K35"),
            ("Septic shock", "A41"),
            ("Acute ST-elevation myocardial infarction", "I21"),
            ("Adenocarcinoma of colon", "C18"),
            ("Acute pyelonephritis", "N10"),
            ("Systemic lupus erythematosus", "M32"),
            ("Displaced femoral neck fracture", "S72"),
            ("Ulcerative colitis", "K51"),
        ],
    )
    def test_representative_diagnoses(self, diagnosis: str, expected: str) -> None:
        assert weak_label(f"Diagnosis: {diagnosis}\n").code == expected

    def test_several_matching_categories_are_declined(self) -> None:
        """Nothing in the record says which of these a coder would sequence first."""
        outcome = weak_label("Diagnosis: COVID-19 pneumonia with respiratory failure\n")
        assert outcome.code is None
        assert "ambiguous" in outcome.reason
        assert len(outcome.competing_codes) > 1

    def test_an_unmatched_diagnosis_is_declined(self) -> None:
        outcome = weak_label("Diagnosis: Killian-Jamieson diverticulum\n")
        assert outcome.code is None
        assert outcome.reason == "no rule matched"

    def test_a_missing_diagnosis_section_is_declined(self) -> None:
        outcome = weak_label("Hospital Course: the patient improved and went home.")
        assert outcome.code is None
        assert outcome.reason == "no usable diagnosis section"

    def test_a_long_problem_list_is_declined(self) -> None:
        note = (
            "Diagnosis: 1. Consolidation behind the heart 2. Anomalies of the aortic arch "
            "3. Kyphoscoliosis 4. Arterial hypertension 5. Chronic respiratory failure\n"
        )
        outcome = weak_label(note)
        assert outcome.code is None
        assert "problem list" in outcome.reason

    def test_a_short_enumeration_is_still_labelled(self) -> None:
        note = "Diagnosis: 1. Acute appendicitis 2. Postoperative ileus\n"
        assert weak_label(note, max_items=2).code is not None

    def test_the_item_limit_is_configurable(self) -> None:
        note = "Diagnosis: 1. Acute appendicitis 2. Localised peritonitis\n"
        assert weak_label(note, max_items=1).code is None
        assert weak_label(note, max_items=3).code is not None

    def test_subcutaneous_emphysema_is_not_copd(self) -> None:
        """A rule that fires on unrelated wording silently poisons the labels."""
        assert weak_label("Diagnosis: Postpartum subcutaneous emphysema\n").code != "J44"


class TestItemCounting:
    """The problem-list guard."""

    @pytest.mark.parametrize(
        "evidence, expected",
        [
            ("Acute appendicitis", 1),
            ("1. Sepsis 2. Acute kidney injury", 2),
            ("1. Sepsis 2. AKI 3. Hypertension 4. Anaemia", 4),
            ("- Sepsis - Acute kidney injury", 2),
            ("Sepsis | Acute kidney injury | Anaemia", 3),
        ],
    )
    def test_item_counts(self, evidence: str, expected: int) -> None:
        assert count_diagnosis_items(evidence) == expected

    def test_a_decimal_inside_a_value_is_not_an_item_marker(self) -> None:
        assert count_diagnosis_items("Haemoglobin of 7.2 g/dL with anaemia") == 1
