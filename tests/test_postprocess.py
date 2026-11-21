"""Tests for code extraction.

Extraction is the step that makes or breaks the pipeline: a wrong answer and an
unparsed answer look identical in the accuracy column unless the extractor is
right, so these tests cover each failure mode the model actually produces.
"""

from __future__ import annotations

import pytest

from icdprompt.postprocess import extract_code, is_valid_category, normalise_candidate


class TestWellFormedCodes:
    """Outputs that are already a bare category."""

    @pytest.mark.parametrize("text", ["J44", "C79", "I21", "A41", "U07", "Z51"])
    def test_bare_code_is_returned_unchanged(self, text: str) -> None:
        result = extract_code(text)
        assert result.code == text
        assert result.status == "VALID"
        assert result.method == "bare"

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("  J44  ", "J44"),
            ("`J44`", "J44"),
            ('"J44"', "J44"),
            ("**J44**", "J44"),
            ("J44.", "J44"),
            ("J 44", "J44"),
            ("J-44", "J44"),
        ],
    )
    def test_surrounding_punctuation_is_stripped(self, text: str, expected: str) -> None:
        assert extract_code(text).code == expected


class TestCodesInsideSentences:
    """Outputs that bury the code in prose."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("The principal diagnosis code is J44 for this admission.", "J44"),
            ("I would assign C79 here.", "C79"),
            ("Based on the hospital course, the answer is N17.", "N17"),
            ("Principal diagnosis: acute appendicitis (K35).", "K35"),
            ("This patient should be coded as I21, acute myocardial infarction.", "I21"),
            ("CODE: M05", "M05"),
            ("Final answer: E11", "E11"),
            ("ICD-10-AM category: J18", "J18"),
        ],
    )
    def test_code_is_recovered_from_prose(self, text: str, expected: str) -> None:
        result = extract_code(text)
        assert result.code == expected
        assert result.status == "VALID"

    def test_icd_10_am_is_not_read_as_a_code(self) -> None:
        """'ICD-10-AM' must not yield 'D10'; it is the name of the classification."""
        result = extract_code("The ICD-10-AM classification is used in Australia.")
        assert result.code is None
        assert result.status == "INVALID"

    def test_covid_19_is_not_read_as_a_code(self) -> None:
        result = extract_code("The patient had COVID-19 pneumonia.")
        assert result.code is None

    def test_vitamin_b12_is_not_read_as_a_code(self) -> None:
        result = extract_code("Investigations showed a low vitamin B12 level.")
        assert result.code is None

    def test_an_age_is_not_read_as_a_code(self) -> None:
        result = extract_code("The patient is a 72 year old man.")
        assert result.code is None


class TestJsonWrappers:
    """Outputs produced by the JSON-output template."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ('{"code": "J44", "condition": "COPD"}', "J44"),
            ('{"code":"c79"}', "C79"),
            ('{"icd10am_code": "K35", "condition": "appendicitis"}', "K35"),
            ('{"condition": "sepsis", "code": "A41"}', "A41"),
            ("```json\n{\"code\": \"N10\"}\n```", "N10"),
            ("Here is the result: {\"code\": \"I50\"} -- hope that helps.", "I50"),
            ("{'code': 'M86'}", "M86"),
            ('{"code": "G40",}', "G40"),
            ('{"answer": "J45"}', "J45"),
        ],
    )
    def test_json_value_is_preferred(self, text: str, expected: str) -> None:
        result = extract_code(text)
        assert result.code == expected
        assert result.status == "VALID"

    def test_json_code_wins_over_an_earlier_prose_code(self) -> None:
        """The structured field is authoritative even if prose mentions another code."""
        text = 'It could be J18, but the structured answer is {"code": "J44"}'
        assert extract_code(text).code == "J44"

    def test_unterminated_json_falls_back_to_the_regex_scan(self) -> None:
        result = extract_code('{"code": "J44", "condition": "chronic obstruct')
        assert result.code == "J44"
        assert result.status == "VALID"

    def test_json_with_a_placeholder_is_invalid(self) -> None:
        result = extract_code('{"code": "<letter><digit><digit>"}')
        assert result.code is None
        assert result.status == "INVALID"


class TestNoCodePresent:
    """Outputs with nothing extractable, which must be reported as INVALID."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "\n\n",
            "I cannot determine the diagnosis code from this summary.",
            "The patient has chronic obstructive pulmonary disease.",
            "Sorry, I am unable to help with medical coding.",
            "Principal diagnosis: community-acquired pneumonia",
        ],
    )
    def test_missing_code_is_invalid(self, text: str) -> None:
        result = extract_code(text)
        assert result.code is None
        assert result.status == "INVALID"
        assert result.method == "none"
        assert not result.is_valid

    def test_none_input_is_invalid(self) -> None:
        result = extract_code(None)
        assert result.code is None
        assert result.status == "INVALID"


class TestMultipleCodes:
    """Outputs that hedge between several codes."""

    def test_first_candidate_is_taken_by_default(self) -> None:
        result = extract_code("This could be J44 or J18 or possibly J96.")
        assert result.code == "J44"
        assert result.candidates == ("J44", "J18", "J96")
        assert result.is_ambiguous

    def test_last_candidate_is_taken_when_requested(self) -> None:
        """Chain-of-thought templates conclude after reasoning, so prefer the last."""
        text = "Could be J18. Consolidation is chronic, so J44 fits better."
        assert extract_code(text, prefer="last").code == "J44"
        assert extract_code(text, prefer="first").code == "J18"

    def test_an_answer_cue_beats_the_position_rule_in_prose(self) -> None:
        """An explicit cue is authoritative, so prefer= does not change the result."""
        text = "Could be J18, but the answer is J44"
        assert extract_code(text, prefer="first").code == "J44"
        assert extract_code(text, prefer="last").code == "J44"

    def test_an_answer_cue_overrides_the_position_rule(self) -> None:
        text = "Possibly J18 or N17 or I50.\nCODE: K35"
        assert extract_code(text, prefer="first").code == "K35"
        assert extract_code(text, prefer="last").code == "K35"

    def test_the_last_cue_wins_when_the_model_restates_itself(self) -> None:
        text = "CODE: J18\nOn reflection that is wrong.\nCODE: J44"
        assert extract_code(text).code == "J44"

    def test_repeating_one_code_is_not_ambiguous(self) -> None:
        result = extract_code("J44. The answer is J44, that is, J44.")
        assert result.code == "J44"
        assert not result.is_ambiguous


class TestLowercase:
    """Models frequently drop the capital."""

    @pytest.mark.parametrize(
        "text, expected",
        [("j44", "J44"), ("c79", "C79"), ("code: n17", "N17"), ("the answer is i21.", "I21")],
    )
    def test_lowercase_is_normalised(self, text: str, expected: str) -> None:
        assert extract_code(text).code == expected


class TestLongerCodesAreTruncated:
    """Four- and five-character codes are truncated to the three-character category."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("J44.1", "J44"),
            ("J440", "J44"),
            ("C79.5", "C79"),
            ("M85.80", "M85"),
            ("S72.001", "S72"),
            ("The code is J44.19 for this episode.", "J44"),
            ('{"code": "K35.80"}', "K35"),
            ("CODE: I21.4", "I21"),
        ],
    )
    def test_subdivisions_are_dropped(self, text: str, expected: str) -> None:
        result = extract_code(text)
        assert result.code == expected
        assert result.status == "VALID"
        assert len(result.code or "") == 3


class TestHelpers:
    """The small validity and normalisation helpers."""

    @pytest.mark.parametrize("code", ["J44", "A00", "Z99", "U07"])
    def test_valid_shapes(self, code: str) -> None:
        assert is_valid_category(code)

    @pytest.mark.parametrize("code", ["", None, "J4", "J444", "j44", "44J", "J4A", "JJ4"])
    def test_invalid_shapes(self, code: str | None) -> None:
        assert not is_valid_category(code)

    def test_normalise_uppercases_and_truncates(self) -> None:
        assert normalise_candidate("j", "4419") == "J44"

    def test_chapter_letter_is_exposed(self) -> None:
        assert extract_code("J44").chapter_letter == "J"
        assert extract_code("no code here").chapter_letter is None
