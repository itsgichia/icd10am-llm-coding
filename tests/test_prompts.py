"""Integrity tests for the prompt registry.

The registry doubles as the experimental design and as the source of the
README's strategy table, so it is checked for completeness and for the
properties the evaluation code relies on.
"""

from __future__ import annotations

import pytest

from icdprompt.postprocess import extract_code, is_valid_category
from icdprompt.prompts import TEMPLATES, PromptTemplate, get_template, template_names

ALL_TEMPLATES = list(TEMPLATES.values())


class TestRegistryShape:
    """Structural guarantees about the registry itself."""

    def test_at_least_eight_strategies_are_registered(self) -> None:
        assert len(TEMPLATES) >= 8

    def test_names_are_unique_and_match_their_keys(self) -> None:
        assert len(set(template_names())) == len(TEMPLATES)
        for name, template in TEMPLATES.items():
            assert template.name == name

    def test_every_required_family_is_covered(self) -> None:
        required = {
            "zero-shot",
            "format-constrained",
            "persona",
            "few-shot",
            "chain-of-thought",
            "context-rich",
            "structured-output",
            "hybrid",
        }
        assert required <= {template.family for template in ALL_TEMPLATES}

    def test_get_template_round_trips(self) -> None:
        for name in template_names():
            assert get_template(name).name == name

    def test_unknown_name_raises_with_the_valid_names_listed(self) -> None:
        with pytest.raises(KeyError) as error:
            get_template("not_a_template")
        assert "zero_shot_minimal" in str(error.value)


class TestTemplateFields:
    """Field-level requirements on every registered template."""

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_rationale_is_substantive(self, template: PromptTemplate) -> None:
        """Every strategy must state, in advance, why it might work."""
        assert len(template.rationale.split()) >= 12

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_instruction_mentions_the_classification(self, template: PromptTemplate) -> None:
        assert "ICD-10-AM" in template.instruction or "ICD-10-AM" in (template.system or "")

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_token_budget_is_positive_and_bounded(self, template: PromptTemplate) -> None:
        assert 0 < template.max_new_tokens <= 512

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_prefer_is_a_known_value(self, template: PromptTemplate) -> None:
        assert template.prefer in {"first", "last"}


class TestRendering:
    """Rendering must place the note and keep the output contract visible."""

    NOTE = "Hospital Course: the patient was treated for an exacerbation of airflow limitation."

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_note_appears_exactly_once(self, template: PromptTemplate) -> None:
        rendered = template.render_user(self.NOTE)
        assert rendered.count(self.NOTE) == 1

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_no_unfilled_placeholders_remain(self, template: PromptTemplate) -> None:
        rendered = template.render_user(self.NOTE)
        assert "{note}" not in rendered

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_messages_end_with_the_user_turn(self, template: PromptTemplate) -> None:
        messages = template.render_messages(self.NOTE)
        assert messages[-1]["role"] == "user"
        assert all(message["content"].strip() for message in messages)
        roles = [message["role"] for message in messages]
        assert roles in (["user"], ["system", "user"])

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_examples_are_rendered_when_present(self, template: PromptTemplate) -> None:
        rendered = template.render_user(self.NOTE)
        if template.n_examples:
            assert f"Example {template.n_examples}" in rendered
        else:
            assert "Example 1" not in rendered


class TestExampleCodes:
    """Example codes drive the contamination check, so they must be sound."""

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_example_codes_are_well_formed(self, template: PromptTemplate) -> None:
        for code in template.example_codes:
            assert is_valid_category(code)

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_example_codes_are_distinct(self, template: PromptTemplate) -> None:
        """Duplicate demonstrated codes would understate measured contamination."""
        codes = [example.code for example in template.examples]
        assert len(codes) == len(set(codes))

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_rendered_examples_contain_their_codes(self, template: PromptTemplate) -> None:
        rendered = template.render_user("note text")
        for code in template.example_codes:
            assert code in rendered

    def test_zero_shot_templates_demonstrate_no_codes(self) -> None:
        """A zero-shot template cannot be contaminated, by definition."""
        for name in ("zero_shot_minimal", "zero_shot_strict_format", "persona_clinical_coder"):
            assert get_template(name).example_codes == ()

    def test_few_shot_six_extends_few_shot_three(self) -> None:
        """The two few-shot sizes must be nested so the comparison is clean."""
        three = get_template("few_shot_three").example_codes
        six = get_template("few_shot_six").example_codes
        assert six[: len(three)] == three
        assert len(six) > len(three)

    @pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
    def test_each_demonstrated_answer_survives_extraction(
        self, template: PromptTemplate
    ) -> None:
        """The examples model the output format, so they must parse as that format."""
        for example in template.examples:
            if template.show_reasoning and example.reasoning:
                shown = f"{example.reasoning}\nCODE: {example.code}"
            else:
                shown = example.code
            assert extract_code(shown, prefer=template.prefer).code == example.code


class TestChainOfThought:
    """The reasoning template has requirements the others do not."""

    def test_it_prefers_the_last_code_and_has_room_to_reason(self) -> None:
        template = get_template("cot_then_extract")
        assert template.prefer == "last"
        assert template.max_new_tokens >= 64
        assert template.show_reasoning

    def test_its_examples_display_reasoning(self) -> None:
        template = get_template("cot_then_extract")
        rendered = template.render_user("note")
        assert template.examples
        for example in template.examples:
            assert example.reasoning
            assert example.reasoning in rendered
