"""Metrics for prompt strategies, including a few-shot contamination check.

Seven quantities are reported per template:

``accuracy``
    Exact agreement between the predicted three-character category and the weak
    label.  Records where no code could be extracted count as wrong.
``format_validity``
    Share of records from which a well-formed category could be extracted at
    all.  A template can score high here and zero on accuracy.
``chapter_agreement``
    Share of records where the first character matches.  This is a much weaker
    bar than exact match and is reported separately because a model can know the
    body system without knowing the category.
``code_diversity``
    Distinct predicted codes divided by the number of records.  A value near
    zero means the model collapsed onto one or two favourite codes, which is the
    characteristic failure mode of a small model on a large label space.
``example_echo_rate``
    Of the well-formed predictions, the share whose code the prompt's own
    few-shot block displayed.  This is defined even when nothing is correct, so
    it measures prompt-echoing in the regime where the contamination share
    cannot be computed.
``contaminated_correct``
    Of the correct predictions, how many are for a code that the prompt's own
    few-shot examples displayed.  Those predictions may be copying rather than
    coding.
``clean_accuracy``
    Accuracy over only the records whose label is *not* one of the template's
    demonstrated codes, which is the fair comparison against a zero-shot
    template.

The contamination check is the point of the study.  A few-shot template that
beats a zero-shot one only on its own demonstrated codes has not learned to
code; it has learned to guess from a shortlist.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .data import ClinicalRecord
from .infer import Generation
from .postprocess import extract_code
from .prompts import PromptTemplate

__all__ = [
    "Prediction",
    "TemplateMetrics",
    "build_predictions",
    "chapter_confusion",
    "evaluate_predictions",
    "flag_contamination",
]


@dataclass(frozen=True)
class Prediction:
    """One template's prediction for one record.

    Attributes:
        record_id: Record identifier.
        template: Template name.
        split: Split the record came from.
        true_code: Weak label.
        predicted_code: Extracted category, or ``None``.
        status: ``"VALID"`` or ``"INVALID"`` from extraction.
        method: Which extraction layer produced the code.
        n_candidates: How many distinct category-shaped tokens the output held.
        in_prompt_examples: Whether the predicted code appears among the
            template's few-shot example codes.
        label_in_prompt_examples: Whether the *true* code appears among them.
        raw_output: The raw generated text, truncated for the results file.
    """

    record_id: str
    template: str
    split: str
    true_code: str
    predicted_code: str | None
    status: str
    method: str
    n_candidates: int
    in_prompt_examples: bool
    label_in_prompt_examples: bool
    raw_output: str

    @property
    def is_correct(self) -> bool:
        """Whether the prediction exactly matches the weak label."""
        return self.predicted_code is not None and self.predicted_code == self.true_code

    @property
    def chapter_correct(self) -> bool:
        """Whether the predicted chapter letter matches the label's."""
        return self.predicted_code is not None and self.predicted_code[0] == self.true_code[0]

    @property
    def is_contaminated_correct(self) -> bool:
        """Whether this is a correct prediction of a demonstrated code.

        These are the predictions that the model could have produced by copying
        a code out of its own prompt, so they are reported separately.
        """
        return self.is_correct and self.in_prompt_examples

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON- and CSV-friendly representation."""
        payload = asdict(self)
        payload["is_correct"] = self.is_correct
        payload["chapter_correct"] = self.chapter_correct
        payload["contaminated_correct"] = self.is_contaminated_correct
        return payload


@dataclass(frozen=True)
class TemplateMetrics:
    """Aggregate metrics for one template on one split.

    Attributes:
        template: Template name.
        family: Strategy family from the registry.
        split: Split evaluated.
        n: Number of records.
        n_correct: Exact matches.
        accuracy: ``n_correct / n``.
        format_validity: Share of records yielding a well-formed category.
        chapter_agreement: Share of records with a matching chapter letter.
        code_diversity: Distinct predicted codes divided by ``n``.
        n_unique_codes: Distinct predicted codes.
        modal_code: Most frequently predicted code, or ``None``.
        modal_share: Share of records given the modal code.
        n_example_codes: How many codes the template demonstrates.
        example_echo_rate: Of the records with a well-formed prediction, the
            share whose code is one the prompt displayed. Unlike
            ``contamination_share`` this is defined even when nothing is
            correct, so it still measures prompt-echoing when accuracy is zero.
        contaminated_correct: Correct predictions of a demonstrated code.
        contamination_share: ``contaminated_correct / n_correct``.
        n_clean: Records whose label is not a demonstrated code.
        clean_accuracy: Accuracy restricted to those records.
        n_ambiguous: Outputs offering more than one category.
        mean_new_tokens: Mean generated length, when generations were supplied.
    """

    template: str
    family: str
    split: str
    n: int
    n_correct: int
    accuracy: float
    format_validity: float
    chapter_agreement: float
    code_diversity: float
    n_unique_codes: int
    modal_code: str | None
    modal_share: float
    n_example_codes: int
    example_echo_rate: float
    contaminated_correct: int
    contamination_share: float
    n_clean: int
    clean_accuracy: float
    n_ambiguous: int
    mean_new_tokens: float = 0.0
    notes: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        """Return a flat representation for CSV output."""
        return asdict(self)


def build_predictions(
    template: PromptTemplate,
    records: Sequence[ClinicalRecord],
    generations: Sequence[Generation],
    raw_chars: int = 400,
) -> list[Prediction]:
    """Post-process raw generations into predictions.

    Args:
        template: Template that produced the generations, used for its
            extraction preference and its few-shot example codes.
        records: Records in the same order as ``generations``.
        generations: Raw model outputs.
        raw_chars: How much raw text to retain on each prediction.

    Returns:
        One :class:`Prediction` per record.

    Raises:
        ValueError: If ``records`` and ``generations`` disagree in length or
            order.
    """
    if len(records) != len(generations):
        raise ValueError(
            f"{len(records)} records but {len(generations)} generations; "
            "they must be aligned"
        )

    example_codes = set(template.example_codes)
    predictions: list[Prediction] = []
    for record, generation in zip(records, generations, strict=True):
        if record.record_id != generation.record_id:
            raise ValueError(
                f"generation for {generation.record_id!r} does not match "
                f"record {record.record_id!r}; order must be preserved"
            )
        result = extract_code(generation.text, prefer=template.prefer)
        predictions.append(
            Prediction(
                record_id=record.record_id,
                template=template.name,
                split=record.split,
                true_code=record.label_code,
                predicted_code=result.code,
                status=result.status,
                method=result.method,
                n_candidates=len(set(result.candidates)),
                in_prompt_examples=result.code in example_codes if result.code else False,
                label_in_prompt_examples=record.label_code in example_codes,
                raw_output=generation.text[:raw_chars].replace("\n", " \\n "),
            )
        )
    return predictions


def flag_contamination(predictions: Iterable[Prediction]) -> list[Prediction]:
    """Return the correct predictions whose code the prompt itself displayed.

    A non-empty result means that part of this template's accuracy could be the
    model echoing a code from its own few-shot block rather than classifying the
    note.

    Args:
        predictions: Predictions to inspect.

    Returns:
        The subset flagged as contaminated-correct.
    """
    return [prediction for prediction in predictions if prediction.is_contaminated_correct]


def evaluate_predictions(
    predictions: Sequence[Prediction],
    template: PromptTemplate,
    split: str,
    generations: Sequence[Generation] | None = None,
    notes: str = "",
) -> TemplateMetrics:
    """Aggregate predictions into :class:`TemplateMetrics`.

    Args:
        predictions: Predictions for a single template and split.
        template: The template evaluated, for its name and family.
        split: Split label to record.
        generations: Optional raw generations, for the mean output length.
        notes: Free-text note carried into the summary table.

    Returns:
        The aggregate metrics.

    Raises:
        ValueError: If ``predictions`` is empty.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("cannot evaluate an empty prediction set")

    n_correct = sum(prediction.is_correct for prediction in predictions)
    n_valid = sum(prediction.status == "VALID" for prediction in predictions)
    n_chapter = sum(prediction.chapter_correct for prediction in predictions)
    predicted = [p.predicted_code for p in predictions if p.predicted_code is not None]
    counts = Counter(predicted)
    modal_code, modal_count = counts.most_common(1)[0] if counts else (None, 0)

    # Share of well-formed predictions that simply repeat a demonstrated code.
    # This stays meaningful when accuracy is zero, which is when the
    # contamination share goes undefined.
    n_echoed = sum(p.in_prompt_examples for p in predictions if p.status == "VALID")

    contaminated = flag_contamination(predictions)
    clean = [p for p in predictions if not p.label_in_prompt_examples]
    n_clean = len(clean)
    clean_correct = sum(p.is_correct for p in clean)

    mean_new_tokens = 0.0
    if generations:
        mean_new_tokens = sum(g.n_new_tokens for g in generations) / len(generations)

    return TemplateMetrics(
        template=template.name,
        family=template.family,
        split=split,
        n=n,
        n_correct=n_correct,
        accuracy=n_correct / n,
        format_validity=n_valid / n,
        chapter_agreement=n_chapter / n,
        code_diversity=len(counts) / n,
        n_unique_codes=len(counts),
        modal_code=modal_code,
        modal_share=modal_count / n,
        n_example_codes=len(template.example_codes),
        example_echo_rate=n_echoed / n_valid if n_valid else 0.0,
        contaminated_correct=len(contaminated),
        contamination_share=len(contaminated) / n_correct if n_correct else 0.0,
        n_clean=n_clean,
        clean_accuracy=clean_correct / n_clean if n_clean else 0.0,
        n_ambiguous=sum(p.n_candidates > 1 for p in predictions),
        mean_new_tokens=mean_new_tokens,
        notes=notes,
    )


def chapter_confusion(predictions: Sequence[Prediction]) -> dict[str, dict[str, int]]:
    """Cross-tabulate true against predicted chapter letters.

    Args:
        predictions: Predictions to tabulate.

    Returns:
        A nested mapping ``{true_letter: {predicted_letter: count}}``, where an
        unextractable prediction is counted under the key ``"?"``.
    """
    table: dict[str, dict[str, int]] = {}
    for prediction in predictions:
        true_letter = prediction.true_code[0]
        predicted_letter = prediction.predicted_code[0] if prediction.predicted_code else "?"
        table.setdefault(true_letter, {}).setdefault(predicted_letter, 0)
        table[true_letter][predicted_letter] += 1
    return table
