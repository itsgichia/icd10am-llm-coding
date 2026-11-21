"""Prompt strategies for three-character ICD-10-AM category assignment.

Each strategy is a :class:`PromptTemplate` carrying its own rationale, so the
registry doubles as the experimental design: the table in the README is
generated from these objects rather than written separately.

Few-shot examples are short clinical vignettes written by the author for this
study.  They are *not* drawn from the Asclepius dataset, which keeps the
evaluation records unseen, and their codes are recorded on the template so
:mod:`icdprompt.evaluate` can test whether apparent accuracy is really the model
copying a code it was just shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from .postprocess import Preference

__all__ = [
    "FewShotExample",
    "PromptTemplate",
    "TEMPLATES",
    "get_template",
    "template_names",
]

Family = Literal[
    "zero-shot",
    "format-constrained",
    "persona",
    "few-shot",
    "chain-of-thought",
    "context-rich",
    "structured-output",
    "hybrid",
]

@dataclass(frozen=True)
class FewShotExample:
    """One worked example shown to the model.

    Attributes:
        vignette: Short hospital-course summary written for this study.
        code: The three-character category the example teaches.
        reasoning: Optional one-line justification, used by chain-of-thought
            templates to demonstrate the reasoning format.
    """

    vignette: str
    code: str
    reasoning: str = ""


# Author-written vignettes.  Kept deliberately terse so that few-shot prompts
# stay inside a 0.5B model's usable context alongside a real discharge summary.
_EXAMPLES: Final[tuple[FewShotExample, ...]] = (
    FewShotExample(
        vignette=(
            "A 72-year-old man with a 50 pack-year smoking history was admitted with "
            "three days of worsening breathlessness and purulent sputum. Spirometry "
            "showed an FEV1/FVC ratio of 0.52. He was treated with nebulised "
            "bronchodilators, oral prednisolone and antibiotics for an infective "
            "exacerbation of his long-standing airflow limitation."
        ),
        code="J44",
        reasoning=(
            "Irreversible airflow limitation in a heavy smoker with an infective "
            "exacerbation is chronic obstructive pulmonary disease, chapter J, "
            "category J44."
        ),
    ),
    FewShotExample(
        vignette=(
            "A 58-year-old man presented with crushing central chest pain radiating to "
            "the jaw. The electrocardiogram showed ST elevation in the anterior leads "
            "and high-sensitivity troponin rose from 40 to 2100 ng/L. He underwent "
            "primary percutaneous coronary intervention with a stent to the proximal "
            "left anterior descending artery."
        ),
        code="I21",
        reasoning=(
            "ST elevation with a rising troponin and culprit-vessel stenting is an "
            "acute myocardial infarction, chapter I, category I21."
        ),
    ),
    FewShotExample(
        vignette=(
            "A 34-year-old woman presented with fever, rigors, dysuria and right flank "
            "tenderness. Urine microscopy showed pyuria and culture grew Escherichia "
            "coli. Computed tomography showed a swollen right kidney with no abscess. "
            "She improved on intravenous ceftriaxone."
        ),
        code="N10",
        reasoning=(
            "Fever with flank pain and an inflamed kidney on imaging is acute "
            "pyelonephritis, chapter N, category N10."
        ),
    ),
    FewShotExample(
        vignette=(
            "An 81-year-old woman was brought in after a fall at home. She could not "
            "weight-bear and the right leg was shortened and externally rotated. A "
            "pelvic radiograph showed a displaced subcapital fracture of the right "
            "femoral neck. She underwent hemiarthroplasty on day one."
        ),
        code="S72",
        reasoning=(
            "A displaced fracture through the femoral neck is a fracture of the femur, "
            "chapter S, category S72."
        ),
    ),
    FewShotExample(
        vignette=(
            "A 19-year-old man presented with 18 hours of periumbilical pain that "
            "migrated to the right iliac fossa, with anorexia and a low-grade fever. "
            "There was guarding at McBurney's point and the white cell count was "
            "15.2. Laparoscopy confirmed an inflamed, non-perforated appendix."
        ),
        code="K35",
        reasoning=(
            "Migratory right iliac fossa pain with an inflamed appendix at operation is "
            "acute appendicitis, chapter K, category K35."
        ),
    ),
    FewShotExample(
        vignette=(
            "A 64-year-old woman with a history of oestrogen-receptor-positive breast "
            "carcinoma treated five years earlier presented with progressive thoracic "
            "back pain. Magnetic resonance imaging showed deposits in T7 and T9 with "
            "no cord compression. Biopsy confirmed deposits of breast origin and she "
            "received palliative radiotherapy to the spine."
        ),
        code="C79",
        reasoning=(
            "Admission was for management of tumour deposits in bone rather than the "
            "breast primary, so the principal diagnosis is a secondary malignant "
            "neoplasm at another site, chapter C, category C79."
        ),
    ),
)

_STRUCTURE_BRIEFING: Final[str] = (
    "ICD-10-AM is the Australian modification of the International Statistical "
    "Classification of Diseases and Related Health Problems, 10th revision. Its "
    "disease codes are arranged in chapters, and a three-character category is "
    "written as one capital letter followed by two digits. The letter indicates "
    "the chapter or block: A and B are infectious diseases, C is malignant "
    "neoplasms, D is other neoplasms and blood disorders, E is endocrine, "
    "nutritional and metabolic disease, F is mental and behavioural disorders, G "
    "is the nervous system, H is the eye and ear, I is the circulatory system, J "
    "is the respiratory system, K is the digestive system, L is the skin, M is the "
    "musculoskeletal system, N is the genitourinary system, O is pregnancy and "
    "childbirth, P is the perinatal period, Q is congenital malformations, R is "
    "symptoms and abnormal findings, S and T are injury and poisoning, and Z is "
    "factors influencing health status. The two digits narrow the chapter to a "
    "specific category, for example J44 is chronic obstructive pulmonary disease "
    "and N17 is acute kidney failure."
)

_PRINCIPAL_BRIEFING: Final[str] = (
    "The principal diagnosis is the single condition established after "
    "investigation to be chiefly responsible for the patient's episode of "
    "admitted care. It is not the patient's most serious comorbidity, not the "
    "presenting symptom, and not the procedure performed."
)


@dataclass(frozen=True)
class PromptTemplate:
    """A named prompting strategy and the reason it is being tested.

    Attributes:
        name: Registry key, also used in cache keys and result files.
        family: Broad strategy family, for grouping in the results table.
        rationale: Why this strategy might help, stated before the experiment.
        system: System message, or ``None`` to omit the system turn.
        instruction: Instruction text placed before the note.
        closing: Text placed after the note, typically the output contract.
        examples: Few-shot examples shown between instruction and note.
        prefer: Which candidate :func:`icdprompt.postprocess.extract_code`
            should keep when several codes appear.
        max_new_tokens: Generation budget.  Reasoning strategies need more.
        show_reasoning: Whether the examples display their reasoning lines.
    """

    name: str
    family: Family
    rationale: str
    instruction: str
    closing: str = ""
    system: str | None = None
    examples: tuple[FewShotExample, ...] = field(default=())
    prefer: Preference = "first"
    max_new_tokens: int = 16
    show_reasoning: bool = False

    @property
    def example_codes(self) -> tuple[str, ...]:
        """Distinct codes appearing in this template's few-shot examples."""
        return tuple(dict.fromkeys(example.code for example in self.examples))

    @property
    def n_examples(self) -> int:
        """Number of few-shot examples carried by this template."""
        return len(self.examples)

    def _render_examples(self) -> str:
        """Render the few-shot block, empty when there are no examples."""
        if not self.examples:
            return ""
        blocks: list[str] = []
        for index, example in enumerate(self.examples, start=1):
            answer = example.code
            if self.show_reasoning and example.reasoning:
                answer = f"{example.reasoning}\nCODE: {example.code}"
            blocks.append(
                f"Example {index}\nHospital course: {example.vignette}\nAnswer: {answer}"
            )
        return "\n\n".join(blocks)

    def render_user(self, note: str) -> str:
        """Build the user turn for one note.

        Args:
            note: Hospital-course text with stated diagnoses redacted.

        Returns:
            The fully rendered user message.
        """
        parts: list[str] = [self.instruction.strip()]
        examples = self._render_examples()
        if examples:
            parts.append(examples)
        parts.append(f"Hospital course:\n{note.strip()}")
        if self.closing.strip():
            parts.append(self.closing.strip())
        return "\n\n".join(parts)

    def render_messages(self, note: str) -> list[dict[str, str]]:
        """Build the chat message list for one note.

        Args:
            note: Hospital-course text with stated diagnoses redacted.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dictionaries suitable for
            a tokenizer chat template.
        """
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system.strip()})
        messages.append({"role": "user", "content": self.render_user(note)})
        return messages


_TEMPLATE_LIST: Final[tuple[PromptTemplate, ...]] = (
    PromptTemplate(
        name="zero_shot_minimal",
        family="zero-shot",
        rationale=(
            "Baseline. Asks the question in the plainest possible way, with no "
            "format contract, no persona and no examples. Any gain from the other "
            "strategies has to be measured against this."
        ),
        instruction="What is the ICD-10-AM principal diagnosis code for this patient?",
        max_new_tokens=48,
    ),
    PromptTemplate(
        name="zero_shot_strict_format",
        family="format-constrained",
        rationale=(
            "Tests whether the baseline's failures are formatting rather than "
            "knowledge. A hard output contract plus an explicit shape definition "
            "should raise format validity even if accuracy does not move."
        ),
        system=(
            "You are a precise classification engine. You reply with a code and "
            "nothing else: no explanation, no punctuation, no preamble."
        ),
        instruction=(
            "Assign the three-character ICD-10-AM principal diagnosis category for "
            "the episode of care described below. A three-character category is one "
            "capital letter followed by exactly two digits, for example J44."
        ),
        closing="Reply with the three characters only.",
        max_new_tokens=8,
    ),
    PromptTemplate(
        name="persona_clinical_coder",
        family="persona",
        rationale=(
            "Tests whether role framing recruits more domain-appropriate "
            "behaviour. Persona prompts are widely reported to help on "
            "professional tasks, and this is the cheapest version of that claim to "
            "falsify on a 0.5B model."
        ),
        system=(
            "You are a senior clinical coder in an Australian public hospital with "
            "fifteen years of experience coding inpatient episodes in ICD-10-AM. You "
            "are careful, conservative, and you never invent a code you are not sure "
            "of the chapter for."
        ),
        instruction=(
            "Code the episode of care below. " + _PRINCIPAL_BRIEFING + " Give the "
            "three-character ICD-10-AM category for the principal diagnosis."
        ),
        closing="Answer in the form CODE: <letter><digit><digit>",
        max_new_tokens=24,
    ),
    PromptTemplate(
        name="few_shot_three",
        family="few-shot",
        rationale=(
            "Tests in-context learning with three worked examples spanning three "
            "different chapters. This is also the first template where a correct "
            "answer may be a copy of a demonstrated code, which the contamination "
            "check is designed to detect."
        ),
        instruction=(
            "Assign the three-character ICD-10-AM principal diagnosis category. "
            "Here are worked examples, then the case to code."
        ),
        examples=_EXAMPLES[:3],
        closing="Answer with the three-character category only.",
        max_new_tokens=8,
    ),
    PromptTemplate(
        name="few_shot_six",
        family="few-shot",
        rationale=(
            "Doubles the demonstrations to six chapters to separate two effects: "
            "learning the output format, which should saturate quickly, from "
            "learning the task, which should keep improving if in-context learning "
            "is doing real work."
        ),
        instruction=(
            "Assign the three-character ICD-10-AM principal diagnosis category. "
            "Here are worked examples, then the case to code."
        ),
        examples=_EXAMPLES,
        closing="Answer with the three-character category only.",
        max_new_tokens=8,
    ),
    PromptTemplate(
        name="cot_then_extract",
        family="chain-of-thought",
        rationale=(
            "Tests whether forcing an explicit reasoning chain before the answer "
            "helps. The risk on a small model is that reasoning tokens crowd out "
            "the answer or drift, so extraction takes the last code rather than "
            "the first."
        ),
        instruction=(
            "Work out the ICD-10-AM principal diagnosis category for the episode "
            "below. " + _PRINCIPAL_BRIEFING + "\n\nThink in three steps: first name "
            "the condition chiefly responsible for the admission, second name the "
            "ICD-10-AM chapter letter it belongs to, third give the three-character "
            "category."
        ),
        examples=_EXAMPLES[:2],
        show_reasoning=True,
        closing=(
            "Write your reasoning in at most three short sentences, then a final "
            "line of exactly 'CODE: <letter><digit><digit>'."
        ),
        prefer="last",
        # Three short sentences plus the CODE line fit comfortably; the budget
        # is capped because every decoded token costs CPU time.
        max_new_tokens=128,
    ),
    PromptTemplate(
        name="context_rich_structure",
        family="context-rich",
        rationale=(
            "Tests whether the classification's structure can be supplied in the "
            "prompt instead of relied on from pretraining. If the model knows the "
            "clinical condition but not the code system, a chapter-letter briefing "
            "should at least raise chapter-level agreement."
        ),
        system="You are an expert in clinical classification systems.",
        instruction=_STRUCTURE_BRIEFING + "\n\n" + _PRINCIPAL_BRIEFING + "\n\nUsing that "
        "structure, assign the three-character ICD-10-AM principal diagnosis "
        "category for the episode below. Choose the chapter letter first, then the "
        "two digits.",
        closing="Answer in the form CODE: <letter><digit><digit>",
        max_new_tokens=32,
    ),
    PromptTemplate(
        name="json_output",
        family="structured-output",
        rationale=(
            "Tests whether a machine-readable contract improves parseability. JSON "
            "is easier to validate than prose, but a small model may spend its "
            "budget on syntax, produce unterminated objects, or echo the schema "
            "back with the placeholder still in it."
        ),
        system=(
            "You are an API endpoint. Every response is a single JSON object and "
            "nothing else."
        ),
        instruction=(
            "Assign the three-character ICD-10-AM principal diagnosis category for "
            'the episode below. Respond with exactly {"code": "<letter><digit><digit>", '
            '"condition": "<short name>"} and no other text.'
        ),
        closing='Respond with the JSON object only, starting with { and ending with }.',
        max_new_tokens=48,
    ),
    PromptTemplate(
        name="hybrid_coder_structured",
        family="hybrid",
        rationale=(
            "Combines the components that are individually plausible: the coder "
            "persona, the chapter-structure briefing, the principal-diagnosis "
            "definition, six demonstrations, and a hard single-line output "
            "contract. If prompt engineering alone can carry this task, this is the "
            "template that should show it."
        ),
        system=(
            "You are a senior clinical coder in an Australian public hospital, "
            "coding inpatient episodes in ICD-10-AM. You answer in exactly the "
            "format requested and add nothing else."
        ),
        instruction=_STRUCTURE_BRIEFING
        + "\n\n"
        + _PRINCIPAL_BRIEFING
        + "\n\nAssign the three-character ICD-10-AM principal diagnosis category. "
        "Here are worked examples, then the case to code.",
        examples=_EXAMPLES,
        closing="Answer with one line of exactly 'CODE: <letter><digit><digit>'.",
        max_new_tokens=16,
    ),
)

#: The prompt registry, keyed by template name.
TEMPLATES: Final[dict[str, PromptTemplate]] = {
    template.name: template for template in _TEMPLATE_LIST
}


def template_names() -> tuple[str, ...]:
    """Return registered template names in declaration order."""
    return tuple(TEMPLATES)


def get_template(name: str) -> PromptTemplate:
    """Look up a template by name.

    Args:
        name: Registry key.

    Returns:
        The matching :class:`PromptTemplate`.

    Raises:
        KeyError: If ``name`` is not registered, with the valid names listed.
    """
    try:
        return TEMPLATES[name]
    except KeyError as error:
        valid = ", ".join(template_names())
        raise KeyError(f"unknown template {name!r}; registered templates: {valid}") from error
