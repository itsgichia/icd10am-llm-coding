"""Extraction of a single three-character ICD-10-AM category from free text.

A three-character ICD-10-AM category is one uppercase letter followed by two
digits (for example ``J44``, ``C79``, ``I21``).  Language models rarely emit
exactly that and nothing else: they wrap it in prose, in JSON, in Markdown
fences, in a chain of reasoning, or they emit a four- or five-character code
such as ``J44.1``.  Every downstream metric in this study depends on turning
that output into one category or an explicit "no code found", so the extractor
is deliberately layered and heavily tested.

Extraction order (first layer that succeeds wins):

1. ``bare``  -- the whole (stripped) output is a single code.
2. ``json``  -- a JSON object whose key looks like a code field.
3. ``cue``   -- an explicit answer cue such as ``CODE:`` or ``Final answer:``.
4. ``regex`` -- a scan of the remaining text, taking the first or last match
   depending on ``prefer``.

The first three layers are authoritative: a code supplied in a structured field
or after an explicit cue wins over position, so ``prefer`` only decides between
bare regex matches.

Each layer normalises the candidate: it is uppercased, internal separators are
dropped, and anything beyond the first two digits is truncated, so ``j44.1``
and ``J440`` both become ``J44``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Final, Literal

__all__ = [
    "CATEGORY_PATTERN",
    "ExtractionResult",
    "extract_code",
    "is_valid_category",
    "normalise_candidate",
]

Preference = Literal["first", "last"]
Status = Literal["VALID", "INVALID"]

#: Shape of a well-formed three-character ICD-10-AM category.
CATEGORY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z][0-9]{2}$")

# A candidate is a word-initial letter, an optional single separator, two
# digits, and then any trailing sub-division characters that we discard.  The
# leading ``\b`` is what keeps "ICD-10-AM" from being read as code "D10": the
# ``D`` there is not at a word boundary.
_CANDIDATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Za-z])(?P<sep>[\s.\-]?)([0-9]{2})(?:[0-9]|\.[0-9]{1,2})?\b"
)

# Letters that are also standalone English words.  A space-separated candidate
# starting with one of these is almost always prose, not a code: without this
# guard "the patient is a 72 year old man" extracts as A72.
_WORD_LETTERS: Final[frozenset[str]] = frozenset({"A", "I", "O"})

# Keys that, inside a JSON object, are understood to hold the answer.
_JSON_CODE_KEYS: Final[tuple[str, ...]] = (
    "code",
    "icd_code",
    "icd10am_code",
    "icd10_code",
    "icd",
    "category",
    "principal_diagnosis_code",
    "diagnosis_code",
    "answer",
)

# Phrases that introduce the final answer in prose or in a template's required
# output format.  Longer, more specific cues come first.
_ANSWER_CUES: Final[tuple[str, ...]] = (
    "final answer",
    "final code",
    "answer code",
    "icd-10-am code",
    "icd-10-am category",
    "icd-10-am",
    "icd10am",
    "icd code",
    "category",
    "code",
    "answer",
)

# Text that reliably produces spurious candidates.  Removing it before the
# regex scan is cheaper and more transparent than trying to out-clever it.
_NOISE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"ICD\s*[-–]?\s*10\s*[-–]?\s*AM", re.IGNORECASE),
    re.compile(r"ICD\s*[-–]?\s*10\s*[-–]?\s*CM", re.IGNORECASE),
    re.compile(r"ICD\s*[-–]?\s*(?:9|10|11)\b", re.IGNORECASE),
    re.compile(r"\bvitamin\s*[-–]?\s*[A-K]\s*[-–]?\s*[0-9]{1,2}\b", re.IGNORECASE),
    re.compile(r"\bCOVID\s*[-–]?\s*19\b", re.IGNORECASE),
    re.compile(r"\bSARS\s*[-–]?\s*CoV\s*[-–]?\s*2\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of trying to pull one category out of one model output.

    Attributes:
        code: The normalised three-character category, or ``None``.
        status: ``"VALID"`` when a well-formed category was recovered,
            ``"INVALID"`` otherwise.  Format-validity rate is the mean of
            ``status == "VALID"`` across records.
        method: Which extraction layer produced the code (``"json"``, ``"cue"``,
            ``"bare"``, ``"regex"``) or ``"none"`` when nothing was found.
        candidates: Every distinct normalised candidate seen, in order of
            appearance.  More than one means the model hedged.
        note: Short human-readable explanation, for error analysis.
    """

    code: str | None
    status: Status
    method: str
    candidates: tuple[str, ...] = field(default=())
    note: str = ""

    @property
    def is_valid(self) -> bool:
        """Whether a well-formed category was recovered."""
        return self.status == "VALID"

    @property
    def is_ambiguous(self) -> bool:
        """Whether the output offered more than one distinct category."""
        return len(set(self.candidates)) > 1

    @property
    def chapter_letter(self) -> str | None:
        """First character of the code, which identifies the ICD chapter block."""
        return self.code[0] if self.code else None


def is_valid_category(code: str | None) -> bool:
    """Return whether ``code`` has the shape of a three-character category."""
    if not code:
        return False
    return bool(CATEGORY_PATTERN.match(code))


def normalise_candidate(letter: str, digits: str) -> str:
    """Build a canonical category from a matched letter and digit pair."""
    return f"{letter.upper()}{digits[:2]}"


def _strip_noise(text: str) -> str:
    """Blank out tokens that look like codes but never are."""
    cleaned = text
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def _all_candidates(text: str) -> list[str]:
    """Return every normalised candidate in ``text``, in order of appearance.

    Space-separated candidates beginning with a letter that is also an English
    word are dropped, so that ordinary prose such as "a 72 year old man" does
    not yield a code.
    """
    candidates: list[str] = []
    for match in _CANDIDATE_PATTERN.finditer(text):
        letter, digits = match.group(1), match.group(3)
        separator = match.group("sep")
        if separator.isspace() and letter.upper() in _WORD_LETTERS:
            continue
        candidates.append(normalise_candidate(letter, digits))
    return candidates


def _iter_json_objects(text: str) -> list[dict[str, object]]:
    """Yield JSON objects parsed from ``text``, tolerating surrounding prose.

    Handles Markdown fences and a single object embedded in a sentence.  Nested
    objects are not needed for this study, so the scan is intentionally shallow.
    """
    objects: list[dict[str, object]] = []
    fence_stripped = re.sub(r"```(?:json)?", " ", text, flags=re.IGNORECASE)
    for match in re.finditer(r"\{[^{}]*\}", fence_stripped, flags=re.DOTALL):
        snippet = match.group(0)
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            # Single quotes and trailing commas are common model slips.
            repaired = snippet.replace("'", '"')
            repaired = re.sub(r",\s*\}", "}", repaired)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def _from_json(text: str) -> tuple[str, str] | None:
    """Try the JSON layer.  Returns ``(code, note)`` on success."""
    for obj in _iter_json_objects(text):
        lowered = {str(key).strip().lower().replace(" ", "_"): value for key, value in obj.items()}
        for key in _JSON_CODE_KEYS:
            if key not in lowered:
                continue
            value = lowered[key]
            if not isinstance(value, (str, int)):
                continue
            candidates = _all_candidates(_strip_noise(str(value)))
            if candidates:
                return candidates[0], f"json key {key!r}"
    return None


def _from_cue(text: str) -> tuple[str, str] | None:
    """Try the answer-cue layer.  Returns ``(code, note)`` on success."""
    lowered = text.lower()
    best: tuple[int, str, str] | None = None
    for cue in _ANSWER_CUES:
        start = 0
        while True:
            position = lowered.find(cue, start)
            if position == -1:
                break
            start = position + len(cue)
            # Only the text shortly after the cue can be the value it introduces;
            # a code further away belongs to a different sentence.
            tail = text[start : start + 48]
            candidates = _all_candidates(_strip_noise(tail))
            if candidates:
                # Prefer the last cue in the text: models restate their answer.
                if best is None or position >= best[0]:
                    best = (position, candidates[0], f"cue {cue!r}")
    if best is None:
        return None
    return best[1], best[2]


def _from_bare(text: str) -> tuple[str, str] | None:
    """Try the bare-output layer, where the response is only a code."""
    stripped = text.strip().strip("`\"'*.,:;()[]{} \n\t")
    match = re.fullmatch(r"([A-Za-z])[\s.\-]?([0-9]{2})(?:[0-9]|\.[0-9]{1,2})?", stripped)
    if match is None:
        return None
    return normalise_candidate(match.group(1), match.group(2)), "bare code"


def extract_code(text: str | None, prefer: Preference = "first") -> ExtractionResult:
    """Extract one three-character ICD-10-AM category from model output.

    Args:
        text: Raw generated text.  ``None`` and whitespace are tolerated.
        prefer: Which regex match to keep when the model lists several codes and
            offers no JSON or cue.  Chain-of-thought prompts should pass
            ``"last"`` because the conclusion comes after the reasoning;
            direct-answer prompts should keep the default ``"first"``.

    Returns:
        An :class:`ExtractionResult`.  ``status`` is ``"VALID"`` only when a
        well-formed category was recovered.
    """
    if text is None:
        return ExtractionResult(None, "INVALID", "none", (), "empty output")

    raw = text.strip()
    if not raw:
        return ExtractionResult(None, "INVALID", "none", (), "empty output")

    cleaned = _strip_noise(raw)
    candidates = tuple(_all_candidates(cleaned))

    for layer in (_from_bare, _from_json, _from_cue):
        found = layer(raw)
        if found is None:
            continue
        code, note = found
        if is_valid_category(code):
            return ExtractionResult(code, "VALID", _method_for(layer), candidates or (code,), note)

    if candidates:
        code = candidates[-1] if prefer == "last" else candidates[0]
        if is_valid_category(code):
            note = f"regex scan, {len(candidates)} candidate(s), prefer={prefer}"
            return ExtractionResult(code, "VALID", "regex", candidates, note)

    return ExtractionResult(None, "INVALID", "none", candidates, "no category-shaped token found")


def _method_for(layer: object) -> str:
    """Map an extraction layer function to its short method name."""
    return {
        "_from_json": "json",
        "_from_cue": "cue",
        "_from_bare": "bare",
    }.get(getattr(layer, "__name__", ""), "regex")
