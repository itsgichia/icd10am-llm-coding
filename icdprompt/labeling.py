"""Transparent weak labelling of synthetic discharge summaries.

The Asclepius synthetic notes ship no ICD-10-AM codes, so this module derives a
*weak* label from each note's own stated diagnosis section using an ordered,
inspectable keyword table.  Three properties make the result auditable:

1. The diagnosis section is located by an explicit heading regex, and the exact
   matched text is carried through to ``data/labels.jsonl`` so a human can check
   every label by eye.
2. A note is only labelled when the rule table fires on exactly one distinct
   category.  Notes listing several codeable conditions are dropped, because
   nothing in the record identifies which one a coder would have made the
   principal diagnosis.
3. The diagnosis section is redacted from the text shown to the model, so the
   model must infer the category from the hospital course rather than read the
   string the label came from.

These are weak labels, not clinical coding.  They encode one obvious category
per condition and ignore the ICD-10-AM Australian Coding Standards entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DIAGNOSIS_HEADING_PATTERN",
    "LABEL_RULES",
    "LabelOutcome",
    "LabelRule",
    "count_diagnosis_items",
    "find_diagnosis_section",
    "label_rules_by_code",
    "redact_diagnosis_sections",
    "weak_label",
]

#: Headings that introduce a stated diagnosis in these synthetic notes.
DIAGNOSIS_HEADING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?mi)^[ \t]*(?:[-*\d.)\s]*)?"
    r"(?P<heading>(?:discharge|admitting|admission|final|principal|primary|"
    r"secondary|working|clinical)?[ \t]*diagnos[ei]s)"
    r"[ \t]*:[ \t]*(?P<body>.*(?:\n(?![ \t]*[A-Z][A-Za-z /&()-]{2,45}[ \t]*:).*)*)"
)

#: Diagnosis bodies that carry no information.
_EMPTY_BODY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:n/?a|none|nil|not applicable|not available|unknown|deferred|"
    r"no new diagnos\w*|see (?:above|below)|as above)\.?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LabelRule:
    """One keyword rule mapping diagnosis wording to an ICD-10-AM category.

    Attributes:
        code: Three-character ICD-10-AM category assigned by this rule.
        title: Short plain-English title of that category.
        pattern: Case-insensitive regex matched against the diagnosis text.
    """

    code: str
    title: str
    pattern: re.Pattern[str]

    def matches(self, text: str) -> re.Match[str] | None:
        """Return the first match of this rule in ``text``, if any."""
        return self.pattern.search(text)


@dataclass(frozen=True)
class LabelOutcome:
    """Result of applying the rule table to one diagnosis section.

    Attributes:
        code: Assigned category, or ``None`` when the note was not labelled.
        title: Title of the assigned category, or ``None``.
        evidence: The diagnosis text the decision was made from.
        matched_text: The exact substring the winning rule matched.
        reason: Why the note was accepted or rejected.
        competing_codes: Distinct codes that fired, for rejected ambiguous notes.
    """

    code: str | None
    title: str | None
    evidence: str
    matched_text: str
    reason: str
    competing_codes: tuple[str, ...] = ()

    @property
    def is_labelled(self) -> bool:
        """Whether a single category was assigned."""
        return self.code is not None


def _rule(code: str, title: str, pattern: str) -> LabelRule:
    """Build a :class:`LabelRule` with case-insensitive matching."""
    return LabelRule(code, title, re.compile(pattern, re.IGNORECASE))


# Ordered keyword table.  Each entry is one three-character category and the
# wording that, in these notes, reliably denotes it.  The table is deliberately
# narrow: it covers common, unambiguously codeable conditions and declines
# everything else rather than guessing.
LABEL_RULES: Final[tuple[LabelRule, ...]] = (
    # --- Certain infectious and parasitic diseases (A, B) ---
    _rule("A41", "Other sepsis", r"\b(?:sepsis|septic shock|septicaemia|septicemia|bacteraemia|bacteremia)\b"),
    _rule("A15", "Respiratory tuberculosis", r"\b(?:pulmonary tuberculosis|pulmonary TB)\b"),
    _rule("A09", "Infectious gastroenteritis", r"\b(?:infectious |acute )?gastroenteritis\b"),
    _rule("B20", "HIV disease", r"\bHIV(?:/AIDS)? (?:disease|infection)\b|\bacquired immunodeficiency syndrome\b|\b(?-i:AIDS)\b"),
    _rule("B18", "Chronic viral hepatitis", r"\bchronic (?:viral )?hepatitis\s*[BC]\b"),
    # --- Neoplasms (C, D0-D4) ---
    _rule("C34", "Malignant neoplasm of bronchus and lung", r"\b(?:lung|bronchogenic|pulmonary)\s+(?:adeno)?carcinoma\b|\blung cancer\b|\bnon-?small cell lung\b|\bsmall cell lung\b"),
    _rule("C50", "Malignant neoplasm of breast", r"\bbreast\s+(?:cancer|carcinoma|malignancy)\b|\b(?:invasive|infiltrating)\s+(?:ductal|lobular)\s+(?:breast\s+)?carcinoma\b|\blobular breast cancer\b"),
    _rule("C16", "Malignant neoplasm of stomach", r"\b(?:gastric|stomach)\s+(?:cancer|carcinoma|adenocarcinoma)\b"),
    _rule("C18", "Malignant neoplasm of colon", r"\b(?:colon|colonic|caecal|cecal|sigmoid)\s+(?:cancer|carcinoma|adenocarcinoma)\b|\badenocarcinoma of (?:the )?colon\b|\bcolon cancer\b"),
    _rule("C20", "Malignant neoplasm of rectum", r"\brect(?:al|um)\s+(?:cancer|carcinoma|adenocarcinoma)\b|\badenocarcinoma of (?:the )?rectum\b"),
    _rule("C22", "Malignant neoplasm of liver", r"\bhepatocellular carcinoma\b|\b(?-i:HCC)\b|\bliver cancer\b"),
    _rule("C25", "Malignant neoplasm of pancreas", r"\bpancrea(?:tic|s)\s+(?:cancer|carcinoma|adenocarcinoma)\b|\badenocarcinoma of (?:the )?pancreas\b"),
    _rule("C43", "Malignant melanoma of skin", r"\b(?:malignant |cutaneous )?melanoma\b"),
    _rule("C53", "Malignant neoplasm of cervix uteri", r"\bcervical (?:cancer|carcinoma)\b"),
    _rule("C56", "Malignant neoplasm of ovary", r"\bovarian (?:cancer|carcinoma)\b"),
    _rule("C61", "Malignant neoplasm of prostate", r"\bprostate\s+(?:cancer|carcinoma|adenocarcinoma)\b|\badenocarcinoma of (?:the )?prostate\b"),
    _rule("C64", "Malignant neoplasm of kidney", r"\brenal cell carcinoma\b|\bkidney cancer\b|\b(?-i:RCC)\b"),
    _rule("C67", "Malignant neoplasm of bladder", r"\bbladder\s+(?:cancer|carcinoma)\b|\burothelial carcinoma\b"),
    _rule("C71", "Malignant neoplasm of brain", r"\bglioblastoma\b|\b(?:high|low)-grade glioma\b|\banaplastic astrocytoma\b|\bbrain tumou?r\b"),
    _rule("C73", "Malignant neoplasm of thyroid gland", r"\bthyroid\s+(?:cancer|carcinoma)\b|\bpapillary thyroid\b"),
    _rule("C78", "Secondary neoplasm of respiratory and digestive organs", r"\b(?:hepatic|liver|lung|pulmonary) metastas[ei]s\b|\bmetastas[ei]s to the (?:liver|lung)\b"),
    _rule("C79", "Secondary neoplasm of other sites", r"\b(?:bone|brain|cerebral|vertebral|osseous) metastas[ei]s\b|\bmetastas[ei]s to the (?:bone|brain|spine)\b"),
    _rule("C81", "Hodgkin lymphoma", r"\bHodgkin(?:'s)? lymphoma\b"),
    _rule("C83", "Non-follicular lymphoma", r"\bdiffuse large B-?cell lymphoma\b|\b(?-i:DLBCL)\b"),
    _rule("C85", "Other non-Hodgkin lymphoma", r"\bnon-?Hodgkin(?:'s)? lymphoma\b"),
    _rule("C90", "Multiple myeloma", r"\bmultiple myeloma\b"),
    _rule("C91", "Lymphoid leukaemia", r"\b(?:acute lymphoblastic|acute lymphocytic|chronic lymphocytic) leuk(?:a)?emia\b|\b(?-i:ALL)\b|\b(?-i:CLL)\b"),
    _rule("C92", "Myeloid leukaemia", r"\b(?:acute|chronic) myelo(?:id|genous|blastic) leuk(?:a)?emia\b|\b(?-i:AML)\b|\b(?-i:CML)\b"),
    # --- Blood and blood-forming organs (D5-D8) ---
    _rule("D50", "Iron deficiency anaemia", r"\biron[- ]deficiency an(?:a)?emia\b"),
    _rule("D69", "Purpura and other haemorrhagic conditions", r"\b(?:immune |idiopathic )?thrombocytopenic purpura\b|\b(?-i:ITP)\b"),
    # --- Endocrine, nutritional and metabolic (E) ---
    _rule("E03", "Other hypothyroidism", r"\bhypothyroidism\b"),
    _rule("E05", "Thyrotoxicosis", r"\b(?:thyrotoxicosis|hyperthyroidism|Graves(?:'s|')? disease)\b"),
    _rule("E10", "Type 1 diabetes mellitus", r"\btype\s*(?:1|I|one)\s*diabetes\b"),
    _rule("E11", "Type 2 diabetes mellitus", r"\btype\s*(?:2|II|two)\s*diabetes\b"),
    _rule("E86", "Volume depletion", r"\b(?:severe )?dehydration\b|\bvolume depletion\b"),
    _rule("E87", "Other disorders of fluid and electrolyte balance", r"\b(?:hyponatr(?:a)?emia|hypernatr(?:a)?emia|hypokal(?:a)?emia|hyperkal(?:a)?emia)\b"),
    # --- Mental and behavioural (F) ---
    _rule("F10", "Mental disorders due to alcohol", r"\balcohol (?:use disorder|dependence|withdrawal)\b|\bdelirium tremens\b"),
    _rule("F20", "Schizophrenia", r"\bschizophrenia\b"),
    _rule("F32", "Depressive episode", r"\bmajor depress(?:ive|ion)(?: disorder| episode)?\b"),
    # --- Nervous system (G) ---
    _rule("G00", "Bacterial meningitis", r"\b(?:bacterial|pneumococcal|meningococcal) meningitis\b"),
    _rule("G20", "Parkinson disease", r"\bParkinson(?:'s)? disease\b"),
    _rule("G35", "Multiple sclerosis", r"\bmultiple sclerosis\b"),
    _rule("G40", "Epilepsy", r"\b(?:epilepsy|status epilepticus|epileptic seizures?)\b"),
    _rule("G61", "Inflammatory polyneuropathy", r"\bGuillain-?Barr[eé]\b|\bchronic inflammatory demyelinating polyneuropathy\b|\b(?-i:CIDP)\b"),
    _rule("G93", "Other disorders of brain", r"\b(?:metabolic |toxic |hypoxic )?encephalopathy\b"),
    # --- Circulatory system (I) ---
    _rule("I21", "Acute myocardial infarction", r"\b(?:acute )?myocardial infarction\b|\b(?:ST[- ]elevation|non[- ]ST[- ]elevation)\b|\b(?-i:STEMI)\b|\b(?-i:NSTEMI)\b"),
    _rule("I26", "Pulmonary embolism", r"\bpulmonary embolis\w+\b|\bpulmonary embolus\b|\b(?-i:PE)\b"),
    _rule("I33", "Acute endocarditis", r"\b(?:infective|bacterial|acute) endocarditis\b|\bendocarditis\b"),
    _rule("I42", "Cardiomyopathy", r"\b(?:dilated|hypertrophic|takotsubo|peripartum|arrhythmogenic) cardiomyopathy\b"),
    _rule("I48", "Atrial fibrillation and flutter", r"\batrial (?:fibrillation|flutter)\b"),
    _rule("I50", "Heart failure", r"\b(?:congestive |acute |decompensated |chronic )*heart failure\b|\bpulmonary o?edema\b"),
    _rule("I61", "Intracerebral haemorrhage", r"\bintracerebral h(?:a)?emorrhage\b|\bintraparenchymal h(?:a)?emorrhage\b"),
    _rule("I60", "Subarachnoid haemorrhage", r"\bsubarachnoid h(?:a)?emorrhage\b|\b(?-i:SAH)\b"),
    _rule("I63", "Cerebral infarction", r"\b(?:acute |ischa?emic )?(?:cerebral infarction|ischa?emic stroke)\b|\bstroke\b"),
    _rule("I71", "Aortic aneurysm and dissection", r"\baortic (?:aneurysm|dissection)\b|\babdominal aortic aneurysm\b|\b(?-i:AAA)\b"),
    _rule("I80", "Phlebitis and thrombophlebitis", r"\bsuperficial (?:vein )?thrombophlebitis\b"),
    _rule("I82", "Other venous embolism and thrombosis", r"\bdeep vein thrombosis\b|\bdeep venous thrombosis\b|\b(?-i:DVT)\b"),
    # --- Respiratory system (J) and COVID-19 ---
    _rule("U07", "COVID-19 (emergency use code)", r"\bCOVID-?19\b|\bSARS-?CoV-?2 infection\b"),
    _rule("J18", "Pneumonia, organism unspecified", r"\b(?:community-?acquired |hospital-?acquired |aspiration |bilateral |lobar )?pneumonia\b"),
    _rule("J44", "Chronic obstructive pulmonary disease", r"\bchronic obstructive pulmonary disease\b|\b(?-i:COPD)\b|\b(?:pulmonary|centrilobular|panlobular) emphysema\b"),
    _rule("J45", "Asthma", r"\basthma\b"),
    _rule("J69", "Pneumonitis due to solids and liquids", r"\baspiration pneumonitis\b"),
    _rule("J80", "Acute respiratory distress syndrome", r"\b(?:acute )?respiratory distress syndrome\b|\b(?-i:ARDS)\b"),
    _rule("J90", "Pleural effusion", r"\bpleural effusion\b"),
    _rule("J93", "Pneumothorax", r"\b(?:spontaneous |tension )?pneumothorax\b"),
    _rule("J96", "Respiratory failure", r"\b(?:acute |hypoxa?emic |hypercapnic )*respiratory failure\b"),
    # --- Digestive system (K) ---
    _rule("K25", "Gastric ulcer", r"\bgastric ulcer\b"),
    _rule("K35", "Acute appendicitis", r"\b(?:acute |perforated )?appendicitis\b"),
    _rule("K40", "Inguinal hernia", r"\binguinal hernia\b"),
    _rule("K50", "Crohn disease", r"\bCrohn(?:'s)? disease\b"),
    _rule("K51", "Ulcerative colitis", r"\bulcerative colitis\b"),
    _rule("K56", "Paralytic ileus and intestinal obstruction", r"\b(?:small )?bowel obstruction\b|\bintestinal obstruction\b|\bparalytic ileus\b|\bvolvulus\b|\bintussusception\b"),
    _rule("K57", "Diverticular disease of intestine", r"\bdiverticulitis\b|\bdiverticular disease\b"),
    _rule("K70", "Alcoholic liver disease", r"\balcoholic (?:hepatitis|liver disease|cirrhosis)\b"),
    _rule("K74", "Fibrosis and cirrhosis of liver", r"\b(?:hepatic )?cirrhosis\b"),
    _rule("K80", "Cholelithiasis", r"\bcholelithiasis\b|\bgallstones?\b"),
    _rule("K81", "Cholecystitis", r"\b(?:acute |chronic )?cholecystitis\b"),
    _rule("K85", "Acute pancreatitis", r"\bacute pancreatitis\b"),
    _rule("K92", "Other diseases of digestive system", r"\b(?:upper |lower )?gastrointestinal (?:bleeding|h(?:a)?emorrhage)\b|\b(?-i:GI) bleed\w*\b|\bmelaena\b|\bhaematemesis\b"),
    # --- Skin (L) ---
    _rule("L03", "Cellulitis", r"\bcellulitis\b"),
    _rule("L89", "Pressure ulcer", r"\b(?:pressure|decubitus) ulcer\b"),
    # --- Musculoskeletal (M) ---
    _rule("M05", "Rheumatoid arthritis", r"\brheumatoid arthritis\b"),
    _rule("M10", "Gout", r"\bgout(?:y arthritis)?\b"),
    _rule("M16", "Osteoarthritis of hip", r"\b(?:hip|coxarthrosis) osteoarthritis\b|\bosteoarthritis of (?:the )?hip\b"),
    _rule("M17", "Osteoarthritis of knee", r"\bknee osteoarthritis\b|\bosteoarthritis of (?:the )?knee\b|\bgonarthrosis\b"),
    _rule("M31", "Other necrotising vasculopathies", r"\bgranulomatosis with polyangiitis\b|\b(?-i:GPA)\b|\bANCA-?associated vasculitis\b"),
    _rule("M32", "Systemic lupus erythematosus", r"\bsystemic lupus erythematosus\b|\b(?-i:SLE)\b"),
    _rule("M54", "Dorsalgia", r"\b(?:low(?:er)? )?back pain\b|\bsciatica\b"),
    _rule("M62", "Other disorders of muscle", r"\brhabdomyolysis\b"),
    _rule("M80", "Osteoporosis with pathological fracture", r"\bosteoporo(?:sis|tic) (?:with )?(?:pathological |fragility |vertebral )?fracture\b|\bfragility fracture\b"),
    _rule("M86", "Osteomyelitis", r"\b(?:acute |chronic |vertebral )?osteomyelitis\b"),
    # --- Genitourinary (N) ---
    _rule("N10", "Acute tubulo-interstitial nephritis", r"\b(?:acute )?pyelonephritis\b"),
    _rule("N17", "Acute kidney failure", r"\bacute kidney (?:injury|failure)\b|\b(?-i:AKI)\b|\bacute renal failure\b"),
    _rule("N18", "Chronic kidney disease", r"\bchronic kidney disease\b|\bend-?stage renal disease\b|\b(?-i:CKD)\b|\b(?-i:ESRD)\b"),
    _rule("N20", "Calculus of kidney and ureter", r"\b(?:renal|ureteric|kidney) (?:calculus|calculi|stone)s?\b|\bnephrolithiasis\b|\burolithiasis\b"),
    _rule("N39", "Other disorders of urinary system", r"\burinary tract infection\b|\b(?-i:UTI)\b"),
    _rule("N40", "Hyperplasia of prostate", r"\b(?:benign )?prostatic hyperplasia\b|\b(?-i:BPH)\b"),
    # --- Pregnancy (O) ---
    _rule("O00", "Ectopic pregnancy", r"\bectopic pregnancy\b|\bheterotopic pregnancy\b"),
    _rule("O14", "Pre-eclampsia", r"\bpre-?eclampsia\b"),
    # --- Injury and poisoning (S, T) ---
    _rule("S06", "Intracranial injury", r"\btraumatic brain injury\b|\b(?-i:TBI)\b|\b(?:acute )?subdural h(?:a)?ematoma\b|\bepidural h(?:a)?ematoma\b"),
    _rule("S22", "Fracture of rib(s), sternum and thoracic spine", r"\brib fractures?\b"),
    _rule("S32", "Fracture of lumbar spine and pelvis", r"\bpelvic fracture\b|\bfracture of (?:the )?pelvis\b"),
    _rule("S72", "Fracture of femur", r"\b(?:femoral neck|neck of femur|femur|femoral shaft|intertrochanteric|subcapital) fracture\b|\bfracture of (?:the )?femur\b|\bhip fracture\b"),
    _rule("S82", "Fracture of lower leg, including ankle", r"\b(?:tibial|tibia|ankle) fractures?\b"),
    _rule("T31", "Burns classified by extent of body surface", r"\bburns? (?:covering|involving|of)\b.{0,40}\bTBSA\b|\b\d{1,3}%\s*TBSA\b"),
)


def label_rules_by_code() -> dict[str, LabelRule]:
    """Return the rule table keyed by category code."""
    return {rule.code: rule for rule in LABEL_RULES}


def find_diagnosis_section(note: str) -> str:
    """Return the concatenated text of every diagnosis section in ``note``.

    Args:
        note: Full note text.

    Returns:
        The diagnosis text with headings removed, or an empty string when the
        note has no diagnosis heading or only an uninformative one.
    """
    bodies: list[str] = []
    for match in DIAGNOSIS_HEADING_PATTERN.finditer(note):
        body = " ".join(match.group("body").split()).strip()
        if not body or _EMPTY_BODY_PATTERN.match(body):
            continue
        if body not in bodies:
            bodies.append(body)
    return " | ".join(bodies)


def redact_diagnosis_sections(note: str, placeholder: str = "[diagnosis section withheld]") -> str:
    """Remove stated-diagnosis sections from ``note``.

    The weak label is derived from those sections, so they are withheld from the
    model's input.  Without this the task collapses into string matching and the
    reported accuracy would measure label leakage, not clinical inference.

    Args:
        note: Full note text.
        placeholder: Text substituted for each removed section body.

    Returns:
        The note with each diagnosis section body replaced by ``placeholder``.
    """

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group('heading')}: {placeholder}"

    redacted = DIAGNOSIS_HEADING_PATTERN.sub(_replace, note)
    return re.sub(r"\n{3,}", "\n\n", redacted).strip()


def count_diagnosis_items(evidence: str) -> int:
    """Count enumerated conditions in a diagnosis section.

    A section reading "1. Sepsis 2. Acute kidney injury 3. Hypertension" states
    a problem list, not a principal diagnosis, and nothing in the record says
    which entry a coder would have sequenced first.  Counting the enumeration
    markers lets :func:`weak_label` decline those notes.

    Args:
        evidence: Diagnosis text as returned by :func:`find_diagnosis_section`.

    Returns:
        The number of enumerated items, with a minimum of one.
    """
    markers = re.findall(r"(?:(?<=^)|(?<=[\s|]))(?:\d{1,2}[.)]|[-*•])\s+", evidence)
    sections = evidence.count(" | ") + 1
    return max(1, len(markers), sections)


def weak_label(note: str, max_items: int = 2) -> LabelOutcome:
    """Derive a weak ICD-10-AM category label for one note.

    Args:
        note: Full note text, including its diagnosis section.
        max_items: Maximum number of enumerated conditions the diagnosis
            section may list.  Above this the note is declined, because a long
            problem list does not identify a single principal diagnosis.

    Returns:
        A :class:`LabelOutcome`.  ``code`` is ``None`` when the note has no
        usable diagnosis section, when that section is a long problem list, when
        no rule fires, or when rules for more than one distinct category fire
        (ambiguous principal diagnosis).
    """
    evidence = find_diagnosis_section(note)
    if not evidence:
        return LabelOutcome(None, None, "", "", "no usable diagnosis section")

    n_items = count_diagnosis_items(evidence)
    if n_items > max_items:
        return LabelOutcome(
            None,
            None,
            evidence,
            "",
            f"problem list of {n_items} conditions, no single principal diagnosis",
        )

    hits: list[tuple[LabelRule, str]] = []
    for rule in LABEL_RULES:
        match = rule.matches(evidence)
        if match is not None:
            hits.append((rule, match.group(0)))

    distinct = tuple(dict.fromkeys(rule.code for rule, _ in hits))
    if not distinct:
        return LabelOutcome(None, None, evidence, "", "no rule matched")
    if len(distinct) > 1:
        return LabelOutcome(
            None, None, evidence, "", "ambiguous: several categories matched", distinct
        )

    rule, matched_text = hits[0]
    return LabelOutcome(rule.code, rule.title, evidence, matched_text, "single rule matched")
