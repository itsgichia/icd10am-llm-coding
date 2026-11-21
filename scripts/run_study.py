"""Run the prompt-engineering study end to end.

Stages:

1. Load the dev and test splits, building and caching them from the Hub on the
   first run.
2. Run every registered template on the development split and rank them.
3. Take the best development template plus four comparators and evaluate them on
   the held-out test split.
4. Write ``results/summary.csv``, ``results/predictions.csv``,
   ``results/chapter_confusion.csv``, ``results/study_meta.json`` and three
   figures to ``figures/``.

The script degrades cleanly.  If the model cannot be loaded or generation
fails, it writes whatever it completed, records the failure in
``results/study_meta.json`` and exits with status 0 so that the tests and the
code remain the deliverable.  No metric is ever invented.

Usage::

    python scripts/run_study.py                 # full study
    python scripts/run_study.py --dev-only      # development stage only
    python scripts/run_study.py --figures-only  # redraw from existing results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from icdprompt.data import ClinicalRecord, ScanConfig, load_splits  # noqa: E402
from icdprompt.evaluate import (  # noqa: E402
    Prediction,
    TemplateMetrics,
    build_predictions,
    chapter_confusion,
    evaluate_predictions,
)
from icdprompt.infer import (  # noqa: E402
    DEFAULT_MODEL_ID,
    GenerationConfig,
    ModelUnavailableError,
    QwenRunner,
)
from icdprompt.prompts import TEMPLATES, get_template, template_names  # noqa: E402

#: Comparators always carried to the test split alongside the dev winner.
#: Chosen before seeing results: the plain baseline, the format contract, the
#: two few-shot sizes that the contamination check compares, and the structural
#: briefing.
TEST_COMPARATORS: tuple[str, ...] = (
    "zero_shot_minimal",
    "zero_shot_strict_format",
    "few_shot_six",
    "context_rich_structure",
    "hybrid_coder_structured",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20260321, help="split and decoding seed")
    parser.add_argument("--dev-n", type=int, default=10, help="development split size")
    parser.add_argument("--test-n", type=int, default=10, help="held-out test split size")
    parser.add_argument("--max-chars", type=int, default=2000, help="longest redacted note")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--batch-size", type=int, default=5, help="prompts per generate call")
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device; the study is specified and reported on cpu",
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="intra-op thread count for CPU inference"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling mass")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="sample instead of decoding greedily (greedy is the default for reproducibility)",
    )
    parser.add_argument("--dev-only", action="store_true", help="skip the test stage")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="redraw figures from existing results without running the model",
    )
    parser.add_argument(
        "--dev-templates",
        nargs="*",
        default=None,
        help="templates for the development stage (default: all registered)",
    )
    parser.add_argument(
        "--test-templates",
        nargs="*",
        default=None,
        help=(
            "comparators for the test stage (default: the fixed list in "
            "TEST_COMPARATORS). The development winner is always evaluated first."
        ),
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write dictionaries to ``path`` as CSV using the first row's keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_confusion_csv(predictions: Sequence[Prediction], path: Path) -> Path:
    """Write the true-vs-predicted chapter cross-tabulation as long-form CSV."""
    table = chapter_confusion(predictions)
    rows = [
        {"true_chapter": true_letter, "predicted_chapter": predicted_letter, "n": count}
        for true_letter, row in sorted(table.items())
        for predicted_letter, count in sorted(row.items())
    ]
    return write_csv(rows, path)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style() -> Any:
    """Import matplotlib with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def figure_accuracy_vs_validity(
    metrics: Sequence[TemplateMetrics], path: Path, split: str
) -> Path | None:
    """Scatter accuracy against format validity, one point per template."""
    if not metrics:
        return None
    plt = _style()
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    # Templates often score identically, so coincident labels are staggered
    # vertically instead of being drawn on top of each other.
    seen_at_point: dict[tuple[float, float], int] = {}
    for item in metrics:
        marker = "D" if item.n_example_codes else "o"
        axis.scatter(
            item.format_validity,
            item.accuracy,
            s=70,
            marker=marker,
            color="#2f6f8f" if item.n_example_codes else "#8f4b2f",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        point = (round(item.format_validity, 3), round(item.accuracy, 3))
        rank = seen_at_point.get(point, 0)
        seen_at_point[point] = rank + 1
        axis.annotate(
            item.template,
            (item.format_validity, item.accuracy),
            textcoords="offset points",
            xytext=(8, 4 - 10 * rank),
            fontsize=7,
            color="#444444",
        )
    axis.set_xlabel("Format validity (share of outputs yielding a well-formed category)")
    axis.set_ylabel("Exact-match accuracy vs weak label")
    axis.set_title(f"Prompt strategies on the {split} split ({metrics[0].n} records)")
    axis.set_xlim(-0.05, 1.05)
    top = max(0.2, max(item.accuracy for item in metrics) * 1.4)
    axis.set_ylim(-0.02, top)
    handles = [
        plt.Line2D([], [], marker="D", color="#2f6f8f", linestyle="", label="has few-shot examples"),
        plt.Line2D([], [], marker="o", color="#8f4b2f", linestyle="", label="no examples"),
    ]
    axis.legend(handles=handles, frameon=False, loc="upper left", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_code_distribution(
    predictions: Sequence[Prediction], path: Path, top_n: int = 16
) -> Path | None:
    """Bar chart of the most frequently predicted codes, against the labels."""
    if not predictions:
        return None
    from collections import Counter

    plt = _style()
    predicted = Counter(p.predicted_code or "(none)" for p in predictions)
    true_counts = Counter(p.true_code for p in predictions)
    # Show the true labels as well as the predictions. When the two sets do not
    # overlap at all - which is what a collapsed model produces - a chart of the
    # predictions alone shows a row of empty comparison bars and hides the fact
    # that the labels themselves were spread over many distinct categories.
    labels = [code for code, _ in predicted.most_common(top_n)]
    labels += [code for code, _ in true_counts.most_common(top_n) if code not in labels]
    positions = range(len(labels))
    width = 0.42

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(
        [x - width / 2 for x in positions],
        [predicted[code] for code in labels],
        width=width,
        label="predicted",
        color="#2f6f8f",
    )
    axis.bar(
        [x + width / 2 for x in positions],
        [true_counts.get(code, 0) for code in labels],
        width=width,
        label="weak label",
        color="#c9ab6a",
    )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=60, ha="right")
    axis.set_ylabel("records")
    # Count only real codes, so the title agrees with n_unique_codes in
    # summary.csv; unparseable outputs are shown as their own "(none)" bar.
    n_distinct = len({p.predicted_code for p in predictions if p.predicted_code})
    axis.set_title(
        f"Predicted codes vs weak labels, {predictions[0].template}: "
        f"{n_distinct} distinct codes predicted "
        f"over {len(predictions)} records carrying "
        f"{len(true_counts)} distinct labels"
    )
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_chapter_confusion(predictions: Sequence[Prediction], path: Path) -> Path | None:
    """Heatmap of true against predicted ICD chapter letters."""
    if not predictions:
        return None
    plt = _style()
    table = chapter_confusion(predictions)
    true_letters = sorted(table)
    predicted_letters = sorted({letter for row in table.values() for letter in row})
    grid = [[table[t].get(p, 0) for p in predicted_letters] for t in true_letters]

    fig, axis = plt.subplots(
        figsize=(max(4.0, 0.52 * len(predicted_letters) + 2.2), max(3.2, 0.42 * len(true_letters) + 1.8))
    )
    image = axis.imshow(grid, cmap="BuPu", aspect="auto")
    axis.set_xticks(range(len(predicted_letters)), predicted_letters)
    axis.set_yticks(range(len(true_letters)), true_letters)
    axis.set_xlabel("predicted chapter letter ('?' = no code extracted)")
    axis.set_ylabel("weak-label chapter letter")
    axis.set_title(f"Chapter agreement, {predictions[0].template}")
    axis.grid(False)
    peak = max((count for row in grid for count in row), default=0)
    for row_index, row in enumerate(grid):
        for column_index, count in enumerate(row):
            if count:
                axis.text(
                    column_index,
                    row_index,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if count > peak * 0.6 else "#222222",
                )
    fig.colorbar(image, ax=axis, shrink=0.8, label="records")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def run_stage(
    runner: QwenRunner,
    template_name: str,
    records: Sequence[ClinicalRecord],
    split: str,
) -> tuple[TemplateMetrics, list[Prediction]]:
    """Run one template over one split and evaluate it."""
    template = get_template(template_name)
    generations = runner.run(template, records)
    predictions = build_predictions(template, records, generations)
    metrics = evaluate_predictions(predictions, template, split, generations)
    return metrics, predictions


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study, returning a process exit code."""
    args = parse_args(argv)
    results_dir = REPO_ROOT / "results"
    figures_dir = REPO_ROOT / "figures"
    meta: dict[str, Any] = {
        "model": args.model,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_ran": False,
        "failure": None,
        "stages_completed": [],
    }

    config = ScanConfig(
        seed=args.seed, dev_n=args.dev_n, test_n=args.test_n, max_chars=args.max_chars
    )
    try:
        splits = load_splits(config=config)
    except Exception as error:
        print(f"could not load splits: {error}", file=sys.stderr)
        meta["failure"] = f"data: {error}"
        (results_dir / "study_meta.json").parent.mkdir(parents=True, exist_ok=True)
        (results_dir / "study_meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
        return 0

    meta["scan_config"] = asdict(config)
    meta["n_dev"] = len(splits["dev"])
    meta["n_test"] = len(splits["test"])
    print(f"dev: {len(splits['dev'])} records, test: {len(splits['test'])} records")

    if args.figures_only:
        return _figures_only(results_dir, figures_dir)

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.sample,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    runner = QwenRunner(
        model_id=args.model,
        config=generation_config,
        device=args.device,
        threads=args.threads,
    )
    meta["generation_config"] = asdict(generation_config)
    meta["device"] = args.device

    all_metrics: list[TemplateMetrics] = []
    all_predictions: list[Prediction] = []
    dev_metrics: list[TemplateMetrics] = []
    dev_names = list(args.dev_templates or template_names())

    def flush() -> None:
        """Write results after each template, so a long run is never lost."""
        if all_metrics:
            write_csv([m.to_dict() for m in all_metrics], results_dir / "summary.csv")
        if all_predictions:
            write_csv([p.to_dict() for p in all_predictions], results_dir / "predictions.csv")
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "study_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    # ---- development stage ------------------------------------------------
    for name in dev_names:
        if name not in TEMPLATES:
            print(f"skipping unknown template {name!r}", file=sys.stderr)
            continue
        print(f"[dev] {name}", flush=True)
        try:
            metrics, predictions = run_stage(runner, name, splits["dev"], "dev")
        except ModelUnavailableError as error:
            print(f"model unavailable: {error}", file=sys.stderr)
            meta["failure"] = f"model unavailable: {error}"
            break
        except Exception as error:  # pragma: no cover - runtime dependent
            print(f"[dev] {name} failed: {error}", file=sys.stderr)
            traceback.print_exc()
            meta.setdefault("template_failures", {})[name] = str(error)
            continue
        meta["model_ran"] = True
        dev_metrics.append(metrics)
        all_metrics.append(metrics)
        all_predictions.extend(predictions)
        print(
            f"      accuracy={metrics.accuracy:.3f} validity={metrics.format_validity:.3f} "
            f"chapter={metrics.chapter_agreement:.3f} diversity={metrics.code_diversity:.3f} "
            f"contaminated_correct={metrics.contaminated_correct}",
            flush=True,
        )
        flush()

    if dev_metrics:
        meta["stages_completed"].append("dev")

    # ---- test stage -------------------------------------------------------
    test_predictions: list[Prediction] = []
    if dev_metrics and not args.dev_only:
        # Rank on accuracy, then on format validity, then on chapter agreement.
        best = max(
            dev_metrics,
            key=lambda m: (m.accuracy, m.format_validity, m.chapter_agreement),
        )
        meta["dev_winner"] = best.template
        meta["dev_winner_accuracy"] = best.accuracy
        print(f"\ndev winner: {best.template} (accuracy {best.accuracy:.3f})\n")

        comparators = tuple(args.test_templates) if args.test_templates else TEST_COMPARATORS
        test_names = [best.template] + [
            name for name in comparators if name != best.template and name in TEMPLATES
        ]
        test_names = test_names[:5]
        meta["test_templates"] = test_names

        for name in test_names:
            print(f"[test] {name}", flush=True)
            try:
                metrics, predictions = run_stage(runner, name, splits["test"], "test")
            except ModelUnavailableError as error:
                meta["failure"] = f"model unavailable during test stage: {error}"
                break
            except Exception as error:  # pragma: no cover - runtime dependent
                print(f"[test] {name} failed: {error}", file=sys.stderr)
                meta.setdefault("template_failures", {})[f"test:{name}"] = str(error)
                continue
            all_metrics.append(metrics)
            all_predictions.extend(predictions)
            test_predictions.extend(predictions)
            print(
                f"      accuracy={metrics.accuracy:.3f} validity={metrics.format_validity:.3f} "
                f"chapter={metrics.chapter_agreement:.3f} diversity={metrics.code_diversity:.3f} "
                f"contaminated_correct={metrics.contaminated_correct}",
                flush=True,
            )
            flush()
        if test_predictions:
            meta["stages_completed"].append("test")

    # ---- outputs ----------------------------------------------------------
    flush()

    focus = test_predictions or all_predictions
    if focus:
        # Templates frequently tie (often at zero correct), so the candidates are
        # ordered before the max to keep the chosen figure template deterministic.
        best_name = max(
            sorted(dict.fromkeys(p.template for p in focus)),
            key=lambda name: sum(p.is_correct for p in focus if p.template == name),
        )
        focus_subset = [p for p in focus if p.template == best_name]
        write_confusion_csv(focus_subset, results_dir / "chapter_confusion.csv")
        split_label = focus_subset[0].split
        figure_accuracy_vs_validity(
            [m for m in all_metrics if m.split == split_label],
            figures_dir / "accuracy_vs_format_validity.png",
            split_label,
        )
        figure_code_distribution(focus_subset, figures_dir / "predicted_code_distribution.png")
        figure_chapter_confusion(focus_subset, figures_dir / "chapter_confusion.png")
        meta["figure_template"] = best_name

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    flush()

    if not meta["model_ran"]:
        print(
            "\nThe model did not run in this environment; no metrics were produced.\n"
            "The pipeline and tests are unaffected. See results/study_meta.json.",
            file=sys.stderr,
        )
    else:
        print(f"\nwrote {results_dir}/summary.csv and figures to {figures_dir}")
    return 0


def _figures_only(results_dir: Path, figures_dir: Path) -> int:
    """Redraw figures from an existing ``predictions.csv``."""
    predictions_path = results_dir / "predictions.csv"
    summary_path = results_dir / "summary.csv"
    if not predictions_path.exists():
        print(f"no {predictions_path}; run the study first", file=sys.stderr)
        return 0

    def _to_bool(value: str) -> bool:
        return value.strip().lower() in {"true", "1", "yes"}

    with predictions_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    predictions = [
        Prediction(
            record_id=row["record_id"],
            template=row["template"],
            split=row["split"],
            true_code=row["true_code"],
            predicted_code=row["predicted_code"] or None,
            status=row["status"],
            method=row["method"],
            n_candidates=int(row["n_candidates"]),
            in_prompt_examples=_to_bool(row["in_prompt_examples"]),
            label_in_prompt_examples=_to_bool(row["label_in_prompt_examples"]),
            raw_output=row["raw_output"],
        )
        for row in rows
    ]
    test = [p for p in predictions if p.split == "test"] or predictions
    split_label = test[0].split
    best_name = max(
        sorted(dict.fromkeys(p.template for p in test)),
        key=lambda name: sum(p.is_correct for p in test if p.template == name),
    )
    subset = [p for p in test if p.template == best_name]

    metrics: list[TemplateMetrics] = []
    if summary_path.exists():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != split_label:
                    continue
                metrics.append(
                    TemplateMetrics(
                        template=row["template"],
                        family=row["family"],
                        split=row["split"],
                        n=int(row["n"]),
                        n_correct=int(row["n_correct"]),
                        accuracy=float(row["accuracy"]),
                        format_validity=float(row["format_validity"]),
                        chapter_agreement=float(row["chapter_agreement"]),
                        code_diversity=float(row["code_diversity"]),
                        n_unique_codes=int(row["n_unique_codes"]),
                        modal_code=row["modal_code"] or None,
                        modal_share=float(row["modal_share"]),
                        n_example_codes=int(row["n_example_codes"]),
                        example_echo_rate=float(row["example_echo_rate"]),
                        contaminated_correct=int(row["contaminated_correct"]),
                        contamination_share=float(row["contamination_share"]),
                        n_clean=int(row["n_clean"]),
                        clean_accuracy=float(row["clean_accuracy"]),
                        n_ambiguous=int(row["n_ambiguous"]),
                        mean_new_tokens=float(row["mean_new_tokens"]),
                    )
                )
    figure_accuracy_vs_validity(
        metrics, figures_dir / "accuracy_vs_format_validity.png", split_label
    )
    figure_code_distribution(subset, figures_dir / "predicted_code_distribution.png")
    figure_chapter_confusion(subset, figures_dir / "chapter_confusion.png")
    print(f"redrew figures in {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
