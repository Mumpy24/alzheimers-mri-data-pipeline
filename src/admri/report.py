"""Human-readable Markdown report for a pipeline run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from admri.pipeline import PipelineResult


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(card: dict[str, Any], result: PipelineResult) -> str:
    c = card["counts"]
    dx_labels = sorted({dx for v in c["baseline_diagnosis_by_split"].values() for dx in v})
    split_rows = [
        [
            name,
            c["subjects_by_split"][name],
            *[c["baseline_diagnosis_by_split"][name].get(dx, 0) for dx in dx_labels],
            c["pairs_by_split"].get(name, 0),
        ]
        for name in c["subjects_by_split"]
    ]
    failed = [s for s in result.scans if s.qc_status == "fail"]
    warned = [s for s in result.scans if s.qc_status == "warn"]

    parts = [
        f"# Dataset report `{card['dataset_version']}`",
        f"Generated {card['created_at']} with admri {card['software']['admri']}.",
        "## Splits (subject-level)",
        _table(
            ["split", "subjects", *[f"{dx} (baseline)" for dx in dx_labels], "pairs"], split_rows
        ),
        f"Converter pairs (diagnosis worsened between visits): **{c['converter_pairs']}**",
        "## Quality control",
        _table(["status", "scans"], [[k, v] for k, v in c["scans_by_qc_status"].items()]),
    ]
    if failed:
        parts += [
            "### Excluded scans",
            _table(
                ["subject", "session", "reason"],
                [[s.subject_id, s.session_id, s.qc_reasons] for s in failed],
            ),
        ]
    if warned:
        parts += [
            "### Warnings",
            _table(
                ["subject", "session", "reason"],
                [[s.subject_id, s.session_id, s.qc_reasons] for s in warned],
            ),
        ]
    if result.excluded_pairs:
        parts += [
            "### Excluded pairs",
            _table(
                ["pair", "reason"], [[p["pair_id"], p["reason"]] for p in result.excluded_pairs]
            ),
        ]
    parts += [
        "## Leakage checks",
        "\n".join(f"- [x] {k.replace('_', ' ')}" for k, v in card["leakage_checks"].items() if v),
        "Run `admri audit <output_dir>` to re-verify every hash and invariant.",
    ]
    return "\n\n".join(parts) + "\n"
