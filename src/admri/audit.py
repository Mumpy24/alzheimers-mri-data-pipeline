"""Independent verification of a pipeline output directory.

``admri audit`` re-derives every invariant from the files on disk rather
than trusting the pipeline's own bookkeeping: manifest and image hashes,
split disjointness, pair/split consistency, duplicate content and the
dataset version. It is suitable as a CI gate before training.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from admri.config import PipelineConfig
from admri.io import sha256_file
from admri.lineage import dataset_version
from admri.pipeline import CARD_JSON, PAIRS_CSV, SCANS_CSV, SPLITS_JSON, ScanRecord


@dataclass
class AuditResult:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def add(self, name: str, problems: list[str]) -> None:
        detail = "; ".join(problems[:5]) + (
            f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""
        )
        self.checks.append((name, not problems, detail))


def _read_csv(path: Path) -> list[dict[str, str]]:
    text = path.read_text()
    return list(csv.DictReader(text.splitlines())) if text else []


def audit(output_dir: str | Path, bids_root: str | Path | None = None) -> AuditResult:
    out = Path(output_dir)
    result = AuditResult()

    missing = [n for n in (CARD_JSON, SCANS_CSV, PAIRS_CSV, SPLITS_JSON) if not (out / n).exists()]
    result.add("manifests_present", missing)
    if missing:
        return result

    card = json.loads((out / CARD_JSON).read_text())
    scans = _read_csv(out / SCANS_CSV)
    pairs = _read_csv(out / PAIRS_CSV)
    splits: dict[str, list[str]] = json.loads((out / SPLITS_JSON).read_text())

    result.add(
        "manifest_hashes",
        [
            f"{name} modified"
            for name, digest in card["manifests"].items()
            if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest
        ],
    )

    file_problems = []
    for row, path_key, hash_key in [(s, "processed_path", "processed_sha256") for s in scans] + [
        (p, "delta_path", "delta_sha256") for p in pairs
    ]:
        if not row[path_key]:
            continue
        f = out / row[path_key]
        if not f.exists():
            file_problems.append(f"missing {row[path_key]}")
        elif sha256_file(f) != row[hash_key]:
            file_problems.append(f"hash mismatch {row[path_key]}")
    result.add("output_file_hashes", file_problems)

    root = Path(bids_root) if bids_root else Path(card["source"]["bids_root"])
    if root.is_dir():
        result.add(
            "source_file_hashes",
            [
                f"source changed: {s['source_path']}"
                for s in scans
                if not (root / s["source_path"]).exists()
                or sha256_file(root / s["source_path"]) != s["source_sha256"]
            ],
        )

    seen: dict[str, str] = {}
    overlap = []
    for name, ids in splits.items():
        for sid in ids:
            if sid in seen:
                overlap.append(f"{sid} in {seen[sid]} and {name}")
            seen[sid] = name
    result.add("splits_disjoint", overlap)

    result.add(
        "scan_split_consistency",
        [
            f"{s['subject_id']}/{s['session_id']} split={s['split']!r}"
            for s in scans
            if s["qc_status"] != "fail" and seen.get(s["subject_id"]) != s["split"]
        ],
    )
    result.add(
        "pair_split_consistency",
        [
            f"{p['pair_id']} in {p['split']}"
            for p in pairs
            if seen.get(p["subject_id"]) != p["split"]
        ],
    )

    by_hash: dict[str, set[str]] = defaultdict(set)
    for s in scans:
        if s["qc_status"] != "fail":
            by_hash[s["source_sha256"]].add(f"{s['subject_id']}/{s['session_id']}")
    result.add(
        "no_duplicate_content", [", ".join(sorted(v)) for v in by_hash.values() if len(v) > 1]
    )

    records = [ScanRecord(**{k: s[k] for k in ScanRecord.__dataclass_fields__}) for s in scans]
    expected = dataset_version(PipelineConfig.from_dict(card["config"]), records)
    result.add(
        "dataset_version",
        [] if expected == card["dataset_version"] else [f"{card['dataset_version']} != {expected}"],
    )
    return result
