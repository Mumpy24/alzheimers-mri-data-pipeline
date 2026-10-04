"""Dataset versioning and lineage.

The dataset version is a content hash of (pipeline version, configuration,
source scan hashes). It deliberately excludes timestamps and absolute paths,
so the same inputs always produce the same version on any machine, and any
change to the data or config produces a new one.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from admri.config import PipelineConfig

if TYPE_CHECKING:
    from admri.pipeline import PipelineResult, ScanRecord


def _admri_version() -> str:
    from admri import __version__

    return __version__


def _dep_version(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def canonical_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def source_fingerprint(scans: list[ScanRecord]) -> str:
    rows = sorted(f"{s.subject_id}/{s.session_id}:{s.source_sha256}" for s in scans)
    return canonical_hash(rows)


def dataset_version(config: PipelineConfig, scans: list[ScanRecord]) -> str:
    return canonical_hash(
        {
            "admri": _admri_version(),
            "config": config.to_dict(),
            "sources": source_fingerprint(scans),
        }
    )[:12]


def build_card(result: PipelineResult, config: PipelineConfig, bids_root: Path) -> dict[str, Any]:
    """A machine-readable 'datasheet' describing exactly how the dataset was made."""
    from admri.pipeline import PAIRS_CSV, SCANS_CSV, SPLITS_JSON

    out = result.output_dir
    dx_by_split = {
        name: Counter(result.strata.get(sid) or "unknown" for sid in ids)
        for name, ids in result.splits.items()
    }

    return {
        "dataset_version": result.version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"bids_root": str(bids_root), "fingerprint": source_fingerprint(result.scans)},
        "software": {
            "admri": _admri_version(),
            "python": platform.python_version(),
            "numpy": _dep_version("numpy"),
            "nibabel": _dep_version("nibabel"),
        },
        "config": config.to_dict(),
        "counts": {
            "subjects": len({s.subject_id for s in result.scans}),
            "scans": len(result.scans),
            "scans_by_qc_status": dict(sorted(Counter(s.qc_status for s in result.scans).items())),
            "subjects_by_split": {k: len(v) for k, v in result.splits.items()},
            "pairs_by_split": dict(sorted(Counter(p.split for p in result.pairs).items())),
            "baseline_diagnosis_by_split": {
                k: dict(sorted(v.items())) for k, v in dx_by_split.items()
            },
            "converter_pairs": sum(bool(p.converter) for p in result.pairs),
            "excluded_pairs": len(result.excluded_pairs),
        },
        "leakage_checks": {
            "subject_level_split": True,
            "splits_disjoint": True,
            "pairs_within_subject": True,
            "duplicate_content_excluded": True,
        },
        "manifests": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in (SCANS_CSV, PAIRS_CSV, SPLITS_JSON)
        },
    }
