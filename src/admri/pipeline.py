"""End-to-end pipeline: discover -> QC -> normalize -> split -> pair -> delta -> manifest.

Images are processed one scan (or one pair) at a time, so memory use is
bounded by the size of two volumes regardless of dataset size.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from admri import lineage
from admri.bids import Session, Subject, discover_bids
from admri.config import PipelineConfig
from admri.io import load_nifti, save_nifti, sha256_file
from admri.pairs import DAYS_PER_YEAR, RegistrationError, compute_delta, generate_pairs
from admri.preprocessing import robust_zscore
from admri.qc import Status, run_qc
from admri.splits import check_disjoint, split_subjects

logger = logging.getLogger(__name__)

SCANS_CSV = "scans.csv"
PAIRS_CSV = "pairs.csv"
SPLITS_JSON = "splits.json"
CARD_JSON = "dataset_card.json"
REPORT_MD = "report.md"


@dataclass
class ScanRecord:
    subject_id: str
    session_id: str
    source_path: str
    source_sha256: str
    shape: str = ""
    voxel_spacing: str = ""
    qc_status: str = ""
    qc_reasons: str = ""
    processed_path: str = ""
    processed_sha256: str = ""
    split: str = ""

    @property
    def usable(self) -> bool:
        return self.qc_status != Status.FAIL.value


@dataclass
class PairRecord:
    pair_id: str
    subject_id: str
    split: str
    baseline_session: str
    followup_session: str
    interval_days: float | None
    interval_years: float | None
    baseline_diagnosis: str | None
    followup_diagnosis: str | None
    converter: bool | None
    delta_path: str = ""
    delta_sha256: str = ""
    delta_mean_abs: float | None = None


@dataclass
class PipelineResult:
    output_dir: Path
    version: str
    scans: list[ScanRecord]
    pairs: list[PairRecord]
    splits: dict[str, list[str]]
    strata: dict[str, str | None] = field(default_factory=dict)
    excluded_pairs: list[dict[str, str]] = field(default_factory=list)

    @property
    def card_path(self) -> Path:
        return self.output_dir / CARD_JSON


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def process_scan(
    subject_id: str, session: Session, bids_root: Path, out_dir: Path, config: PipelineConfig
) -> ScanRecord:
    """Load, QC, normalize and save a single scan. Never raises on bad data."""
    record = ScanRecord(
        subject_id=subject_id,
        session_id=session.session_id,
        source_path=_relative(session.path, bids_root),
        source_sha256=sha256_file(session.path),
    )
    try:
        data, affine, spacing = load_nifti(session.path)
    except Exception as exc:  # corrupt header, truncated gzip, ...
        record.qc_status, record.qc_reasons = Status.FAIL.value, f"load: {exc}"
        return record

    record.shape = "x".join(map(str, data.shape))
    record.voxel_spacing = "x".join(f"{s:g}" for s in spacing)
    report = run_qc(data, spacing, affine, config.qc)
    record.qc_status = report.status.value
    record.qc_reasons = "; ".join(report.reasons(Status.FAIL) + report.reasons(Status.WARN))
    if not report.passed:
        return record

    try:
        normalized = robust_zscore(data, config.normalization)
    except ValueError as exc:
        record.qc_status, record.qc_reasons = Status.FAIL.value, f"normalization: {exc}"
        return record

    rel = Path("processed") / subject_id / f"{subject_id}_{session.session_id}_T1w_norm.nii.gz"
    record.processed_sha256 = save_nifti(normalized, affine, out_dir / rel)
    record.processed_path = rel.as_posix()
    return record


def _process_scan_args(args: tuple) -> ScanRecord:
    return process_scan(*args)


def flag_duplicate_content(scans: list[ScanRecord]) -> int:
    """Exclude scans whose bytes are identical to another scan.

    The same image filed under two subject IDs is a classic source of
    train/test leakage that subject-level splitting alone cannot catch.
    """
    by_hash: dict[str, list[ScanRecord]] = defaultdict(list)
    for s in scans:
        by_hash[s.source_sha256].append(s)
    n = 0
    for group in by_hash.values():
        if len(group) < 2:
            continue
        ids = ", ".join(f"{s.subject_id}/{s.session_id}" for s in group)
        for s in group:
            s.qc_status = Status.FAIL.value
            s.qc_reasons = "; ".join(filter(None, [s.qc_reasons, f"duplicate_content: {ids}"]))
            n += 1
    return n


def _build_pair(
    spec_args: tuple[Any, ...], out_dir: Path, config: PipelineConfig
) -> PairRecord | dict[str, str]:
    spec, split, base_rec, fu_rec = spec_args
    interval = spec.interval_days
    record = PairRecord(
        pair_id=spec.pair_id,
        subject_id=spec.subject_id,
        split=split,
        baseline_session=spec.baseline.session_id,
        followup_session=spec.followup.session_id,
        interval_days=interval,
        interval_years=round(interval / DAYS_PER_YEAR, 4) if interval is not None else None,
        baseline_diagnosis=spec.baseline.diagnosis,
        followup_diagnosis=spec.followup.diagnosis,
        converter=spec.converter,
    )
    if not config.write_deltas:
        return record
    b, b_aff, _ = load_nifti(out_dir / base_rec.processed_path)
    f, f_aff, _ = load_nifti(out_dir / fu_rec.processed_path)
    try:
        delta = compute_delta(b, f, b_aff, f_aff, atol=config.affine_tolerance)
    except RegistrationError as exc:
        return {"pair_id": spec.pair_id, "reason": str(exc)}
    rel = Path("deltas") / split / f"{spec.pair_id}_delta.nii.gz"
    record.delta_sha256 = save_nifti(delta, b_aff, out_dir / rel)
    record.delta_path = rel.as_posix()
    mask = (b != 0) | (f != 0)
    record.delta_mean_abs = round(float(abs(delta[mask]).mean()), 6) if mask.any() else 0.0
    return record


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    bids_root: str | Path,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
    workers: int = 1,
) -> PipelineResult:
    config = config or PipelineConfig()
    bids_root, out_dir = Path(bids_root), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_bids(bids_root)
    n_scans = sum(len(s.sessions) for s in subjects)
    logger.info("Discovered %d subjects / %d T1w scans in %s", len(subjects), n_scans, bids_root)

    # 1. Per-scan ingestion, QC and normalization.
    jobs = [(s.subject_id, ses, bids_root, out_dir, config) for s in subjects for ses in s.sessions]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scans = list(pool.map(_process_scan_args, jobs, chunksize=4))
    else:
        scans = [_process_scan_args(j) for j in jobs]
    n_dup = flag_duplicate_content(scans)
    n_failed = sum(not s.usable for s in scans)
    logger.info("QC: %d/%d scans failed (%d duplicate-content)", n_failed, len(scans), n_dup)

    # 2. Subject-level split over subjects with at least one usable scan.
    usable: dict[str, list[ScanRecord]] = defaultdict(list)
    for s in scans:
        if s.usable:
            usable[s.subject_id].append(s)
    by_id: dict[str, Subject] = {s.subject_id: s for s in subjects}
    strata = {sid: by_id[sid].baseline_diagnosis for sid in usable}
    splits = split_subjects(usable.keys(), config.split, strata)
    split_of = {sid: name for name, members in splits.items() for sid in members}
    for s in scans:
        s.split = split_of.get(s.subject_id, "")

    # 3. Longitudinal pairs and delta images, built only within a subject.
    pair_jobs = []
    for subject in subjects:
        sid = subject.subject_id
        if sid not in usable:
            continue
        recs = {r.session_id: r for r in usable[sid]}
        sessions = [ses for ses in subject.sessions if ses.session_id in recs]
        for spec in generate_pairs(sid, sessions, config.pair_strategy):
            pair_jobs.append(
                (
                    spec,
                    split_of[sid],
                    recs[spec.baseline.session_id],
                    recs[spec.followup.session_id],
                )
            )
    pairs: list[PairRecord] = []
    excluded_pairs: list[dict[str, str]] = []
    for job in pair_jobs:
        built = _build_pair(job, out_dir, config)
        if isinstance(built, PairRecord):
            pairs.append(built)
        else:
            excluded_pairs.append(built)
    logger.info("Built %d longitudinal pairs (%d excluded)", len(pairs), len(excluded_pairs))

    # 4. Leakage gate: hard failure, never a warning.
    check_disjoint(splits)
    for p in pairs:
        if split_of[p.subject_id] != p.split:
            raise RuntimeError(f"Pair {p.pair_id} assigned to wrong split")

    # 5. Manifests, lineage and report.
    version = lineage.dataset_version(config, scans)
    result = PipelineResult(out_dir, version, scans, pairs, splits, strata, excluded_pairs)
    _write_csv(out_dir / SCANS_CSV, [asdict(s) for s in scans])
    _write_csv(out_dir / PAIRS_CSV, [asdict(p) for p in pairs])
    (out_dir / SPLITS_JSON).write_text(json.dumps(splits, indent=2) + "\n")
    card = lineage.build_card(result, config, bids_root)
    (out_dir / CARD_JSON).write_text(json.dumps(card, indent=2) + "\n")

    from admri.report import render_report

    (out_dir / REPORT_MD).write_text(render_report(card, result))
    logger.info("Dataset version %s written to %s", version, out_dir)
    return result
