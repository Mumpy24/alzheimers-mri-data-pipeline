"""BIDS dataset discovery and the subject/session data model.

Expected layout (a subset of BIDS)::

    root/
      participants.tsv                    # participant_id, diagnosis, age, sex, ...
      sub-001/
        sub-001_sessions.tsv              # session_id, days_since_baseline, diagnosis
        ses-01/anat/sub-001_ses-01_T1w.nii.gz
        ses-02/anat/sub-001_ses-02_T1w.nii.gz

Only paths and tabular metadata are read here; image data is loaded lazily
by the pipeline, one scan at a time.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from admri.io import is_nifti

logger = logging.getLogger(__name__)

NA_VALUES = {"", "n/a", "na", "nan"}
DIAGNOSIS_ORDER = {"CN": 0, "MCI": 1, "AD": 2}


@dataclass(frozen=True)
class Session:
    session_id: str
    path: Path
    days_since_baseline: float | None = None
    diagnosis: str | None = None


@dataclass
class Subject:
    subject_id: str
    sessions: list[Session] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sort_sessions()

    def sort_sessions(self) -> None:
        """Order sessions chronologically.

        Uses ``days_since_baseline`` when every session has it, otherwise a
        natural sort of the session label (so ``ses-10`` follows ``ses-9``).
        """
        if self.sessions and all(s.days_since_baseline is not None for s in self.sessions):
            self.sessions.sort(key=lambda s: (s.days_since_baseline, natural_key(s.session_id)))
        else:
            self.sessions.sort(key=lambda s: natural_key(s.session_id))
        ids = [s.session_id for s in self.sessions]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate session IDs for {self.subject_id}: {ids}")

    @property
    def baseline_diagnosis(self) -> str | None:
        dx = self.attributes.get("diagnosis")
        if dx:
            return dx
        return self.sessions[0].diagnosis if self.sessions else None


def natural_key(label: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", label))


def _clean(value: str | None) -> str | None:
    if value is None or value.strip().lower() in NA_VALUES:
        return None
    return value.strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _read_sessions_tsv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["session_id"]: row for row in read_tsv(path)}


def _find_t1w(anat_dir: Path) -> Path | None:
    candidates = sorted(p for p in anat_dir.glob("*_T1w.nii*") if is_nifti(p))
    if len(candidates) > 1:
        logger.warning("%d T1w runs in %s; using %s", len(candidates), anat_dir, candidates[0].name)
    return candidates[0] if candidates else None


def discover_bids(root: str | Path) -> list[Subject]:
    """Discover T1w scans in a BIDS dataset, sorted by subject ID."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS root not found: {root}")

    participants: dict[str, dict[str, str]] = {}
    if (root / "participants.tsv").exists():
        for row in read_tsv(root / "participants.tsv"):
            pid = row.pop("participant_id")
            participants[pid] = {k: v for k, v in row.items() if _clean(v) is not None}

    subjects = []
    for sub_dir in sorted(root.glob("sub-*"), key=lambda p: natural_key(p.name)):
        if not sub_dir.is_dir():
            continue
        sid = sub_dir.name
        session_meta = _read_sessions_tsv(sub_dir / f"{sid}_sessions.tsv")
        sessions = []
        ses_dirs = sorted(sub_dir.glob("ses-*")) or [sub_dir]
        for ses_dir in ses_dirs:
            ses_id = ses_dir.name if ses_dir is not sub_dir else "ses-01"
            t1w = _find_t1w(ses_dir / "anat")
            if t1w is None:
                continue
            meta = session_meta.get(ses_id, {})
            days = _clean(meta.get("days_since_baseline"))
            sessions.append(
                Session(
                    session_id=ses_id,
                    path=t1w,
                    days_since_baseline=float(days) if days is not None else None,
                    diagnosis=_clean(meta.get("diagnosis")),
                )
            )
        if sessions:
            subjects.append(Subject(sid, sessions, participants.get(sid, {})))
        else:
            logger.warning("No T1w scans found for %s", sid)
    return subjects
