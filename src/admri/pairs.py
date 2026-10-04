"""Longitudinal pair generation and delta-MRI computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from admri.bids import DIAGNOSIS_ORDER, Session

DAYS_PER_YEAR = 365.25


class RegistrationError(ValueError):
    """Raised when two scans are not in the same voxel space."""


@dataclass(frozen=True)
class PairSpec:
    subject_id: str
    baseline: Session
    followup: Session

    @property
    def pair_id(self) -> str:
        return f"{self.subject_id}_{self.baseline.session_id}_to_{self.followup.session_id}"

    @property
    def interval_days(self) -> float | None:
        b, f = self.baseline.days_since_baseline, self.followup.days_since_baseline
        return None if b is None or f is None else f - b

    @property
    def converter(self) -> bool | None:
        """True if diagnosis worsened (e.g. MCI -> AD) between the two visits."""
        b = DIAGNOSIS_ORDER.get(self.baseline.diagnosis or "")
        f = DIAGNOSIS_ORDER.get(self.followup.diagnosis or "")
        return None if b is None or f is None else f > b


def generate_pairs(
    subject_id: str, sessions: list[Session], strategy: str = "consecutive"
) -> list[PairSpec]:
    """Build (baseline, follow-up) pairs from chronologically ordered sessions.

    ``consecutive``: t0->t1, t1->t2, ...  ``baseline``: t0->t1, t0->t2, ...
    ``all``: every forward-in-time combination.
    """
    n = len(sessions)
    if strategy == "consecutive":
        idx = [(i, i + 1) for i in range(n - 1)]
    elif strategy == "baseline":
        idx = [(0, j) for j in range(1, n)]
    elif strategy == "all":
        idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(f"Unknown pair strategy: {strategy!r}")
    return [PairSpec(subject_id, sessions[i], sessions[j]) for i, j in idx]


def compute_delta(
    baseline: np.ndarray,
    followup: np.ndarray,
    baseline_affine: np.ndarray,
    followup_affine: np.ndarray,
    atol: float = 1e-3,
) -> np.ndarray:
    """Voxel-wise ``followup - baseline``.

    A voxel-wise difference is only meaningful if both volumes share the same
    grid, so shape and affine (orientation, spacing, origin) are verified.
    Scans must be registered to a common space upstream (e.g. ANTs, FSL FLIRT).
    """
    if baseline.shape != followup.shape:
        raise RegistrationError(f"Shape mismatch: {baseline.shape} vs {followup.shape}")
    if not np.allclose(baseline_affine, followup_affine, atol=atol):
        raise RegistrationError(
            "Affine mismatch: scans are not co-registered (register to a common space first)"
        )
    return (followup - baseline).astype(np.float32)
