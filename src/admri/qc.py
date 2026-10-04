"""Scan-level quality control.

QC runs on the *raw* volume, before normalization, and returns a structured
report instead of raising, so that failing scans can be excluded with an
auditable reason rather than crashing the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from admri.config import QCConfig
from admri.preprocessing import foreground_mask


def background_snr(image: np.ndarray) -> float:
    """Foreground mean over the std of the eight corner patches (assumed air).

    Mirrors background-based SNR metrics used by MRIQC. A volume of pure
    noise scores ~1-2, while a usable structural scan scores far higher.
    """
    k = [max(2, n // 8) for n in image.shape]
    corners = [
        image[xs, ys, zs]
        for xs in (slice(0, k[0]), slice(-k[0], None))
        for ys in (slice(0, k[1]), slice(-k[1], None))
        for zs in (slice(0, k[2]), slice(-k[2], None))
    ]
    bg_std = float(np.concatenate([c.ravel() for c in corners]).std())
    mask = foreground_mask(image)
    if not mask.any():
        return 0.0
    return float(image[mask].mean()) / max(bg_std, 1e-6)


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class QCCheck:
    name: str
    status: Status
    message: str = ""


@dataclass(frozen=True)
class QCReport:
    checks: tuple[QCCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.status is not Status.FAIL for c in self.checks)

    @property
    def status(self) -> Status:
        if not self.passed:
            return Status.FAIL
        if any(c.status is Status.WARN for c in self.checks):
            return Status.WARN
        return Status.PASS

    def reasons(self, status: Status = Status.FAIL) -> list[str]:
        return [f"{c.name}: {c.message}" for c in self.checks if c.status is status]


def run_qc(
    image: np.ndarray,
    spacing: Sequence[float],
    affine: np.ndarray,
    config: QCConfig | None = None,
) -> QCReport:
    config = config or QCConfig()
    checks: list[QCCheck] = []

    def add(name: str, ok: bool, message: str, fail: Status = Status.FAIL) -> None:
        checks.append(QCCheck(name, Status.PASS if ok else fail, "" if ok else message))

    add("ndim", image.ndim == 3, f"expected 3D volume, got shape {image.shape}")
    if image.ndim != 3:
        return QCReport(tuple(checks))

    finite = bool(np.isfinite(image).all())
    add("finite", finite, "volume contains NaN or Inf values")
    add(
        "min_dim",
        min(image.shape) >= config.min_dim,
        f"shape {image.shape} below minimum dimension {config.min_dim}",
    )

    sp = np.asarray(spacing, dtype=float)
    add(
        "spacing_range",
        bool(np.all((sp >= config.min_spacing_mm) & (sp <= config.max_spacing_mm))),
        f"voxel spacing {tuple(sp.round(3))} outside "
        f"[{config.min_spacing_mm}, {config.max_spacing_mm}] mm",
    )
    if np.all(sp > 0):
        add(
            "anisotropy",
            float(sp.max() / sp.min()) <= config.max_anisotropy,
            f"anisotropy ratio {sp.max() / sp.min():.2f} > {config.max_anisotropy}",
            fail=Status.WARN,
        )
    add(
        "affine",
        affine.shape == (4, 4) and abs(float(np.linalg.det(affine[:3, :3]))) > 1e-8,
        "affine is singular or malformed",
    )

    if finite:
        ratio = float(foreground_mask(image).mean())
        add(
            "foreground_ratio",
            ratio >= config.min_foreground_ratio,
            f"foreground ratio {ratio:.4f} < {config.min_foreground_ratio} (empty/corrupt scan?)",
        )
        snr = background_snr(image)
        add("snr", snr >= config.min_snr, f"background SNR {snr:.1f} < {config.min_snr}")

    return QCReport(tuple(checks))
