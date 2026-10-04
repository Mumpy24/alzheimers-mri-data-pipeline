"""MRI-safe intensity preprocessing.

MRI intensities are in arbitrary scanner units, so scans must be normalized
before they can be compared across sessions. Statistics are computed on the
head/brain foreground only: in a typical T1w volume more than half of the
voxels are background air, which would otherwise dominate the percentiles.
"""

from __future__ import annotations

import numpy as np

from admri.config import NormalizationConfig


def foreground_mask(image: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Voxels brighter than ``fraction`` of the robust (99.5th pct) maximum."""
    robust_max = float(np.percentile(image, 99.5))
    if robust_max <= 0:
        return np.zeros(image.shape, dtype=bool)
    return image > fraction * robust_max


def robust_zscore(
    image: np.ndarray,
    config: NormalizationConfig | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Percentile-clip then z-score using foreground statistics.

    Background voxels are set to 0 so they do not contribute to downstream
    difference images.
    """
    config = config or NormalizationConfig()
    if mask is None:
        mask = foreground_mask(image, config.foreground_fraction)
    if not mask.any():
        raise ValueError("Empty foreground mask: cannot normalize")

    fg = image[mask]
    low, high = np.percentile(fg, [config.lower_percentile, config.upper_percentile])
    if high <= low:
        raise ValueError("Degenerate intensity range in foreground")

    clipped = np.clip(fg, low, high)
    mean, std = float(clipped.mean()), float(clipped.std())
    if std == 0:
        raise ValueError("Zero foreground variance after clipping")

    out = np.zeros_like(image, dtype=np.float32)
    out[mask] = (clipped - mean) / std
    return out
