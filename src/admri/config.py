"""Pipeline configuration.

Every parameter that can change the produced dataset lives here, so that the
dataset version hash (see :mod:`admri.lineage`) can be derived from it.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SPLIT_METHODS = ("stratified", "hash")
PAIR_STRATEGIES = ("consecutive", "baseline", "all")


@dataclass(frozen=True)
class QCConfig:
    """Thresholds for scan-level quality control."""

    min_dim: int = 32
    min_foreground_ratio: float = 0.05
    min_spacing_mm: float = 0.5
    max_spacing_mm: float = 3.0
    max_anisotropy: float = 3.0
    min_snr: float = 5.0


@dataclass(frozen=True)
class NormalizationConfig:
    """Robust, foreground-only intensity normalization."""

    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    foreground_fraction: float = 0.05


@dataclass(frozen=True)
class SplitConfig:
    """Subject-level split configuration."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    method: str = "stratified"
    seed: int = 42

    def __post_init__(self) -> None:
        if abs(self.train + self.val + self.test - 1.0) > 1e-6:
            raise ValueError("train/val/test fractions must sum to 1")
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("split fractions must be non-negative")
        if self.method not in SPLIT_METHODS:
            raise ValueError(f"split method must be one of {SPLIT_METHODS}, got {self.method!r}")


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for :func:`admri.pipeline.run_pipeline`."""

    qc: QCConfig = field(default_factory=QCConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    pair_strategy: str = "consecutive"
    affine_tolerance: float = 1e-3
    write_deltas: bool = True

    def __post_init__(self) -> None:
        if self.pair_strategy not in PAIR_STRATEGIES:
            raise ValueError(
                f"pair_strategy must be one of {PAIR_STRATEGIES}, got {self.pair_strategy!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        nested = {"qc": QCConfig, "normalization": NormalizationConfig, "split": SplitConfig}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            kwargs[key] = nested[key](**value) if key in nested else value
        return cls(**kwargs)

    @classmethod
    def from_toml(cls, path: str | Path) -> PipelineConfig:
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))
