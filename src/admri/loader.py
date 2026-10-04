"""Framework-agnostic access to pipeline outputs for model training.

:class:`DeltaPairDataset` implements ``__len__``/``__getitem__``, so it can be
passed directly to ``torch.utils.data.DataLoader`` without this package
depending on PyTorch. It refuses to open a split the audit has not cleared.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from admri.io import load_nifti
from admri.pipeline import PAIRS_CSV


class DeltaPairDataset:
    def __init__(
        self,
        output_dir: str | Path,
        split: str,
        transform: Callable[[np.ndarray], Any] | None = None,
        verify: bool = True,
    ) -> None:
        self.root = Path(output_dir)
        if verify:
            from admri.audit import audit

            result = audit(self.root)
            if not result.ok:
                failed = [name for name, ok, _ in result.checks if not ok]
                raise RuntimeError(f"Audit failed for {self.root}: {failed}")
        with open(self.root / PAIRS_CSV, newline="") as fh:
            self.rows = [r for r in csv.DictReader(fh) if r["split"] == split and r["delta_path"]]
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        delta, _, _ = load_nifti(self.root / row["delta_path"])
        x: Any = delta[np.newaxis]  # channel-first
        if self.transform is not None:
            x = self.transform(x)
        return {
            "delta": x,
            "converter": row["converter"] == "True",
            "interval_years": float(row["interval_years"]) if row["interval_years"] else np.nan,
            "subject_id": row["subject_id"],
            "pair_id": row["pair_id"],
        }
