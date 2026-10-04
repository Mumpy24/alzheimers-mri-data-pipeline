"""Render docs/delta_example.png from real pipeline outputs on synthetic data.

python scripts/make_figure.py
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from admri.config import PipelineConfig
from admri.io import load_nifti
from admri.pipeline import run_pipeline
from admri.synthetic import make_synthetic_bids

OUT = Path(__file__).resolve().parents[1] / "docs" / "delta_example.png"
TEXT, TEXT_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#f0efec", "#b3322f"])
GRAY = plt.get_cmap("gray").with_extremes(bad="black")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bids = make_synthetic_bids(Path(tmp) / "bids", n_subjects=40, seed=0)
        out = Path(tmp) / "out"
        run_pipeline(bids, out, PipelineConfig(pair_strategy="baseline"))
        with open(out / "pairs.csv") as fh:
            pairs = list(csv.DictReader(fh))

        def longest(dx: str) -> dict[str, str]:
            rows = [p for p in pairs if p["baseline_diagnosis"] == dx]
            return max(rows, key=lambda p: float(p["interval_years"]))

        chosen = [("CN", longest("CN")), ("AD", longest("AD"))]
        fig, axes = plt.subplots(2, 3, figsize=(9, 6.4), facecolor=SURFACE, layout="constrained")
        lim = 3.0
        for row, (dx, pair) in enumerate(chosen):
            sid = pair["subject_id"]
            base, _, _ = load_nifti(
                out / "processed" / sid / f"{sid}_{pair['baseline_session']}_T1w_norm.nii.gz"
            )
            fu, _, _ = load_nifti(
                out / "processed" / sid / f"{sid}_{pair['followup_session']}_T1w_norm.nii.gz"
            )
            delta, _, _ = load_nifti(out / pair["delta_path"])
            z = base.shape[2] // 2 + 1
            years = float(pair["interval_years"])
            panels = [
                (
                    np.ma.masked_equal(base, 0),
                    f"Baseline ({pair['baseline_session']})",
                    GRAY,
                    (-2.5, 2.5),
                ),
                (np.ma.masked_equal(fu, 0), f"Follow-up (+{years:.1f} y)", GRAY, (-2.5, 2.5)),
                (delta, "Δ = follow-up − baseline", DIVERGING, (-lim, lim)),  # noqa: RUF001
            ]
            for col, (vol, title, cmap, (lo, hi)) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(vol[:, :, z].T, cmap=cmap, vmin=lo, vmax=hi, origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_title(title, color=TEXT, fontsize=10)
            axes[row, 0].set_ylabel(f"{dx} · {sid}", color=TEXT_2, fontsize=11)

        cbar = fig.colorbar(im, ax=axes[:, 2], shrink=0.7)
        cbar.set_label("Intensity change (z-units)", color=TEXT_2)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=TEXT_2)
        fig.suptitle(
            "Longitudinal Δ-MRI: ventricles enlarge and the brain shrinks faster in AD",
            color=TEXT,
            fontsize=12,
        )
        OUT.parent.mkdir(exist_ok=True)
        fig.savefig(OUT, dpi=130, facecolor=SURFACE, bbox_inches="tight")
        print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
