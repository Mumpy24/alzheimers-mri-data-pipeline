"""Synthetic longitudinal BIDS dataset with simulated neurodegeneration.

Real Alzheimer's cohorts (ADNI, OASIS) require a data use agreement, so this
module generates a small, fully reproducible stand-in for development, tests
and CI. Each subject gets a brain phantom (white matter, gray matter,
ventricles, CSF, skull) that atrophies over time at a diagnosis-dependent
rate: ventricles enlarge and the brain shrinks fastest in AD, as observed in
longitudinal MRI studies. Scanner gain/offset and noise vary per scan, which
is what intensity normalization has to undo.

``inject_faults=True`` adds the kinds of problems real datasets contain, so
the QC and audit layers have something to catch.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

# (fractional brain shrinkage per year, fractional ventricle growth per year)
ATROPHY_RATES = {"CN": (0.004, 0.02), "MCI": (0.010, 0.05), "AD": (0.020, 0.09)}
CONVERTER_RATES = (0.016, 0.075)
INTENSITY = {"wm": 700.0, "gm": 480.0, "csf": 140.0, "skull": 300.0}


def _ellipsoid_radius(
    shape: tuple[int, int, int],
    radii: tuple[float, float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    axes = [np.linspace(-1, 1, n) for n in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return np.sqrt(
        ((x - center[0]) / radii[0]) ** 2
        + ((y - center[1]) / radii[1]) ** 2
        + ((z - center[2]) / radii[2]) ** 2
    )


def render_phantom(
    shape: tuple[int, int, int],
    head_scale: float,
    brain_scale: float,
    ventricle_scale: float,
    rng: np.random.Generator,
    gain: float = 1.0,
    offset: float = 0.0,
    noise_sd: float = 18.0,
) -> np.ndarray:
    """Render one T1w-like volume (bright WM, darker GM, dark CSF)."""
    s = head_scale
    img = np.zeros(shape, dtype=np.float32)

    head = _ellipsoid_radius(shape, (0.92 * s, 0.95 * s, 0.88 * s))
    img[head <= 1.0] = INTENSITY["skull"]
    img[head <= 0.93] = INTENSITY["csf"]

    b = s * brain_scale
    brain = _ellipsoid_radius(shape, (0.78 * b, 0.84 * b, 0.74 * b))
    img[brain <= 1.0] = INTENSITY["gm"]
    img[brain <= 0.80] = INTENSITY["wm"]

    v = s * ventricle_scale
    for cx in (-0.09, 0.09):
        vent = _ellipsoid_radius(shape, (0.07 * v, 0.22 * v, 0.10 * v), center=(cx, 0.02, 0.05))
        img[vent <= 1.0] = INTENSITY["csf"]

    img = img * gain + offset
    img = img + rng.normal(0.0, noise_sd, size=shape).astype(np.float32)
    img[head > 1.02] = np.abs(rng.normal(0.0, 4.0, size=int((head > 1.02).sum())))
    return np.clip(img, 0, None)


def _write(img: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(np.round(img).astype(np.int16), affine)
    nii.header.set_xyzt_units("mm")
    nib.save(nii, str(path))


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def make_synthetic_bids(
    root: str | Path,
    n_subjects: int = 40,
    shape: tuple[int, int, int] = (48, 56, 48),
    spacing_mm: float = 3.0,
    seed: int = 0,
    inject_faults: bool = False,
    overwrite: bool = False,
) -> Path:
    root = Path(root)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} exists (use overwrite=True)")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    affine[:3, 3] = -np.array(shape) * spacing_mm / 2

    participants = []
    diagnoses = rng.choice(["CN", "MCI", "AD"], size=n_subjects, p=[0.4, 0.35, 0.25])
    for i, dx in enumerate(diagnoses, start=1):
        sid = f"sub-{i:03d}"
        n_sessions = int(rng.integers(2, 5))
        converts_at = (
            int(rng.integers(1, n_sessions)) if dx == "MCI" and rng.random() < 0.4 else None
        )
        brain_rate, vent_rate = CONVERTER_RATES if converts_at else ATROPHY_RATES[str(dx)]
        head_scale = float(rng.uniform(0.95, 1.03))
        vent0 = float(rng.uniform(0.9, 1.1)) * {"CN": 1.0, "MCI": 1.15, "AD": 1.35}[str(dx)]
        age = float(rng.uniform(60, 85))
        participants.append(
            {
                "participant_id": sid,
                "diagnosis": dx,
                "age": round(age, 1),
                "sex": rng.choice(["F", "M"]),
            }
        )

        sessions = []
        days = 0.0
        for k in range(n_sessions):
            if k:
                days += float(rng.normal(365, 30))
            years = days / 365.25
            ses_dx = "AD" if converts_at is not None and k >= converts_at else str(dx)
            img = render_phantom(
                shape,
                head_scale=head_scale,
                brain_scale=(1 - brain_rate) ** years,
                ventricle_scale=vent0 * (1 + vent_rate) ** years,
                rng=rng,
                gain=float(rng.uniform(0.8, 1.25)),
                offset=float(rng.uniform(0, 40)),
            )
            ses = f"ses-{k + 1:02d}"
            _write(img, affine, root / sid / ses / "anat" / f"{sid}_{ses}_T1w.nii.gz")
            sessions.append(
                {"session_id": ses, "days_since_baseline": round(days), "diagnosis": ses_dx}
            )
        _write_tsv(root / sid / f"{sid}_sessions.tsv", sessions)

    if inject_faults:
        _inject_faults(root, rng, affine, participants)

    _write_tsv(root / "participants.tsv", participants)
    (root / "dataset_description.json").write_text(
        json.dumps(
            {
                "Name": "admri synthetic longitudinal AD phantom",
                "BIDSVersion": "1.9.0",
                "DatasetType": "raw",
                "GeneratedBy": [{"Name": "admri.synthetic", "Description": f"seed={seed}"}],
            },
            indent=2,
        )
        + "\n"
    )
    return root


def _inject_faults(
    root: Path, rng: np.random.Generator, affine: np.ndarray, participants: list[dict]
) -> None:
    """Corrupt a few scans in realistic ways."""
    subs = sorted(p for p in root.glob("sub-*") if p.is_dir())

    # 1. NaN slab (e.g. failed reconstruction).
    f = subs[0] / "ses-02" / "anat" / f"{subs[0].name}_ses-02_T1w.nii.gz"
    img = nib.load(str(f))
    data = np.asarray(img.dataobj, dtype=np.float32)
    data[:, :, data.shape[2] // 2] = np.nan
    nib.save(nib.Nifti1Image(data, img.affine), str(f))

    # 2. Near-empty volume (aborted acquisition).
    f = subs[1] / "ses-01" / "anat" / f"{subs[1].name}_ses-01_T1w.nii.gz"
    _write(np.abs(rng.normal(0, 3, size=img.shape)), affine, f)

    # 3. Same scan filed under a second subject ID (re-enrolled participant).
    dup = f"sub-{len(subs) + 1:03d}"
    src = subs[2] / "ses-01" / "anat" / f"{subs[2].name}_ses-01_T1w.nii.gz"
    dst = root / dup / "ses-01" / "anat" / f"{dup}_ses-01_T1w.nii.gz"
    dst.parent.mkdir(parents=True)
    shutil.copyfile(src, dst)
    participants.append({**participants[2], "participant_id": dup})

    # 4. Follow-up that was never registered to the baseline (shifted origin).
    f = subs[3] / "ses-02" / "anat" / f"{subs[3].name}_ses-02_T1w.nii.gz"
    img = nib.load(str(f))
    shifted = img.affine.copy()
    shifted[:3, 3] += 4.5
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), shifted), str(f))
