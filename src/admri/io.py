"""NIfTI input/output with content hashing.

Outputs are written as gzip streams with a fixed mtime so that identical
inputs and configuration produce byte-identical files (and hashes).
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: str | Path) -> bool:
    return str(path).endswith(NIFTI_SUFFIXES)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Load a NIfTI file as ``(float32 data, 4x4 affine, voxel spacing)``.

    A trailing singleton 4th dimension (common in T1w exports) is squeezed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MRI file not found: {path}")
    if not is_nifti(path):
        raise ValueError(f"Only NIfTI (.nii / .nii.gz) files are supported: {path}")

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, np.asarray(img.affine, dtype=np.float64), spacing


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str | Path) -> str:
    """Write a float32 NIfTI deterministically and return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    img.header.set_xyzt_units("mm")
    payload = img.to_bytes()
    if path.name.endswith(".gz"):
        payload = gzip.compress(payload, compresslevel=6, mtime=0)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
