"""
MRI Ingestion Module

Handles safe loading of brain MRI scans (NIfTI format)
and extracts core metadata needed for Alzheimer’s research.
"""

from pathlib import Path
import nibabel as nib
import numpy as np


def ingest_nifti(path):
    """
    Load a NIfTI MRI file and extract image + metadata.

    Parameters
    ----------
    path : str or Path
        Path to .nii or .nii.gz file

    Returns
    -------
    image : np.ndarray
        3D MRI volume (X, Y, Z)
    metadata : dict
        Core imaging metadata
    """
    path = Path(path)

    # Safety checks
    if not path.exists():
        raise FileNotFoundError(f"MRI file not found: {path}")

    if not (path.suffix == ".nii" or path.name.endswith(".nii.gz")):
        raise ValueError("Only NIfTI (.nii / .nii.gz) files are supported")

    # Load MRI
    nifti = nib.load(str(path))
    image = nifti.get_fdata(dtype=np.float32)

    # Extract metadata
    header = nifti.header
    affine = nifti.affine

    metadata = {
        "shape": image.shape,
        "voxel_spacing": header.get_zooms()[:3],
        "datatype": str(image.dtype),
        "orientation_affine": affine,
    }

    return image, metadata
