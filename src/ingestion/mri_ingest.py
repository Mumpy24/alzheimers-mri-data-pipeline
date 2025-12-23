from pathlib import Path
import nibabel as nib
import numpy as np

def robust_intensity_normalization(image, lower=1, upper=99):
    """
    MRI-safe intensity normalization using robust percentiles.

    Parameters
    ----------
    image : np.ndarray
        3D MRI volume
    lower : int
        Lower percentile (default 1)
    upper : int
        Upper percentile (default 99)

    Returns
    -------
    norm_image : np.ndarray
        Normalized MRI volume
    """
    # Compute robust intensity bounds
    low = np.percentile(image, lower)
    high = np.percentile(image, upper)

    if high <= low:
        raise ValueError("Invalid intensity range for normalization")

    # Clip extreme values
    clipped = np.clip(image, low, high)

    # Z-score normalization
    mean = clipped.mean()
    std = clipped.std()

    if std == 0:
        raise ValueError("Zero variance after clipping")

    norm_image = (clipped - mean) / std

    return norm_image

def validate_mri_volume(image, min_nonzero_ratio=0.05):
    """
    Basic sanity checks for MRI volumes.

    Parameters
    ----------
    image : np.ndarray
        3D MRI volume
    min_nonzero_ratio : float
        Minimum fraction of non-zero voxels required

    Raises
    ------
    ValueError if validation fails
    """
    # Check dimensionality
    if image.ndim != 3:
        raise ValueError(f"Expected 3D MRI volume, got shape {image.shape}")

    # Check for NaNs or Infs
    if not np.isfinite(image).all():
        raise ValueError("MRI volume contains NaNs or infinite values")

    # Check for empty / near-empty scans
    nonzero_ratio = np.count_nonzero(image) / image.size
    if nonzero_ratio < min_nonzero_ratio:
        raise ValueError(
            f"MRI volume appears empty or corrupted "
            f"(non-zero ratio={nonzero_ratio:.4f})"
        )

    # Basic shape sanity (very conservative)
    min_dim = min(image.shape)
    if min_dim < 32:
        raise ValueError(f"MRI volume too small: shape={image.shape}")

    return True

def ingest_nifti(path):
    """
    Load a NIfTI MRI file, normalize intensities,
    validate the volume, and extract metadata.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"MRI file not found: {path}")

    if not (path.suffix == ".nii" or path.name.endswith(".nii.gz")):
        raise ValueError("Only NIfTI (.nii / .nii.gz) files are supported")

    nifti = nib.load(str(path))

    raw_image = nifti.get_fdata(dtype=np.float32)
    image = robust_intensity_normalization(raw_image)

    validate_mri_volume(image)

    header = nifti.header
    affine = nifti.affine

    metadata = {
        "shape": image.shape,
        "voxel_spacing": header.get_zooms()[:3],
        "datatype": str(image.dtype),
        "orientation_affine": affine,
        "normalization": "robust_zscore_percentile_1_99",
        "validation": "basic_mri_sanity_checks_v1",
    }

    return image, metadata

