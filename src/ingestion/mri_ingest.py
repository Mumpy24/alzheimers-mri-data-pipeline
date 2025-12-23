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
