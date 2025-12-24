"""
Delta MRI Computation

Computes longitudinal change representations (Δ-MRI)
from baseline → follow-up MRI pairs.
"""

import numpy as np


def compute_delta_mri(baseline_image, followup_image, baseline_metadata, followup_metadata):
    """
    Compute delta MRI (follow-up minus baseline).

    Parameters
    ----------
    baseline_image : np.ndarray
    followup_image : np.ndarray
    baseline_metadata : dict
    followup_metadata : dict

    Returns
    -------
    delta_image : np.ndarray
        Follow-up minus baseline MRI
    delta_metadata : dict
        Metadata describing the delta computation
    """

    # ----------------------------
    # Shape consistency check
    # ----------------------------
    if baseline_image.shape != followup_image.shape:
        raise ValueError(
            f"Shape mismatch: baseline {baseline_image.shape} vs "
            f"follow-up {followup_image.shape}"
        )

    # ----------------------------
    # Voxel spacing consistency
    # ----------------------------
    if baseline_metadata["voxel_spacing"] != followup_metadata["voxel_spacing"]:
        raise ValueError("Voxel spacing mismatch between baseline and follow-up")

    # ----------------------------
    # Compute delta
    # ----------------------------
    delta_image = followup_image - baseline_image

    # ----------------------------
    # Delta metadata
    # ----------------------------
    delta_metadata = {
        "operation": "followup_minus_baseline",
        "voxel_spacing": baseline_metadata["voxel_spacing"],
        "baseline_normalization": baseline_metadata.get("normalization"),
        "followup_normalization": followup_metadata.get("normalization"),
        "validation": "delta_mri_shape_and_spacing_checked",
    }

    return delta_image, delta_metadata
