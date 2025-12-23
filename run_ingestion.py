"""
Run MRI Ingestion Pipeline

This script demonstrates how to ingest a single MRI scan
using the Alzheimer’s MRI data engineering pipeline.
"""

from pathlib import Path
from src.ingestion.mri_ingest import ingest_nifti


def main():
    """
    Main entry point for MRI ingestion.
    """

    # ------------------------------------------------------------------
    # CHANGE THIS PATH LATER TO A REAL MRI FILE (.nii or .nii.gz)
    # ------------------------------------------------------------------
    mri_path = Path("example_data/sample_mri.nii.gz")

    if not mri_path.exists():
        print(f"[INFO] MRI file not found at {mri_path}")
        print("[INFO] This is expected for now.")
        print("[INFO] Replace the path with a real MRI file later.")
        return

    # Run ingestion
    image, metadata = ingest_nifti(mri_path)

    # Report results
    print("MRI ingestion successful")
    print("Image shape:", image.shape)
    print("Voxel spacing:", metadata["voxel_spacing"])
    print("Normalization:", metadata["normalization"])
    print("Validation:", metadata["validation"])


if __name__ == "__main__":
    main()
