import nibabel as nib
import numpy as np
import pytest

from admri.config import NormalizationConfig
from admri.io import load_nifti, save_nifti, sha256_file
from admri.preprocessing import foreground_mask, robust_zscore


def test_save_load_roundtrip_is_deterministic(tmp_path, phantom):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    h1 = save_nifti(phantom, affine, tmp_path / "a.nii.gz")
    h2 = save_nifti(phantom, affine, tmp_path / "b.nii.gz")
    assert h1 == h2 == sha256_file(tmp_path / "a.nii.gz")

    data, aff, spacing = load_nifti(tmp_path / "a.nii.gz")
    np.testing.assert_allclose(data, phantom)
    np.testing.assert_allclose(aff, affine)
    assert spacing == (2.0, 2.0, 2.0)


def test_load_squeezes_singleton_4th_dim(tmp_path):
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 1), np.float32), np.eye(4)), tmp_path / "x.nii")
    data, _, _ = load_nifti(tmp_path / "x.nii")
    assert data.shape == (4, 4, 4)


def test_load_rejects_missing_and_non_nifti(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_nifti(tmp_path / "missing.nii.gz")
    (tmp_path / "scan.mgz").write_bytes(b"")
    with pytest.raises(ValueError, match="NIfTI"):
        load_nifti(tmp_path / "scan.mgz")


def test_zscore_is_invariant_to_scanner_gain(phantom):
    np.testing.assert_allclose(robust_zscore(phantom), robust_zscore(phantom * 1.7), atol=1e-5)


def test_zscore_is_nearly_invariant_to_offset(phantom):
    a, b = robust_zscore(phantom), robust_zscore(phantom * 1.7 + 25)
    common = foreground_mask(phantom) & foreground_mask(phantom * 1.7 + 25)
    assert common.sum() / foreground_mask(phantom).sum() > 0.99  # only boundary voxels differ
    assert np.abs(a - b)[common].max() < 0.05


def test_zscore_uses_foreground_only(phantom):
    out = robust_zscore(phantom)
    mask = foreground_mask(phantom)
    assert np.all(out[~mask] == 0)
    assert abs(out[mask].mean()) < 0.1
    assert abs(out[mask].std() - 1) < 0.1
    # Adding background (a larger field of view) barely changes foreground values.
    padded = robust_zscore(np.pad(phantom, 10))[10:-10, 10:-10, 10:-10]
    both = mask & (padded != 0)
    assert np.abs(padded - out)[both].max() < 0.05


def test_zscore_rejects_empty_or_constant():
    with pytest.raises(ValueError):
        robust_zscore(np.zeros((8, 8, 8), np.float32))
    with pytest.raises(ValueError):
        robust_zscore(np.full((8, 8, 8), 5.0, np.float32), NormalizationConfig())
