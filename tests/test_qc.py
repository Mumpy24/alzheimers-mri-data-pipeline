import numpy as np

from admri.qc import Status, run_qc

AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
SPACING = (2.0, 2.0, 2.0)


def failed(report):
    return {c.name for c in report.checks if c.status is Status.FAIL}


def test_clean_phantom_passes(phantom):
    report = run_qc(phantom, SPACING, AFFINE)
    assert report.passed and report.status is Status.PASS


def test_nan_fails(phantom):
    phantom[5, 5, 5] = np.nan
    assert failed(run_qc(phantom, SPACING, AFFINE)) == {"finite"}


def test_pure_noise_fails_snr():
    noise = np.abs(np.random.default_rng(0).normal(0, 3, (40, 40, 40))).astype(np.float32)
    assert "snr" in failed(run_qc(noise, SPACING, AFFINE))


def test_empty_volume_fails(phantom):
    assert {"foreground_ratio", "snr"} <= failed(run_qc(np.zeros_like(phantom), SPACING, AFFINE))


def test_wrong_dimensionality_short_circuits():
    report = run_qc(np.ones((40, 40)), SPACING, AFFINE)
    assert failed(report) == {"ndim"} and len(report.checks) == 1


def test_small_volume_and_bad_spacing(phantom):
    assert "min_dim" in failed(run_qc(phantom[:20], SPACING, AFFINE))
    assert "spacing_range" in failed(run_qc(phantom, (5.0, 5.0, 5.0), AFFINE))


def test_singular_affine_fails(phantom):
    assert "affine" in failed(run_qc(phantom, SPACING, np.zeros((4, 4))))


def test_anisotropy_is_a_warning(phantom):
    report = run_qc(phantom, (0.8, 0.8, 2.9), AFFINE)
    assert report.passed and report.status is Status.WARN
    assert any("anisotropy" in r for r in report.reasons(Status.WARN))
