from pathlib import Path

import numpy as np
import pytest

from admri.bids import Session
from admri.config import PipelineConfig, SplitConfig
from admri.pairs import RegistrationError, compute_delta, generate_pairs
from admri.splits import LeakageError, check_disjoint, split_subjects
from admri.synthetic import render_phantom

IDS = [f"sub-{i:03d}" for i in range(100)]
STRATA = {sid: ["CN", "MCI", "AD"][i % 3] for i, sid in enumerate(IDS)}


@pytest.mark.parametrize("method", ["stratified", "hash"])
def test_split_is_complete_disjoint_and_deterministic(method):
    cfg = SplitConfig(method=method, seed=3)
    a = split_subjects(IDS, cfg, STRATA)
    assert sorted(sid for members in a.values() for sid in members) == IDS
    check_disjoint(a)
    assert split_subjects(reversed(IDS), cfg, STRATA) == a
    assert split_subjects(IDS, SplitConfig(method=method, seed=4), STRATA) != a


def test_stratified_split_preserves_class_balance():
    splits = split_subjects(IDS, SplitConfig(), STRATA)
    for members in splits.values():
        counts = [sum(STRATA[s] == dx for s in members) for dx in ("CN", "MCI", "AD")]
        assert max(counts) - min(counts) <= 1
    assert [len(v) for v in splits.values()] == [70, 15, 15]


def test_hash_split_is_stable_when_dataset_grows():
    cfg = SplitConfig(method="hash")
    before = split_subjects(IDS[:60], cfg)
    after = split_subjects(IDS, cfg)
    for name, members in before.items():
        assert set(members) <= set(after[name])


def test_check_disjoint_detects_leakage():
    with pytest.raises(LeakageError, match="sub-1"):
        check_disjoint({"train": ["sub-1", "sub-2"], "test": ["sub-1"]})


def test_invalid_configs():
    with pytest.raises(ValueError):
        SplitConfig(train=0.8, val=0.3, test=0.1)
    with pytest.raises(ValueError):
        SplitConfig(method="random")
    with pytest.raises(ValueError):
        PipelineConfig(pair_strategy="every")


def _sessions():
    return [
        Session(f"ses-0{i + 1}", Path("x"), days_since_baseline=365.0 * i, diagnosis=dx)
        for i, dx in enumerate(["MCI", "MCI", "AD", "AD"])
    ]


@pytest.mark.parametrize("strategy,n", [("consecutive", 3), ("baseline", 3), ("all", 6)])
def test_pair_strategies(strategy, n):
    pairs = generate_pairs("sub-1", _sessions(), strategy)
    assert len(pairs) == n
    assert all(p.interval_days > 0 for p in pairs)


def test_pair_labels():
    pairs = generate_pairs("sub-1", _sessions(), "consecutive")
    assert [p.converter for p in pairs] == [False, True, False]
    assert pairs[0].pair_id == "sub-1_ses-01_to_ses-02"
    assert pairs[0].interval_days == 365.0
    assert generate_pairs("sub-1", _sessions()[:1]) == []
    with pytest.raises(ValueError):
        generate_pairs("sub-1", _sessions(), "nope")


def test_delta_requires_coregistration():
    a = np.zeros((4, 4, 4), np.float32)
    aff = np.eye(4)
    np.testing.assert_array_equal(compute_delta(a, a + 1, aff, aff), np.ones_like(a))
    with pytest.raises(RegistrationError, match="Shape"):
        compute_delta(a, a[:3], aff, aff)
    shifted = aff.copy()
    shifted[0, 3] = 2.0
    with pytest.raises(RegistrationError, match="co-registered"):
        compute_delta(a, a, aff, shifted)


def test_synthetic_atrophy_is_faster_in_ad():
    """The phantom's ground truth: AD loses more tissue per year than CN."""

    def changed_voxels(brain_rate, vent_rate, years=2.0):
        rng = np.random.default_rng(0)
        t0 = render_phantom((48, 48, 48), 1.0, 1.0, 1.0, rng, noise_sd=0)
        t1 = render_phantom(
            (48, 48, 48), 1.0, (1 - brain_rate) ** years, (1 + vent_rate) ** years, rng, noise_sd=0
        )
        return int((np.abs(t1 - t0) > 50).sum())

    assert changed_voxels(0.020, 0.09) > 2 * changed_voxels(0.004, 0.02)
