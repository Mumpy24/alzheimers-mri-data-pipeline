from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from admri.pipeline import PipelineResult, run_pipeline
from admri.synthetic import make_synthetic_bids, render_phantom

SHAPE = (40, 44, 40)


@pytest.fixture(scope="session")
def bids_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data") / "bids"
    return make_synthetic_bids(root, n_subjects=12, shape=SHAPE, seed=7, inject_faults=True)


@pytest.fixture(scope="session")
def pipeline_run(bids_root: Path, tmp_path_factory: pytest.TempPathFactory) -> PipelineResult:
    return run_pipeline(bids_root, tmp_path_factory.mktemp("out"))


@pytest.fixture
def phantom() -> np.ndarray:
    rng = np.random.default_rng(0)
    return render_phantom(SHAPE, 1.0, 1.0, 1.0, rng)
