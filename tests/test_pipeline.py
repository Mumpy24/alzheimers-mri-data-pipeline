import csv
import json
import shutil

import numpy as np
import pytest

from admri.audit import audit
from admri.cli import main
from admri.config import PipelineConfig, SplitConfig
from admri.loader import DeltaPairDataset
from admri.pipeline import run_pipeline


def _audit_failures(out):
    return {name for name, ok, _ in audit(out).checks if not ok}


def test_injected_faults_are_excluded(pipeline_run):
    failed = {
        (s.subject_id, s.session_id): s.qc_reasons for s in pipeline_run.scans if not s.usable
    }
    assert "finite" in failed[("sub-001", "ses-02")]
    assert "snr" in failed[("sub-002", "ses-01")]
    assert "duplicate_content" in failed[("sub-003", "ses-01")]
    assert "duplicate_content" in failed[("sub-013", "ses-01")]
    assert len(failed) == 4
    excluded = {p["pair_id"] for p in pipeline_run.excluded_pairs}
    assert "sub-004_ses-01_to_ses-02" in excluded


def test_no_pair_uses_a_failed_scan(pipeline_run):
    bad = {(s.subject_id, s.session_id) for s in pipeline_run.scans if not s.usable}
    for p in pipeline_run.pairs:
        assert (p.subject_id, p.baseline_session) not in bad
        assert (p.subject_id, p.followup_session) not in bad


def test_pairs_and_scans_respect_subject_split(pipeline_run):
    split_of = {sid: n for n, ids in pipeline_run.splits.items() for sid in ids}
    assert all(p.split == split_of[p.subject_id] for p in pipeline_run.pairs)
    assert all(s.split == split_of[s.subject_id] for s in pipeline_run.scans if s.usable)
    assert "sub-013" not in split_of  # only scan was a duplicate


def test_outputs_and_audit(pipeline_run):
    out = pipeline_run.output_dir
    for name in ("scans.csv", "pairs.csv", "splits.json", "dataset_card.json", "report.md"):
        assert (out / name).exists()
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["dataset_version"] == pipeline_run.version
    assert card["counts"]["scans_by_qc_status"]["fail"] == 4
    assert audit(out).ok


def test_rerun_is_byte_identical_regardless_of_workers(bids_root, pipeline_run, tmp_path):
    rerun = run_pipeline(bids_root, tmp_path, workers=2)
    assert rerun.version == pipeline_run.version
    for name in ("scans.csv", "pairs.csv", "splits.json"):
        assert (tmp_path / name).read_bytes() == (pipeline_run.output_dir / name).read_bytes()


def test_version_changes_with_config(bids_root, pipeline_run, tmp_path):
    cfg = PipelineConfig(split=SplitConfig(seed=1), write_deltas=False)
    assert run_pipeline(bids_root, tmp_path, cfg).version != pipeline_run.version


@pytest.fixture
def run_copy(pipeline_run, tmp_path):
    out = tmp_path / "copy"
    shutil.copytree(pipeline_run.output_dir, out)
    return out


def test_audit_detects_modified_delta(run_copy):
    delta = next((run_copy / "deltas").rglob("*.nii.gz"))
    delta.write_bytes(delta.read_bytes() + b"\0")
    assert _audit_failures(run_copy) == {"output_file_hashes"}


def test_audit_detects_split_leakage(run_copy):
    splits = json.loads((run_copy / "splits.json").read_text())
    splits["test"].append(splits["train"][0])
    (run_copy / "splits.json").write_text(json.dumps(splits))
    assert {"manifest_hashes", "splits_disjoint"} <= _audit_failures(run_copy)


def test_audit_detects_changed_source(run_copy, bids_root, tmp_path):
    src = tmp_path / "bids"
    shutil.copytree(bids_root, src)
    f = next(src.rglob("sub-005_ses-01_T1w.nii.gz"))
    f.write_bytes(f.read_bytes()[:-1] + b"\1")
    result = audit(run_copy, bids_root=src)
    assert {n for n, ok, _ in result.checks if not ok} == {"source_file_hashes"}


def test_audit_missing_manifests(tmp_path):
    assert _audit_failures(tmp_path) == {"manifests_present"}


def test_loader(pipeline_run):
    ds = DeltaPairDataset(pipeline_run.output_dir, "train")
    n_train = sum(p.split == "train" for p in pipeline_run.pairs)
    assert len(ds) == n_train > 0
    item = ds[0]
    assert item["delta"].ndim == 4 and item["delta"].shape[0] == 1
    assert np.isfinite(item["delta"]).all()
    assert item["interval_years"] > 0


def test_loader_refuses_tampered_output(run_copy):
    with open(run_copy / "pairs.csv", "a") as fh:
        fh.write("\n")
    with pytest.raises(RuntimeError, match="Audit failed"):
        DeltaPairDataset(run_copy, "train")


def test_config_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('pair_strategy = "all"\n[split]\nmethod = "hash"\nseed = 9\n[qc]\nmin_snr = 3.0\n')
    cfg = PipelineConfig.from_toml(p)
    assert cfg.pair_strategy == "all" and cfg.split.seed == 9 and cfg.qc.min_snr == 3.0
    assert PipelineConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="Unknown"):
        PipelineConfig.from_dict({"bogus": 1})


def test_cli_end_to_end(tmp_path, capsys):
    bids, out = tmp_path / "bids", tmp_path / "out"
    assert main(["synth", str(bids), "--subjects", "5", "--seed", "1"]) == 0
    assert main(["run", str(bids), str(out)]) == 0
    assert main(["audit", str(out)]) == 0
    assert "Audit passed" in capsys.readouterr().out
    with open(out / "scans.csv") as fh:
        assert all(r["qc_status"] == "pass" for r in csv.DictReader(fh))
    (out / "splits.json").write_text("{}")
    assert main(["audit", str(out)]) == 1


def test_shipped_config_matches_defaults():
    from pathlib import Path

    path = Path(__file__).parents[1] / "configs" / "default.toml"
    assert PipelineConfig.from_toml(path) == PipelineConfig()
