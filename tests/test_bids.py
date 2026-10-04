from pathlib import Path

import pytest

from admri.bids import Session, Subject, discover_bids, natural_key


def test_discovers_subjects_sessions_and_metadata(bids_root):
    subjects = discover_bids(bids_root)
    assert [s.subject_id for s in subjects][:3] == ["sub-001", "sub-002", "sub-003"]
    s = subjects[0]
    assert s.attributes["diagnosis"] in {"CN", "MCI", "AD"}
    assert s.sessions[0].days_since_baseline == 0
    days = [x.days_since_baseline for x in s.sessions]
    assert days == sorted(days)
    assert all(x.path.exists() for x in s.sessions)


def test_natural_session_ordering():
    sessions = [Session(f"ses-{i}", Path(f"{i}.nii")) for i in (10, 2, 1)]
    assert [s.session_id for s in Subject("sub-x", sessions).sessions] == [
        "ses-1",
        "ses-2",
        "ses-10",
    ]
    assert natural_key("ses-9") < natural_key("ses-10")


def test_days_take_precedence_over_labels():
    sessions = [
        Session("ses-a", Path("a.nii"), days_since_baseline=400),
        Session("ses-b", Path("b.nii"), days_since_baseline=0),
    ]
    assert Subject("sub-x", sessions).sessions[0].session_id == "ses-b"


def test_duplicate_sessions_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        Subject("sub-x", [Session("ses-1", Path("a")), Session("ses-1", Path("b"))])


def test_missing_root_and_empty_subjects(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_bids(tmp_path / "nope")
    (tmp_path / "sub-001" / "ses-01" / "anat").mkdir(parents=True)
    assert discover_bids(tmp_path) == []
