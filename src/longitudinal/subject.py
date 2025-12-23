"""
Longitudinal Subject Abstraction

Represents a single subject with multiple MRI sessions
for neurodegenerative disease analysis.
"""

from pathlib import Path
from collections import OrderedDict
from src.ingestion.mri_ingest import ingest_nifti


class Subject:
    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.sessions = OrderedDict()

    def add_session(self, session_id, mri_path):
        """
        Add a session MRI to the subject.

        Parameters
        ----------
        session_id : str
            Session identifier (e.g., 'ses-01')
        mri_path : Path or str
            Path to MRI file
        """
        if session_id in self.sessions:
            raise ValueError(
                f"Session {session_id} already exists for subject {self.subject_id}"
            )

        image, metadata = ingest_nifti(mri_path)

        self.sessions[session_id] = {
            "image": image,
            "metadata": metadata,
        }

    def get_session_ids(self):
        return list(self.sessions.keys())

    def get_session(self, session_id):
        if session_id not in self.sessions:
            raise KeyError(
                f"Session {session_id} not found for subject {self.subject_id}"
            )
        return self.sessions[session_id]

    def num_sessions(self):
        return len(self.sessions)
