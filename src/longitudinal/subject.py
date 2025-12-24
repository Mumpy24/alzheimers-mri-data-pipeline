"""
Longitudinal Subject Abstraction

Represents a single subject with multiple MRI sessions
for neurodegenerative disease (e.g., Alzheimer's) analysis.

Design principles:
- Subject-centric (no MRI without a subject)
- Longitudinal ordering enforced via visit index
- MRI ingestion, normalization, and validation handled centrally
"""

from pathlib import Path
from collections import OrderedDict
from src.ingestion.mri_ingest import ingest_nifti


class Subject:
    """
    Represents a single subject with multiple longitudinal MRI sessions.
    """

    def __init__(self, subject_id):
        """
        Initialize a Subject.

        Parameters
        ----------
        subject_id : str
            Unique identifier for the subject (e.g., 'sub-001')
        """
        self.subject_id = subject_id
        self.sessions = OrderedDict()

    def add_session(self, session_id, mri_path):
        """
        Add a session MRI to the subject in strict longitudinal order.

        Parameters
        ----------
        session_id : str
            Session identifier (e.g., 'ses-01', 'ses-02')
        mri_path : str or Path
            Path to MRI file (.nii or .nii.gz)

        Raises
        ------
        ValueError
            If session already exists or violates visit ordering
        """
        if session_id in self.sessions:
            raise ValueError(
                f"Session {session_id} already exists for subject {self.subject_id}"
            )

        # Enforce visit index ordering (ses-01 < ses-02 < ...)
        if self.sessions:
            last_session = list(self.sessions.keys())[-1]

            try:
                last_idx = int(last_session.split("-")[1])
                current_idx = int(session_id.split("-")[1])
            except (IndexError, ValueError):
                raise ValueError(
                    f"Session IDs must follow 'ses-XX' format (got {session_id})"
                )

            if current_idx <= last_idx:
                raise ValueError(
                    f"Session {session_id} must come after {last_session}"
                )

        # Ingest MRI (includes normalization + validation)
        image, metadata = ingest_nifti(mri_path)

        self.sessions[session_id] = {
            "image": image,
            "metadata": metadata,
        }

    def get_session_ids(self):
        """
        Return all session IDs in longitudinal order.
        """
        return list(self.sessions.keys())

    def get_session(self, session_id):
        """
        Retrieve data for a specific session.

        Parameters
        ----------
        session_id : str

        Returns
        -------
        dict
            Dictionary containing 'image' and 'metadata'
        """
        if session_id not in self.sessions:
            raise KeyError(
                f"Session {session_id} not found for subject {self.subject_id}"
            )
        return self.sessions[session_id]

    def num_sessions(self):
        """
        Return the number of sessions for this subject.
        """
        return len(self.sessions)

    def summary(self):
        """
        Return a lightweight summary of the subject.
        """
        return {
            "subject_id": self.subject_id,
            "num_sessions": self.num_sessions(),
            "sessions": self.get_session_ids(),
        }
