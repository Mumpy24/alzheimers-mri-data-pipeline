"""
Dataset Manager for Longitudinal Neuroimaging Data

Manages multiple subjects and enforces:
- Subject-level integrity
- Leakage-free dataset splits
- Reproducible experiment structure
"""

import random
from collections import OrderedDict
from src.longitudinal.subject import Subject


class LongitudinalDataset:
    """
    Dataset-level manager for longitudinal MRI subjects.
    """

    def __init__(self, seed=42):
        """
        Initialize dataset.

        Parameters
        ----------
        seed : int
            Random seed for reproducible splits
        """
        self.subjects = OrderedDict()
        self.seed = seed
        random.seed(seed)

    def add_subject(self, subject):
        """
        Add a Subject to the dataset.

        Parameters
        ----------
        subject : Subject

        Raises
        ------
        ValueError if subject ID already exists
        """
        if not isinstance(subject, Subject):
            raise TypeError("Only Subject instances can be added")

        if subject.subject_id in self.subjects:
            raise ValueError(
                f"Subject {subject.subject_id} already exists in dataset"
            )

        self.subjects[subject.subject_id] = subject

    def subject_ids(self):
        """
        Return all subject IDs.
        """
        return list(self.subjects.keys())

    def num_subjects(self):
        """
        Return number of subjects in dataset.
        """
        return len(self.subjects)

    def split_subjects(self, train=0.7, val=0.15, test=0.15):
        """
        Split subjects into train/val/test sets.

        Parameters
        ----------
        train : float
        val : float
        test : float

        Returns
        -------
        dict with keys: 'train', 'val', 'test'
        """
        if not abs(train + val + test - 1.0) < 1e-6:
            raise ValueError("Train/val/test fractions must sum to 1")

        subject_ids = list(self.subjects.keys())
        random.shuffle(subject_ids)

        n = len(subject_ids)
        n_train = int(n * train)
        n_val = int(n * val)

        splits = {
            "train": subject_ids[:n_train],
            "val": subject_ids[n_train : n_train + n_val],
            "test": subject_ids[n_train + n_val :],
        }

        return splits

    def get_split(self, split_ids):
        """
        Retrieve Subject objects for a given split.

        Parameters
        ----------
        split_ids : list of subject IDs

        Returns
        -------
        list of Subject
        """
        return [self.subjects[sid] for sid in split_ids]

    def summary(self):
        """
        Dataset summary.
        """
        return {
            "num_subjects": self.num_subjects(),
            "subject_ids": self.subject_ids(),
        }
    
    def generate_pairs_for_split(self, split_ids):
        """
        Generate longitudinal baseline → follow-up pairs
        for a given subject split.

        Parameters
        ----------
        split_ids : list of subject IDs

        Returns
        -------
        list of longitudinal pairs (dicts)
        """
        from src.longitudinal.pairs import generate_longitudinal_pairs

        all_pairs = []

        for subject_id in split_ids:
            subject = self.subjects[subject_id]
            pairs = generate_longitudinal_pairs(subject)
            all_pairs.extend(pairs)

        return all_pairs
