"""
Longitudinal Pair Generation

Creates baseline → follow-up MRI pairs for
neurodegenerative disease progression modeling.
"""

from typing import List, Dict
from src.longitudinal.subject import Subject
from src.longitudinal.delta import compute_delta_mri

def generate_longitudinal_pairs(subject: Subject) -> List[Dict]:
    """
    Generate consecutive baseline → follow-up pairs for a subject.

    Parameters
    ----------
    subject : Subject

    Returns
    -------
    List of dictionaries containing:
        - subject_id
        - baseline_session
        - followup_session
        - baseline_image
        - followup_image
        - baseline_metadata
        - followup_metadata
    """
    session_ids = subject.get_session_ids()

    if len(session_ids) < 2:
        return []

    pairs = []

    for i in range(len(session_ids) - 1):
        baseline_id = session_ids[i]
        followup_id = session_ids[i + 1]

        baseline = subject.get_session(baseline_id)
        followup = subject.get_session(followup_id)

        pair = {
            "subject_id": subject.subject_id,
            "baseline_session": baseline_id,
            "followup_session": followup_id,
            "baseline_image": baseline["image"],
            "followup_image": followup["image"],
            "baseline_metadata": baseline["metadata"],
            "followup_metadata": followup["metadata"],
        }

        pairs.append(pair)

    return pairs

def generate_delta_pairs(subject: Subject):
    """
    Generate delta-MRI representations for a subject.

    Returns
    -------
    List of dicts containing:
        - subject_id
        - baseline_session
        - followup_session
        - delta_image
        - delta_metadata
    """
    from src.longitudinal.pairs import generate_longitudinal_pairs

    raw_pairs = generate_longitudinal_pairs(subject)
    delta_pairs = []

    for pair in raw_pairs:
        delta_image, delta_metadata = compute_delta_mri(
            pair["baseline_image"],
            pair["followup_image"],
            pair["baseline_metadata"],
            pair["followup_metadata"],
        )

        delta_pairs.append({
            "subject_id": pair["subject_id"],
            "baseline_session": pair["baseline_session"],
            "followup_session": pair["followup_session"],
            "delta_image": delta_image,
            "delta_metadata": delta_metadata,
        })

    return delta_pairs
