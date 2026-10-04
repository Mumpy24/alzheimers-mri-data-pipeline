"""Subject-level train/val/test splitting with leakage guarantees.

Splitting at the *scan* level lets follow-up scans of a training subject
appear in the test set, which inflates reported accuracy. Every function here
operates on subject IDs, and :func:`check_disjoint` is used as a hard gate.

Two methods are provided:

``stratified``
    Seeded shuffle within each diagnosis stratum, so class balance is kept
    across splits. Best for a fixed dataset.
``hash``
    Each subject is assigned by ``sha256(seed:subject_id)``. Adding new
    subjects never moves existing ones to a different split, which keeps
    test sets stable as a dataset grows.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping

from admri.config import SplitConfig

SPLIT_NAMES = ("train", "val", "test")


class LeakageError(RuntimeError):
    """Raised when the same subject (or scan content) appears in multiple splits."""


def _allocate(n: int, fractions: tuple[float, float, float]) -> list[int]:
    """Largest-remainder allocation of ``n`` items to fractions."""
    raw = [n * f for f in fractions]
    counts = [int(r) for r in raw]
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return counts


def split_subjects(
    subject_ids: Iterable[str],
    config: SplitConfig | None = None,
    strata: Mapping[str, str | None] | None = None,
) -> dict[str, list[str]]:
    config = config or SplitConfig()
    ids = sorted(set(subject_ids))
    fractions = (config.train, config.val, config.test)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    if config.method == "hash":
        bounds = (config.train, config.train + config.val)
        for sid in ids:
            h = hashlib.sha256(f"{config.seed}:{sid}".encode()).digest()
            u = int.from_bytes(h[:8], "big") / 2**64
            name = "train" if u < bounds[0] else "val" if u < bounds[1] else "test"
            splits[name].append(sid)
    else:
        groups: dict[str, list[str]] = defaultdict(list)
        for sid in ids:
            groups[str((strata or {}).get(sid) or "unknown")].append(sid)
        rng = random.Random(config.seed)
        for key in sorted(groups):
            members = groups[key]
            rng.shuffle(members)
            n_train, n_val, _ = _allocate(len(members), fractions)
            splits["train"] += members[:n_train]
            splits["val"] += members[n_train : n_train + n_val]
            splits["test"] += members[n_train + n_val :]

    out = {name: sorted(members) for name, members in splits.items()}
    check_disjoint(out)
    return out


def check_disjoint(splits: Mapping[str, Iterable[str]]) -> None:
    seen: dict[str, str] = {}
    for name, members in splits.items():
        for sid in members:
            if sid in seen:
                raise LeakageError(f"{sid} appears in both '{seen[sid]}' and '{name}'")
            seen[sid] = name
