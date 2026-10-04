"""Command-line interface: ``admri synth | run | audit``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from admri import __version__
from admri.config import PipelineConfig


def _cmd_synth(args: argparse.Namespace) -> int:
    from admri.synthetic import make_synthetic_bids

    root = make_synthetic_bids(
        args.output,
        n_subjects=args.subjects,
        seed=args.seed,
        inject_faults=args.inject_faults,
        overwrite=args.overwrite,
    )
    print(f"Synthetic BIDS dataset written to {root}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from admri.pipeline import run_pipeline

    config = PipelineConfig.from_toml(args.config) if args.config else PipelineConfig()
    result = run_pipeline(args.bids, args.output, config, workers=args.workers)
    card = json.loads(result.card_path.read_text())
    c = card["counts"]
    print(f"Dataset version : {result.version}")
    print(f"Scans           : {c['scans']} {c['scans_by_qc_status']}")
    print(f"Subjects/split  : {c['subjects_by_split']}")
    print(f"Pairs/split     : {c['pairs_by_split']} (excluded: {c['excluded_pairs']})")
    print(f"Report          : {Path(args.output) / 'report.md'}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from admri.audit import audit

    result = audit(args.output, bids_root=args.bids)
    for name, ok, detail in result.checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    print("Audit passed" if result.ok else "Audit FAILED")
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="admri", description=__doc__)
    parser.add_argument("--version", action="version", version=f"admri {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="generate a synthetic longitudinal BIDS dataset")
    p.add_argument("output", type=Path)
    p.add_argument("--subjects", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inject-faults", action="store_true", help="add corrupt/duplicate scans")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_synth)

    p = sub.add_parser("run", help="run the pipeline on a BIDS dataset")
    p.add_argument("bids", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--config", type=Path, help="TOML config file")
    p.add_argument("--workers", type=int, default=1)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("audit", help="verify hashes and leakage invariants of an output dir")
    p.add_argument("output", type=Path)
    p.add_argument("--bids", type=Path, help="override source BIDS root")
    p.set_defaults(func=_cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
