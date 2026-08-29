#!/usr/bin/env python3
"""Audit an explicitly named Phase 2A unlabeled candidate pool.

This command does not download data and has no testing-image or testing-label
argument.  Cross-split comparison accepts only an earlier sealed fingerprint
CSV.  The per-file manifest must stay private; the public output contains only
aggregate counts and status flags.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geoequi_ld.data.intake import (  # noqa: E402
    SUPPORTED_IMAGE_SUFFIXES,
    ArchiveIntegrity,
    IntakeStatus,
    build_intake_payloads,
    check_archive_integrity,
    extract_allowed_zip_members,
    load_sealed_fingerprints,
    plan_allowed_zip_members,
    reject_raw_testing_path,
    scan_candidate_images,
    write_intake_payloads,
)

DEFAULT_ALLOWED_PREFIX = "Dataset/Training/Unlabeled cases"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Explicit extracted unlabeled candidate directory; raw testing paths are refused",
    )
    parser.add_argument(
        "--archive",
        help="Optional complete official ZIP to verify; this command never downloads it",
    )
    parser.add_argument("--expected-size", type=int, help="Exact official archive byte size")
    parser.add_argument("--expected-md5", help="Exact official archive MD5")
    parser.add_argument(
        "--allowed-prefix",
        action="append",
        help=(
            "ZIP subtree permitted for extraction; repeatable. Default: "
            f"{DEFAULT_ALLOWED_PREFIX!r}"
        ),
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract only validated allowed-prefix members into --candidate-dir",
    )
    parser.add_argument(
        "--sealed-fingerprint-csv",
        help="Optional earlier sealed split/file/pixel fingerprint inventory",
    )
    parser.add_argument("--private-manifest", required=True, help="Protected per-file JSON output")
    parser.add_argument("--public-aggregate", required=True, help="Shareable aggregate JSON output")
    parser.add_argument(
        "--expected-final-count",
        type=int,
        help="Expected post-dedup/overlap count for READY_FULL; omit for named-subset status",
    )
    parser.add_argument(
        "--source-id",
        default="zenodo-17355570-v1",
        help="Public source/version label without a local path",
    )
    return parser.parse_args(argv)


def _validate_archive_args(args: argparse.Namespace) -> None:
    supplied_integrity = args.expected_size is not None or args.expected_md5 is not None
    if bool(args.archive) != supplied_integrity:
        raise ValueError("--archive requires both --expected-size and --expected-md5")
    if args.archive and (args.expected_size is None or args.expected_md5 is None):
        raise ValueError("--archive requires both --expected-size and --expected-md5")
    if args.extract and not args.archive:
        raise ValueError("--extract requires a verified --archive")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", args.source_id):
        raise ValueError("--source-id must be a short source/version label, not a path")


def run(args: argparse.Namespace) -> int:
    _validate_archive_args(args)
    reject_raw_testing_path(args.candidate_dir, purpose="Candidate directory")
    prefixes = tuple(args.allowed_prefix or [DEFAULT_ALLOWED_PREFIX])
    archive_result: ArchiveIntegrity | None = None
    discovered_images: int | None = None
    downloaded_images = 0

    if args.archive:
        archive_result = check_archive_integrity(
            args.archive,
            expected_size=args.expected_size,
            expected_md5=args.expected_md5,
        )
        if archive_result.complete:
            plans = plan_allowed_zip_members(args.archive, prefixes)
            discovered_images = sum(
                not plan.is_directory
                and Path(plan.relative_parts[-1]).suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
                for plan in plans
            )
            downloaded_images = discovered_images
            if args.extract:
                extract_allowed_zip_members(args.archive, args.candidate_dir, plans)

    records = scan_candidate_images(args.candidate_dir)
    sealed = load_sealed_fingerprints(args.sealed_fingerprint_csv)
    private, public = build_intake_payloads(
        candidate_dir=args.candidate_dir,
        records=records,
        sealed=sealed,
        archive=archive_result,
        downloaded_images=downloaded_images,
        discovered_images=discovered_images,
        expected_final_count=args.expected_final_count,
        source_id=args.source_id,
    )
    write_intake_payloads(
        candidate_dir=args.candidate_dir,
        private_manifest_path=args.private_manifest,
        public_aggregate_path=args.public_aggregate,
        private_payload=private,
        public_payload=public,
    )
    status = str(public["status"])
    counts = public["counts"]
    print(f"Phase 2A intake status: {status}")
    print(f"Aggregate counts: {counts}")
    print("Raw testing images and labels were not available to this command.")
    ready_states = {IntakeStatus.READY_FULL.value, IntakeStatus.READY_NAMED_SUBSET.value}
    return 0 if status in ready_states else 1


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileExistsError, FileNotFoundError, OSError, PermissionError, ValueError) as exc:
        print(f"phase2a intake error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
