from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from geoequi_ld.data.intake import (
    ArchiveIntegrity,
    IntakeStatus,
    build_intake_payloads,
    check_archive_integrity,
    extract_allowed_zip_members,
    load_sealed_fingerprints,
    plan_allowed_zip_members,
    public_payload_contains_private_evidence,
    reject_raw_testing_path,
    scan_candidate_images,
    write_intake_payloads,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _save_image(path: Path, value: int, *, image_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 6), color=value).save(path, format=image_format)


def _write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "file_sha256", "pixel_sha256", "relative_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_required_intake_status_enum_is_complete() -> None:
    assert {status.value for status in IntakeStatus} == {
        "READY_FULL",
        "READY_NAMED_SUBSET",
        "DOWNLOADING",
        "BLOCKED_ACCESS",
        "BLOCKED_INTEGRITY",
        "BLOCKED_OVERLAP_AUDIT",
    }


def test_archive_integrity_requires_size_md5_and_readable_zip(tmp_path: Path) -> None:
    archive = tmp_path / "Dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Dataset/Training/Unlabeled cases/a.txt", b"safe")
    payload = archive.read_bytes()
    expected_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()

    passed = check_archive_integrity(
        archive,
        expected_size=len(payload),
        expected_md5=expected_md5,
    )
    assert passed.complete
    assert passed.size_match and passed.md5_match and passed.zip_readable

    wrong_size = check_archive_integrity(
        archive,
        expected_size=len(payload) + 1,
        expected_md5=expected_md5,
    )
    assert not wrong_size.complete
    assert not wrong_size.size_match

    wrong_md5 = check_archive_integrity(
        archive,
        expected_size=len(payload),
        expected_md5="0" * 32,
    )
    assert not wrong_md5.complete
    assert not wrong_md5.md5_match


def test_zip_plan_rejects_traversal_even_outside_selected_prefix(tmp_path: Path) -> None:
    archive = tmp_path / "Dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Dataset/Training/Unlabeled cases/good.png", b"not-decoded-here")
        handle.writestr("../escape.txt", b"unsafe")

    with pytest.raises(ValueError, match="path-traversal"):
        plan_allowed_zip_members(archive, ["Dataset/Training/Unlabeled cases"])


def test_zip_extracts_only_allowed_prefix_and_refuses_overwrite(tmp_path: Path) -> None:
    archive = tmp_path / "Dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Dataset/Training/Unlabeled cases/kept.txt", b"kept")
        handle.writestr("Dataset/Validation/ignored.txt", b"ignored")

    plans = plan_allowed_zip_members(archive, ["Dataset/Training/Unlabeled cases"])
    assert len(plans) == 1
    destination = tmp_path / "extracted"
    assert extract_allowed_zip_members(archive, destination, plans) == 1
    assert (destination / "Dataset/Training/Unlabeled cases/kept.txt").read_bytes() == b"kept"
    assert not (destination / "Dataset/Validation/ignored.txt").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_allowed_zip_members(archive, destination, plans)


def test_raw_testing_sources_and_prefixes_are_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="raw testing"):
        reject_raw_testing_path(tmp_path / "Testing", purpose="Candidate directory")

    archive = tmp_path / "Dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("Dataset/Training/Unlabeled cases/a.png", b"x")
    with pytest.raises(PermissionError, match="raw testing"):
        plan_allowed_zip_members(archive, ["Dataset/Testing"])


def test_scan_records_decode_properties_hashes_and_corruption(tmp_path: Path) -> None:
    pool = tmp_path / "unlabeled_pool"
    _save_image(pool / "one.jpg", 40, image_format="JPEG")
    _save_image(pool / "Example1.png", 40, image_format="PNG")
    (pool / "broken.jpg").write_bytes(b"not an image")
    (pool / "notes.txt").write_text("ignored", encoding="utf-8")

    records = scan_candidate_images(pool)
    assert len(records) == 3
    assert sum(record.decodable for record in records) == 2
    assert sum(record.is_example_named for record in records) == 1
    assert all(len(record.file_sha256) == 64 for record in records)
    formats = {record.decoded_format for record in records if record.decodable}
    assert formats == {"JPEG", "PNG"}
    assert {(record.width, record.height) for record in records if record.decodable} == {(8, 6)}
    assert {record.channels for record in records if record.decodable} == {1}
    assert any(record.error_type == "UnidentifiedImageError" for record in records)


def test_example_collection_directory_is_flagged_without_automatic_exclusion(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "unlabeled_pool"
    _save_image(pool / "Examples" / "ordinary_name.png", 40)

    records = scan_candidate_images(pool)

    assert len(records) == 1
    assert records[0].is_example_named is True


def test_sealed_overlap_is_aggregate_only_and_examples_are_not_auto_removed(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "unlabeled_pool"
    _save_image(pool / "same.bmp", 60, image_format="BMP")
    _save_image(pool / "Example2.png", 60, image_format="PNG")
    _save_image(pool / "unique.png", 120, image_format="PNG")
    records = scan_candidate_images(pool)
    unique = next(record for record in records if record.relative_path == "unique.png")

    inventory = tmp_path / "sealed_inventory.csv"
    _write_inventory(
        inventory,
        [
            {
                "split": "train",
                "file_sha256": unique.file_sha256,
                "pixel_sha256": unique.pixel_sha256 or "",
                "relative_path": "private-train-name",
            },
            {
                "split": "validation",
                "file_sha256": "1" * 64,
                "pixel_sha256": "2" * 64,
                "relative_path": "private-validation-name",
            },
            {
                "split": "testing",
                "file_sha256": "3" * 64,
                "pixel_sha256": "4" * 64,
                "relative_path": "private-testing-name",
            },
        ],
    )
    sealed = load_sealed_fingerprints(inventory)
    private, public = build_intake_payloads(
        candidate_dir=pool,
        records=records,
        sealed=sealed,
        downloaded_images=3,
        discovered_images=3,
        source_id="synthetic-v1",
        expected_sealed_row_counts={"train": 1, "validation": 1, "testing": 1},
    )

    assert public["status"] == "READY_NAMED_SUBSET"
    assert public["counts"] == {
        "discovered": 3,
        "downloaded": 3,
        "extracted": 3,
        "decodable": 3,
        "byte_unique": 3,
        "pixel_unique": 2,
        "excluded_overlap": 1,
        "final": 1,
    }
    assert public["deduplication"]["example_duplicate_risk_groups"] == 1
    assert public["deduplication"]["example_files_excluded_automatically"] is False
    assert public["overlap_audit"]["testing"] == "checked_sealed_fingerprints_only"
    assert public["group_independence"]["patient"] == "unknown"
    assert not public_payload_contains_private_evidence(public)

    public_text = json.dumps(public, ensure_ascii=False)
    assert "private-train-name" not in public_text
    assert unique.file_sha256 not in public_text
    assert unique.pixel_sha256 not in public_text
    assert any(row["overlap_splits"] == ["train"] for row in private["records"])
    assert private["source_root"] == str(pool.resolve())


def test_missing_testing_fingerprints_stays_unknown_and_blocks_clearance(tmp_path: Path) -> None:
    pool = tmp_path / "unlabeled_pool"
    _save_image(pool / "one.png", 80)
    records = scan_candidate_images(pool)
    inventory = tmp_path / "sealed_inventory.csv"
    _write_inventory(
        inventory,
        [
            {
                "split": "train",
                "file_sha256": "1" * 64,
                "pixel_sha256": "2" * 64,
                "relative_path": "private",
            },
            {
                "split": "validation",
                "file_sha256": "3" * 64,
                "pixel_sha256": "4" * 64,
                "relative_path": "private",
            },
        ],
    )
    _, public = build_intake_payloads(
        candidate_dir=pool,
        records=records,
        sealed=load_sealed_fingerprints(inventory),
        expected_sealed_row_counts={"train": 1, "validation": 1, "testing": 1},
    )
    assert public["status"] == "BLOCKED_OVERLAP_AUDIT"
    assert public["overlap_audit"]["testing"] == "unknown"
    assert public["overlap_audit"]["raw_testing_entry_available"] is False


def test_corrupt_image_or_failed_archive_integrity_blocks_ready_status(tmp_path: Path) -> None:
    pool = tmp_path / "unlabeled_pool"
    _save_image(pool / "one.png", 80)
    (pool / "broken.png").write_bytes(b"broken")
    records = scan_candidate_images(pool)
    sealed = load_sealed_fingerprints(None)
    failed_archive = ArchiveIntegrity(True, 10, False, False, False)
    _, public = build_intake_payloads(
        candidate_dir=pool,
        records=records,
        sealed=sealed,
        archive=failed_archive,
    )
    assert public["status"] == "BLOCKED_INTEGRITY"
    assert public["image_audit"]["corrupt"] == 1


def test_empty_candidate_pool_does_not_claim_overlap_comparison_ran(tmp_path: Path) -> None:
    pool = tmp_path / "unlabeled_pool"
    pool.mkdir()
    inventory = tmp_path / "sealed_inventory.csv"
    _write_inventory(
        inventory,
        [
            {
                "split": split,
                "file_sha256": str(index) * 64,
                "pixel_sha256": str(index + 3) * 64,
                "relative_path": "private",
            }
            for index, split in enumerate(("train", "validation", "testing"), start=1)
        ],
    )

    _, public = build_intake_payloads(
        candidate_dir=pool,
        records=scan_candidate_images(pool),
        sealed=load_sealed_fingerprints(inventory),
        archive=ArchiveIntegrity(True, 10, True, True, False),
        expected_sealed_row_counts={"train": 1, "validation": 1, "testing": 1},
    )

    assert public["status"] == "BLOCKED_INTEGRITY"
    assert public["overlap_audit"]["comparison_executed"] is False
    assert public["overlap_audit"]["testing"] == "not_run_no_candidate_pool"


def test_expected_count_promotes_only_audited_pool_to_ready_full(tmp_path: Path) -> None:
    pool = tmp_path / "unlabeled_pool"
    _save_image(pool / "one.png", 80)
    records = scan_candidate_images(pool)
    inventory = tmp_path / "sealed_inventory.csv"
    _write_inventory(
        inventory,
        [
            {
                "split": split,
                "file_sha256": str(index) * 64,
                "pixel_sha256": str(index + 3) * 64,
                "relative_path": "private",
            }
            for index, split in enumerate(("train", "validation", "testing"), start=1)
        ],
    )
    _, public = build_intake_payloads(
        candidate_dir=pool,
        records=records,
        sealed=load_sealed_fingerprints(inventory),
        archive=ArchiveIntegrity(True, 10, True, True, True),
        expected_final_count=1,
        expected_sealed_row_counts={"train": 1, "validation": 1, "testing": 1},
    )
    assert public["status"] == "READY_FULL"


def test_private_and_public_outputs_must_be_separate_from_source(tmp_path: Path) -> None:
    pool = tmp_path / "unlabeled_pool"
    _save_image(pool / "one.png", 80)
    private, public = build_intake_payloads(
        candidate_dir=pool,
        records=scan_candidate_images(pool),
        sealed=load_sealed_fingerprints(None),
    )
    with pytest.raises(ValueError, match="inside candidate"):
        write_intake_payloads(
            candidate_dir=pool,
            private_manifest_path=pool / "private.json",
            public_aggregate_path=tmp_path / "public.json",
            private_payload=private,
            public_payload=public,
        )

    private_path = tmp_path / "protected" / "private.json"
    public_path = tmp_path / "reports" / "aggregate.json"
    write_intake_payloads(
        candidate_dir=pool,
        private_manifest_path=private_path,
        public_aggregate_path=public_path,
        private_payload=private,
        public_payload=public,
    )
    assert "file_sha256" in private_path.read_text(encoding="utf-8")
    assert "file_sha256" not in public_path.read_text(encoding="utf-8")


def test_phase2a_intake_cli_help_has_no_raw_testing_argument() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_phase2a_unlabeled.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--sealed-fingerprint-csv" in result.stdout
    assert "--testing-image-dir" not in result.stdout
    assert "--testing-label" not in result.stdout
