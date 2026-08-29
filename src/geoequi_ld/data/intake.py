"""Privacy-preserving Phase 2A intake helpers for unlabeled image archives.

The module deliberately has no testing-image input.  Testing overlap can only
be checked against a previously sealed fingerprint CSV.  Per-file paths and
digests belong to the private manifest; the public aggregate contains counts
and status flags only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import stat
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image, UnidentifiedImageError

from geoequi_ld.utils.hashing import sha256_file, sha256_image_pixels
from geoequi_ld.utils.io import write_json

SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
FORBIDDEN_RAW_SPLIT_NAMES = frozenset({"test", "testing"})
SEALED_SPLITS = ("train", "validation", "testing")
DEFAULT_SEALED_ROW_COUNTS = {"train": 300, "validation": 100, "testing": 501}
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_EXAMPLE_STEM = re.compile(r"^examples?\d*$", re.IGNORECASE)
_WINDOWS_RESERVED = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)


class IntakeStatus(str, Enum):
    """Required Phase 2A data intake states."""

    READY_FULL = "READY_FULL"
    READY_NAMED_SUBSET = "READY_NAMED_SUBSET"
    DOWNLOADING = "DOWNLOADING"
    BLOCKED_ACCESS = "BLOCKED_ACCESS"
    BLOCKED_INTEGRITY = "BLOCKED_INTEGRITY"
    BLOCKED_OVERLAP_AUDIT = "BLOCKED_OVERLAP_AUDIT"


@dataclass(frozen=True)
class ArchiveIntegrity:
    """Local archive integrity result without embedding expected secrets."""

    present: bool
    actual_size: int
    size_match: bool
    md5_match: bool
    zip_readable: bool

    @property
    def complete(self) -> bool:
        return self.present and self.size_match and self.md5_match and self.zip_readable


@dataclass(frozen=True)
class ZipMemberPlan:
    """One validated member selected for extraction."""

    archive_name: str
    relative_parts: tuple[str, ...]
    size_bytes: int
    is_directory: bool


@dataclass(frozen=True)
class CandidateRecord:
    """Private per-file audit record."""

    relative_path: str
    extension: str
    size_bytes: int
    decodable: bool
    decoded_format: str | None
    width: int | None
    height: int | None
    mode: str | None
    channels: int | None
    file_sha256: str
    pixel_sha256: str | None
    is_example_named: bool
    error_type: str | None


@dataclass(frozen=True)
class SealedFingerprints:
    """Digest sets loaded from the earlier sealed image inventory."""

    file_digests: dict[str, frozenset[str]]
    pixel_digests: dict[str, frozenset[str]]
    row_counts: dict[str, int]

    @classmethod
    def empty(cls) -> SealedFingerprints:
        return cls(
            file_digests={split: frozenset() for split in SEALED_SPLITS},
            pixel_digests={split: frozenset() for split in SEALED_SPLITS},
            row_counts={split: 0 for split in SEALED_SPLITS},
        )


def _path_components(value: str | Path) -> tuple[str, ...]:
    return tuple(part.casefold() for part in re.split(r"[\\/]+", str(value)) if part)


def _is_example_path(relative_path: Path) -> bool:
    """Flag explicit Example/Examples collections without inferring their identity."""

    return bool(_EXAMPLE_STEM.fullmatch(relative_path.stem)) or any(
        _EXAMPLE_STEM.fullmatch(part) for part in relative_path.parts[:-1]
    )


def reject_raw_testing_path(path: str | Path, *, purpose: str) -> None:
    """Fail before filesystem access when a raw path names testing data."""

    if FORBIDDEN_RAW_SPLIT_NAMES.intersection(_path_components(path)):
        raise PermissionError(f"{purpose} cannot point to a raw testing split")


def _md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_archive_integrity(
    archive_path: str | Path,
    *,
    expected_size: int,
    expected_md5: str,
) -> ArchiveIntegrity:
    """Verify exact byte size, MD5, and ZIP readability for a complete archive."""

    archive = Path(archive_path)
    reject_raw_testing_path(archive, purpose="Archive")
    if expected_size <= 0:
        raise ValueError("expected_size must be positive")
    normalized_md5 = expected_md5.strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{32}", normalized_md5):
        raise ValueError("expected_md5 must be 32 hexadecimal characters")
    if not archive.is_file():
        return ArchiveIntegrity(False, 0, False, False, False)
    actual_size = archive.stat().st_size
    actual_md5 = _md5_file(archive)
    zip_readable = zipfile.is_zipfile(archive)
    return ArchiveIntegrity(
        present=True,
        actual_size=actual_size,
        size_match=actual_size == expected_size,
        md5_match=actual_md5 == normalized_md5,
        zip_readable=zip_readable,
    )


def _normalize_member(name: str) -> tuple[str, ...]:
    if not name or "\x00" in name:
        raise ValueError("ZIP contains an empty or NUL-bearing member name")
    normalized = name.replace("\\", "/")
    windows = PureWindowsPath(normalized)
    posix = PurePosixPath(normalized)
    if windows.is_absolute() or posix.is_absolute() or windows.drive:
        raise ValueError("ZIP contains an absolute member path")
    parts = tuple(part for part in posix.parts if part not in {"", "."})
    if not parts or ".." in parts:
        raise ValueError("ZIP contains a path-traversal member")
    for part in parts:
        stem = part.rstrip(" .").split(".", maxsplit=1)[0].casefold()
        if ":" in part or stem in _WINDOWS_RESERVED:
            raise ValueError("ZIP contains a platform-unsafe member name")
    return parts


def _normalize_allowed_prefix(prefix: str) -> tuple[str, ...]:
    parts = _normalize_member(prefix.rstrip("/\\"))
    if FORBIDDEN_RAW_SPLIT_NAMES.intersection(part.casefold() for part in parts):
        raise PermissionError("Allowed ZIP prefixes cannot select raw testing data")
    return parts


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def plan_allowed_zip_members(
    archive_path: str | Path,
    allowed_prefixes: tuple[str, ...] | list[str],
) -> tuple[ZipMemberPlan, ...]:
    """Validate all member paths and return only members below allowed prefixes.

    No unselected member name is returned or logged.  Raw testing prefixes are
    rejected before the archive is opened.
    """

    archive = Path(archive_path)
    reject_raw_testing_path(archive, purpose="Archive")
    prefixes = tuple(_normalize_allowed_prefix(value) for value in allowed_prefixes)
    if not prefixes:
        raise ValueError("At least one allowed ZIP prefix is required")
    selected: list[ZipMemberPlan] = []
    normalized_selected: set[str] = set()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            parts = _normalize_member(info.filename)
            matched = any(parts[: len(prefix)] == prefix for prefix in prefixes)
            if not matched:
                continue
            if FORBIDDEN_RAW_SPLIT_NAMES.intersection(part.casefold() for part in parts):
                raise PermissionError("Selected ZIP subtree contains raw testing data")
            if _is_zip_symlink(info):
                raise ValueError("Selected ZIP subtree contains a symbolic link")
            collision_key = "/".join(parts).casefold()
            if collision_key in normalized_selected:
                raise ValueError("Selected ZIP subtree contains a normalized path collision")
            normalized_selected.add(collision_key)
            selected.append(
                ZipMemberPlan(
                    archive_name=info.filename,
                    relative_parts=parts,
                    size_bytes=info.file_size,
                    is_directory=info.is_dir(),
                )
            )
    return tuple(selected)


def extract_allowed_zip_members(
    archive_path: str | Path,
    destination: str | Path,
    plans: tuple[ZipMemberPlan, ...] | list[ZipMemberPlan],
) -> int:
    """Extract validated members without ``extractall`` and without overwrites."""

    archive = Path(archive_path)
    target_root = Path(destination)
    reject_raw_testing_path(target_root, purpose="Extraction destination")
    target_root.mkdir(parents=True, exist_ok=True)
    resolved_root = target_root.resolve(strict=True)
    prepared: list[tuple[ZipMemberPlan, Path]] = []
    file_parts = {plan.relative_parts for plan in plans if not plan.is_directory}
    for plan in plans:
        if any(
            plan.relative_parts[: len(parent)] == parent and plan.relative_parts != parent
            for parent in file_parts
        ):
            raise ValueError("Selected ZIP subtree places a member below a file")
        target = (resolved_root / Path(*plan.relative_parts)).resolve(strict=False)
        if not target.is_relative_to(resolved_root):
            raise PermissionError("Validated ZIP member escaped extraction root")
        if plan.is_directory:
            if target.exists() and not target.is_dir():
                raise FileExistsError("Extraction directory collides with an existing file")
        elif target.exists():
            raise FileExistsError("Extraction refuses to overwrite an existing file")
        for parent in target.parents:
            if parent == resolved_root:
                break
            if parent.exists() and not parent.is_dir():
                raise FileExistsError("Extraction parent collides with an existing file")
        prepared.append((plan, target))

    extracted_files = 0
    with zipfile.ZipFile(archive) as handle:
        for plan, target in prepared:
            if plan.is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(plan.archive_name, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if target.stat().st_size != plan.size_bytes:
                raise OSError("Extracted member size does not match ZIP metadata")
            extracted_files += 1
    return extracted_files


def scan_candidate_images(candidate_dir: str | Path) -> tuple[CandidateRecord, ...]:
    """Decode and hash candidate images while refusing raw testing paths."""

    root = Path(candidate_dir)
    reject_raw_testing_path(root, purpose="Candidate directory")
    if not root.is_dir():
        raise FileNotFoundError("Candidate directory is unavailable")
    resolved_root = root.resolve(strict=True)
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PermissionError("Candidate directory contains a symbolic link")
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise PermissionError("Candidate image escapes its declared root")
        candidates.append(path)
    candidates.sort(key=lambda value: value.relative_to(root).as_posix().casefold())

    records: list[CandidateRecord] = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        decoded_format: str | None = None
        width: int | None = None
        height: int | None = None
        mode: str | None = None
        channels: int | None = None
        pixel_digest: str | None = None
        error_type: str | None = None
        decodable = False
        try:
            with Image.open(path) as image:
                image.load()
                decoded_format = image.format or "UNKNOWN"
                width, height = image.size
                mode = image.mode
                channels = len(image.getbands())
                pixel_digest = sha256_image_pixels(image)
                decodable = True
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            error_type = type(exc).__name__
        records.append(
            CandidateRecord(
                relative_path=relative,
                extension=path.suffix.casefold(),
                size_bytes=path.stat().st_size,
                decodable=decodable,
                decoded_format=decoded_format,
                width=width,
                height=height,
                mode=mode,
                channels=channels,
                file_sha256=file_digest,
                pixel_sha256=pixel_digest,
                is_example_named=_is_example_path(path.relative_to(root)),
                error_type=error_type,
            )
        )
    return tuple(records)


def load_sealed_fingerprints(path: str | Path | None) -> SealedFingerprints:
    """Load only split and digest columns from the earlier sealed inventory."""

    if path is None:
        return SealedFingerprints.empty()
    inventory = Path(path)
    if not inventory.is_file():
        return SealedFingerprints.empty()
    file_sets: dict[str, set[str]] = {split: set() for split in SEALED_SPLITS}
    pixel_sets: dict[str, set[str]] = {split: set() for split in SEALED_SPLITS}
    row_counts = {split: 0 for split in SEALED_SPLITS}
    with inventory.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "file_sha256", "pixel_sha256"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Sealed fingerprint CSV lacks required digest columns")
        for row in reader:
            split = str(row.get("split", "")).strip().casefold()
            if split not in SEALED_SPLITS:
                continue
            file_digest = str(row.get("file_sha256", "")).strip().casefold()
            pixel_digest = str(row.get("pixel_sha256", "")).strip().casefold()
            if not file_digest or not _HEX_64.fullmatch(file_digest):
                raise ValueError("Sealed fingerprint CSV contains an invalid file digest")
            if not pixel_digest or not _HEX_64.fullmatch(pixel_digest):
                raise ValueError("Sealed fingerprint CSV contains an invalid pixel digest")
            row_counts[split] += 1
            file_sets[split].add(file_digest)
            pixel_sets[split].add(pixel_digest)
    return SealedFingerprints(
        file_digests={split: frozenset(values) for split, values in file_sets.items()},
        pixel_digests={split: frozenset(values) for split, values in pixel_sets.items()},
        row_counts=row_counts,
    )


def _overlap_splits(record: CandidateRecord, sealed: SealedFingerprints) -> tuple[str, ...]:
    matches: list[str] = []
    for split in SEALED_SPLITS:
        file_match = record.file_sha256 in sealed.file_digests[split]
        pixel_match = bool(
            record.pixel_sha256 and record.pixel_sha256 in sealed.pixel_digests[split]
        )
        if file_match or pixel_match:
            matches.append(split)
    return tuple(matches)


def _distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_intake_payloads(
    *,
    candidate_dir: str | Path,
    records: tuple[CandidateRecord, ...] | list[CandidateRecord],
    sealed: SealedFingerprints,
    archive: ArchiveIntegrity | None = None,
    downloaded_images: int = 0,
    discovered_images: int | None = None,
    expected_final_count: int | None = None,
    source_id: str = "unspecified-official-source",
    expected_sealed_row_counts: dict[str, int] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build strictly separated private and public Phase 2A artifacts."""

    if downloaded_images < 0:
        raise ValueError("downloaded_images cannot be negative")
    if expected_final_count is not None and expected_final_count <= 0:
        raise ValueError("expected_final_count must be positive")
    audited = list(records)
    decoded = [record for record in audited if record.decodable and record.pixel_sha256]
    pixel_groups: dict[str, list[CandidateRecord]] = defaultdict(list)
    for record in decoded:
        assert record.pixel_sha256 is not None
        pixel_groups[record.pixel_sha256].append(record)

    excluded_groups = 0
    overlap_group_counts = {split: 0 for split in SEALED_SPLITS}
    private_rows: list[dict[str, object]] = []
    for group_index, (_, members) in enumerate(sorted(pixel_groups.items()), start=1):
        group_overlaps = sorted(
            {split for member in members for split in _overlap_splits(member, sealed)}
        )
        if group_overlaps:
            excluded_groups += 1
            for split in group_overlaps:
                overlap_group_counts[split] += 1
        for member in members:
            row = asdict(member)
            row["pixel_group"] = group_index
            row["overlap_splits"] = group_overlaps
            row["included_after_exact_audit"] = not group_overlaps
            private_rows.append(row)
    for record in audited:
        if record.decodable and record.pixel_sha256:
            continue
        row = asdict(record)
        row["pixel_group"] = None
        row["overlap_splits"] = []
        row["included_after_exact_audit"] = False
        private_rows.append(row)
    private_rows.sort(key=lambda row: str(row["relative_path"]).casefold())

    pixel_unique = len(pixel_groups)
    byte_unique = len({record.file_sha256 for record in audited})
    final_count = pixel_unique - excluded_groups
    corrupt = sum(not record.decodable for record in audited)
    expected_sealed = expected_sealed_row_counts or DEFAULT_SEALED_ROW_COUNTS
    if set(expected_sealed) != set(SEALED_SPLITS) or any(
        count <= 0 for count in expected_sealed.values()
    ):
        raise ValueError("Expected sealed row counts must cover all three fixed splits")
    split_known = {
        split: sealed.row_counts[split] == expected_sealed[split] for split in SEALED_SPLITS
    }
    testing_known = split_known["testing"]
    train_known = split_known["train"]
    validation_known = split_known["validation"]
    comparison_executed = bool(decoded)
    archive_complete = archive.complete if archive is not None else False
    if (
        corrupt
        or not audited
        or not decoded
        or final_count <= 0
        or (archive is not None and not archive_complete)
    ):
        status = IntakeStatus.BLOCKED_INTEGRITY
    elif not testing_known:
        status = IntakeStatus.BLOCKED_OVERLAP_AUDIT
    elif (
        archive_complete
        and expected_final_count is not None
        and final_count == expected_final_count
    ):
        status = IntakeStatus.READY_FULL
    else:
        status = IntakeStatus.READY_NAMED_SUBSET

    example_records = [record for record in decoded if record.is_example_named]
    example_duplicate_risk_groups = sum(
        len(members) > 1 and any(member.is_example_named for member in members)
        for members in pixel_groups.values()
    )
    integrity_public = {
        "archive_checked": archive is not None,
        "archive_present": bool(archive and archive.present),
        "actual_size_bytes": archive.actual_size if archive is not None else 0,
        "size_match": bool(archive and archive.size_match),
        "md5_match": bool(archive and archive.md5_match),
        "zip_readable": bool(archive and archive.zip_readable),
    }
    public: dict[str, object] = {
        "schema_version": 1,
        "phase": "phase2a-data-intake",
        "source_id": source_id,
        "status": status.value,
        "counts": {
            "discovered": len(audited) if discovered_images is None else discovered_images,
            "downloaded": downloaded_images,
            "extracted": len(audited),
            "decodable": len(decoded),
            "byte_unique": byte_unique,
            "pixel_unique": pixel_unique,
            "excluded_overlap": excluded_groups,
            "final": final_count,
        },
        "count_definitions": {
            "discovered": (
                "selected image members in a verified archive when supplied; otherwise "
                "supported image files found in the candidate directory"
            ),
            "downloaded": "selected image members only after complete archive verification",
            "extracted": "supported image files present in the candidate directory",
            "decodable": "candidate images fully decoded by Pillow",
            "byte_unique": "unique SHA-256 file-content groups, reported as a count only",
            "pixel_unique": (
                "unique decoded RGB+width+height pixel groups, reported as a count only"
            ),
            "excluded_overlap": "unique pixel groups matching any sealed labeled split",
            "final": "pixel-unique groups after exact sealed-overlap exclusion",
        },
        "integrity": integrity_public,
        "image_audit": {
            "corrupt": corrupt,
            "decoded_format_distribution": _distribution(
                [str(record.decoded_format) for record in decoded]
            ),
            "size_distribution": _distribution(
                [f"{record.width}x{record.height}" for record in decoded]
            ),
            "mode_distribution": _distribution([str(record.mode) for record in decoded]),
            "channel_distribution": _distribution([str(record.channels) for record in decoded]),
        },
        "deduplication": {
            "exact_byte_duplicate_excess": len(audited) - byte_unique,
            "exact_pixel_duplicate_excess": len(decoded) - pixel_unique,
            "example_named_files": len(example_records),
            "example_duplicate_risk_groups": example_duplicate_risk_groups,
            "example_files_excluded_automatically": False,
            "near_duplicates_removed": False,
        },
        "overlap_audit": {
            "train": (
                "checked"
                if comparison_executed and train_known
                else "not_run_no_candidate_pool"
                if not comparison_executed
                else "unknown"
            ),
            "validation": (
                "checked"
                if comparison_executed and validation_known
                else "not_run_no_candidate_pool"
                if not comparison_executed
                else "unknown"
            ),
            "testing": (
                "checked_sealed_fingerprints_only"
                if comparison_executed and testing_known
                else "not_run_no_candidate_pool"
                if not comparison_executed
                else "unknown"
            ),
            "comparison_executed": comparison_executed,
            "excluded_unique_groups_by_split": overlap_group_counts,
            "raw_testing_entry_available": False,
            "testing_labels_read": False,
            "testing_images_read": False,
            "sealed_fingerprint_rows": sealed.row_counts,
            "expected_sealed_fingerprint_rows": expected_sealed,
        },
        "group_independence": {
            "patient": "unknown",
            "case": "unknown",
            "video": "unknown",
            "inferred_from_filename": False,
        },
        "privacy": {
            "public_contains_paths": False,
            "public_contains_digests": False,
            "private_manifest_required_for_per_file_evidence": True,
        },
    }
    private: dict[str, object] = {
        "schema_version": 1,
        "phase": "phase2a-data-intake-private",
        "source_root": str(Path(candidate_dir).resolve()),
        "records": private_rows,
        "sealed_inventory_counts": sealed.row_counts,
    }
    _assert_public_payload_safe(public)
    return private, public


def _assert_public_payload_safe(payload: dict[str, object]) -> None:
    forbidden_keys = {
        "relative_path",
        "source_root",
        "file_sha256",
        "pixel_sha256",
        "filename",
        "archive_name",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise ValueError("Public aggregate contains a private field")
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
        elif isinstance(value, str) and _HEX_64.fullmatch(value.casefold()):
            raise ValueError("Public aggregate contains a raw SHA-256 digest")

    walk(payload)


def write_intake_payloads(
    *,
    candidate_dir: str | Path,
    private_manifest_path: str | Path,
    public_aggregate_path: str | Path,
    private_payload: dict[str, object],
    public_payload: dict[str, object],
) -> None:
    """Write separated artifacts outside the input tree."""

    root = Path(candidate_dir).resolve()
    private_path = Path(private_manifest_path).resolve()
    public_path = Path(public_aggregate_path).resolve()
    if private_path == public_path:
        raise ValueError("Private manifest and public aggregate must be separate files")
    for output in (private_path, public_path):
        if output == root or output.is_relative_to(root):
            raise ValueError("Audit outputs cannot be written inside candidate data")
    _assert_public_payload_safe(public_payload)
    write_json(private_path, private_payload)
    write_json(public_path, public_payload)


def public_payload_contains_private_evidence(payload: dict[str, object]) -> bool:
    """Convenience predicate used by tests and report gates."""

    try:
        _assert_public_payload_safe(json.loads(json.dumps(payload)))
    except ValueError:
        return True
    return False
