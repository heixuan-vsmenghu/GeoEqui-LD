from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from PIL import Image

PUBLIC_BINARY_ALLOWLIST = {
    "reports/phase05/curves/confirmation_validation_metrics.png",
    "reports/phase05/curves/seed42_validation_metrics.png",
}
PUBLIC_PHASE05_TEXT = {
    "reports/phase05/PHASE05_SUMMARY.md",
    "reports/phase05/SUPERVISED_ABLATION.md",
    "reports/phase05/aggregate_results.json",
}
PUBLIC_FORBIDDEN_FIELDS = {
    "protocol_sha256",
    "git_commit",
    "initialization_sha256",
    "best_checkpoint_sha256",
    "labels_sha256",
    "aggregate_sha256",
}


def _tracked_files() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line]


def test_restricted_artifacts_are_not_tracked() -> None:
    forbidden_roots = ("data/", "runs/", "artifacts/", "checkpoints/")
    forbidden_suffixes = (".pt", ".pth", ".ckpt", ".onnx")
    for path in _tracked_files():
        assert not path.startswith(forbidden_roots)
        assert not path.endswith(forbidden_suffixes)
        assert path not in {"configs/phase0_local.yaml", "configs/phase05_local.yaml"}


def test_tracked_text_does_not_contain_machine_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    windows_drive = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
    for relative in _tracked_files():
        path = root / relative
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf"}:
            assert relative in PUBLIC_BINARY_ALLOWLIST, f"Unexpected public binary: {relative}"
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert windows_drive.search(text) is None, f"Machine path found in {relative}"


def test_phase05_public_outputs_are_sanitized() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = set(_tracked_files())
    digest = re.compile(r"\b(?:[0-9a-f]{40}|[0-9a-f]{64})\b")
    unix_machine_path = re.compile(r"(?<![A-Za-z0-9:])/(?:home|Users|mnt|tmp|var/tmp)/")
    unc_path = re.compile(r"\\\\[^\\\s]+\\[^\\\s]+")
    for relative in PUBLIC_PHASE05_TEXT:
        assert relative in tracked, f"Missing tracked Phase 0.5 output: {relative}"
        text = (root / relative).read_text(encoding="utf-8")
        assert digest.search(text) is None, f"Digest found in {relative}"
        assert unix_machine_path.search(text) is None, f"Unix machine path found in {relative}"
        assert unc_path.search(text) is None, f"UNC path found in {relative}"
        assert not any(field in text for field in PUBLIC_FORBIDDEN_FIELDS)

    aggregate = json.loads(
        (root / "reports/phase05/aggregate_results.json").read_text(encoding="utf-8")
    )

    def check_keys(value: object) -> None:
        if isinstance(value, dict):
            assert not (set(value) & PUBLIC_FORBIDDEN_FIELDS)
            for item in value.values():
                check_keys(item)
        elif isinstance(value, list):
            for item in value:
                check_keys(item)

    check_keys(aggregate)


def test_phase05_public_png_metadata_is_sanitized() -> None:
    root = Path(__file__).resolve().parents[1]
    tracked = set(_tracked_files())
    machine_path = re.compile(
        r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|(?<![A-Za-z0-9:])/(?:home|Users|mnt|tmp)/|"
        r"\\\\[^\\\s]+\\[^\\\s]+"
    )
    digest = re.compile(r"\b(?:[0-9a-f]{40}|[0-9a-f]{64})\b")
    for relative in PUBLIC_BINARY_ALLOWLIST:
        assert relative in tracked, f"Missing tracked Phase 0.5 curve: {relative}"
        with Image.open(root / relative) as image:
            metadata = json.dumps(image.info, sort_keys=True, default=str)
        assert machine_path.search(metadata) is None, f"Machine path found in {relative} metadata"
        assert digest.search(metadata) is None, f"Digest found in {relative} metadata"
