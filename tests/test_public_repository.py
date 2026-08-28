from __future__ import annotations

import re
import subprocess
from pathlib import Path


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
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert windows_drive.search(text) is None, f"Machine path found in {relative}"
