"""Streaming and decoded-pixel hashes used by the read-only data audit."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_image_pixels(image: Image.Image) -> str:
    """Hash decoded pixels together with mode and dimensions."""

    canonical = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{canonical.width}x{canonical.height}:".encode("ascii"))
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def stable_identifier(value: str, length: int = 16) -> str:
    """Create a deterministic non-reversible identifier for report rows."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def sequence_signature(relative_path: str) -> tuple[str, int] | None:
    """Parse a final ``prefix_frame`` or ``prefix-frame`` pattern without interpreting it."""

    stem = Path(relative_path).stem
    match = re.fullmatch(r"(.+?)[_-](\d+)", stem)
    if match is None:
        return None
    return match.group(1), int(match.group(2))
