"""Small deterministic writers for generated reports."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def ensure_output_directory(path: str | Path, protected_roots: Sequence[Path] = ()) -> Path:
    """Create an output directory after ensuring it is outside protected data roots."""

    output = Path(path).resolve()
    for root in protected_roots:
        protected = root.resolve()
        if output == protected or protected in output.parents:
            raise ValueError(f"Refusing to write generated files inside input data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: object) -> None:
    """Atomically write stable UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_csv_rows(
    path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Atomically write deterministic CSV rows."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", dir=destination.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_text(path: str | Path, text: str) -> None:
    """Atomically write UTF-8 text with a final newline."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = text.rstrip() + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, destination)
