"""Validated YAML configuration loading without machine-specific defaults."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is incomplete or ambiguous."""


@dataclass(frozen=True)
class SplitPaths:
    """Resolved local inputs for one dataset split."""

    name: str
    image_dir: Path
    csv: Path | None
    aop_csv: Path | None
    required: bool


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        missing = sorted({name for name in _ENV_PATTERN.findall(value) if name not in os.environ})
        if missing:
            raise ConfigError(
                "Configuration references unset environment variables: " + ", ".join(missing)
            )
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and expand explicit ``${ENV_VAR}`` references."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ConfigError("The top level of the configuration must be a mapping")
    return _expand_environment(payload)


def find_project_root(start: str | Path) -> Path:
    """Locate the nearest parent containing ``pyproject.toml``."""

    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(f"Could not locate pyproject.toml above: {start}")


def _resolve_path(value: object, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_splits(config: Mapping[str, Any], project_root: Path) -> list[SplitPaths]:
    """Resolve enabled split paths relative to the configured data root."""

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ConfigError("Missing data configuration")
    root = _resolve_path(data.get("root"), project_root, "data.root")
    split_config = data.get("splits")
    if not isinstance(split_config, Mapping) or not split_config:
        raise ConfigError("data.splits must be a non-empty mapping")

    splits: list[SplitPaths] = []
    for name, raw in split_config.items():
        if not isinstance(raw, Mapping):
            raise ConfigError(f"data.splits.{name} must be a mapping")
        if not bool(raw.get("enabled", True)):
            continue
        image_dir = _resolve_path(raw.get("image_dir"), root, f"{name}.image_dir")
        csv_value = raw.get("csv")
        aop_value = raw.get("aop_csv")
        csv_path = (
            None if csv_value in (None, "") else _resolve_path(csv_value, root, f"{name}.csv")
        )
        aop_path = (
            None if aop_value in (None, "") else _resolve_path(aop_value, root, f"{name}.aop_csv")
        )
        splits.append(
            SplitPaths(
                name=str(name),
                image_dir=image_dir,
                csv=csv_path,
                aop_csv=aop_path,
                required=bool(raw.get("required", True)),
            )
        )
    return splits
