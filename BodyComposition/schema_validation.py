"""Packaged JSON-schema validation for public manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import jsonschema


class SchemaValidationError(ValueError):
    """Raised when a public artifact does not satisfy its versioned schema."""


@cache
def load_schema(name: str) -> dict[str, Any]:
    resource = files("BodyComposition").joinpath("schemas").joinpath(name)
    value = json.loads(resource.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(value)
    return value


def validate_payload(payload: Mapping[str, Any], schema_name: str) -> None:
    validator = jsonschema.Draft202012Validator(load_schema(schema_name))
    errors = sorted(validator.iter_errors(dict(payload)), key=lambda error: list(error.path))
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(value) for value in error.absolute_path) or "<root>"
    raise SchemaValidationError(
        f"{schema_name} validation failed at {location}: {error.message}"
    )


def resolve_bundle_path(root: Path, relative: str, *, field: str) -> Path:
    """Resolve a manifest-owned relative path without permitting traversal."""

    value = Path(relative)
    if value.is_absolute():
        raise SchemaValidationError(f"{field} must be relative to its manifest bundle.")
    resolved_root = root.resolve()
    candidate = (resolved_root / value).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise SchemaValidationError(f"{field} escapes its manifest bundle.") from error
    return candidate
