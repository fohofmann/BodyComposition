# Contributing

BodyComposition changes must preserve physical geometry, scientific identity,
privacy, licensing, and immutable-result behavior. Open an issue before a
change that alters measurement definitions, model assets, schemas, or clinical
interpretation.

## Environment

Use Python 3.11 and uv. Do not install project dependencies globally.

```bash
uv sync --frozen --extra test --extra release
```

Model weights and clinical data must remain outside the repository. Unit tests
use synthetic arrays. The opt-in real-world fixture downloads a checksum-pinned
public CT under its documented license into a temporary/external location.

Study protocols, cohort manifests, linkage tables, notebooks, local validation
receipts, and generated results belong in a separate governed analysis
workspace. This repository contains only reusable pipeline code, tests,
schemas, release tooling, and public documentation. Public documentation assets
require explicit source, license, and derivative attribution.

## Required checks

```bash
uv run ruff check BodyComposition tests scripts
uv run mypy
uv run pytest
uv run pytest --cov
uv run python -m build
uv run python scripts/audit_repository.py --history
uv run python scripts/audit_distribution.py dist/*.whl dist/*.tar.gz
```

Use `scripts/release_checks.py --allow-dirty` while developing. The actual
release gate requires a clean reviewed tree and no flag.

## Test expectations

- geometry tests must cover anisotropic/oblique metadata and the SimpleITK
  `xyz`/array `zyx` boundary;
- backend tests must prove no download or fallback during inference;
- service tests must cover API/CLI equivalence, atomicity, resume, failure,
  cancellation, shared-worker fencing, and privacy;
- measurement changes require numerical tests with units and missingness;
- reporting changes require page-count, digest, numerical, privacy, and both
  layout tests; and
- model changes require exact source/revision, license/citation, byte size,
  hash, compatibility, and controlled integration validation.

Do not weaken a gate by excluding the changed behavior. Coverage exclusions
are limited to generated code, upstream/model execution boundaries, and
visual/model integration that has a separate controlled test.

## Style and scope

Prefer small conventional changes. Remove obsolete paths instead of adding
compatibility shims unless a public contract explicitly requires one. Comments
explain scientific or software invariants. Do not include local paths,
generated clinical output, credentials, model weights, or machine-specific
state.

## License

Contributions are accepted under Apache-2.0. Third-party code and model assets
retain their own terms; do not copy or relicense them without explicit review.
