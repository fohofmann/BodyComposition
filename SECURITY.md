# Security and privacy

## Reporting a vulnerability

Do not open a public issue containing patient data, credentials, private paths,
or an exploitable vulnerability. Use the repository's enabled
[private vulnerability reporting](https://github.com/fohofmann/BodyComposition/security/advisories/new)
channel to contact the maintainer. Include a minimal synthetic reproducer,
affected version/commit, impact, and proposed embargo window. Do not attach
clinical images.

## Supported versions

Only the latest reviewed release candidate or release is supported. The
current `1.0.0rc1` source is pre-release software and must not be represented as
a clinically validated or production-secure medical device.

## Data boundary

BodyComposition accepts governed NIfTI or DICOM CT inputs. Apply the required
de-identification/pseudonymization before processing or within the surrounding
controlled workflow: BodyComposition does not perform DICOM de-identification,
OCR, or burned-in pixel-text removal. CTs, derived masks, review PNGs, and PDFs
remain sensitive medical data even when manifests omit source paths and raw
identifiers.

Keep linkage tables, credentials, tokens, raw clinical exports, and model
weights outside the repository. Use least-privilege mounts, encrypted
transport/storage, access logging, and governed retention. Mount inputs and a
verified model cache read-only during inference where practical.

## Software supply chain

- dependencies are frozen by `uv.lock`;
- container bases and CI actions are pinned by immutable digest/SHA;
- model assets are synchronized only from recorded original sources and
  verified by byte size/SHA-256;
- inference-time downloads and silent model fallback are forbidden;
- built archives are scanned for weights, images, secrets, private paths, and
  unintended large files; and
- release artifacts receive checksums, a CycloneDX SBOM, and a dependency
  vulnerability report.

Never publish an artifact when a required gate is unresolved. A vulnerability
scan is evidence for known dependency advisories, not proof that the package is
secure.
