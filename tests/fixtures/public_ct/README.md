# Public CT test fixture

The real-world integration test uses `ct_org_volume-0_0000.nii.gz` from the
[CT-ORG collection](https://www.cancerimagingarchive.net/collection/ct-org/).
CT-ORG is distributed under the
[Creative Commons Attribution 3.0 Unported license](https://creativecommons.org/licenses/by/3.0/).

The CT file is **not** stored in this repository, included in wheels, or covered
by BodyComposition's Apache-2.0 license. The opt-in test downloads the unchanged
file from a revision-pinned public mirror, or accepts an existing local copy,
and verifies its byte count and SHA-256 digest before processing it. Full
provenance, attribution, citations, license links, and integrity values are in
[`ct_org_volume_0.json`](ct_org_volume_0.json).

Required source citation:

> Rister, B., Shivakumar, K., Nobashi, T., & Rubin, D. L. (2019). CT-ORG: A
> Dataset of CT Volumes With Multiple Organ Segmentations (Version 1)
> [dataset]. The Cancer Imaging Archive.
> https://doi.org/10.7937/TCIA.2019.TT7F4V7O

The immutable download is hosted by the
[CADS dataset mirror](https://huggingface.co/datasets/huggingface/CADS-dataset),
which should also be cited when that mirror is used:

> Xu, M. et al. CADS: A Comprehensive Anatomical Dataset and Segmentation for
> Whole-Body Anatomy in Computed Tomography. arXiv:2507.22953 (2025).

TCIA notes that several CT-ORG volumes may have a left-right flip. This scan is
therefore an execution, geometry, and orientation-stage regression fixture,
not independent orientation ground truth and not evidence of clinical
accuracy.
