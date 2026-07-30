# QC and manual review

Execution status and scientific QC are separate. A case can complete and still
require review. The pipeline continues readable uncertain cases by default,
retains structured reasons, and never clears flags automatically.
Informational findings document expected coverage or acquisition-boundary
conditions without requesting adjudication. Warnings and errors enter the
manual-review queue; errors also make scientific QC fail.

## Where to start

For a cohort, inspect:

```bash
bodycomposition results review-queue output/runs/<run-id> --json
```

Then open the linked case manifest and canonical artifacts. If PDF reporting is
enabled, affected cases retain plain-language Notes on their individual page.
The combined PDF always begins with a run-summary cover; its review count and
issue totals summarize the current completed-case snapshot without adjudicating
the findings. The PDF is a convenience view; JSON, NIfTI, and Parquet remain
authoritative.

## Orientation states

- `PASS_METADATA_MATCH`: anatomy supports the physical metadata.
- `PASS_METADATA_UNCERTAIN`: no safe contradiction was established; metadata
  was retained and uncertainty requires review when configured.
- `MISMATCH_REPAIRED`: a supported lossless proper rotation was applied.
  Segmentation continues, but manual review is mandatory.
- `MISMATCH_UNCERTAIN`: anatomy and metadata disagree without enough evidence
  for a safe repair; metadata is retained and review is mandatory.
- `HEADER_UNCERTAIN`: the physical header itself is questionable.
- `ORIENTATION_FAILED`: no valid prepared-image contract was produced.

Review `orientation/orientation_review.png` and
`orientation/orientation_report.json`. For repaired cases verify the superior,
inferior, anterior, posterior, left, and right anatomy against the scout and
the transformation record. Do not infer correctness solely from a successful
downstream segmentation.

## Vertebral review

Use `qc/spine_review.png`, `qc/vertebral_result.json`, the prepared CT, and
`masks/vertebral_bodies.nii.gz`. Review at least:

- enumeration and cranio-caudal monotonicity;
- T13, L6, rib-remnant, and sacral conventions;
- missing/internal sequence gaps;
- fragments or merged corpora;
- material fragmentation or clinically relevant field-of-view truncation; and
- physical alignment of every corpus mask.

Only corpus labels are accepted downstream. The presence of a full-vertebra
upstream artifact does not justify using posterior elements for territories.

## Measurement review

Use `qc/measurement_review.png`, `qc/qc.json`, and the five canonical tables.
Important conditions include body/trunk contour contact with the image edge,
fragmentation, internal gaps, abrupt circumference changes, incomplete
vertebral territories, insufficient coverage, invalid landmarks, and extrema
at a search boundary.

Missing measurements remain null with validity and reason columns. Observed
anthropometric extrema from incomplete searches retain a numeric value but are
marked ineligible with a reason and coverage. Do not replace either state with
zero. A valid zero means the tissue was assessed and absent.

FOV contact is orthogonal to numerical availability. The slice table records
full-body and primary-trunk contact separately; the case summary records their
counts. A sacral maximum measured from a boundary-cropped but closed,
nonfragmented trunk contour remains numeric and propagates to the ratios, but
is marked FOV-cropped and ineligible as an exact anthropometric value.
Routine cranial or caudal vertebral contact is retained as informational QC.
It does not by itself request manual review, and observed measurements from a
valid truncated vertebral territory remain available with explicit incomplete
coverage. Material fragmentation above the configured 5% largest-component
limit remains a warning.

## Adjudication

The generated review queue contains empty `reviewer`, `reviewed_at`,
`adjudication`, and `review_comment` fields. Adjudication belongs in a separate
governed derivative linked by `run_id`, `case_id`, `analysis_id`, and flag code;
do not edit the immutable case bundle in place. Preserve the original flag and
record acceptance, exclusion, relabeling, or rerun decisions explicitly.

## Privacy

Review artifacts are created from the supplied CT pixels and may contain
burned-in text if de-identification failed before the pipeline. The software
does not OCR or scrub pixels. Use only deidentified inputs and inspect artifacts
before moving them outside the controlled environment.
