# QC and manual review

Execution status and scientific QC are separate. A case can complete and still
require review. The pipeline continues readable uncertain cases by default,
retains structured reasons, and never clears flags automatically.

## Where to start

For a cohort, inspect:

```bash
bodycomposition results review-queue output/runs/<run-id> --json
```

Then open the linked case manifest and canonical artifacts. If PDF reporting is
enabled, affected cases have a warning annotation on their individual page and
the combined PDF begins with a manual-review summary when any case is flagged.
The PDF is a convenience view; JSON, NIfTI, and Parquet remain authoritative.

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
- superior/inferior field-of-view truncation; and
- physical alignment of every corpus mask.

Only corpus labels are accepted downstream. The presence of a full-vertebra
upstream artifact does not justify using posterior elements for territories.

## Measurement review

Use `qc/measurement_review.png`, `qc/qc.json`, and the three tables. Important
conditions include body/trunk contour contact with the image edge,
fragmentation, internal gaps, abrupt circumference changes, incomplete
vertebral territories, insufficient coverage, invalid landmarks, and extrema
at a search boundary.

Missing measurements remain null with validity and reason columns. Do not
replace them with zero. A valid zero means the tissue was assessed and absent.

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
