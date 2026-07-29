# Pipeline architecture and geometry

BodyComposition 1.x has one orchestration service and one canonical action
sequence. The Python API and CLI are adapters over `PipelineService`; they do
not implement separate pipelines.

## Stage sequence

1. Validate one explicit three-dimensional NIfTI CT or one selected DICOM CT
   series and calculate content, pixel, and physical-geometry identities.
2. Assess orientation with pinned CTDeepRot. Preserve trustworthy metadata;
   apply only a supported lossless proper-rotation repair when anatomy and
   metadata disagree; retain uncertainty and review provenance.
3. Segment and label vertebral bodies. The default adapter uses upstream
   SPINEPS 2.0.0, its pinned VERIDAH CT-labeling model, and TPTBox 0.7.5
   VibeSeg crop inference.
4. Segment anatomical compartments with the selected internal nnU-Net
   v2 model. In the explicit low-resource scope, localize L3 first and supply
   only an L3-centred z region to the ResEncM tissue model.
5. Derive the body/trunk measurement support, optional TotalSegmentator
   landmarks, per-slice measurements, three-bin native vertebral summaries,
   established case summaries, and QC.
6. Optionally create a one-page report from the already computed scientific
   artifacts.
7. Validate schemas, inventory every artifact, write the case manifest, and
   atomically promote the complete bundle.

There is no automatic backend fallback. A selected model failure is an
observable failed result. Reporting never reruns scientific inference.

## Prepared-image boundary

For DICOM input, the service first uses SimpleITK/GDCM to assemble the selected
series and writes a pixel/geometry-verified temporary NIfTI in the attempt
workspace. Conversion neither canonicalizes nor repairs orientation. The
temporary file is not promoted into the result bundle; standalone conversion
is available through `bodycomposition convert` and `convert_dicom`.

The orientation stage owns orientation. It produces one immutable prepared-image object with
the image, transform, pixel digest, physical-domain provenance, orientation
state, and review flags. Downstream stages consume that object and must not
independently canonicalize, transpose, or relabel the input.

The explicit convention is:

- SimpleITK size, spacing, origin, indices, and physical points: `x-y-z`;
- arrays returned by `SimpleITK.GetArrayFromImage`: `z-y-x`;
- physical coordinates: DICOM LPS millimetres; and
- longitudinal ordering: the physical superior axis, not storage index.

Every segmentation output is checked against the prepared CT's full physical
domain. The low-resource z crop is an index region with an affine-derived
origin; it does not resize pixels. nnU-Net returns the cropped prediction at
the input shape, after which only the L3 target is pasted onto the full
prepared grid. Label resampling elsewhere uses nearest-neighbour
interpolation. Geometry is not accepted merely because array shapes match.

## Vertebral contract

The downstream anatomical input is `masks/vertebral_bodies.nii.gz`: the
vertebral corpora only. SPINEPS semantic or full-vertebra instance outputs may
be retained for provenance/QC, but posterior elements do not define
territories, measurements, or PDF overlays.

Native labels are preserved. T13 and L6 are variants, not errors, and sacrum
is an ordinary supported territory. Sequence gaps, non-monotonic order,
fragments, edge truncation, and low-confidence enumeration remain explicit QC
conditions. They never cause relabeling by silent heuristic fallback.

## Execution and atomicity

One process reuses compatible model objects across cases. A stage-aware memory
barrier releases earlier segmentation bundles before the large optional
measurement-support predictor and releases that predictor immediately after
use. Multiple identical processes can join a filesystem-backed whole-case
queue. Claims have heartbeats, bounded stale takeover, fencing tokens, and a
maximum attempt count. A stale process cannot overwrite a reclaimed case.

Case work occurs in an attempt directory on the same filesystem. A successful
bundle is renamed atomically to its content-addressed destination only after
the manifest and artifact inventory are complete. Failed or cancelled bundles
are retained under `failed/`; partial work is never presented as success.

A recognized accelerator OOM records the failure and enables a simple
low-memory strategy on the next compatible invocation using equivalent
hardware. It unloads model bundles between segmentation stages. A worker
timeout remains eligible for bounded takeover by another compatible worker.
Neither path silently changes the scientific plan.

See [execution.md](execution.md), [output-schema.md](output-schema.md), and
[reproducibility.md](reproducibility.md).
