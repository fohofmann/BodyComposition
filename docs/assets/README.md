# README assets

The README contains two BodyComposition-generated derivatives of CT-ORG volume
0:

- `pipeline-report-example.png` is the complete current patient-page report
  screenshot.
- `pipeline-overview.png` is a deterministic recomposition for legibility at
  README width. The spine and tissue-area columns retain their shared
  superior-inferior transform from the current report, and the axial overlay
  was regenerated from the same retained CT and segmentation masks on a finer
  display grid. The full table, technical metadata, and notes were omitted. It
  is an illustrative overview, not an authoritative numerical result.

> Rister, B., Shivakumar, K., Nobashi, T., & Rubin, D. L. (2019). CT-ORG: A
> Dataset of CT Volumes With Multiple Organ Segmentations (Version 1)
> [dataset]. The Cancer Imaging Archive.
> <https://doi.org/10.7937/TCIA.2019.TT7F4V7O>

The source dataset is licensed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). The screenshot was
created by rendering a one-page BodyComposition report; its automated
segmentations, annotations, and measurements are pipeline outputs rather than
CT-ORG ground truth. These changes and the source attribution are also stated
in the README caption. Both image assets remain subject to CC BY 3.0 and are
not relicensed under BodyComposition's Apache-2.0 license. Attribution does not
imply endorsement by the dataset authors or The Cancer Imaging Archive.
