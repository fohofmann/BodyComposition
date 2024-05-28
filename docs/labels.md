# Labels

In this pipeline, we differentiate between `labels` and `masks`:

- `labels` are the unprocessed and unchanged output of the respective segmentation model. Thus, they differ between models and are not directly comparable. For instance, for body composition `labels` refer to compartments (e.g., muscle comparment, visceral compartment, subcutaneous compartment, etc.) as identified by the respective model.
- `masks` are the remapped and postprocessed labels used for the calculation of the outcome parameters. For instance, for body composition `masks` refer to the postprocessed version compartments by using a combination of thresholding and size filters to subset a specific proportion within the compartment (e.g. muscle tissue, intermuscular adipose tissue, visceral adipose tissue, etc.).

## Definitions: Compartments vs Tissue
We distinguish between compartments (defined by anatomical boundaries) and tissue (representing the tissue composition within a compartment). The terms *muscle compartment*, *iliopsoas muscle compartment*, *visceral compartment*, and *subcutaneous compartment* refer to regions delineated by surrounding structures such as skin, fascia, bone, vessels, and organs. The terms total *skeletal muscle tissue* (SM<sub>total</sub>), *psoas skeletal muscle tissue* (SM<sub>psoas</sub>), *intermuscular adipose tissue* (IMAT), *visceral adipose tissue* (VAT), and *subcutaneous adipose tissue* (SAT) refer to a tissue-specific fraction of these compartments. This differentiation is based on the assignment of each voxel based on Hounsfield Units (HU) thresholds. Pre- and postprocessing are [configurable](./config.md), but are kept to a minimum and use established trehsholds by default:

| Compartment | Definition | Tissue | Threshold (HU) |
| --- | --- | --- | --- |
| Muscle | Regions predominantly including muscles, without separating individual muscles. Includes smaller vessels and inter- and intramuscular connective tissues such as fascia. | SM<sub>total</sub> | -29 to 150 |
| | | IMAT | -190 to -30 |
| Iliospoas | Region including the iliopsoas muscle (psoas major & iliacus), without separating individual muscle parts. Includes smaller vessels and inter- and intramuscular connective tissues such as fascia. | SM<sub>psoas</sub> | -29 to 150 |
| | | IMAT | -190 to -30 |
| Visceral | Regions beneath the thoracic cage (defined by ribs, costal cartilages, intercostal muscles, sternum, and spine) and muscles of the abdominal wall (rectus abdominis, external and internal obliques, transversus abdominis, psoas, quadratus lumborum). Includes pelvis and retroperitoneum. Excludes solid organs (e.g., stomach, liver, gallbladder, spleen, pancreas, small intestine, colon, kidneys) and larger vessels (e.g., aorta, vena cava, portal vein, splenic vein). | VAT | -190 to -30 |
| Subcutaneous | Regions beneath the skin, overlaying fascia (e.g., around muscles) or periosteum. Excludes foreign material (e.g., implants). | SAT | -190 to -30 |


## TotalSegmentator

### Labels

The labels are equivalent to the (v2-) labels as described in the [TotalSegmentator repository](https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py).

### Masks

#### Vertebral Bodies

| Label | Name |
| --- | --- |
| 1 | T1 |
| 2 | T2 |
| 3 | T3 |
| 4 | T4 |
| 5 | T5 |
| 6 | T6 |
| 7 | T7 |
| 8 | T8 |
| 9 | T9 |
| 10 | T10 |
| 11 | T11 |
| 12 | T12 |
| 13 | L1 |
| 14 | L2 |
| 15 | L3 |
| 16 | L4 |
| 17 | L5 |
| 18 | L6 |
| 19 | SACRUM |
| 20 | COCCYX |
| 21 | T13 |

#### Tissue

| Label | Name |
| --- | --- |
| 1 | SAT |
| 2 | VAT |
| 3 | SM |
| 4 | PSOAS |
| 5 | IMAT |