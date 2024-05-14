# Labels

In this repository, we differentiate between `labels` and `masks`:

- `labels` are the unprocessed and unchanged output of the respective segmentation model. Thus, they differ between models and are not directly comparable.
- `masks` are the remapped and postprocessed labels used for the calculation of the outcome parameters.

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