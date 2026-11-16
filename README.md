# gbt-demo

A demo of gradient boosted trees on the Golub et al. (1999) ALL/AML leukemia
gene-expression dataset.

## Data

Golub TR et al., "Molecular Classification of Cancer: Class Discovery and Class
Prediction by Gene Expression Monitoring", *Science* 286:531–537 (1999).
doi:[10.1126/science.286.5439.531](https://doi.org/10.1126/science.286.5439.531).
A copy of the paper, from the same Broad page, is in
`docs/Golub_et_al_1999.pdf`. It is the publisher's copyrighted PDF, so it is
gitignored; download it from
<https://pubs.broadinstitute.org/mpr/projects/Leukemia/Golub_et_al_1999.pdf>.

Downloaded 2026-11-16 from the Broad Institute's original publication page,
<https://pubs.broadinstitute.org/mpr/projects/Leukemia/>, into `data/raw/`
(checksums in `data/raw/SHA256SUMS`). The old `datasets.genepattern.org` copies
(`all_aml_*.gct`) no longer resolve.

| File | Contents |
| --- | --- |
| `data_set_ALL_AML_train.txt` | Training set: 38 samples (27 ALL, 11 AML) × 7,129 probes |
| `data_set_ALL_AML_independent.txt` | Independent test set: 34 samples (20 ALL, 14 AML) × 7,129 probes |
| `table_ALL_AML_samples.txt` | Per-sample labels: ALL/AML, BM/PB, T/B-cell, FAB, source lab |
| `table_ALL_AML_predic.txt` | The paper's own predictions, for comparison |
| `table_ALL_AML_rfactors.txt` | Per-chip intensity re-scaling factors |
| `Files_descriptions.txt`, `Experimental_protocol.html` | Broad's notes on the files and protocol |

The Broad page also serves `.tsv` copies of the two expression files, which are
byte-identical to the `.txt` versions, so they were not kept.

### Format notes

- Platform: Affymetrix Hu6800 (HuGeneFL). Values are MAS4 "average difference"
  integers, so they include negative values (range about −28k to 71k).
- Each sample is a pair of columns: the expression value, then a `call` column
  (P/A/M, present/absent/marginal). Drop the `call` columns to get the matrix.
- Sample columns are **not** in numeric order (e.g. train runs 1–27, 34–38,
  28–33), so join to labels by sample number, not by position.
- Data rows end with a trailing tab. In pandas use
  `pd.read_csv(path, sep="\t", index_col=False)`. Otherwise the first column is
  silently taken as the index and every column shifts by one.
- Class labels are not in the expression files; take them from
  `table_ALL_AML_samples.txt` (samples 1–38 train, 39–72 test).
