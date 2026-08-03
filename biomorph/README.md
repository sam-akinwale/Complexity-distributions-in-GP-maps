# Biomorph Model – Data and Reproducibility Notes

This directory contains the code and processed data used to generate the Biomorph model figures presented in the manuscript.

## Repository contents

- `Biomorph_master_revised.ipynb` — analysis and figure-generation notebook.
- `plot_a/`, `plot_b/`, `plot_c/`, `plot_d/` — processed data files required to reproduce the published figures.

## Data files

The notebook uses the following processed datasets:

- `plot_a/plot_a_files/unified_plotA_complexity.csv`
- `plot_b/plot_b_files/plotB_biomorph_entropy_global_vs_local.txt`
- `plot_c/plotC_data_equalN.txt`
- `plot_d/plotD_data_incl0.txt`

These files are included in the repository and are sufficient to reproduce the published figures.

## Regenerating the Plot A processed data

The processed file

```
plot_a/plot_a_files/unified_plotA_complexity.csv
```

can be regenerated from the full raw distribution file

```
plot_a/plot_a_files/unified_plotA.txt
```

The raw distribution file is approximately 1 GB and is not included in this repository because of GitHub file size limits.

The raw dataset is available from the authors (University of Oxford) upon reasonable request. To regenerate the processed Plot A data, place `unified_plotA.txt` in

```
plot_a/plot_a_files/
```

and run the preprocessing section of `Biomorph_master_revised.ipynb`.
