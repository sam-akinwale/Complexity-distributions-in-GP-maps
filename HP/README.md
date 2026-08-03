# HP Model – Data and Reproducibility Notes

This directory contains the code and processed data used to generate the HP model figures presented in the manuscript.

## Repository contents

- `HP_master_revised.ipynb` — analysis and figure-generation notebook.
- `plot_a/`, `plot_c/`, and `plot_d/` — processed datasets required to reproduce the published figures.

## Data files

The notebook uses the following processed datasets:

- `plot_a/plot_a_files/unified_plotA_B_D.txt`
- `plot_c/plotC_data.txt`
- `plot_d/plotD_data_incl0.txt`

These files are included in the repository and are sufficient to reproduce the published figures.

## Raw distribution files

The Plot A comparison also uses

```
forsam3/HP_sizes/random_sample_mc_n15.txt
```

In addition, the supplementary preprocessing section of the notebook reads all files matching

```
forsam3/HP_sizes/random_sample_mc_n*.txt
```

to analyse the complexity distributions across HP sequence lengths.

These raw distribution files are large and are therefore not included in this repository because of GitHub file size limits.

The raw datasets are available from the authors (University of Oxford) upon reasonable request. To rerun the preprocessing, place the files in

```
forsam3/HP_sizes/
```

before executing the corresponding notebook cells.
