# Robustness and Evolvability Across Genotype–Phenotype Maps

This repository contains the code and processed data used to reproduce the figures presented in the manuscript.

## Repository structure

The repository is organised by genotype–phenotype map:

```
.
├── biomorph/
├── HP/
├── RNA/
├── teeth/
└── Polyominos/
```

Each directory contains:

- a Jupyter notebook used to generate the figures;
- the processed datasets required by that notebook;
- a model-specific `README.md` describing the required input files and repository structure.

The `Polyominos` directory contains the processed data used for the analyses presented in the manuscript. The simulation code for the Polyomino model is maintained separately and is available at:

https://github.com/agrawalprayer/the_polycube_model

## Reproducing the figures

Each notebook is self-contained and reproduces the figures for a single genotype–phenotype map.

To reproduce a figure:

1. Navigate to the appropriate model directory.
2. Open the corresponding notebook.
3. Run all notebook cells.

Unless otherwise noted in the model-specific `README.md`, the processed datasets included in this repository are sufficient to reproduce the published figures.

## Data availability

The repository contains the processed datasets required to reproduce the figures presented in the manuscript.

Where raw simulation outputs are omitted because of their size, the corresponding model-specific `README.md` explains which files are unavailable and how they may be obtained.

## Requirements

The notebooks were developed in Python using standard scientific computing libraries, including:

- NumPy
- Pandas
- Matplotlib
- SciPy

Additional package requirements, if any, are documented within the individual notebooks.
