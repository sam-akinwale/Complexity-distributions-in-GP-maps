#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Files to process
# ---------------------------------------------------

files = [
    "/Users/sam/Documents/Oxford/Physics/sloppiness/circadian/mut_project_updates/figures/tooth model/new_range/plot_a/filtered_10000_opc_cusp.txt",

    "/Users/sam/Documents/Oxford/Physics/sloppiness/circadian/mut_project_updates/figures/tooth model/new_range/plot_a/filtered_6000_opc_cusp.txt",

    "/Users/sam/Documents/Oxford/Physics/sloppiness/circadian/mut_project_updates/figures/tooth model/new_range/plot_a/filtered_2000_opc_cusp.txt",
]

# ---------------------------------------------------
# Process each file
# ---------------------------------------------------

for path_str in files:

    path = Path(path_str)

    print(f"\nProcessing: {path.name}")

    # Read whitespace-separated table
    df = pd.read_csv(path, sep=r"\s+")

    if "OPC" not in df.columns:
        raise ValueError(f"No OPC column found in {path}")

    # Count occurrences of each OPC value
    counts = df["OPC"].value_counts()

    # Keep only OPC values occurring more than once
    keep_values = counts[counts > 1].index

    filtered_df = df[df["OPC"].isin(keep_values)].copy()

    n_removed = len(df) - len(filtered_df)

    print(f"Original rows : {len(df)}")
    print(f"Removed rows  : {n_removed}")
    print(f"Remaining rows: {len(filtered_df)}")

    # Output filename
    output_path = path.with_name(
        path.stem + "_nosingletons.txt"
    )

    # Save
    filtered_df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(f"Saved: {output_path}")
