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

    required_cols = {"OPC", "CUSPS"}

    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_cols} in {path}"
        )

    original_n = len(df)

    # ---------------------------------------------------
    # Remove singleton OPC values
    # ---------------------------------------------------

    opc_counts = df["OPC"].value_counts()

    keep_opc = opc_counts[opc_counts > 1].index

    df = df[df["OPC"].isin(keep_opc)].copy()

    after_opc_n = len(df)

    # ---------------------------------------------------
    # Remove singleton CUSPS values
    # ---------------------------------------------------

    cusp_counts = df["CUSPS"].value_counts()

    keep_cusps = cusp_counts[cusp_counts > 1].index

    df = df[df["CUSPS"].isin(keep_cusps)].copy()

    final_n = len(df)

    # ---------------------------------------------------
    # Report
    # ---------------------------------------------------

    print(f"Original rows            : {original_n}")
    print(f"After OPC filtering      : {after_opc_n}")
    print(f"Final rows after CUSPS   : {final_n}")

    print(f"Removed total            : {original_n - final_n}")

    # ---------------------------------------------------
    # Output filename
    # ---------------------------------------------------

    output_path = path.with_name(
        path.stem + "_nosingletons.txt"
    )

    # Save
    df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(f"Saved: {output_path}")
