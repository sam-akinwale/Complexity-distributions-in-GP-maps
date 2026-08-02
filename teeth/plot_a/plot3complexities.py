#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def read_opc_values(path):
    """
    Read the OPC column from a file with header like:
    ID    OPC    CUSPS
    """
    opc_values = []

    with open(path, "r") as f:
        lines = f.readlines()

    if not lines:
        return np.array([])

    header = lines[0].strip().split()
    header_upper = [h.upper() for h in header]

    if "OPC" not in header_upper:
        raise ValueError(f"Could not find OPC column in header of {path}")

    opc_col = header_upper.index("OPC")

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) <= opc_col:
            continue
        try:
            opc_values.append(float(parts[opc_col]))
        except ValueError:
            continue

    return np.array(opc_values, dtype=float)


def binned_probability(values, bin_edges):
    """
    Convert raw values into a normalized histogram probability.
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts / float(total)


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python3 plot3_complexities.py <input_file1> <input_file2> <input_file3> [bin_width] [title]"
        )
        print(
            "Example: python3 plot3_complexities.py filtered_10000_opc_cusp.txt filtered_6000_opc_cusp.txt filtered_2000_opc_cusp.txt 2.0 \"OPC distributions\""
        )
        sys.exit(1)

    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    input_file3 = sys.argv[3]

    bin_width = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    title = sys.argv[5] if len(sys.argv) > 5 else "OPC distributions"

    files = [input_file1, input_file2, input_file3]

    # Read all OPC data
    datasets = []
    for path in files:
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}")
            sys.exit(1)

        opc_values = read_opc_values(path)
        if len(opc_values) == 0:
            print(f"Error: no valid OPC values found in {path}")
            sys.exit(1)

        datasets.append(opc_values)

    # Build shared bin edges across all files
    all_values = np.concatenate(datasets)
    xmin = np.floor(all_values.min())
    xmax = np.ceil(all_values.max())

    bin_edges = np.arange(xmin, xmax + bin_width, bin_width)
    if len(bin_edges) < 2:
        bin_edges = np.array([xmin, xmin + bin_width])

    plt.figure(figsize=(16, 8))

    for path, values in zip(files, datasets):
        probs = binned_probability(values, bin_edges)
        label = os.path.splitext(os.path.basename(path))[0]
        plt.step(bin_edges[:-1], probs, where="post", linewidth=2.5, label=label)

    plt.xlabel("OPC", fontsize=18)
    plt.ylabel("Probability", fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(fontsize=12, frameon=False)
    plt.grid(True, alpha=0.25)
    plt.tick_params(axis="both", labelsize=14)
    plt.xlim(bin_edges[0], bin_edges[-1])
    plt.tight_layout()

    base_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    output_name = "_vs_".join(base_names) + f"_bw{bin_width:g}.png"

    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_name}")
    plt.show()


if __name__ == "__main__":
    main()
