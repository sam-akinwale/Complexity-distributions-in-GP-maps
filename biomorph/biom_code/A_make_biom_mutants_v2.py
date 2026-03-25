#!/usr/bin/env python3
"""
Generate random biomorphs and calculate their complexity.

Usage:
    python3 make_biom.py [n_biomorphs] [genemax] [g9min] [g9max] [resolution] [threshold] [output_file]
    
If output_file is not provided, generates filename: biom_[n_biomorphs]_[genemax]_[g9min]_[g9max]_[resolution]_[threshold].txt

Examples:
    python3 make_biom.py 100000 4 1 10 40 6
    → creates: biom_100000_4_1_10_40_6.txt
    
    python3 make_biom.py 100000 4 1 10 40 6 custom_output.txt
    → creates: custom_output.txt
"""

import numpy as np
import sys
import pandas as pd
from pathlib import Path
from os import cpu_count
from multiprocessing import Pool
from functools import partial
from copy import deepcopy

# Importing from functions_for_GP_analysis
import_failed = False
try:
    from functions_for_GP_analysis.genotype_to_raster import get_biomorphs_phenotype
    from functions_for_GP_analysis.GPproperties import LZ_complexity_array
    print("Successfully imported from functions_for_GP_analysis")
except ImportError:
    try:
        from genotype_to_raster import get_biomorphs_phenotype
        from GPproperties import LZ_complexity_array
        print("Successfully imported from local directory")
    except ImportError as e:
        print(f"Import error: {e}")
        import_failed = True

# If imports still fail, provide error message
if import_failed:
    print("\nERROR: Could not import required modules.")
    print("Please ensure the script is run from the correct directory with the module files.")
    sys.exit(1)

def generate_random_genotype(genemax, g9min, g9max):
    """Generate a random genotype within specified bounds."""
    genes_1_8 = tuple(np.random.randint(-genemax, genemax + 1, size=8))
    gene_9 = np.random.randint(g9min, g9max + 1)
    return genes_1_8 + (gene_9,)

def mutate_genotype(genotype, genemax, g9min, g9max):
    """
    Introduce one random mutation to a genotype.
    Mutates the selected gene to any value within its valid range.
    Returns mutated genotype.
    """
    genotype_list = list(genotype)
    gene_index = np.random.randint(0, 9)  # Choose random gene to mutate
    
    if gene_index < 8:  # Genes 1-8: can be any value in [-genemax, genemax]
        genotype_list[gene_index] = np.random.randint(-genemax, genemax + 1)
    else:  # Gene 9 (recursion order): can be any value in [g9min, g9max]
        genotype_list[8] = np.random.randint(g9min, g9max + 1)
    
    return tuple(genotype_list)

def get_one_genotype_per_phenotype(results_list):
    """
    From results (genotype, genotype_str, phenotype_str, complexity),
    find one genotype per unique phenotype.
    Returns list of (genotype_tuple, genotype_str, phenotype_str, complexity).
    """
    phenotype_dict = {}
    for genotype_tuple, genotype_str, phenotype_str, complexity in results_list:
        if phenotype_str not in phenotype_dict:
            phenotype_dict[phenotype_str] = (genotype_tuple, genotype_str, phenotype_str, complexity)
    
    return list(phenotype_dict.values())

def calculate_phenotype_and_complexity(genotype, resolution, threshold, genemax, g9min):
    """
    Calculate phenotype and complexity for a single genotype.
    Returns: (genotype_tuple, genotype_str, phenotype_str, complexity)
    """
    try:
        # Get phenotype as binary string
        phenotype_str = get_biomorphs_phenotype(
            genotype, 
            resolution=resolution, 
            threshold=threshold
        )
        
        # Calculate LZ complexity
        complexity = LZ_complexity_array(phenotype_str)
        
        # Format genotype as string (comma-separated)
        genotype_str = ','.join([str(g) for g in genotype])
        
        return (genotype, genotype_str, phenotype_str, complexity)
    except Exception as e:
        print(f"Error processing genotype {genotype}: {str(e)[:50]}")
        return None

def calculate_phenotype_worker_mutant(args):
    """
    Worker function for parallel phenotype calculation of mutant genotypes.
    Args: (genotype_tuple, resolution, threshold)
    Returns: (phenotype_string, complexity_value)
    """
    genotype_tuple, resolution, threshold = args
    try:
        phenotype = get_biomorphs_phenotype(genotype_tuple, resolution, threshold)
        complexity = LZ_complexity_array(phenotype)
        return phenotype, complexity
    except Exception as e:
        return f"ERROR: {str(e)}", -1

def main():
    # Parse command line arguments
    n_biomorphs = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    genemax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    g9min = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    g9max = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    resolution = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    threshold = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    
    # Auto-generate filename from parameters if not provided
    if len(sys.argv) > 7:
        output_file = sys.argv[7]
    else:
        output_file = f'biom_{n_biomorphs}_{genemax}_{g9min}_{g9max}_{resolution}_{threshold}_v2.txt'
    
    print("="*70)
    print("BIOMORPH COMPLEXITY CALCULATOR")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Number of biomorphs: {n_biomorphs}")
    print(f"  Gene parameters: genemax={genemax}, g9min={g9min}, g9max={g9max}")
    print(f"  Phenotype parameters: resolution={resolution}, threshold={threshold}")
    print(f"  Output file: {output_file}")
    print(f"  CPU cores available: {cpu_count()}\n")
    
    # Step 1: Generate random genotypes
    print("Step 1: Generating random genotypes...")
    np.random.seed(42)  # For reproducibility, remove if you want different runs
    genotypes = [generate_random_genotype(genemax, g9min, g9max) 
                 for _ in range(n_biomorphs)]
    print(f"  ✓ Generated {len(genotypes)} genotypes\n")
    
    # Step 2: Calculate phenotypes and complexity in parallel
    print("Step 2: Computing phenotypes and complexity in parallel...")
    parallel_function = partial(
        calculate_phenotype_and_complexity,
        resolution=resolution,
        threshold=threshold,
        genemax=genemax,
        g9min=g9min
    )
    
    with Pool(cpu_count()) as pool:
        results = pool.map(parallel_function, genotypes, chunksize=100)
    
    # Filter out failed calculations
    results = [r for r in results if r is not None]
    print(f"  ✓ Successfully calculated {len(results)}/{len(genotypes)} biomorphs\n")
    
    # Step 3: Save original random genotypes before deduplication
    print("Step 3: Saving original random genotypes...")
    
    # Create base filename without extension
    if output_file.endswith('.txt'):
        output_base = output_file[:-4]
    else:
        output_base = output_file
    
    df_random = pd.DataFrame(
        results,
        columns=['genotype_tuple', 'genotype', 'phenotype', 'complexity']
    )
    path_random = Path(f'{output_base}_random.txt')
    df_random[['genotype', 'phenotype', 'complexity']].to_csv(path_random, index=False, sep=' ')
    print(f"  ✓ Saved {len(df_random)} random genotypes to: {path_random}\n")
    
    # Step 4: Find one genotype per unique phenotype
    print("Step 4: Extracting one genotype per phenotype...")
    unique_genotypes = get_one_genotype_per_phenotype(results)
    print(f"  ✓ Found {len(unique_genotypes)} unique phenotypes\n")
    
    # Step 5: Create mutation variants with recalculated phenotypes
    print("Step 5: Generating mutation variants with phenotype recalculation...")
    
    # 0 mutations (original)
    data_0mut = unique_genotypes
    
    # Pass 5a: Generate 1-mutation variants
    print("  Pass 5a: Generating 1-mutation variants...")
    mutant_1_genotypes = []
    mutant_1_info = []
    for genotype_tuple, genotype_str, phenotype_str, complexity in unique_genotypes:
        mutated_genotype = mutate_genotype(genotype_tuple, genemax, g9min, g9max)
        mutated_genotype_str = ','.join([str(g) for g in mutated_genotype])
        mutant_1_genotypes.append((mutated_genotype, resolution, threshold))
        mutant_1_info.append((mutated_genotype, mutated_genotype_str))
    
    # Pass 5b: Generate 2-mutation variants
    print("  Pass 5b: Generating 2-mutation variants...")
    mutant_2_genotypes = []
    mutant_2_info = []
    for genotype_tuple, genotype_str, phenotype_str, complexity in unique_genotypes:
        mutated_genotype = mutate_genotype(genotype_tuple, genemax, g9min, g9max)
        mutated_genotype = mutate_genotype(mutated_genotype, genemax, g9min, g9max)
        mutated_genotype_str = ','.join([str(g) for g in mutated_genotype])
        mutant_2_genotypes.append((mutated_genotype, resolution, threshold))
        mutant_2_info.append((mutated_genotype, mutated_genotype_str))
    
    # Pass 5c: Recalculate phenotypes in parallel for 1-mutation variants
    print("  Pass 5c: Recalculating phenotypes for 1-mutation variants in parallel...")
    with Pool(cpu_count()) as pool:
        phenotypes_1mut = pool.map(calculate_phenotype_worker_mutant, mutant_1_genotypes)
    
    # Pass 5d: Recalculate phenotypes in parallel for 2-mutation variants
    print("  Pass 5d: Recalculating phenotypes for 2-mutation variants in parallel...")
    with Pool(cpu_count()) as pool:
        phenotypes_2mut = pool.map(calculate_phenotype_worker_mutant, mutant_2_genotypes)
    
    # Build final result lists
    data_1mut = []
    for (genotype_tuple, genotype_str), (phenotype_str, complexity) in zip(mutant_1_info, phenotypes_1mut):
        data_1mut.append((genotype_tuple, genotype_str, phenotype_str, complexity))
    
    data_2mut = []
    for (genotype_tuple, genotype_str), (phenotype_str, complexity) in zip(mutant_2_info, phenotypes_2mut):
        data_2mut.append((genotype_tuple, genotype_str, phenotype_str, complexity))
    
    print(f"  ✓ Generated and recalculated {len(data_1mut)} 1-mutation variants")
    print(f"  ✓ Generated and recalculated {len(data_2mut)} 2-mutation variants\n")
    
    # Step 6: Save all variant files
    print("Step 6: Saving variant files...")
    
    # Save 0 mutations
    df_0mut = pd.DataFrame(
        data_0mut,
        columns=['genotype_tuple', 'genotype', 'phenotype', 'complexity']
    )
    path_0mut = Path(f'{output_base}_0mut.txt')
    df_0mut[['genotype', 'phenotype', 'complexity']].to_csv(path_0mut, index=False, sep=' ')
    print(f"  ✓ Saved {len(df_0mut)} 0-mutation genotypes to: {path_0mut}")
    
    # Save 1 mutation
    df_1mut = pd.DataFrame(
        data_1mut,
        columns=['genotype_tuple', 'genotype', 'phenotype', 'complexity']
    )
    path_1mut = Path(f'{output_base}_1mut.txt')
    df_1mut[['genotype', 'phenotype', 'complexity']].to_csv(path_1mut, index=False, sep=' ')
    print(f"  ✓ Saved {len(df_1mut)} 1-mutation genotypes to: {path_1mut}")
    
    # Save 2 mutations
    df_2mut = pd.DataFrame(
        data_2mut,
        columns=['genotype_tuple', 'genotype', 'phenotype', 'complexity']
    )
    path_2mut = Path(f'{output_base}_2mut.txt')
    df_2mut[['genotype', 'phenotype', 'complexity']].to_csv(path_2mut, index=False, sep=' ')
    print(f"  ✓ Saved {len(df_2mut)} 2-mutation genotypes to: {path_2mut}\n")
    
    # Print summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total random genotypes generated: {len(df_random)}")
    print(f"Total unique phenotypes found: {len(df_0mut)}")
    print(f"\nComplexity (LZ) for 0-mutation genotypes:")
    print(f"  Mean:    {df_0mut['complexity'].mean():.4f}")
    print(f"  Std dev: {df_0mut['complexity'].std():.4f}")
    print(f"  Min:     {df_0mut['complexity'].min():.4f}")
    print(f"  Max:     {df_0mut['complexity'].max():.4f}")
    print(f"  Median:  {df_0mut['complexity'].median():.4f}")
    
    print(f"\nFiles created:")
    print(f"  {path_random.stat().st_size / (1024):.1f} KB - {path_random} (random)")
    print(f"  {path_0mut.stat().st_size / (1024):.1f} KB - {path_0mut}")
    print(f"  {path_1mut.stat().st_size / (1024):.1f} KB - {path_1mut}")
    print(f"  {path_2mut.stat().st_size / (1024):.1f} KB - {path_2mut}")
    
    print(f"\nFirst 10 rows (0-mutation):")
    print(df_0mut[['genotype', 'phenotype', 'complexity']].head(10).to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    return df_random, df_0mut, df_1mut, df_2mut

if __name__ == "__main__":
    df_random, df0, df1, df2 = main()
