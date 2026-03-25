#!/usr/bin/env python3
"""
Load a biomorph file, divide by complexity quintiles, apply cumulative mutations,
and save results with quintile and mutation count tracking.
PHENOTYPES RECALCULATED IN PARALLEL USING ALL CPU CORES.

IMPORTANT NOTES:
  1. Parameters are automatically extracted from the filename!
  2. Genotypes ARE mutated and phenotypes/complexity are RECALCULATED
  3. Phenotype calculation uses all available CPU cores (parallel)

Filename format: biom_[n]_[genemax]_[g9min]_[g9max]_[resolution]_[threshold]...

Usage:
    python3 analyze_quintile_mutations.py <input_file> [max_mutations] [min_mutations] [output_file]

Example:
    # Mutations 0 to 5
    python3 analyze_quintile_mutations.py biom_100000_3_1_8_30_5_random.txt 5
    → Creates: biom_100000_3_1_8_30_5_random_quintile_mutations_mut0to5.txt
    
    # Mutations 3 to 10 only (skip 0-2)
    python3 analyze_quintile_mutations.py biom_100000_3_1_8_30_5_random.txt 10 3
    → Creates: biom_100000_3_1_8_30_5_random_quintile_mutations_mut3to10.txt
    → Applies mutations from 1 to 10, records only 3-10
    
    → Extracts: genemax=3, g9min=1, g9max=8, resolution=30, threshold=5
    → Uses all CPU cores for phenotype recalculation

The output file has columns:
    genotype phenotype_original complexity_original phenotype_mutated complexity_mutated mutations initial_quintile

Where:
    genotype = mutated genes
    phenotype_original = original phenotype (from input)
    complexity_original = original LZ complexity (from input)
    phenotype_mutated = RECALCULATED phenotype of mutated genotype
    complexity_mutated = RECALCULATED LZ complexity of mutated phenotype
    mutations = number of mutations applied (0, 1, 2, 3, ...)
    initial_quintile = quintile of the original genotype (1-5)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from genotype_to_raster import get_biomorphs_phenotype

# Import phenotype calculation functions
try:
    from genotype_to_raster import get_biomorphs_phenotype
    from GPproperties import LZ_complexity_array
    PHENOTYPE_CALC_AVAILABLE = True
except ImportError:
    PHENOTYPE_CALC_AVAILABLE = False
    print("WARNING: Could not import phenotype calculation modules")
    print("Make sure genotype_to_raster.py and GPproperties.py are available")

def calculate_phenotype_worker(args):
    """
    Worker function for parallel phenotype calculation.
    Must be at module level for multiprocessing pickling.
    Args: (genotype_tuple, resolution, threshold)
    Returns: (phenotype_string, complexity_value)
    """
    genotype_tuple, resolution, threshold = args
    try:
        from genotype_to_raster import get_biomorphs_phenotype
        from GPproperties import LZ_complexity_array
        
        phenotype = get_biomorphs_phenotype(genotype_tuple, resolution, threshold)
        complexity = LZ_complexity_array(phenotype)
        return phenotype, complexity
    except Exception as e:
        return f"ERROR: {str(e)}", -1

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

def parse_genotype_string(genotype_str):
    """Parse genotype string like '-2,1,0,1,2,1,0,1,5' to tuple."""
    return tuple(int(g) for g in genotype_str.split(','))

def extract_params_from_filename(filename):
    """Extract parameters from filename like: biom_100000_3_1_8_30_5_random.txt"""
    basename = filename.split('/')[-1]
    if basename.endswith('.txt'):
        basename = basename[:-4]
    
    if basename.startswith('biom_'):
        basename = basename[5:]
    
    for suffix in ['_random', '_0mut', '_1mut', '_2mut', '_quintile_mutations']:
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
    
    parts = basename.split('_')
    
    if len(parts) >= 6:
        try:
            n = int(parts[0])
            genemax = int(parts[1])
            g9min = int(parts[2])
            g9max = int(parts[3])
            resolution = int(parts[4])
            threshold = int(parts[5])
            return genemax, g9min, g9max, resolution, threshold
        except (ValueError, IndexError):
            return None
    
    return None

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_file = sys.argv[1]
    max_mutations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    min_mutations = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    output_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Calculate total mutation levels in the range
    n_mutation_levels = max_mutations - min_mutations + 1
    
    # Extract parameters from filename
    extracted_params = extract_params_from_filename(input_file)
    if extracted_params:
        genemax, g9min, g9max, resolution, threshold = extracted_params
        print(f"Parameters extracted from filename:")
        print(f"  genemax={genemax}, g9min={g9min}, g9max={g9max}")
        print(f"  resolution={resolution}, threshold={threshold}\n")
    else:
        genemax, g9min, g9max = 3, 1, 8
        print(f"Could not extract parameters from filename.")
        print(f"Using defaults: genemax={genemax}, g9min={g9min}, g9max={g9max}\n")
    
    if output_file is None:
        if input_file.endswith('.txt'):
            input_base = input_file[:-4]
        else:
            input_base = input_file
        output_file = f'{input_base}_quintile_mutations_mut{min_mutations}to{max_mutations}.txt'
    
    print("="*70)
    print("QUINTILE-BASED CUMULATIVE MUTATION ANALYSIS (PARALLEL)")
    print("="*70)
    print(f"\nInput file: {input_file}")
    print(f"Mutations per individual: {min_mutations} to {max_mutations}")
    print(f"Output file: {output_file}\n")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    try:
        df = pd.read_csv(input_file, sep=' ')
    except FileNotFoundError:
        print(f"ERROR: File '{input_file}' not found.")
        sys.exit(1)
    
    print(f"  ✓ Loaded {len(df)} genotypes\n")
    
    # Step 2: Divide into quintiles
    print("Step 2: Dividing by complexity quintiles...")
    df['quintile'] = pd.qcut(df['complexity'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    print(f"  Quintile sizes:")
    for q in sorted(df['quintile'].unique()):
        count = (df['quintile'] == q).sum()
        min_c = df[df['quintile'] == q]['complexity'].min()
        max_c = df[df['quintile'] == q]['complexity'].max()
        print(f"    Q{q}: {count:6d} individuals (complexity {min_c:.4f} - {max_c:.4f})")
    print()
    
    # Step 3: Apply cumulative mutations and recalculate phenotypes in parallel
    print(f"Step 3: Applying cumulative mutations ({min_mutations} to {max_mutations})...")
    
    if not PHENOTYPE_CALC_AVAILABLE:
        print("ERROR: Cannot recalculate phenotypes - phenotype calculation modules not available")
        print("Make sure genotype_to_raster.py and GPproperties.py are in the same directory")
        sys.exit(1)
    
    num_cores = cpu_count()
    print(f"  Using {num_cores} CPU cores for parallel phenotype calculation\n")
    
    all_results = []
    genotypes_to_calculate = []
    genotype_info = []
    
    # Pass 1: Generate mutations and prepare for parallel calculation
    print("  Pass 1: Generating mutations...")
    for idx, row in df.iterrows():
        genotype_str = row['genotype']
        phenotype_original = row['phenotype']
        complexity_original = row['complexity']
        quintile = int(row['quintile'])
        
        try:
            current_genotype = parse_genotype_string(genotype_str)
        except:
            print(f"ERROR parsing genotype at row {idx}: {genotype_str}")
            continue
        
        # Store original (0 mutations) only if min_mutations <= 0
        if min_mutations <= 0:
            all_results.append({
                'genotype': genotype_str,
                'phenotype_original': phenotype_original,
                'complexity_original': complexity_original,
                'phenotype_mutated': phenotype_original,
                'complexity_mutated': complexity_original,
                'mutations': 0,
                'initial_quintile': quintile
            })
        
        # Generate mutation variants
        temp_genotype = current_genotype
        for n_mut in range(1, max_mutations + 1):
            temp_genotype = mutate_genotype(temp_genotype, genemax, g9min, g9max)
            
            # Only record if at or above minimum threshold
            if n_mut >= min_mutations:
                mutated_genotype_str = ','.join([str(g) for g in temp_genotype])
                
                # Queue for parallel calculation
                genotypes_to_calculate.append((temp_genotype, resolution, threshold))
                genotype_info.append({
                    'genotype_str': mutated_genotype_str,
                    'phenotype_original': phenotype_original,
                    'complexity_original': complexity_original,
                    'mutations': n_mut,
                    'initial_quintile': quintile
                })
        
        if (idx + 1) % 10000 == 0:
            print(f"    Prepared {idx + 1}/{len(df)} individuals...")
    
    # Pass 2: Parallel phenotype calculation with memory-efficient streaming
    print(f"\n  Pass 2: Calculating phenotypes for {len(genotypes_to_calculate)} mutant genotypes in parallel...")
    
    # Use imap (not imap_unordered) to preserve order - critical for matching results!
    batch_size = max(100, num_cores * 4)
    
    with Pool(num_cores) as pool:
        phenotype_results_iter = pool.imap(
            calculate_phenotype_worker,
            genotypes_to_calculate,
            chunksize=batch_size
        )
        
        # Pass 3: Process results as they arrive (streaming to avoid memory buildup)
        print(f"  Pass 3: Processing results as they arrive...")
        for i, (phenotype_mutated, complexity_mutated) in enumerate(phenotype_results_iter):
            info = genotype_info[i]
            all_results.append({
                'genotype': info['genotype_str'],
                'phenotype_original': info['phenotype_original'],
                'complexity_original': info['complexity_original'],
                'phenotype_mutated': phenotype_mutated,
                'complexity_mutated': complexity_mutated,
                'mutations': info['mutations'],
                'initial_quintile': info['initial_quintile']
            })
            
            # Progress checkpoint every 50k mutations
            if (i + 1) % 50000 == 0:
                print(f"      Processed {i + 1}/{len(genotypes_to_calculate)} mutations...")
    
    print(f"  ✓ Generated {len(all_results)} total rows ({len(df)} individuals × {n_mutation_levels} mutation levels)\n")
    
    # Step 4: Create DataFrame and save
    print("Step 4: Saving results...")
    df_results = pd.DataFrame(all_results)
    
    output_path = Path(output_file)
    df_results.to_csv(output_path, index=False, sep=' ')
    print(f"  ✓ Results saved to: {output_path}\n")
    
    # Step 5: Summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total rows in output: {len(df_results)}")
    print(f"Original individuals: {len(df)}")
    print(f"Max mutations per individual: {max_mutations}")
    print(f"CPU cores used: {num_cores}")
    
    print(f"\nPhenotype changes:")
    changed = (df_results['phenotype_mutated'] != df_results['phenotype_original']).sum()
    unchanged = (df_results['phenotype_mutated'] == df_results['phenotype_original']).sum()
    print(f"  Total rows: {len(df_results)}")
    print(f"  Phenotypes CHANGED: {changed} ({changed/len(df_results)*100:.2f}%)")
    print(f"  Phenotypes UNCHANGED: {unchanged} ({unchanged/len(df_results)*100:.2f}%)")
    
    print(f"\n  Changes by mutation level:")
    for n_mut in sorted(df_results['mutations'].unique()):
        subset = df_results[df_results['mutations'] == n_mut]
        n_changed = (subset['phenotype_mutated'] != subset['phenotype_original']).sum()
        pct = n_changed / len(subset) * 100 if len(subset) > 0 else 0
        print(f"    {n_mut} mutations: {n_changed}/{len(subset)} changed ({pct:.2f}%)")
    
     
    
    print(f"\nQuintile distribution and phenotype change rates:")
    for q in sorted(df_results['initial_quintile'].unique()):
        subset = df_results[df_results['initial_quintile'] == q]
        n_indiv = len(subset) // n_mutation_levels  # ← USE n_mutation_levels
        n_changed = (subset['phenotype_mutated'] != subset['phenotype_original']).sum()
        pct_changed = n_changed / len(subset) * 100 if len(subset) > 0 else 0
        avg_complexity_change = (subset['complexity_mutated'] - subset['complexity_original']).mean()
        print(f"  Quintile {q}: {n_indiv:6d} individuals | {pct_changed:5.2f}% phenotypes changed | Avg ΔComplexity: {avg_complexity_change:+.4f}")
    
    
    print(f"\nFile size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    print(f"\nFirst individual's mutation trajectory (with phenotype changes):")
    first_rows = df_results[df_results['initial_quintile'] == df_results.iloc[0]['initial_quintile']].head(n_mutation_levels)
    for idx, (_, row) in enumerate(first_rows.iterrows()):
        pheno_same = "SAME" if row['phenotype_mutated'] == row['phenotype_original'] else "CHANGED"
        complexity_change = row['complexity_mutated'] - row['complexity_original']
        print(f"    {int(row['mutations'])} muts: Genotype: {row['genotype']}")
        print(f"              Phenotype: {pheno_same} | Complexity: {row['complexity_mutated']:.4f} (Δ {complexity_change:+.4f})")
    
    print(f"\nFirst 10 rows:")
    print(df_results.head(10).to_string(index=False))
    
    print("\n" + "="*70 + "\n")
    
    return df_results

if __name__ == "__main__":
    df = main()
