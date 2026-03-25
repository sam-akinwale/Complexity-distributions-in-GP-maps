#!/usr/bin/env python3
"""
Generate random biomorphs and calculate their complexity.
Alternative version with flexible imports - works even if module structure isn't set up.

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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Try importing from functions_for_GP_analysis, fall back to local if needed
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

def calculate_phenotype_and_complexity(genotype, resolution, threshold, genemax, g9min):
    """
    Calculate phenotype and complexity for a single genotype.
    Returns: (genotype, genotype_str, phenotype_str, complexity)
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

def expand_array(array_concatenated):
    """Convert binary string to 2D array (matching GPproperties.py)."""
    array = np.array([int(c) for c in array_concatenated])
    resolution = int(round(np.sqrt(len(array)*2)))
    return np.reshape(array, (resolution, resolution//2))

def create_biomorph_visualization(genotype, phenotype_str, complexity, resolution, output_path):
    """
    Create a visualization of a biomorph with:
    - Left: the drawn biomorph structure
    - Middle: phenotype heatmap
    - Right: binary representation and info
    """
    try:
        from biomorph_functions import draw_biomorph_in_subplot
        
        fig = plt.figure(figsize=(14, 5))
        
        # Left: Draw biomorph
        ax1 = plt.subplot(1, 3, 1)
        draw_biomorph_in_subplot(genotype, ax1, linewidth=2)
        ax1.set_title('Biomorph Structure', fontsize=12, fontweight='bold')
        
        # Middle: Phenotype heatmap
        ax2 = plt.subplot(1, 3, 2)
        ph_array = expand_array(phenotype_str)
        ph_display = np.concatenate((ph_array[:, ::-1], ph_array), axis=1)
        ax2.imshow(ph_display, cmap='Greys', origin='lower')
        ax2.set_title('Phenotype (Binary)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Right: Info
        ax3 = plt.subplot(1, 3, 3)
        ax3.axis('off')
        
        # Create info text
        genotype_str = ', '.join([str(g) for g in genotype])
        info_text = f"""
GENOTYPE
{genotype_str}

COMPLEXITY (LZ)
{complexity:.4f}

PHENOTYPE LENGTH
{len(phenotype_str)} pixels
(resolution: {resolution}x{resolution//2})

PHENOTYPE BITS
{phenotype_str[:50]}{'...' if len(phenotype_str) > 50 else ''}
        """
        
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False

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
        output_file = f'biom_{n_biomorphs}_{genemax}_{g9min}_{g9max}_{resolution}_{threshold}.txt'
    
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
    
    # Step 3: Create DataFrame and save to TXT
    print("Step 3: Saving results to TXT...")
    df = pd.DataFrame(
        results,
        columns=['genotype_tuple', 'genotype', 'phenotype', 'complexity']
    )
    
    output_path = Path(output_file)
    # Save only the string columns (not the genotype tuple)
    df[['genotype', 'phenotype', 'complexity']].to_csv(output_path, index=False, sep=' ')
    print(f"  ✓ Results saved to: {output_path}\n")
    
    # Step 4: Create visualizations for interesting biomorphs
    print("Step 4: Generating visualizations for selected biomorphs...")
    visuals_dir = Path('biomorph_visuals')
    visuals_dir.mkdir(exist_ok=True)
    
    # Select interesting biomorphs
    df_sorted = df.sort_values('complexity').reset_index(drop=True)
    n_samples = min(12, len(df))
    
    # Min, max, quartiles, median, and some random
    sample_indices = [
        0,  # Minimum complexity
        len(df_sorted) // 4,  # Q1
        len(df_sorted) // 2,  # Median
        3 * len(df_sorted) // 4,  # Q3
        len(df_sorted) - 1,  # Maximum complexity
    ]
    
    # Add random samples to reach n_samples
    remaining = n_samples - len(sample_indices)
    if remaining > 0:
        random_indices = np.random.choice(len(df_sorted), size=remaining, replace=False)
        sample_indices.extend(random_indices)
    
    sample_indices = sorted(set(sample_indices))[:n_samples]
    
    visuals_created = 0
    for i, idx in enumerate(sample_indices):
        row = df_sorted.iloc[idx]
        genotype_tuple = row['genotype_tuple']
        phenotype_str = row['phenotype']
        complexity = row['complexity']
        
        output_png = visuals_dir / f'biomorph_{i:02d}_complexity_{complexity:.4f}.png'
        
        if create_biomorph_visualization(genotype_tuple, phenotype_str, complexity, resolution, output_png):
            visuals_created += 1
    
    print(f"  ✓ Created {visuals_created} visualizations in '{visuals_dir}' directory\n")
    
    # Print summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total biomorphs: {len(df)}")
    print(f"\nComplexity (LZ):")
    print(f"  Mean:    {df['complexity'].mean():.4f}")
    print(f"  Std dev: {df['complexity'].std():.4f}")
    print(f"  Min:     {df['complexity'].min():.4f}")
    print(f"  Max:     {df['complexity'].max():.4f}")
    print(f"  Median:  {df['complexity'].median():.4f}")
    
    print(f"\nFile size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Visualizations saved to: {visuals_dir.absolute()}")
    print(f"\nFirst 10 rows:")
    print(df[['genotype', 'phenotype', 'complexity']].head(10).to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    return df

if __name__ == "__main__":
    df = main()
