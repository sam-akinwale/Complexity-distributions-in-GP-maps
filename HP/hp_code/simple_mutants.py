"""
Simplified HP protein folding analysis.

For a given sequence length:
1. Enumerate all unique structures
2. Generate single and double mutants from a random sequence
3. Generate random sequences and fold them

Usage:
    python simple_mutants.py <sequence_length> <num_mutants> <num_random>
    
Example:
    python simple_mutants.py 9 100 1000
    (generates lenght 9 genotypes, 100 single mutants, 100 double mutants, 1000 random sequences)
"""

import sys
import os
import random
import time
import numpy as np
from itertools import product
from copy import deepcopy
from HP_pfh import direction_to_int, int_to_direction, up_down_to_contact_map, find_mfe
from KC import calc_KC


# ============================================================================
# STRUCTURE ENUMERATION
# ============================================================================

index_to_new_index_90deg = {old_index: direction_to_int[(-old_dir[1], old_dir[0])] 
                            for old_dir, old_index in direction_to_int.items()}
mirror_image_x_axis = {0: 0, 1: 1, 2: 3, 3: 2}


def find_all_walks_given_start(startposition, n, L, walk_so_far):
    """Recursively find all valid walks on lattice."""
    if len(walk_so_far) == L:
        return [deepcopy(walk_so_far)]
    
    list_walks = []
    for next_move in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        new_pos = (startposition[0] + next_move[0], startposition[1] + next_move[1])
        if max(new_pos) <= n - 1 and min(new_pos) >= 0 and new_pos not in walk_so_far:
            list_walks += find_all_walks_given_start(new_pos, n, L, walk_so_far + [new_pos])
    return list_walks


def coordinates_to_updown(coords, L):
    """Convert coordinate path to up-down notation."""
    updown = [direction_to_int[(coords[i + 1][0] - coords[i][0], 
                                coords[i + 1][1] - coords[i][1])] 
              for i in range(L - 1)]
    # Normalize to start pointing right (direction 0)
    while updown[0] != 0:
        updown = [index_to_new_index_90deg[i] for i in updown]
    return tuple(updown)


def remove_mirror_images(structure_list):
    """Remove mirror-image duplicates."""
    mirror_images = [tuple([mirror_image_x_axis[i] for i in s]) for s in structure_list]
    result = []
    for i, s in enumerate(structure_list):
        if s not in mirror_images[:i]:
            result.append(s)
    return result


def enumerate_structures(n):
    """Enumerate all unique structures for n×n lattice."""
    L = n * n
    structures = []
    
    for startpos in product(np.arange(n), repeat=2):
        if sum(startpos) % 2 == 0 or n % 2 == 0:
            structures += find_all_walks_given_start(tuple(startpos), n, L, [startpos])
    
    # Convert to updown notation and remove duplicates/mirrors
    updown_structures = list(set([coordinates_to_updown(s, L) for s in structures]))
    unique_structures = remove_mirror_images(updown_structures)
    
    return unique_structures


# ============================================================================
# MUTANT GENERATION
# ============================================================================

def generate_all_single_mutants(sequence):
    """Generate all possible single-point mutations."""
    mutants = []
    for i in range(len(sequence)):
        mutant = list(sequence)
        mutant[i] = 1 - mutant[i]
        mutants.append(tuple(mutant))
    return mutants


def generate_all_double_mutants(sequence):
    """Generate all possible two-point mutations."""
    mutants = []
    seq_len = len(sequence)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mutant = list(sequence)
            mutant[i] = 1 - mutant[i]
            mutant[j] = 1 - mutant[j]
            mutants.append(tuple(mutant))
    return mutants


def sample_mutants(mutants, contact_map_list, num_to_find):
    """
    Sample mutants that fold into structures.
    
    Parameters:
    mutants: List of mutant sequences
    contact_map_list: List of all structure contact maps
    num_to_find: Number of valid mutants to collect
    
    Returns:
    List of up to num_to_find valid mutants with their folding info
    """
    valid_mutants = []
    seen_seqs = set()
    
    for mutant in mutants:
        if len(valid_mutants) >= num_to_find:
            break
        
        if mutant in seen_seqs:
            continue
        
        # Find which structure it folds into
        structure_id = find_mfe(mutant, contact_map_list)
        
        if structure_id > 0:  # Valid fold (not degenerate)
            valid_mutants.append({
                'sequence': mutant,
                'structure_id': structure_id
            })
            seen_seqs.add(mutant)
    
    return valid_mutants


def multi_seed_mutants(seq_length, contact_map_list, num_to_find, mutation_type='single', timeout_seconds=300):
    """
    Generate mutants from multiple random seeds until target or timeout.
    
    Parameters:
    seq_length: Length of sequences
    contact_map_list: List of all structure contact maps
    num_to_find: Number of valid mutants to collect
    mutation_type: 'single' or 'double'
    timeout_seconds: Maximum time to spend (default 5 minutes = 300s)
    
    Returns:
    List of valid mutants (up to num_to_find)
    """
    start_time = time.time()
    valid_mutants = []
    seen_seqs = set()
    seed_count = 0
    
    while len(valid_mutants) < num_to_find:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"    Timeout ({timeout_seconds}s) reached. Got {len(valid_mutants)}/{num_to_find}")
            break
        
        # Generate new random seed
        seed_seq = tuple(random.randint(0, 1) for _ in range(seq_length))
        seed_count += 1
        
        # Generate mutants from this seed
        if mutation_type == 'single':
            mutants = generate_all_single_mutants(seed_seq)
        else:  # double
            mutants = generate_all_double_mutants(seed_seq)
        
        # Test mutants
        for mutant in mutants:
            if len(valid_mutants) >= num_to_find:
                break
            
            if mutant in seen_seqs:
                continue
            
            structure_id = find_mfe(mutant, contact_map_list)
            
            if structure_id > 0:  # Valid fold
                valid_mutants.append({
                    'sequence': mutant,
                    'structure_id': structure_id
                })
                seen_seqs.add(mutant)
        
        # Progress update every 10 seeds or when target reached
        if seed_count % 10 == 0 or len(valid_mutants) >= num_to_find:
            elapsed = time.time() - start_time
            print(f"    Seeds tried: {seed_count}, Found: {len(valid_mutants)}/{num_to_find} ({elapsed:.1f}s)")
    
    return valid_mutants


# ============================================================================
# RANDOM SEQUENCE GENERATION
# ============================================================================

def generate_random_sequences(seq_length, num_sequences, contact_map_list, timeout_seconds=300):
    """
    Generate random binary sequences and fold them until target or timeout.
    
    Parameters:
    seq_length: Length of sequences
    num_sequences: Number of valid random sequences to collect
    contact_map_list: List of all structure contact maps
    timeout_seconds: Maximum time to spend (default 5 minutes = 300s)
    
    Returns:
    List of valid random sequences with their folding info (up to num_sequences)
    """
    start_time = time.time()
    random_seqs = []
    seen_seqs = set()
    attempts = 0
    
    while len(random_seqs) < num_sequences:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"    Timeout ({timeout_seconds}s) reached. Got {len(random_seqs)}/{num_sequences}")
            break
        
        # Generate random sequence
        seq = tuple(random.randint(0, 1) for _ in range(seq_length))
        attempts += 1
        
        if seq in seen_seqs:
            continue
        
        # Fold it
        structure_id = find_mfe(seq, contact_map_list)
        
        if structure_id > 0:  # Valid fold (not degenerate)
            random_seqs.append({
                'sequence': seq,
                'structure_id': structure_id
            })
            seen_seqs.add(seq)
        
        # Progress update every 1000 attempts or when target reached
        if attempts % 1000 == 0 or len(random_seqs) >= num_sequences:
            elapsed = time.time() - start_time
            print(f"    Attempts: {attempts}, Found: {len(random_seqs)}/{num_sequences} ({elapsed:.1f}s)")
    
    return random_seqs
# ============================================================================
# FILE I/O
# ============================================================================

def save_results(n, structures, single_mutants, double_mutants, random_seqs, output_prefix='results'):
    """Save all results to files."""
    
    # Save structures
    structures_file = f'{output_prefix}_structures_n{n}.txt'
    with open(structures_file, 'w') as f:
        f.write("ID\tUpDownNotation\tContactMapSize\tKC\n")
        for i, struct in enumerate(structures, 1):
            updown_str = ''.join(str(d) for d in struct)
            contact_map = up_down_to_contact_map(struct)
            kc = calc_KC(updown_str)
            f.write(f"{i}\t{updown_str}\t{len(contact_map)}\t{kc:.6f}\n")
    print(f"Saved {len(structures)} structures to {structures_file}")
    
    # Combine all sequences into one file
    combined_file = f'{output_prefix}_sequences_n{n}.txt'
    with open(combined_file, 'w') as f:
        f.write("Type\tBinarySequence\tStructureID\tUpDownNotation\tKC\n")
        
        # Write single mutants
        for mut in single_mutants:
            seq_str = ''.join(str(b) for b in mut['sequence'])
            struct = structures[mut['structure_id'] - 1]
            updown_str = ''.join(str(d) for d in struct)
            kc = calc_KC(updown_str)
            f.write(f"single\t{seq_str}\t{mut['structure_id']}\t{updown_str}\t{kc:.6f}\n")
        
        # Write double mutants
        for mut in double_mutants:
            seq_str = ''.join(str(b) for b in mut['sequence'])
            struct = structures[mut['structure_id'] - 1]
            updown_str = ''.join(str(d) for d in struct)
            kc = calc_KC(updown_str)
            f.write(f"double\t{seq_str}\t{mut['structure_id']}\t{updown_str}\t{kc:.6f}\n")
        
        # Write random sequences
        for rand in random_seqs:
            seq_str = ''.join(str(b) for b in rand['sequence'])
            struct = structures[rand['structure_id'] - 1]
            updown_str = ''.join(str(d) for d in struct)
            kc = calc_KC(updown_str)
            f.write(f"random\t{seq_str}\t{rand['structure_id']}\t{updown_str}\t{kc:.6f}\n")
    
    total_seqs = len(single_mutants) + len(double_mutants) + len(random_seqs)
    print(f"Saved {total_seqs} sequences ({len(single_mutants)} single, {len(double_mutants)} double, {len(random_seqs)} random) to {combined_file}")


# ============================================================================
# MAIN
# ============================================================================

def main(seq_length, num_mutants=100, num_random=1000):
    """
    Main analysis pipeline.
    
    Parameters:
    seq_length: Binary sequence length (determines lattice size n)
    num_mutants: Number of single mutants AND double mutants to find (same for both)
    num_random: Number of random sequences to generate
    """
    
    print(f"\n{'='*60}")
    print(f"HP PROTEIN FOLDING ANALYSIS")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_length}")
    print(f"Single mutants to find: {num_mutants}")
    print(f"Double mutants to find: {num_mutants}")
    print(f"Random sequences: {num_random}\n")
    
    # Calculate lattice size
    n = int(np.ceil(np.sqrt(seq_length)))
    if n < 2:
        n = 2
    print(f"Calculated lattice size: {n}×{n} (seq_length={seq_length})")
    
    # Step 1: Enumerate structures
    print(f"\nEnumerating structures...")
    start = time.time()
    structures = enumerate_structures(n)
    elapsed = time.time() - start
    print(f"Found {len(structures)} unique structures in {elapsed:.2f}s")
    
    # Prepare contact maps for folding
    contact_map_list = [up_down_to_contact_map(s) for s in structures]
    
    # Step 2: Generate mutants from multiple random seeds
    print(f"\nGenerating mutants from random seeds (5 minute timeout per type)...")
    
    # Single mutants
    print(f"  Single mutants...")
    single_mutants = multi_seed_mutants(seq_length, contact_map_list, num_mutants, 
                                        mutation_type='single', timeout_seconds=300)
    print(f"    Found {len(single_mutants)} valid single mutants")
    
    # Double mutants
    print(f"  Double mutants...")
    double_mutants = multi_seed_mutants(seq_length, contact_map_list, num_mutants, 
                                        mutation_type='double', timeout_seconds=300)
    print(f"    Found {len(double_mutants)} valid double mutants")
    
    # Step 3: Generate random sequences
    print(f"\nGenerating {num_random} random sequences (5 minute timeout)...")
    random_seqs = generate_random_sequences(seq_length, num_random, contact_map_list, timeout_seconds=300)
    print(f"  Found {len(random_seqs)} valid random sequences")
    
    # Step 4: Save results
    print(f"\nSaving results...")
    save_results(n, structures, single_mutants, double_mutants, random_seqs)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Example: python simple_mutants.py 9 100 1000\n")
        sys.exit(1)
    
    seq_length = int(sys.argv[1])
    num_mutants = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    num_random = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    main(seq_length, num_mutants, num_random)
