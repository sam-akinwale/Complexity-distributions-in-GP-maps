import sys
import numpy as np
import pandas as pd
from KC import calc_KC
from HP_pfh import (up_down_to_contact_map, free_energy, 
                     direction_to_int, int_to_direction)

def generate_random_saw(L, n):
    """Generate a random self-avoiding walk of length L on an nxn lattice"""
    max_attempts = 10000
    
    for attempt in range(max_attempts):
        walk = [(np.random.randint(0, n), np.random.randint(0, n))]
        occupied = {walk[0]}
        
        for step in range(L - 1):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            np.random.shuffle(directions)
            
            found = False
            for dx, dy in directions:
                new_pos = (walk[-1][0] + dx, walk[-1][1] + dy)
                
                if (0 <= new_pos[0] < n and 0 <= new_pos[1] < n and 
                    new_pos not in occupied):
                    walk.append(new_pos)
                    occupied.add(new_pos)
                    found = True
                    break
            
            if not found:
                break
        
        if len(walk) == L:
            return tuple(walk)
    
    return tuple(walk)


def coordinates_to_up_down(coords):
    """Convert coordinate path to direction sequence"""
    if len(coords) < 2:
        return tuple()
    
    directions = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        directions.append(direction_to_int[(dx, dy)])
    
    structure_up_down = list(directions)
    index_to_new_index_90deg = {old_index: direction_to_int[(-old_dir[1], old_dir[0])] 
                                 for old_dir, old_index in direction_to_int.items()}
    
    while structure_up_down[0] != 0:
        structure_up_down = [index_to_new_index_90deg[i] for i in structure_up_down]
    
    return tuple(structure_up_down)


def up_down_to_coordinates(structure_up_down):
    """Convert direction sequence back to coordinates"""
    coords = [(0, 0)]
    current_pos = (0, 0)
    
    for d in structure_up_down:
        dx, dy = int_to_direction[d]
        current_pos = (current_pos[0] + dx, current_pos[1] + dy)
        coords.append(current_pos)
    
    return coords


def perturb_structure_end_move(structure_up_down, L, n):
    """Perturb structure by attempting to move the end of the chain"""
    coords = up_down_to_coordinates(structure_up_down)
    end_pos = coords[-1]
    
    directions_to_try = list(direction_to_int.values())
    np.random.shuffle(directions_to_try)
    
    for new_dir in directions_to_try:
        dx, dy = int_to_direction[new_dir]
        new_end_pos = (end_pos[0] + dx, end_pos[1] + dy)
        
        if (0 <= new_end_pos[0] < n and 0 <= new_end_pos[1] < n and
            new_end_pos not in set(coords[:-1])):
            
            new_structure = list(structure_up_down)
            new_structure[-1] = new_dir
            return tuple(new_structure)
    
    return None


def perturb_structure_pivot(structure_up_down, L, n, pivot_pos=None):
    """Perturb structure using pivot move"""
    if L < 3 or len(structure_up_down) < 2:
        return None
    
    if pivot_pos is None:
        pivot_pos = np.random.randint(1, L - 1)
    
    coords = up_down_to_coordinates(structure_up_down)
    pivot_coord = coords[pivot_pos]
    
    segment_to_rotate = coords[pivot_pos + 1:]
    rotation = np.random.choice([1, 2, 3])
    
    rotated_segment = []
    for coord in segment_to_rotate:
        dx = coord[0] - pivot_coord[0]
        dy = coord[1] - pivot_coord[1]
        
        for _ in range(rotation):
            dx, dy = -dy, dx
        
        new_coord = (pivot_coord[0] + dx, pivot_coord[1] + dy)
        rotated_segment.append(new_coord)
    
    rotated_set = set(rotated_segment)
    coords_before_pivot = set(coords[:pivot_pos + 1])
    
    if rotated_set & coords_before_pivot:
        return None
    
    for coord in rotated_segment:
        if not (0 <= coord[0] < n and 0 <= coord[1] < n):
            return None
    
    new_coords = coords[:pivot_pos + 1] + rotated_segment
    new_structure = coordinates_to_up_down(new_coords)
    
    return new_structure if len(new_structure) == len(structure_up_down) else None


def perturb_structure(structure_up_down, L, n):
    """Apply a random perturbation to the structure"""
    if np.random.rand() < 0.7:
        return perturb_structure_end_move(structure_up_down, L, n)
    else:
        return perturb_structure_pivot(structure_up_down, L, n)


def fold_sequence_mc(seq, L, n, num_moves=5000, T_init=2.0, cooling_rate=0.995):
    """Fold sequence using Monte Carlo simulated annealing"""
    if isinstance(seq, str):
        seq = tuple(int(c) for c in seq)
    
    current_structure = generate_random_saw(L, n)
    current_up_down = coordinates_to_up_down(current_structure)
    
    try:
        current_cm = up_down_to_contact_map(current_up_down)
        current_energy = free_energy(seq, current_cm)
    except:
        current_structure = generate_random_saw(L, n)
        current_up_down = coordinates_to_up_down(current_structure)
        current_cm = up_down_to_contact_map(current_up_down)
        current_energy = free_energy(seq, current_cm)
    
    best_structure = current_up_down
    best_energy = current_energy
    
    T = T_init
    
    for move in range(num_moves):
        new_up_down = perturb_structure(current_up_down, L, n)
        
        if new_up_down is None:
            continue
        
        try:
            new_cm = up_down_to_contact_map(new_up_down)
            new_energy = free_energy(seq, new_cm)
        except:
            continue
        
        dE = new_energy - current_energy
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            current_up_down = new_up_down
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_structure = current_up_down
                best_energy = current_energy
        
        T *= cooling_rate
    
    structure_str = ''.join(str(d) for d in best_structure)
    
    return structure_str, best_energy


def apply_single_mutation(binary_seq):
    """Apply exactly one random mutation to a binary sequence"""
    seq_list = list(binary_seq)
    seq_length = len(seq_list)
    
    mutation_position = np.random.randint(0, seq_length)
    seq_list[mutation_position] = '1' if seq_list[mutation_position] == '0' else '0'
    
    return ''.join(seq_list)


def apply_n_mutations(binary_seq, n_mutations):
    """Apply exactly n random mutations to a binary sequence (independent positions)"""
    seq_list = list(binary_seq)
    seq_length = len(seq_list)
    n_mutations = min(n_mutations, seq_length)
    
    mutation_positions = np.random.choice(seq_length, size=n_mutations, replace=False)
    
    for pos in mutation_positions:
        seq_list[pos] = '1' if seq_list[pos] == '0' else '0'
    
    return ''.join(seq_list)


def main():
    if len(sys.argv) < 2:
        print("Usage: python fold_quin_mutate.py <input_file> [output_file] [num_mc_moves]")
        print("Example: python fold_quin_mutate.py random_sample_mc_n8.txt mutant_analysis.txt 5000")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "mutant_analysis.txt"
    num_mc_moves = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    
    print(f"Reading sequences from {input_file}...")
    df = pd.read_csv(input_file, sep='\t', dtype={'Binary': str, 'UpDownNotation': str})
    
    print(f"Loaded {len(df)} sequences")
    seq_length = len(df.iloc[0]['Binary'])
    print(f"Sequence length: {seq_length}")
    print(f"KC complexity range: {df['KC'].min():.4f} to {df['KC'].max():.4f}")
    print(f"MC moves per mutant: {num_mc_moves}")
    
    n = seq_length - 1
    if n < 2:
        n = 2
    print(f"Lattice size: {n}x{n}")
    
    # Divide into quintiles by KC complexity
    df['Quintile'] = pd.qcut(df['KC'], q=5, duplicates='drop')
    
    # Assign numeric labels based on actual number of bins created
    quintile_mapping = {q: i for i, q in enumerate(sorted(df['Quintile'].unique()), 1)}
    df['Quintile'] = df['Quintile'].map(quintile_mapping)
    
    print(f"\nQuintile distribution:")
    print(df['Quintile'].value_counts().sort_index())
    
    quintile_stats = df.groupby('Quintile')['KC'].agg(['min', 'max', 'mean', 'count'])
    print("\nQuintile statistics:")
    print(quintile_stats)
    
    print(f"\nSampling 10 sequences per quintile based on quintile position...")
    sampled_dfs = []
    
    quintiles_list = sorted(df['Quintile'].unique())
    
    for quintile in quintiles_list:
        quintile_data = df[df['Quintile'] == quintile].sort_values('KC').reset_index(drop=True)
        n_total = len(quintile_data)
        
        if quintile == quintiles_list[0]:  # Q1 (least complex quintile) -> take 10 LEAST complex
            sampled = quintile_data.head(10).copy()
            print(f"  Q{quintile}: 10 LEAST complex sequences (KC {sampled['KC'].min():.4f}-{sampled['KC'].max():.4f})")
            
        elif quintile == quintiles_list[-1]:  # Q5 (most complex quintile) -> take 10 MOST complex
            sampled = quintile_data.tail(10).copy()
            print(f"  Q{quintile}: 10 MOST complex sequences (KC {sampled['KC'].min():.4f}-{sampled['KC'].max():.4f})")
            
        else:  # Q2, Q3, Q4 -> take 10 MIDDLE complex
            middle_start = max(10, n_total // 2 - 5)
            middle_end = middle_start + 10
            sampled = quintile_data.iloc[middle_start:middle_end].copy()
            print(f"  Q{quintile}: 10 MIDDLE complex sequences (KC {sampled['KC'].min():.4f}-{sampled['KC'].max():.4f})")
        
        sampled_dfs.append(sampled)
    
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Total sequences to analyze: {len(sampled_df)}")
    
    print(f"\nGenerating mutations (1-10 per sequence, {num_mc_moves} MC moves per mutant)...")
    
    results = []
    total_sequences = len(sampled_df)
    sequence_counter = 0
    
    for idx, row in sampled_df.iterrows():
        binary_seq = row['Binary']
        original_quintile = row['Quintile']
        original_kc = row['KC']
        
        for mutation_step in range(1, 11):
            mutant_binary = apply_n_mutations(binary_seq, mutation_step)
            
            mutant_up_down, mfe = fold_sequence_mc(mutant_binary, seq_length, n, 
                                                    num_moves=num_mc_moves)
            
            mutant_kc = calc_KC(mutant_up_down)
            
            results.append({
                'Original_Quintile': original_quintile,
                'Original_KC': original_kc,
                'Mutation_Step': mutation_step,
                'Num_Mutations': mutation_step,
                'Mutant_Binary': mutant_binary,
                'Mutant_UpDown': mutant_up_down,
                'Mutant_KC': mutant_kc,
                'MFE': mfe
            })
        
        sequence_counter += 1
        if sequence_counter % max(1, total_sequences // 10) == 0:
            print(f"  Processed {sequence_counter}/{total_sequences} sequences")
    
    results_df = pd.DataFrame(results)
    
    print(f"\nGenerated {len(results_df)} mutant records ({len(sampled_df)} sequences × 10 mutation steps)")
    print(f"Saving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("Original_Quintile\tOriginal_KC\tMutation_Step\tNum_Mutations\tMutant_Binary\tMutant_UpDown\tMutant_KC\tMFE\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['Original_Quintile']}\t{row['Original_KC']:.6f}\t{row['Mutation_Step']}\t{row['Num_Mutations']}\t{row['Mutant_Binary']}\t{row['Mutant_UpDown']}\t{row['Mutant_KC']:.6f}\t{row['MFE']}\n")
    
    print("\n" + "="*90)
    print("MUTATION ANALYSIS SUMMARY")
    print("(Independent mutations: N random positions mutated per step)")
    print("="*90)
    
    summary = results_df.groupby(['Original_Quintile', 'Num_Mutations'])['Mutant_KC'].agg(['mean', 'std', 'min', 'max', 'count'])
    print("\nMean KC complexity by quintile and number of mutations:")
    print(summary)
    
    print("\n" + "="*90)
    print("COMPLEXITY TRAJECTORY BY QUINTILE")
    print("="*90)
    
    for quintile in sorted(results_df['Original_Quintile'].unique()):
        quintile_data = results_df[results_df['Original_Quintile'] == quintile]
        original_kc = sampled_df[sampled_df['Quintile'] == quintile]['KC'].mean()
        
        print(f"\nQuintile {quintile} (mean original KC: {original_kc:.4f}):")
        print(f"  {'Mutations':<12} {'Mean KC':<12} {'Δ':<12} {'% Change':<12} {'Mean MFE':<12}")
        print(f"  {'-'*63}")
        
        for num_mut in range(1, 11):
            mut_data = quintile_data[quintile_data['Num_Mutations'] == num_mut]
            if len(mut_data) > 0:
                mean_kc = mut_data['Mutant_KC'].mean()
                mean_mfe = mut_data['MFE'].mean()
                delta = mean_kc - original_kc
                pct_change = 100 * delta / original_kc if original_kc != 0 else 0
                print(f"  {num_mut:<12} {mean_kc:<12.4f} {delta:<+12.4f} {pct_change:<+12.2f}% {mean_mfe:<12.4f}")
    
    print(f"\n{'='*90}")
    print(f"Results saved to {output_file}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
