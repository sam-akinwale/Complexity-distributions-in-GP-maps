import sys
import numpy as np
from copy import deepcopy
from HP_pfh import (up_down_to_contact_map, free_energy, binary_to_hp_notation, 
                     direction_to_int, int_to_direction)
from KC import calc_KC

def generate_random_saw(L, n):
    """Generate a random self-avoiding walk of length L on an nxn lattice
    
    Returns:
        tuple of coordinates representing the walk
    """
    max_attempts = 10000
    
    for attempt in range(max_attempts):
        walk = [(np.random.randint(0, n), np.random.randint(0, n))]
        occupied = {walk[0]}
        
        for step in range(L - 1):
            # Try random directions until finding an unoccupied neighbor
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            np.random.shuffle(directions)
            
            found = False
            for dx, dy in directions:
                new_pos = (walk[-1][0] + dx, walk[-1][1] + dy)
                
                # Check bounds and not occupied
                if (0 <= new_pos[0] < n and 0 <= new_pos[1] < n and 
                    new_pos not in occupied):
                    walk.append(new_pos)
                    occupied.add(new_pos)
                    found = True
                    break
            
            if not found:
                break  # Dead end, restart
        
        if len(walk) == L:
            return tuple(walk)
    
    # Fallback: return what we have
    return tuple(walk)


def coordinates_to_up_down(coords):
    """Convert coordinate path to direction sequence (up_down_notation)"""
    if len(coords) < 2:
        return tuple()
    
    # Get direction vectors
    directions = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        directions.append(direction_to_int[(dx, dy)])
    
    # Rotate so first direction is 0 (right)
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
    """Perturb structure by attempting to move the end of the chain
    
    This is the simplest move: try to add a new direction to the chain end
    """
    coords = up_down_to_coordinates(structure_up_down)
    end_pos = coords[-1]
    
    # Try random directions
    directions_to_try = list(direction_to_int.values())
    np.random.shuffle(directions_to_try)
    
    for new_dir in directions_to_try:
        dx, dy = int_to_direction[new_dir]
        new_end_pos = (end_pos[0] + dx, end_pos[1] + dy)
        
        # Check if valid (in bounds and not occupied by another residue)
        if (0 <= new_end_pos[0] < n and 0 <= new_end_pos[1] < n and
            new_end_pos not in set(coords[:-1])):  # Check not occupied except current end
            
            new_structure = list(structure_up_down)
            new_structure[-1] = new_dir
            return tuple(new_structure)
    
    return None


def perturb_structure_pivot(structure_up_down, L, n, pivot_pos=None):
    """Perturb structure using pivot move
    
    Select a residue and rotate the chain segment after it around that residue
    """
    if L < 3 or len(structure_up_down) < 2:
        return None
    
    if pivot_pos is None:
        pivot_pos = np.random.randint(1, L - 1)
    
    coords = up_down_to_coordinates(structure_up_down)
    pivot_coord = coords[pivot_pos]
    
    # Try rotating the chain segment after pivot by 90 degrees around pivot
    segment_to_rotate = coords[pivot_pos + 1:]
    
    # Random rotation: 90, 180, or 270 degrees
    rotation = np.random.choice([1, 2, 3])  # 1=90, 2=180, 3=270
    
    rotated_segment = []
    for coord in segment_to_rotate:
        dx = coord[0] - pivot_coord[0]
        dy = coord[1] - pivot_coord[1]
        
        for _ in range(rotation):
            dx, dy = -dy, dx  # 90 degree rotation
        
        new_coord = (pivot_coord[0] + dx, pivot_coord[1] + dy)
        rotated_segment.append(new_coord)
    
    # Check if rotated segment is valid (in bounds and no self-intersections)
    rotated_set = set(rotated_segment)
    coords_before_pivot = set(coords[:pivot_pos + 1])
    
    if rotated_set & coords_before_pivot:  # Intersection check
        return None
    
    # Check bounds
    for coord in rotated_segment:
        if not (0 <= coord[0] < n and 0 <= coord[1] < n):
            return None
    
    new_coords = coords[:pivot_pos + 1] + rotated_segment
    new_structure = coordinates_to_up_down(new_coords)
    
    return new_structure if len(new_structure) == len(structure_up_down) else None


def perturb_structure(structure_up_down, L, n):
    """Apply a random perturbation to the structure"""
    if np.random.rand() < 0.7:
        # 70% end moves (simpler, faster)
        return perturb_structure_end_move(structure_up_down, L, n)
    else:
        # 30% pivot moves (more dramatic changes)
        return perturb_structure_pivot(structure_up_down, L, n)


def fold_sequence_mc(seq, L, n, num_moves=5000, T_init=2.0, cooling_rate=0.995):
    """Fold sequence using Monte Carlo simulated annealing
    
    Args:
        seq: Binary sequence (tuple of 0s and 1s)
        L: Sequence length
        n: Lattice size (nxn)
        num_moves: Number of MC moves to attempt
        T_init: Initial temperature
        cooling_rate: Temperature decay factor (< 1.0)
    
    Returns:
        (best_structure, num_contacts)
    """
    # Generate initial random structure
    current_structure = generate_random_saw(L, n)
    current_up_down = coordinates_to_up_down(current_structure)
    
    try:
        current_cm = up_down_to_contact_map(current_up_down)
        current_energy = free_energy(seq, current_cm)
    except:
        # If initial structure fails, try again
        current_structure = generate_random_saw(L, n)
        current_up_down = coordinates_to_up_down(current_structure)
        current_cm = up_down_to_contact_map(current_up_down)
        current_energy = free_energy(seq, current_cm)
    
    best_structure = current_up_down
    best_energy = current_energy
    
    T = T_init
    accepted_moves = 0
    
    for move in range(num_moves):
        # Attempt perturbation
        new_up_down = perturb_structure(current_up_down, L, n)
        
        if new_up_down is None:
            continue
        
        # Evaluate new structure
        try:
            new_cm = up_down_to_contact_map(new_up_down)
            new_energy = free_energy(seq, new_cm)
        except:
            continue
        
        # Metropolis criterion
        dE = new_energy - current_energy
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            current_structure = new_up_down
            current_energy = new_energy
            accepted_moves += 1
            
            if current_energy < best_energy:
                best_structure = current_up_down
                best_energy = current_energy
        
        # Cool down
        T *= cooling_rate
    
    # Get final contact map for best structure
    best_cm = up_down_to_contact_map(best_structure)
    
    return best_structure, len(best_cm)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fold_random_mc.py <sequence_length> [num_samples] [num_mc_moves]")
        print("Example: python fold_random_mc.py 5 5000 5000")
        sys.exit(1)
    
    seq_length = int(sys.argv[1])
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    num_mc_moves = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    
    # Determine lattice size
    n = seq_length-1#int(np.ceil(np.sqrt(seq_length))) + 1
    if n < 2:
        n = 2
    
    print(f"Sequence length: {seq_length}")
    print(f"Lattice size: {n}x{n}")
    print(f"Monte Carlo moves per sequence: {num_mc_moves}")
    
    # Output file
    output_file = f'random_sample_mc_n{seq_length}_set4.txt'
    
    print(f"\nGenerating {num_samples} random sequences and folding...")
    
    grid_size = f"{n}x{n}"
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("Binary\tUpDownNotation\tn\tGridSize\tKC\n")
        
        for i in range(num_samples):
            # Generate random sequence
            seq = tuple(np.random.randint(0, 2, seq_length))
            seq_binary = ''.join(str(bit) for bit in seq)
            
            # Fold sequence using MC
            mfe_fold, contact_size = fold_sequence_mc(seq, seq_length, n, 
                                                       num_moves=num_mc_moves)
            updown_notation = ''.join(str(d) for d in mfe_fold)
            
            # Calculate KC complexity
            kc = calc_KC(updown_notation)
            
            # Write to file
            f.write(f"{seq_binary}\t{updown_notation}\t{seq_length}\t{grid_size}\t{kc:.6f}\n")
            
            if (i + 1) % max(1, num_samples // 10) == 0:
                print(f"  Completed {i + 1}/{num_samples}")
    
    print(f"\nSaved {num_samples} sequences to {output_file}")
