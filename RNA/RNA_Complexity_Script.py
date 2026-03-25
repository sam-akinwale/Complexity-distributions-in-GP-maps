#########################################################################################
############################ Install and Import Librerys ################################
    
#pip install ViennaRNA


import itertools
import random
import RNA
from collections import defaultdict
import pandas as pd
import numpy as np
#pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt



#########################################################################################
################################# Complexity Function ###################################

def KC_LZ(string):
    n = len(string)
    s = '0' + string
    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    stop = 0

    while stop == 0:
        if s[i + k] != s[l + k]:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    stop = 1
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
        else:
            k += 1
            if l + k > n:
                c += 1
                stop = 1

    return c

def calc_KC(s):
    L = len(s)
    if s == '0' * L or s == '1' * L:
        return np.log2(L)
    else:
        return np.log2(L) * (KC_LZ(s) + KC_LZ(s[::-1])) / 2.0

def SS2bin(s):
    s = s.replace('.', '00')
    s = s.replace('(', '10')
    s = s.replace(')', '01')
    return s



#########################################################################################
################# Generate Billion random RNA sequence and their structures ###############

# Define parameters
nucleotides = ['A', 'C', 'G', 'U']
L_values = [30] #or 18
n_samples = 1_000_000_000


# Loop through each sequence length
for L in L_values:
    structure_data = defaultdict(lambda: {'count': 0, 'sequences': set(), 'complexity': 0.0})

    for _ in range(n_samples):
        genotype = ''.join(random.choices(nucleotides, k=L))
        structure, mfe = RNA.fold(genotype)
        structure_data[structure]['count'] += 1
        structure_data[structure]['sequences'].add(genotype)

        if structure_data[structure]['complexity'] == 0.0:
            structure_data[structure]['complexity'] = calc_KC(SS2bin(structure))

        if len(structure_data[structure]['sequences']) > 10:
            structure_data[structure]['sequences'].pop()

    # Calculate min and max complexity and normalize
    complexities = [data['complexity'] for data in structure_data.values()]
    min_complexity = min(complexities)
    max_complexity = max(complexities)
    N_O = len(structure_data)  # Number of unique structures
    N0 = 0.13 * 1.76**L  # Predefined value


#Scalling Comlexity in two ways for comparison
    for structure, data in structure_data.items():
        # Normalize complexity using N_O
        if max_complexity != min_complexity:
            normalized_complexity_NO = (
                math.log2(N_O) *
                (data['complexity'] - min_complexity) / (max_complexity - min_complexity)
            )
        else:
            normalized_complexity_NO = 0.0

        # Normalize complexity using N0
        if max_complexity != min_complexity:
            normalized_complexity_N0 = (
                math.log2(N0) *
                (data['complexity'] - min_complexity) / (max_complexity - min_complexity)
            )
        else:
            normalized_complexity_N0 = 0.0

        # Store both complexities in the structure data
        data['normalized_complexity_NO'] = normalized_complexity_NO
        data['normalized_complexity_N0'] = normalized_complexity_N0

    total_structures = sum(data['count'] for data in structure_data.values())

    # Write to the output file
    output_file = f"~PATH/L{L}_data_billion.txt"
    rep_seq_file = f"~PATH/L{L}_representative_sequences_billion.csv"

    with open(output_file, "w") as f:
        f.write("Structure\tFrequency\tProbability\tComplexity\tk_scalled_1\tk_scalled_2\tEntropy_contriburion\tRepresentative_Sequences\n")
        entropy = 0
        for structure, data in structure_data.items():
            frequency = data['count']
            probability = frequency / total_structures
            complexity = data['complexity']
            k_scalled_1 = data['normalized_complexity_NO']
            k_scalled_2 = data['normalized_complexity_N0']
            Entropy_contriburion = -probability * math.log2(probability)
            Entropy += Entropy_contriburion
            representative_sequences = ', '.join(list(data['sequences']))
            f.write(f"{structure}\t{frequency}\t{probability:.6f}\t{complexity:.4f}\t{k_scalled_1:.4f}\t{k_scalled_2:.4f}\t{Entropy_contriburion:.4f}\t{representative_sequences}\n")

        f.write(f"\nEntropy for L{L}: {Entropy:.4f}")

    with open(rep_seq_file, "w") as f:
        f.write("Representative Sequences\n")
        for data in structure_data.values():
            f.writelines(f"{seq}\n" for seq in data['sequences'])


#########################################################################################
############################# multi mutation functions ##################################

def mutate_sequence(sequence, exclude_positions):
    nucleotides = ['A', 'C', 'G', 'U']
    available_positions = [pos for pos in range(len(sequence)) if pos not in exclude_positions]
    random_pos = random.choice(available_positions)  # Ensure no overlap with excluded positions
    original_nucleotide = sequence[random_pos]
    new_nucleotide = random.choice([n for n in nucleotides if n != original_nucleotide])
    mutated_sequence = sequence[:random_pos] + new_nucleotide + sequence[random_pos + 1:]
    return mutated_sequence, random_pos

#Load representative sequencess to be used as input
rep_seq_file = "~PATH/representative_sequences.csv"
Original_RNA_rep_seq_df = pd.read_csv(rep_seq_file)


#create list to store data
combined_data = []


#go through all representative sequence
for _, row in Original_RNA_rep_seq_df.iterrows():
    original_seq = row['Representative Sequences']
    structure_original, mfe_original = RNA.fold(original_seq)
    complexity_original = calc_KC(SS2bin(structure_original))
    
    # First mutation
    mutated_seq1, pos1 = mutate_sequence(original_seq, exclude_positions=[])
    structure_mut1, mfe_mut1 = RNA.fold(mutated_seq1)
    complexity_mut1 = calc_KC(SS2bin(structure_mut1))
    
    # Second mutation
    mutated_seq2, pos2 = mutate_sequence(mutated_seq1, exclude_positions=[pos1])
    structure_mut2, mfe_mut2 = RNA.fold(mutated_seq2)
    complexity_mut2 = calc_KC(SS2bin(structure_mut2))
    
    # Third mutation
    mutated_seq3, pos3 = mutate_sequence(mutated_seq2, exclude_positions=[pos1, pos2])
    structure_mut3, mfe_mut3 = RNA.fold(mutated_seq3)
    complexity_mut3 = calc_KC(SS2bin(structure_mut3))
    
    # Fourth mutation
    mutated_seq4, pos4 = mutate_sequence(mutated_seq3, exclude_positions=[pos1, pos2, pos3])
    structure_mut4, mfe_mut4 = RNA.fold(mutated_seq4)
    complexity_mut4 = calc_KC(SS2bin(structure_mut4))
    
    # Append the data
    combined_data.append({
        'Original Sequence': original_seq,
        'Original Structure': structure_original,
        'Original Complexity': complexity_original,
        
        'First Mutated Sequence': mutated_seq1,
        'First Mutation Position': pos1,
        'First Mutated Structure': structure_mut1,
        'First Mutated Complexity': complexity_mut1,
        
        'Second Mutated Sequence': mutated_seq2,
        'Second Mutation Position': pos2,
        'Second Mutated Structure': structure_mut2,
        'Second Mutated Complexity': complexity_mut2,
        
        'Third Mutated Sequence': mutated_seq3,
        'Third Mutation Position': pos3,
        'Third Mutated Structure': structure_mut3,
        'Third Mutated Complexity': complexity_mut3,
        
        'Fourth Mutated Sequence': mutated_seq4,
        'Fourth Mutation Position': pos4,
        'Fourth Mutated Structure': structure_mut4,
        'Fourth Mutated Complexity': complexity_mut4
    })

# Create a DataFrame for the combined data
combined_data_df = pd.DataFrame(combined_data)

# Save to a single CSV file
output_file = "~PATH/L{L}_original_and_multi_mutated_structures.csv"

combined_data_df.to_csv(output_file, index=False)

print(f"Original and multi-mutated structure results (with 4 mutation) saved to: {output_file}")





















