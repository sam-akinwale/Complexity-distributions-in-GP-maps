import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

# Direction mappings
direction_to_int = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}
int_to_direction = {i: d for d, i in direction_to_int.items()}
contact_energies = {(0, 0): -1, (0, 1): 0, (1, 0): 0, (1, 1): 0} # H is 0

def up_down_to_contact_map(structure_up_down_notation):
    current_point = (0, 0)
    structure_coordinate_notation = [(0, 0)]
    for d in structure_up_down_notation:
        current_point = (current_point[0] + int_to_direction[d][0], current_point[1] + int_to_direction[d][1]) 
        structure_coordinate_notation.append((current_point[0], current_point[1]))
    contact_map = []
    for i, coordi in enumerate(structure_coordinate_notation):
        for j, coordj in enumerate(structure_coordinate_notation):
            if i < j - 1.5 and abs(coordi[0] - coordj[0]) + abs(coordi[1] - coordj[1]) == 1:
               contact_map.append((i, j))
    return tuple(sorted(contact_map))

def contact_map_to_str(cm):
    return '__'.join([str(i) + '_' + str(j) for i, j in cm])

def free_energy(seq, contact_map):
    return sum([contact_energies[(seq[i], seq[j])] for i, j in contact_map])

def find_mfe(seq, contact_map_list):
    free_energy_list = [free_energy(seq, contact_map) for contact_map in contact_map_list]
    sorted_free_energy_list = sorted(free_energy_list)
    if sorted_free_energy_list[0] < sorted_free_energy_list[1] - 0.00001:
       return np.argmin(free_energy_list) + 1
    else:
        return 0

def HPget_Boltzmann_freq_list(seq, contact_map_list, kbT):
    free_energy_list = [free_energy(seq, contact_map) for contact_map in contact_map_list]
    exp_list = np.exp(-1.0 * np.array(free_energy_list)/kbT)
    Z = np.sum(exp_list)
    return [e/Z for e in exp_list]

def binary_to_hp_notation(seq):
    """Convert binary sequence (0=H, 1=P) to HP notation string"""
    return ''.join([{0: 'H', 1: 'P'}[i] for i in seq])

def plot_structure(structure_up_down_notation, ax):
    ax.axis('off')
    ax.axis('equal')
    current_point = (0, 0)
    ax.scatter([current_point[0], ], [current_point[1],], c='r')
    for d in structure_up_down_notation:
        new_pos = (current_point[0] + int_to_direction[d][0], current_point[1] + int_to_direction[d][1]) 
        ax.plot([current_point[0], new_pos[0]], [current_point[1], new_pos[1]], c='k')
        current_point = (new_pos[0], new_pos[1])
