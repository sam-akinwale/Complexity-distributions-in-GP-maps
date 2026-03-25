import numpy as np
import networkx as nx
#from numba import jit
try:
   from GPproperties import neighbours_g_indices_nonperiodic_boundary
except ModuleNotFoundError:
   from .GPproperties import neighbours_g_indices_nonperiodic_boundary
from copy import deepcopy


def NC_rho_c_pq_biomorphs(phenotypes, startgene_ind):
   print( 'mutational neighbourhood for specific NC - rho and c_pq')
   max_int_for_pos = {pos: phenotypes.shape[pos] - 1 for pos in range(len(phenotypes.shape))}
   max_index_by_site = np.array([phenotypes.shape[pos] - 1 for pos in range(len(phenotypes.shape))])
   V = get_NC_genotypes(startgene_ind, phenotypes, max_int_for_pos)
   p_vs_cpq_list, normalisation = {}, 0
   for genotype in V:
     neighbours = neighbours_g_indices_nonperiodic_boundary(tuple(genotype), max_int_for_pos)
     assert len(neighbours) == 2 * len(genotype) - genotype.count(0) - list(max_index_by_site - genotype).count(0)
     normalisation += 1
     for neighbourgene in neighbours:       
         try:
            p_vs_cpq_list[phenotypes[tuple(neighbourgene)]] += 1/float(len(neighbours))
         except KeyError:
            p_vs_cpq_list[phenotypes[tuple(neighbourgene)]] = 1/float(len(neighbours))   
   cpq = {ph: count/float(normalisation) for ph, count in p_vs_cpq_list.items() if ph != phenotypes[startgene_ind]}
   rho = p_vs_cpq_list[phenotypes[startgene_ind]]/float(normalisation)
   return cpq, rho


def find_NCs_specific_pheno(pheno_int, GPmap, max_index_by_site):
   return find_NCs_specific_pheno_given_genotype_set(np.argwhere(GPmap==pheno_int), pheno_int, GPmap, max_index_by_site)

def find_NCs_specific_pheno_with_genotype_set_info(pheno_int, phenotype_vs_genotypelist, GPmap, max_index_by_site):
   return find_NCs_specific_pheno_given_genotype_set(phenotype_vs_genotypelist[pheno_int], pheno_int, GPmap, max_index_by_site)

def find_NCs_specific_pheno_given_genotype_set(genotype_set, pheno_int, GPmap, max_index_by_site):
   neutral_set = nx.Graph()
   neutral_set.add_nodes_from([tuple(g[:]) for g in genotype_set])
   for g in neutral_set.nodes():
      for g2 in neighbours_g_indices_nonperiodic_boundary(g, max_index_by_site):
         if GPmap[tuple(g2)] == pheno_int:
            neutral_set.add_edge(g, tuple(g2))
   geno_vs_NC = {tuple(deepcopy(g[:])): NCindex for NCindex, list_of_geno in enumerate(nx.connected_components(neutral_set)) for g in list_of_geno}
   del neutral_set
   print( 'finished NCs for', pheno_int)
   return geno_vs_NC 
   
def get_NC_genotypes(g0, GPmap, max_index_by_site):
   geno_vs_NC = find_NCs_specific_pheno(GPmap[g0], GPmap, max_index_by_site)
   index = geno_vs_NC[g0]
   return [g[:] for g,  NCindex in geno_vs_NC.items() if NCindex == index]
   

