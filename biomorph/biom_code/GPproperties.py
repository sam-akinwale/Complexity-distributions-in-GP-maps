#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from multiprocessing import Pool
from functools import partial
from os import cpu_count
try:
   from KC import calc_KC
except ModuleNotFoundError:
   from .KC import calc_KC
try:
   from biomorph_functions import draw_biomorph_in_subplot
except ModuleNotFoundError:
   from .biomorph_functions import draw_biomorph_in_subplot
from copy import deepcopy
from scipy.stats import linregress

def decimal_to_scientific_notation(dec, n=1, b=np.nan):
   """ transform floating point number dec to a string in scientific notation
   n is the number of digits for rounding (n=1 means that 233 will be 2.3 * 10**2);
   adaptedfrom phD thesis code (stability chapter)"""
   if dec == 0:
      return '0'
   try:
      if np.isnan(b):
         b = int(np.floor(np.log10(abs(dec))))
      a = dec/float(10**b)
      if n == 0:
         first_part = str(int(round(a, n)))
      else:
         first_part = str(round(a, n))
      if b == 0:
         if isinstance(dec, int):
            return str(dec)
         return first_part
      else:
         return first_part+r'$\times 10^{{{0}}}$'.format(b)
   except OverflowError:
      return str(round(dec, n))

def df_data_to_dict(column_name, df):
   ph_list = df['phenotype int'].tolist()
   data_list = df[column_name].tolist()
   return {ph: data_list[i] for i, ph in enumerate(ph_list)}

def is_tree(ph):
   ph_array = expand_array(ph)
   nonzero_array_indices = [(y, x) for (y, x), a in np.ndenumerate(ph_array) if a > 0]
   if len(nonzero_array_indices) == 0:
      return False
   maxx, miny, maxy = max(list(zip(*nonzero_array_indices))[1]), min(list(zip(*nonzero_array_indices))[0]), max(list(zip(*nonzero_array_indices))[0])
   if maxx == 0 and maxy > miny:
      return True # straight line is a tree
   elif maxy == miny:
      return False
   ph_array_cropped = ph_array[miny:maxy+1, :maxx +1]
   assert ph_array_cropped.shape[0] == maxy - miny + 1 and ph_array_cropped.shape[1] == maxx + 1
   try:
      total_height = max([y for (y, x), a in np.ndenumerate(ph_array_cropped) if a > 0])
   except ValueError:
      total_height = 0
   ph_array_trunk_part = [a for (y, x), a in np.ndenumerate(ph_array_cropped) if y < 0.201 * total_height and x == 0]
   ph_array_empty_part = [a for (y, x), a in np.ndenumerate(ph_array_cropped) if y < 0.201 * total_height and 0 < x < 2]
   if len([a for a in ph_array_empty_part if a > 0]) == 0 and len([a for a in ph_array_trunk_part if a == 0]) == 0:
      return True
   else:
      return False

def plot_phenotype(ph, ax, phenotypes, g9min, color='k', linewidth=3):
   genemax = (phenotypes.shape[0]-1)//2
   g0 = get_startgeno_indices(phenotypes, ph)
   if len(g0) == 8:
      g0 = tuple([g for g in g0] + [0,])
      print('assuming g9 is constant')
   gene = index_to_gene(g0, genemax, g9min)
   draw_biomorph_in_subplot(gene, ax, color=color, linewidth=linewidth)


def get_startgeno_indices(phenotypes, ph):
   return tuple(np.argwhere(phenotypes == ph)[0])


def gene_to_index(gene, genemax, g9min):
   difference_index_gene = [genemax,]*8 + [-g9min]
   return tuple([g + difference_index_gene[i] for i, g in enumerate(gene)])

def index_to_gene(gene_as_index, genemax, g9min):
   difference_index_gene = [genemax,]*8 + [-g9min]
   return tuple([g - difference_index_gene[i] for i, g in enumerate(gene_as_index)])

def mirror_x(gene_not_index):
    sign_change_reflection=[-1, -1, -1, 1, 1, 1, 1, 1, 1]
    return [g*sign_change_reflection[i] for i, g in enumerate(gene_not_index)]
   
     
def neighbours_g_indices_nonperiodic_boundary(g, max_index_by_site):  
   return [tuple(n[:]) for site in range(len(g)) for n in neighbours_g_indices_nonperiodic_boundary_given_site(g, max_index_by_site, site) ] 

def neighbours_g_indices_nonperiodic_boundary_given_site(g, max_index_by_site, site):  
   new_indices = [g[site] -1, g[site] +1]
   return [[g[site_index] if site_index != site else new_site for site_index in range(len(g))] for new_site in  new_indices if new_site >= 0 and new_site <= max_index_by_site[site]]


def random_startgene(genemax, g9min, g9max):
   gene=np.random.randint(0, 2*genemax+1, size=8)
   if abs(g9max-g9min) > 0:
      order_recursion = np.random.randint(0, g9max-g9min+1, size=1)
   else:
      order_recursion = 0
   return [g for g in gene] + [int(order_recursion)]

####################################################################################
# neutral set size
####################################################################################

def get_ph_vs_f(phenotypes):
   print('calculate neutral set size')
   ph_vs_f = {}
   for genotype, ph in np.ndenumerate(phenotypes):
      try:
         ph_vs_f[ph] += 1
      except KeyError:
         ph_vs_f[ph] = 1
   return ph_vs_f

####################################################################################
# phi_pq_and_rho
####################################################################################
def rho_phi_pq_biomorphs(phenotypes, chosen_ph):
   print( 'mutational neighbourhood - rh o and phipq')
   L = phenotypes.ndim
   max_int_for_pos = {pos: phenotypes.shape[pos]-1 for pos in range(L)}
   pairs_phenotypes={} #counts the number of times each relevant pair of phenoypes appears in the sample
   for genotype, ph in np.ndenumerate(phenotypes):
      neighbours = neighbours_g_indices_nonperiodic_boundary(tuple(genotype), max_int_for_pos)
      for neighbourgeno in neighbours:
         neighbourpheno = phenotypes[neighbourgeno]
         pair=(min(ph, neighbourpheno), max(ph, neighbourpheno))
         if chosen_ph in pair or ph == neighbourpheno:
            try:
               pairs_phenotypes[pair] += 1.0/len(neighbours)
            except KeyError:
               pairs_phenotypes[pair] = 1.0/len(neighbours)
   neutralsetsize_dict = get_ph_vs_f(phenotypes)
   N_chosen_ph = neutralsetsize_dict[chosen_ph]
   rho, phi_pq = {}, {}
   for pheno in neutralsetsize_dict:
      if pheno != chosen_ph:
         pair = (min(chosen_ph, pheno), max(chosen_ph, pheno))
         try:
            phi_pq[pheno] = pairs_phenotypes[pair]/float(N_chosen_ph)
         except KeyError:
            phi_pq[pheno] = 0
      try:
        rho[pheno] = pairs_phenotypes[(pheno,pheno)]/float(neutralsetsize_dict[pheno])
      except KeyError:
        rho[pheno] = 0.0
   phi_pq[chosen_ph] = np.nan
   return rho, phi_pq

def nonzero_phi_pq_biomorphs(phenotypes, chosen_ph):
   V = [tuple(g[:]) for g in np.argwhere(phenotypes == chosen_ph)]
   max_int_for_pos = {pos: phenotypes.shape[pos] - 1 for pos in range(len(phenotypes.shape))}
   phenotypes_in_neighbourhood = set([])
   for genotype in V:
     neighbours = neighbours_g_indices_nonperiodic_boundary(tuple(genotype), max_int_for_pos)
     for neighbourgene in neighbours:
         phenotypes_in_neighbourhood.add(phenotypes[tuple(neighbourgene)])
   return phenotypes_in_neighbourhood

def phi_pq_list_initial_ph(phenotypes, ph_vs_f, list_initial_ph):
   print( 'mutational neighbourhood - rh o and phipq')
   L = phenotypes.ndim
   max_int_for_pos = {pos: phenotypes.shape[pos]-1 for pos in range(L)}
   ph_vs_phi_pq = {ph: {} for ph in list_initial_ph}
   ph_vs_in_list = np.zeros(np.amax(phenotypes) + 1, dtype='uint8')
   for ph in list_initial_ph:
      ph_vs_in_list[ph] = 1
   for genotype, ph in np.ndenumerate(phenotypes):
      if ph_vs_in_list[ph] > 0.5:
         neighbours = neighbours_g_indices_nonperiodic_boundary(tuple(genotype), max_int_for_pos)
         for neighbourgeno in neighbours:
            neighbourpheno = phenotypes[neighbourgeno]
            try:
               ph_vs_phi_pq[ph][neighbourpheno] += 1.0/(ph_vs_f[ph] * len(neighbours))
            except KeyError:
               ph_vs_phi_pq[ph][neighbourpheno] = 1.0/(ph_vs_f[ph] * len(neighbours))
   return ph_vs_phi_pq
#####################################################################################
# phenotype evolvability
####################################################################################

def get_Pevolvability(phenotypes):
   print('calculate evolvability')
   L = phenotypes.ndim
   max_int_for_pos = {pos: phenotypes.shape[pos]-1 for pos in range(L)}
   ph_list = set([p for p in phenotypes.copy().flat])
   Pneighbours, Pevolvability = {p: set([]) for p in ph_list}, {} 
   for genotype, ph in np.ndenumerate(phenotypes):
      neighbours = neighbours_g_indices_nonperiodic_boundary(tuple(genotype), max_int_for_pos)
      for neighbourgeno in neighbours:
         neighbourpheno = phenotypes[tuple(neighbourgeno)]
         if neighbourpheno != ph:
            Pneighbours[ph].add(neighbourpheno)
            Pneighbours[neighbourpheno].add(ph)
   for pheno in ph_list:
       Pevolvability[pheno]=len(Pneighbours[pheno])
   return Pevolvability
####################################################################################
# genotype robustness and evolvability as lists
####################################################################################
def get_Grobustness_Gevolvability(phenotypes):
   L = phenotypes.ndim
   max_int_for_pos = {pos: phenotypes.shape[pos]-1 for pos in range(L)}
   Grobustness, Gevolvability = np.zeros_like(phenotypes, dtype=float), np.zeros_like(phenotypes, dtype='uint8')
   for genotype, ph in np.ndenumerate(phenotypes):
      neighbours = neighbours_g_indices_nonperiodic_boundary(tuple(genotype), max_int_for_pos)
      neighbourphenos = [phenotypes[tuple(deepcopy(neighbourgeno))] for neighbourgeno in neighbours]
      Grobustness[genotype] = neighbourphenos.count(ph)/float(len(neighbourphenos))
      Gevolvability[genotype] = len([n for n in set(neighbourphenos) if n != ph])
   return Grobustness, Gevolvability


####################################################################################
# for array: LZ complexity
####################################################################################
def expand_and_transpose(array_concatenated):
   new_array = expand_array(array_concatenated)
   return ''.join([str(n) for n in np.ravel(new_array.T)])

def expand_array(array_concatenated):
   array = np.array([int(c) for c in array_concatenated])
   resolution = int(round(np.sqrt(len(array)*2))) #because half x
   return np.reshape(array, (resolution, resolution//2))

def tuple_to_binary_string(tuple_ph, resolution):
   if tuple_ph[1] > 0:
      bin_array = '0' * tuple_ph[0] + bin(tuple_ph[1])[2:]
   else:
      bin_array = '0' * tuple_ph[0]
   assert len(bin_array) == resolution**2 //2 
   return bin_array

def LZ_complexity_array(array_concatenated):
   array_concatenated_binary = ''.join([c for c in array_concatenated]) 
   array_transposed_concatenated_binary = expand_and_transpose(array_concatenated)
   return np.mean([calc_KC(deepcopy(array_transposed_concatenated_binary)), calc_KC(deepcopy(array_concatenated_binary))])
####################################################################################
# for plots
####################################################################################
def approximate_upper_bound_complexity_plot(c_list_fit, f_list_fit):
   H, xedges, yedges = np.histogram2d(c_list_fit, np.log2(f_list_fit), bins=100)
   x_bin_centres = 0.5 * (np.array(xedges[1:]) + np.array(xedges[:len(xedges)-1]))
   logy_bin_centres = 0.5 * (np.array(yedges[1:]) + np.array(yedges[:len(yedges)-1]))
   try:
      max_logy = [logy_bin_centres[max([yi for yi in range(len(logy_bin_centres)) if H[xi, yi] > 0.0001])] for xi in range(len(x_bin_centres))]
      slope, intercept, r, p, se = linregress(x_bin_centres, max_logy)
      print('found linear regression using 2D histogram')
      return slope, intercept
   except ValueError:
      slope = -1 *(max(np.log2(f_list_fit)) - min(np.log2(f_list_fit))) /(max(c_list_fit) - min(c_list_fit))
      intercept = max([np.log2(f) - c * slope for c, f in zip(c_list_fit, f_list_fit)])
      return slope, intercept
############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   ############################################################################################################
   g = (1 ,2, 0, 6)
   n_site_one = neighbours_g_indices_nonperiodic_boundary_given_site(g, max_index_by_site=(6, 6, 6, 6), site=1)
   assert len(n_site_one) == 2
   assert len([i for i in  range(4) if g[i] != n_site_one[0][i] and i != 1]) == 0
   n_site_two = neighbours_g_indices_nonperiodic_boundary_given_site(g, max_index_by_site=(6, 6, 6, 6), site=2)
   assert len(n_site_two) == 1
   assert len([i for i in  range(4) if g[i] != n_site_two[0][i] and i != 2]) == 0
   n_site_three = neighbours_g_indices_nonperiodic_boundary_given_site(g, max_index_by_site=(6, 6, 6, 6), site=3)
   assert len(n_site_three) == 1
   assert len([i for i in  range(4) if g[i] != n_site_three[0][i] and i != 3]) == 0
   assert len(neighbours_g_indices_nonperiodic_boundary(g, max_index_by_site=(6, 6, 6, 6))) == 6
   ############################################################################################################
   array_test = np.array([[1, 1], [0, 0], [1, 0], [1, 1]])
   array_flattened = ''.join([str(n) for n in np.ravel(array_test)])
   print('array', array_test)
   print('array flattened', array_flattened)
   print('array transposed and flattened', expand_and_transpose(array_flattened))
   ############################################################################################################




