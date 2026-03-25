import numpy as np
from copy import deepcopy
from fractions import Fraction
import math
import networkx as nx
try:
   from biomorph_functions import find_line_coordinates
except ModuleNotFoundError:
   from .biomorph_functions import find_line_coordinates  

def array_to_1Dstring(coarse_grained_array):
   return ''.join([str(n) for n in np.ravel(coarse_grained_array)])


def get_biomorphs_phenotype(gene, resolution, threshold = 5, test=False):
   assert resolution%2 == 0 #only works if there is a pixel boundary on the central axis
   lines = find_line_coordinates(gene)
   edited_lines = merge_overlapping_and_parallel_lines(cut_out_positive_x(delete_zero_lines(lines)))
   if len(edited_lines):
       raster, raster_cg = find_phenotype(edited_lines, resolution, threshold) 
   else:
      raster, raster_cg = np.zeros((resolution, resolution//2), dtype='int8'), np.zeros((resolution, resolution//2), dtype='int8')
   if test:
      return raster, raster_cg
   else:
      return array_to_1Dstring(raster_cg)
 
def cut_out_positive_x(lines):
  lines_positive_half = []
  for l in lines:
     if min(l[:2]) >= 0:
        lines_positive_half.append(l)
     elif max(l[:2]) <= 0:
        pass
     elif l[0] < 0 and l[1] > 0:
        lines_positive_half.append([0, l[1], extrapolate_line_given_x(l, 0), l[3]])
     elif l[0] > 0 and l[1] < 0:
        lines_positive_half.append([l[0], 0, l[2], extrapolate_line_given_x(l, 0)])
  return lines_positive_half

def convert_to_fraction(z, zmin, zmax):
  assert zmin <= z <= zmax
  return Fraction(z-zmin, zmax-zmin) 

def extrapolate_line_given_x(line, x):
  if line[1] != line[0]: #not vertical
     gradient = Fraction(line[3]-line[2], line[1]-line[0])
     return (x-line[0]) * gradient + line[2]
  elif line[1] == line[0]:
     raise RuntimeError('cannot extrapolate a line with deltax = 0')

def extrapolate_line_given_y(line, y):
  if line[3] != line[2]: #not vertical
     gradient_inv = Fraction(line[1]-line[0], line[3]-line[2])
     return (y-line[2])*gradient_inv + line[0]
  elif line[3] == line[2]:
     raise RuntimeError('cannot extrapolate a line with deltay = 0')

def cut_x_window_from_line(line, xmin, xmax):
  assert line[0] <= line[1]
  if min(line[:2]) <= xmax and  max(line[:2]) >= xmin:
     x_components = [max(line[0], xmin), min(line[1], xmax)]
     if abs(line[1] - line[0]) > 0: #not vertical
        y_components = [extrapolate_line_given_x(line, x_components[0]), extrapolate_line_given_x(line, x_components[1])]
        on_boundary = False
     else:
        y_components = [line[2], line[3]]
        if min(line[:2]) == xmax or  max(line[:2]) == xmin:
          on_boundary = True
        else:
          on_boundary = False
     return x_components + y_components, on_boundary
  else:
    return [], False
  

def cut_y_window_from_line(line, ymin, ymax):
  if min(line[2:]) <= ymax and  max(line[2:]) >= ymin:
    y_components = [max(min(line[2:]), ymin), min(max(line[2:]), ymax)]
    if abs(line[3] - line[2]) > 0: #not horizontal
       x_components = [extrapolate_line_given_y(line, y_components[0]), extrapolate_line_given_y(line, y_components[1])]
       on_boundary = False
    else:
       x_components = [line[0], line[1]]
       if min(line[2:]) == ymax or  max(line[2:]) == ymin:
          on_boundary = True
       else:
          on_boundary = False
    return x_components + y_components, on_boundary
  else:
    return [], False

def integer_sqrt(x):
  assert x >= 0
  if x < 1:
    return 0
  first_guess = int(math.floor(math.sqrt(x))) - 1
  assert first_guess ** 2 < x
  while first_guess ** 2 <= x and (first_guess + 1)**2 <= x:
    first_guess += 1
  assert first_guess ** 2 <= x and (first_guess + 1)**2 > x
  return first_guess

def line_length_integersqrt(line):
  return Fraction(integer_sqrt(int(round(10**6*((line[1] - line[0])*(line[1] - line[0]) + (line[3] - line[2])*(line[3] - line[2]))))), 10**3)



def get_squared_length_of_line_per_pixel(lines_list, resolution):
  length_per_pixel_array = np.zeros((resolution, resolution//2))
  dict_with_exact_fraction = {index: Fraction(0, 1) for index, v in np.ndenumerate(length_per_pixel_array)}
  for line_as_frac in lines_list:
     min_x_raster_index, max_x_raster_index = int(line_as_frac[0]*resolution//2), int(line_as_frac[1]*resolution//2)
     assert min_x_raster_index <= max_x_raster_index and min_x_raster_index <= line_as_frac[0]*resolution//2 and max_x_raster_index + 1 >= line_as_frac[1]*resolution//2
     for x_raster_index in range(min_x_raster_index - 1, max_x_raster_index+2):
        line_in_x_window, on_boundary_x = cut_x_window_from_line(line_as_frac, Fraction(x_raster_index, resolution//2), Fraction(x_raster_index+1, resolution//2))
        if len(line_in_x_window):
          y_span_in_x_window = sorted(line_in_x_window[2:])
          min_y_raster_index, max_y_raster_index = int(y_span_in_x_window[0]*resolution),int(y_span_in_x_window[1]*resolution)
          assert min_y_raster_index <= max_y_raster_index and min_y_raster_index <= y_span_in_x_window[0]*resolution and max_y_raster_index + 1 >= y_span_in_x_window[1]*resolution
          for y_raster_index in range(min_y_raster_index -1, max_y_raster_index+2):
             if y_raster_index < resolution and x_raster_index < resolution//2 and min(x_raster_index, y_raster_index) >= 0:
                line_segment_in_raster, on_boundary_y = cut_y_window_from_line(line_in_x_window, Fraction(y_raster_index, resolution), Fraction(y_raster_index+1, resolution))
                if len(line_segment_in_raster):# and min(line_segment_in_raster[:2]) < Fraction(x_raster_index+1, resolution//2) and min(line_segment_in_raster[2:]) < Fraction(y_raster_index+1, resolution): #lines on boundary count for lower boundary
                   v, l = deepcopy(dict_with_exact_fraction[y_raster_index, x_raster_index]), line_length_integersqrt(line_segment_in_raster)
                   if on_boundary_y or on_boundary_x:
                      l = Fraction(deepcopy(l), 2)
                   dict_with_exact_fraction[y_raster_index, x_raster_index] = deepcopy(deepcopy(Fraction(v)) + Fraction(l))
                   assert dict_with_exact_fraction[y_raster_index, x_raster_index] >= 0
  for index, v in np.ndenumerate(length_per_pixel_array):
     length_per_pixel_array[index] = float(dict_with_exact_fraction[index])
     assert dict_with_exact_fraction[index] >= 0
  return length_per_pixel_array, dict_with_exact_fraction

def get_limits_figure(lines):
  x_vals = [l[0] for l in lines] + [l[1] for l in lines]
  y_vals = [l[2] for l in lines] + [l[3] for l in lines]
  return min(x_vals), max(x_vals), min(y_vals), max(y_vals)


def prepare_lines(lines, xmin, xmax, ymin, ymax):
  edited_lines = []
  for x1, x2, y1, y2 in lines:
     x1_frac, x2_frac, y1_frac, y2_frac = [convert_to_fraction(x, xmin, xmax) for x in (x1, x2)] + [convert_to_fraction(y, ymin, ymax) for y in (y1, y2)]
     if x1_frac <= x2_frac:
        edited_lines.append([x1_frac, x2_frac, y1_frac, y2_frac])
     else:
        edited_lines.append([x2_frac, x1_frac, y2_frac, y1_frac])
  return edited_lines
  
  
def coarse_grain_array_binary(dict_with_exact_fraction, resolution, threshold=5):
  cg_array = np.zeros((resolution, resolution//2), dtype=np.uint8)
  for index, fraction in dict_with_exact_fraction.items():
     if fraction > Fraction(1, threshold*resolution):
        cg_array[index] = 1
     else:
        cg_array[index] = 0
  return cg_array

def find_phenotype(lines, resolution, threshold):
  # get xmin, xmax, ymin, ymax of grid (not figure)
   xminfig, xmaxfig, yminfig, ymaxfig = get_limits_figure(lines)
   delta_y, delta_x = ymaxfig - yminfig, xmaxfig - xminfig
   assert xminfig == 0 #negative part has been cut off 
   delta = max(delta_y, 2*delta_x) * Fraction(105, 100)
   ymin = Fraction(yminfig + ymaxfig - delta, 2)
   ymax =  ymin + delta
   xmax, xmin =  Fraction(delta, 2), xminfig
   assert xmax + 0.01 > abs(xmin)
   assert xmin - 0.01 < xminfig and ymin - 0.01 < yminfig
   assert xmax + 0.01 > xmaxfig and ymax + 0.01 > ymaxfig
   length_per_pixel, dict_with_exact_fraction = get_squared_length_of_line_per_pixel(prepare_lines(lines, xmin, xmax, ymin, ymax), resolution)
   cg_array = coarse_grain_array_binary(dict_with_exact_fraction, resolution, threshold)
   return length_per_pixel, cg_array


def parallel_and_overlap(line1, line2):
   """returns True if two lines are parallel, identical and overlapping"""
   v1=[line1[1]-line1[0], line1[3]-line1[2]]
   v2=[line2[1]-line2[0], line2[3]-line2[2]]
   if v1[1]*v2[0] != v1[0]*v2[1]: #parallel
      return False
   else:
      d=[line1[0]-line2[0], line1[2]-line2[2]] #vector between starting points
      if  v1[1]*d[0] == v1[0]*d[1]: #identical lines
         minx_line1, maxx_line1 = min(line1[:2]), max(line1[:2])
         miny_line1, maxy_line1 = min(line1[2:]), max(line1[2:])
         minx_line2, maxx_line2 = min(line2[:2]), max(line2[:2])
         miny_line2, maxy_line2 = min(line2[2:]), max(line2[2:])
         if (minx_line1<=line2[0]<=maxx_line1 and miny_line1<=line2[2]<=maxy_line1) or (minx_line1<=line2[1]<=maxx_line1 and miny_line1<=line2[3]<=maxy_line1):# no gap
            return True
         elif (minx_line2<=line1[0]<=maxx_line2 and miny_line2<=line1[2]<=maxy_line2) or (minx_line2<=line1[1]<=maxx_line2 and miny_line2<=line1[3]<=maxy_line2):# no gap
            return True
         else:
            return False
      else:
         return False

def delete_zero_lines(lines):
   return [line[:] for line in lines if max(abs(line[1]-line[0]), abs(line[3]-line[2])) > 0 ]



def merge_overlapping_and_parallel_lines(lines):  
   groups_overlapping_lines = nx.Graph()
   for lineindex, line in enumerate(lines):
      for lineindex2, line2 in enumerate(lines):
         if lineindex < lineindex2 and parallel_and_overlap(line, line2):
            groups_overlapping_lines.add_edge(lineindex, lineindex2)
   merged_lines = [deepcopy(line[:]) for lineindex, line in enumerate(lines) if lineindex not in groups_overlapping_lines.nodes()]
   for component in nx.connected_components(groups_overlapping_lines):
      lines_x = [lines[n][i] for n in component for i in range(2)]
      lines_y = [lines[n][i] for n in component for i in range(2, 4)]
      line_example = lines[[c for c in component][0]]
      if line_example[1] - line_example[0] != 0:
         gradient = Fraction(line_example[3] - line_example[2], line_example[1] - line_example[0])
         intercept = line_example[2] - line_example[0] * gradient
         merged_lines.append([min(lines_x), max(lines_x), intercept + gradient * min(lines_x), intercept + gradient * max(lines_x)])
      elif line_example[1] - line_example[0] == 0:
         assert min(lines_y) != max(lines_y) and min(lines_x) == max(lines_x)
         merged_lines.append([line_example[0], line_example[0], min(lines_y), max(lines_y)])
   return merged_lines

#######################################################################################################################################################################


if __name__ == "__main__":
  import time
  import matplotlib.pyplot as plt  
  #######################################################################################################################################################################
  print('test functions')
  ########################################################################################################################################################################
  assert extrapolate_line_given_x([0, 1, 2 ,3], 1) == 3
  assert extrapolate_line_given_x([0, 1, 2 ,3], 2) == 4
  assert extrapolate_line_given_x([0, 3, 2 ,3], 6) == 4
  assert extrapolate_line_given_x([-2, 1, 2 ,3], 4) == 4
  ##
  assert extrapolate_line_given_y([0, 3, 2 ,3], 4) == 6
  assert extrapolate_line_given_y([0, 3, 2 ,1], 4) == -6
  ##
  posx = cut_out_positive_x([[0,1,2,3], [-2, 1, 2 ,3], [1, -2, 3, 2]])
  assert posx[0][0] == 0 and posx[0][1] == 1 and posx[0][2] == 2 and posx[0][3] == 3
  assert posx[1][0] == 0 and posx[1][1] == 1 and posx[1][2] == Fraction(8,3) and posx[1][3] == 3
  assert posx[2][0] == 1 and posx[2][1] == 0 and posx[2][2] == 3 and posx[2][3] == Fraction(8,3)
  ##
  line = [0, 3, 2, 1]
  x_segment, on_boundary = cut_x_window_from_line(line, 1, 2)
  assert x_segment[0] == 1 and x_segment[1] == 2 and x_segment[2] == Fraction(5, 3) and x_segment[3] == Fraction(4, 3) and on_boundary == False
  x_segment, on_boundary  = cut_x_window_from_line(line, Fraction(1,3), 2)
  assert x_segment[0] == Fraction(1,3) and x_segment[1] == 2 and x_segment[2] == Fraction(17, 9) and x_segment[3] == Fraction(4, 3) and on_boundary == False
  x_segment, on_boundary  = cut_x_window_from_line(line, 1, 5)
  assert x_segment[0] == 1 and x_segment[1] == 3 and x_segment[2] == Fraction(5, 3) and x_segment[3] == 1 and on_boundary == False
  ## test some vertical lines
  line = [1, 1, 2, 1]
  x_segment, on_boundary = cut_x_window_from_line(line, 1, 2)
  assert x_segment[0] == 1 and x_segment[1] == 1 and x_segment[2] == 2 and x_segment[3] == 1 and on_boundary == True
  line = [1, 1, 2, 1]
  x_segment, on_boundary = cut_x_window_from_line(line, 0, 2)
  assert x_segment[0] == 1 and x_segment[1] == 1 and x_segment[2] == 2 and x_segment[3] == 1 and on_boundary == False
  ##
  line = [0, 3, 2, -4]
  y_segment, on_boundary  = cut_y_window_from_line(line, 1, 2)
  assert y_segment[0] == Fraction(1,2) and y_segment[1] == 0 and y_segment[2] == 1 and y_segment[3] == 2 and on_boundary == False
  y_segment, on_boundary  = cut_y_window_from_line(line, Fraction(1,3), 2)
  assert y_segment[0] == Fraction(15,18) and y_segment[1] == 0 and y_segment[2] == Fraction(1, 3) and y_segment[3] == 2 and on_boundary == False
  y_segment, on_boundary  = cut_y_window_from_line(line, 1, 5)
  assert y_segment[0] == Fraction(1,2) and y_segment[1] == 0 and y_segment[2] == 1 and y_segment[3] == 2 and on_boundary == False
  ### test some horizontal lines
  line = [0, 3, 2, 2]
  y_segment, on_boundary  = cut_y_window_from_line(line, 1, 2)
  assert y_segment[0] == 0 and y_segment[1] == 3 and y_segment[2] == 2 and y_segment[3] == 2 and on_boundary == True  
  y_segment, on_boundary  = cut_y_window_from_line(line, 2, 4)
  assert y_segment[0] == 0 and y_segment[1] == 3 and y_segment[2] == 2 and y_segment[3] == 2 and on_boundary == True 
  y_segment, on_boundary  = cut_y_window_from_line(line, 1, Fraction(3, 2))
  assert len(y_segment) == 0 
  y_segment, on_boundary  = cut_y_window_from_line(line, 1, 4)
  assert y_segment[0] == 0 and y_segment[1] == 3 and y_segment[2] == 2 and y_segment[3] == 2 and on_boundary == False 
  ###
  assert line_length_integersqrt([0, 3, -1, 3]) == 5
  assert abs(line_length_integersqrt([0, Fraction(1, 3), -2, 3]) - np.sqrt(1/3**2 + 5**2)) < 0.01
  ###
  ####
  lines_list = [[0, Fraction(1, 2), 0, Fraction(1, 2)], [Fraction(1, 2), Fraction(9,10), Fraction(3, 4), Fraction(3, 4)], 
                [0, 0, Fraction(3, 4), 1], [Fraction(6, 10), Fraction(7, 10), Fraction(1,10), Fraction(1,10)],
                [Fraction(6, 10), Fraction(7, 10), Fraction(26,100), Fraction(30,100)]]
  dict_computed = get_squared_length_of_line_per_pixel(lines_list, resolution=4)[1]
  print( dict_computed)
  dict_manual = {(y, x): 0 for x in range(2) for y in range(4)} 
  dict_manual [(0, 0)] = np.sqrt(0.5) * 0.5
  dict_manual [(1, 0)] = np.sqrt(0.5) * 0.5
  dict_manual [(3, 1)] = 0.2
  dict_manual [(2, 1)] = 0.2
  dict_manual [(3, 0)] = 0.125
  dict_manual [(0, 1)] = 0.1
  dict_manual [(1, 1)] = np.sqrt(0.1**2 + 0.04**2)
  for x in range(2):
    for y in range(4):
       assert abs(dict_manual[(y, x)] - dict_computed[(y, x)]) < 0.01

  assert len(delete_zero_lines([[1, 1, 1, 1], [1, 2, 1, 1], [2, 2, 3, 2]])) == 2
  ###
  l1 = [0, 3, 2, -4]
  l2 = [4, 3, -6, -4]
  l3 = [0, 1, 2, 0]
  assert parallel_and_overlap(l1, l2) and parallel_and_overlap(l2, l1)
  assert parallel_and_overlap(l1, l3) and parallel_and_overlap(l3, l1)
  assert not parallel_and_overlap(l3, l2) and not parallel_and_overlap(l2, l3)
  merged_lines = merge_overlapping_and_parallel_lines([l1, l2, l3])
  single_line = merged_lines[0]
  assert len(merged_lines) == 1
  single_line_correct = [0, 4, 2, -6]
  single_line_correct_second_option = [single_line_correct[1], single_line_correct[0], single_line_correct[3], single_line_correct[2]]
  assert sum([single_line[i] - single_line_correct[i] for i in range(4)]) == 0 or sum([single_line[i] - single_line_correct_second_option[i] for i in range(4)]) == 0
  ##
  l1 = [0, 0, 2, -4]
  l2 = [0, 0, -6, -4]
  l3 = [0, 0, 1, 0]
  assert parallel_and_overlap(l1, l2) and parallel_and_overlap(l2, l1)
  assert parallel_and_overlap(l1, l3) and parallel_and_overlap(l3, l1)
  assert not parallel_and_overlap(l3, l2) and not parallel_and_overlap(l2, l3)
  merged_lines = merge_overlapping_and_parallel_lines([l1, l2, l3])
  single_line = merged_lines[0]
  assert len(merged_lines) == 1
  single_line_correct = [0, 0, 2, -6]
  single_line_correct_second_option = [single_line_correct[1], single_line_correct[0], single_line_correct[3], single_line_correct[2]]
  assert sum([single_line[i] - single_line_correct[i] for i in range(4)]) == 0 or sum([single_line[i] - single_line_correct_second_option[i] for i in range(4)]) == 0
  ##
  l1 = [0, 3, 1, 1]
  l2 = [4, 3, 1, 1]
  l3 = [0, 1, 1, 1]
  assert parallel_and_overlap(l1, l2) and parallel_and_overlap(l2, l1)
  assert parallel_and_overlap(l1, l3) and parallel_and_overlap(l3, l1)
  assert not parallel_and_overlap(l3, l2) and not parallel_and_overlap(l2, l3)
  merged_lines = merge_overlapping_and_parallel_lines([l1, l2, l3])
  single_line = merged_lines[0]
  assert len(merged_lines) == 1
  single_line_correct = [0, 4, 1, 1]
  single_line_correct_second_option = [single_line_correct[1], single_line_correct[0], single_line_correct[3], single_line_correct[2]]
  assert sum([single_line[i] - single_line_correct[i] for i in range(4)]) == 0 or sum([single_line[i] - single_line_correct_second_option[i] for i in range(4)]) == 0

  #######################################################################################################################################################################
  print('plot fraction of squares filled as histogram in units of square length')
  ########################################################################################################################################################################
  number_tests= 10 ** 2
  resolution = 10
  order_recursion_list = range(2, 8)
  order_recursion_vs_array_element_distribution = {o: [] for o in order_recursion_list}
  for order_recursion in order_recursion_vs_array_element_distribution.keys():
     for i in range(number_tests):
        gene = np.random.randint(-3, 3, size=8)
        gene = tuple(list(np.append(gene, order_recursion)))
        array_by_length = get_biomorphs_phenotype(gene, resolution, test=True)[0]
        order_recursion_vs_array_element_distribution[order_recursion] += list(np.ravel(array_by_length) * resolution)
  for plotnumber, log in enumerate([True, False]):
     f, ax = plt.subplots(ncols = len(order_recursion_vs_array_element_distribution)+1, figsize=(4.2*len(order_recursion_vs_array_element_distribution), 3))    
     for i, order_recursion in enumerate(order_recursion_list):
        n, bins, patches = ax[i].hist(order_recursion_vs_array_element_distribution[order_recursion], bins=100, log=log, color='k')
        ax[i].set_xlabel('length per array square')  
        ax[i].set_ylabel('frequency') 
        ax[i].set_title('order recursion: '+str(order_recursion))  
        #ax[i].vlines(np.arange(max(order_recursion_vs_array_element_distribution[order_recursion]), step =0.5), ymin=0, ymax=10*number_tests, zorder=-1)
     ax[-1].hist([a for l in order_recursion_vs_array_element_distribution.values() for a in l], bins=100, log=log, color='k')
     ax[-1].set_xlabel('length per array square')  
     ax[-1].set_ylabel('frequency') 
     ax[-1].set_title('all')  
     ax[-1].vlines(np.arange(max(order_recursion_vs_array_element_distribution[order_recursion]), step =0.5), ymin=0, ymax=max(n), zorder=-1)
     f.tight_layout()
     f.savefig('quick_plots/statistics_score_per_square_res'+str(resolution)+'_'+str(plotnumber)+'.png', bbox_inches='tight')
     plt.close('all')
  #######################################################################################################################################################################
  print('test symmetry')
  ########################################################################################################################################################################
  number_tests= 10 ** 2
  resolution = 30
  sign_change_reflection=[-1, -1, -1, 1, 1, 1, 1, 1, 1]
  for i in range(number_tests):
    gene=np.random.randint(-3, 3, size=8)
    order_recursion=np.random.randint(1, 8, size=1)
    gene=tuple(list(np.append(gene, order_recursion)))
    gene2=[gene[j]*sign_change_reflection[j] for j in range(9)]
    assert np.allclose(get_biomorphs_phenotype(gene, resolution, test=True)[0], get_biomorphs_phenotype(gene2, resolution, test=True)[0])
    if not get_biomorphs_phenotype(gene, resolution, test=False) == get_biomorphs_phenotype(gene2, resolution, test=False):
      print('rounding error')
      raise RuntimeError('')
  print('passed symmetry test')


  #######################################################################################################################################################################
  print('other tests')
  ########################################################################################################################################################################
  def get_limits(gene):
      lines = merge_overlapping_and_parallel_lines(cut_out_positive_x(delete_zero_lines(find_line_coordinates(gene))))
      xminfig, xmaxfig, yminfig, ymaxfig = get_limits_figure(lines)
      delta_y, delta_x = ymaxfig - yminfig, xmaxfig - xminfig
      delta = max(ymaxfig - yminfig, 2*xmaxfig) * Fraction(105, 100)
      xmax, xmin =  Fraction(delta, 2), 0
      ymin = Fraction(yminfig + ymaxfig - delta, 2)
      ymax =  ymin + delta
      assert xmin == 0
      return [xmin, xmax, ymin, ymax]



  from biomorph_functions import *

  gene=[1,3,-3,-1,3,1,2,1,2]
  for i in range(10):
    gene=np.random.randint(-3, 3, size=8)
    print(gene)
    order_recursion=np.random.randint(7, 9, size=1)
    gene=tuple(list(np.append(gene, order_recursion)))
    array_ph, array_coarsegrained = get_biomorphs_phenotype(gene, resolution = resolution, test=True)
    [xmin, xmax, ymin, ymax] =  get_limits(gene)
    f, ax = plt.subplots(ncols=3)
    im = ax[2].imshow(np.concatenate((array_coarsegrained[:, ::-1], array_coarsegrained), axis=1), origin='lower')
    ax[1].imshow(np.concatenate((array_ph[:, ::-1], array_ph), axis=1), origin='lower')
    draw_biomorph_in_subplot(gene, ax[0], linewidth=0.5)    
    ax[0].plot([xmin, xmax], [ymin, ymin], c='r', ls =':', lw=0.2)
    ax[0].plot([xmin, xmax], [ymax, ymax], c='r', ls =':', lw=0.2)
    ax[0].plot([xmin, xmin], [ymin, ymax], c='r', ls =':', lw=0.2)
    ax[0].plot([xmax, xmax], [ymin, ymax], c='r', ls =':', lw=0.2)
    ax[1].set_title(str(gene))
    f.tight_layout()
    f.savefig('quick_plots/example'+str(resolution)+'_'+str(i)+'.eps', bbox_inches='tight')
    plt.close('all')
  print('finished')

  #######################################################################################################################################################################
  print('test if removing one line changes phenotype')
  ########################################################################################################################################################################
  def find_phenotype_test_without_missing_line(lines, resolution, threshold, lines_for_axis_limits):
    # get xmin, xmax, ymin, ymax of grid (not figure)
     xminfig, xmaxfig, yminfig, ymaxfig = get_limits_figure(lines_for_axis_limits)
     delta_y, delta_x = ymaxfig - yminfig, xmaxfig - xminfig
     assert xminfig == 0 #negative part has been cut off 
     delta = max(delta_y, 2*delta_x) * Fraction(105, 100)
     ymin = Fraction(yminfig + ymaxfig - delta, 2)
     ymax =  ymin + delta
     xmax, xmin =  Fraction(delta, 2), xminfig
     assert xmax + 0.01 > abs(xmin)
     assert xmin - 0.01 < xminfig and ymin - 0.01 < yminfig
     assert xmax + 0.01 > xmaxfig and ymax + 0.01 > ymaxfig
     length_per_pixel, dict_with_exact_fraction = get_squared_length_of_line_per_pixel(prepare_lines(lines, xmin, xmax, ymin, ymax), resolution)
     cg_array = coarse_grain_array_binary(dict_with_exact_fraction, resolution, threshold)
     return length_per_pixel, cg_array, xmax, xmin, ymin, ymax

  for i in range(100):
    gene=np.random.randint(-3, 3, size=8)
    
    order_recursion=np.random.randint(7, 9, size=1)
    gene=tuple(list(np.append(gene, order_recursion)))
    lines = find_line_coordinates(gene) 
    edited_lines = merge_overlapping_and_parallel_lines(cut_out_positive_x(delete_zero_lines(lines)))  
    resolution, threshold = 30, 5
    if len(edited_lines):
       raster, raster_cg, xmax, xmin, ymin, ymax = find_phenotype_test_without_missing_line(edited_lines, resolution=resolution, threshold=threshold, lines_for_axis_limits=edited_lines) 
    if len(edited_lines) > 1:
      index_to_remove = np.random.choice(len(edited_lines) -1)
      new_edited_lines = [l[:] for index, l in enumerate(edited_lines) if index != index_to_remove] 
      if min([l[xindex] for xindex in range(2) for l in new_edited_lines]) > 0:
        continue # only line at zero deleted
      raster2, raster_cg2, xmax, xmin, ymin, ymax = find_phenotype_test_without_missing_line(new_edited_lines, resolution=resolution, threshold=threshold, lines_for_axis_limits=edited_lines) 
      length_difference = np.sum(raster - raster2)
      deleted_line_length = line_length_integersqrt(prepare_lines([edited_lines[index_to_remove][:],], xmin, xmax, ymin, ymax)[0])
      if max(edited_lines[index_to_remove][:2]) == 0:
        deleted_line_length = 0.5 * deleted_line_length #only half falls into this half
      print(gene, np.sum(raster - raster2), float(deleted_line_length), np.sum(raster - raster2)- float(deleted_line_length))
      if np.sum(raster - raster2) <= 0.0001:
        print(edited_lines)
        print(new_edited_lines)
      assert np.sum(raster - raster2) > 0.0001
      assert np.sum(raster - raster2)- float(deleted_line_length) < deleted_line_length * resolution * 0.005 # longer lines are split across more pixels, so more rounding errors
      assert np.sum(raster - raster2)- float(deleted_line_length) < 0.0066 #this is where it would matter if it only concerned one pixel




  #######################################################################################################################################################################
  print('test tree function')
  ########################################################################################################################################################################
  from GPproperties import expand_array, is_tree
  from biomorph_functions import *

  resolution = 30
  for i in range(50):
    if i == 0:
      gene = tuple([0,] *8 + [2,])
    elif i == 1:
      gene = tuple([1,] *8 + [2,])
    elif i == 2:
      gene = tuple([1, 2,] *4 + [2,])
    else:
       gene=np.random.randint(-3, 3, size=8)
       print(gene)
       order_recursion=np.random.randint(7, 9, size=1)
       gene=tuple(list(np.append(gene, order_recursion)))
    array_ph, array_coarsegrained = get_biomorphs_phenotype(gene, resolution = resolution, test=True)
    print( get_biomorphs_phenotype(gene, resolution=30, threshold = 5, test=False))
    f, ax = plt.subplots(ncols=3)
    im = ax[2].imshow(np.concatenate((array_coarsegrained[:, ::-1], array_coarsegrained), axis=1), origin='lower')
    ax[1].imshow(np.concatenate((array_ph[:, ::-1], array_ph), axis=1), origin='lower')
    draw_biomorph_in_subplot(gene, ax=ax[0])
    if is_tree(get_biomorphs_phenotype(gene, resolution = resolution, test=False)):
       ax[0].set_title('a tree-like figure')
    ax[1].set_title(str(gene))
    f.tight_layout()
    f.savefig('quick_plots/example'+str(resolution)+'_'+str(i)+'_testtree.png', bbox_inches='tight')
    plt.close('all')
  print('finished')
  #######################################################################################################################################################################
  print('plot for biomorph construction')
  ########################################################################################################################################################################
  from biomorph_functions import *
  max_g9 = 3
  resolution_for_nice_plots = 30
  for i in range(50):
    if i== 0:
      gene_vector = (1,1,1, 1, 2, 1, 1, 0) #stick to a specific example
    else:
       gene_vector=np.abs(np.random.randint(-3, 3, size=8))
    print(gene_vector)
    lines = find_line_coordinates([g for g in gene_vector] + [max_g9,])
    x_values = [l[0] for l in lines] + [l[1] for l in lines] 
    y_values = [l[2] for l in lines] + [l[3] for l in lines] 
    delta = max(max(x_values) * 2, max(y_values) - min(y_values))
    xlims, ylims = (-0.5 * delta, 0.5 * delta), (min(y_values), min(y_values) + delta)
    
    f, ax = plt.subplots(ncols=max_g9, figsize=(3 * max_g9,2.7))
    for order_recursion in range(1, max_g9 + 1):
       gene=tuple(list(np.append(gene_vector[:], order_recursion)))
       draw_biomorph_in_subplot(gene, ax=ax[order_recursion - 1])
       scale = max_g9/float(order_recursion)
       ax[order_recursion - 1].set_xlim(xlims[0] /scale, xlims[1] /scale)
       ax[order_recursion - 1].set_ylim(ylims[0] /scale, ylims[1] /scale)
    f.suptitle(str(gene))
    f.tight_layout()
    f.savefig('quick_plots/schematic_construction'+str(i)+'.eps', bbox_inches='tight')
    plt.close('all')
    ######
    if i == 0:
       gene = tuple([g for g in gene_vector[:8]] + [5,])
    else:
       gene = tuple([g for g in gene_vector[:8]] + [int(np.random.randint(5, 9, size=1)[0]),])
    print(gene)
    array_ph, array_coarsegrained = get_biomorphs_phenotype(gene, resolution = resolution_for_nice_plots, test=True)
    f, ax = plt.subplots(ncols=2, figsize=(9,4))
    im = ax[1].imshow(np.concatenate((array_coarsegrained[:, ::-1], array_coarsegrained), axis=1), origin='lower', cmap='Greys')
    ax[1].set_xlim(0, resolution_for_nice_plots)
    ax[1].set_ylim(0, resolution_for_nice_plots)
    ax[1].axis('equal')
    ax[1].axis('off')
    draw_biomorph_in_subplot(gene, ax=ax[0])
    lines = find_line_coordinates(gene)
    x_values = [l[0] for l in lines] + [l[1] for l in lines] 
    y_values = [l[2] for l in lines] + [l[3] for l in lines] 
    delta = max(max(x_values) * 2, max(y_values) - min(y_values))
    ax[0].set_xlim(-0.5 * delta, 0.5 * delta)
    ax[0].set_ylim(min(y_values), min(y_values) + delta)
    f.suptitle(str(gene))
    f.tight_layout()
    f.savefig('quick_plots/schematic_coarsegraining'+str(resolution_for_nice_plots)+'_'+str(i)+'.eps', bbox_inches='tight')
    plt.close('all')
  print('finished')
