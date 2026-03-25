"""
All the functions needed to run all the other biomorph programs
"""
import numpy as np
import matplotlib.pyplot as plt

def directions(gene):
    #convert the gene into directions and the order up to which lines are drawn
    order = gene[8]
    dx = [int(-gene[1]), int(-gene[0]), 0, int(gene[0]), int(gene[1]), int(gene[2]), 0, -gene[2]]
    dy = [int(gene[5]), int(gene[4]), int(gene[3]), int(gene[4]), int(gene[5]), int(gene[6]), int(gene[7]), int(gene[6])]
    return dx, dy, order

def draw_biomorph(xstart, ystart, order, dx, dy, direction, ax, color='k', linewidth=3):
    #draws the lines to make a biomorph
    direction = direction%8
    xnew = xstart + order * dx[direction]
    ynew = ystart + order * dy[direction] 
    ax.plot([xstart, xnew], [ystart, ynew], color=color, linestyle='-', linewidth=linewidth) 
    if order>1:
        draw_biomorph(xnew, ynew, order-1, dx, dy, direction-1, ax, color, linewidth=linewidth)
        draw_biomorph(xnew, ynew, order-1, dx, dy, direction+1, ax, color, linewidth=linewidth)
    return 

def draw_biomorph_in_subplot(gene, ax, color='k', linewidth=3):
    dx, dy, order= directions(gene)
    draw_biomorph(0, 0, order, dx, dy, 2, ax, color=color, linewidth=linewidth)
    ax.set_aspect('equal', 'box')
    ax.axis('off')


def find_line_coordinates(gene):
   dx, dy, order= directions(gene)
   return coords_lines(0, 0, order, dx, dy, 2, [])


def coords_lines(xstart, ystart, order, dx, dy, direction, lines):
    #draws the lines to make a biomorph
    direction = direction%8
    xnew = xstart + order * dx[direction]
    ynew = ystart + order * dy[direction] 
    lines.append([xstart, xnew, ystart, ynew])
    if order>1:
        lines = coords_lines(xnew, ynew, order-1, dx, dy, direction-1, lines)
        lines = coords_lines(xnew, ynew, order-1, dx, dy, direction+1, lines)
    return lines
