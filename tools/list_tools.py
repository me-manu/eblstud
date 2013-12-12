"""
Module Containing tools for list search

History of changes:
Version 1.0
- Created 7th December 2010

"""

__version__ = 1.0
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

import numpy as np


def find_best(list, value):
    """Finds the value in list which is closest to value and returns it,
    Note that list must be a sorted list in ascending order"""

    if not len(list):
	raise ValueError('List has zero length!')
    if len(list) < 2:
	return list[0]
    elif len(list) < 3:
	if np.abs(list[0]-value) <= np.abs(list[1]-value):
		return list[0]
	else:
		return list[1]
    mi = 0
    ma = len(list) - 1
    mid = len(list)/2
    if list[mid] > value:
	ma = mid
    elif list[mid] < value:
	mi = mid
    else:
	return list[mid]
    return find_best(list[mi:ma+1],value)

def best_index(list,value):
	""" finds the value in list that is closest to value and returns it together with its index"""

	list_bk = list[:]
	l = []
	for item in list:
	    l.append(item)
	best = find_best(list_bk,value)
	return best, l.index(best)
