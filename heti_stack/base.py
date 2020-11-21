#!/usr/bin/env python3

"""
Base functions and classes
"""

from .hospital import Hospital

# Constants

category_counts = []
with open("category-counts.txt", "r") as categories_infile:
	for line in categories_infile:
		catid, catnum = line.split('\t')
		category_counts.append(int(catnum))
