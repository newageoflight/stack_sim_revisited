# Utility functions and constants

import re

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
sanitise_filename = lambda x: re.sub(r'[<>:"/\|?*,]', '', x)
def uniq(l):
	seen = []
	for i in l:
		if i not in seen:
			seen.append(i)
	return seen
def underscorify(name):
	new_name = re.sub(r"[\s]", "_", name)
	new_name = sanitise_filename(new_name.lower())
	return new_name

POSITIVE_INFINITY = 1e8
NEGATIVE_INFINITY = -POSITIVE_INFINITY

def is_iterable(item):
	try:
		iterable = iter(item)
		if iterable:
			return True
	except TypeError as te:
		return False