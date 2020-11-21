import math
import numpy as np
import cupy as cp

def swap_two(lst) -> None:
    """
    Swap two elements in a cupy array
    """
    new_list = lst.copy()
    app_len = len(new_list) - 1
    a, b = cp.random.choice(app_len, 2, replace=False)
    temp = new_list[b]
    new_list[b] = new_list[a]
    new_list[a] = temp
    return new_list

def accept(energy, new_energy, T):
    if new_energy < energy:
        return 1
    else:
        return math.exp((energy - new_energy)/T)

def one_array(l, n, v=1):
    """
    Generates a 1d np.array of length l with a one at position n 
    Optionally, create a 2d array by specifying the vertical height v
    """
    arr = np.zeros((v,l))
    arr[:, n] = 1
    return arr