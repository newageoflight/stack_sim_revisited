from .utils import POSITIVE_INFINITY

import numba
import numpy as np
import cupy as cp

def swap_two(lst):
    """
    Swap two elements in a cupy array
    """
    new_list = lst.copy()
    app_len = len(new_list) - 1
    a, b = cp.random.choice(app_len, 2, replace=False)
    # won't work properly if you don't do this
    orig, swap = cp.array([a,b]), cp.array([b,a])
    new_list[orig] = new_list[swap]
    return new_list

def accept(energy, new_energy, T):
    """
    Accept a cupy array swap
    """
    if new_energy < energy:
        return 1
    else:
        return cp.exp((energy - new_energy)/T)

def one_array(l, n, v=1):
    """
    Generates a 1d np.array of length l with a one at position n 
    Optionally, create a 2d array by specifying the vertical height v
    """
    arr = np.zeros((v,l))
    arr[:, n] = 1
    return arr

@numba.njit(fastmath=True)
def accept_numba(energy, new_energy, T):
    if new_energy < energy:
        return 1
    else:
        return np.exp((energy - new_energy)/T)

@numba.njit
def swap_two_numba(lst):
    """
    Swap two items in an np.array using numba
    """
    new_list = lst.copy()
    app_len = len(new_list)-1
    a, b = np.random.choice(app_len, 2, replace=False)
    # won't work properly if you don't do this
    orig, swap = np.array([a,b]), np.array([b,a])
    new_list[orig] = new_list[swap]
    return new_list

@numba.njit(parallel=True)
def np_sum_mul(a, b):
    return np.sum(a * b)

@numba.njit(parallel=True)
def np_colsum(a):
    return np.sum(a, axis=0)

@numba.njit(fastmath=True)
def cool_numba(current_state_arr, pref_arr, capacity_arr, T, cool_rate, iterlimit):
    temp = T
    itercount = 0
    unhappiness_log = np.empty((0,2),np.int32)
    min_unhappiness = POSITIVE_INFINITY
    best_state_arr = current_state_arr
    while temp >= 1e-8 and itercount < iterlimit:
        next_state_arr = swap_two_numba(current_state_arr)
        u_current = np_sum_mul(pref_arr, current_state_arr)
        u_next = np_sum_mul(pref_arr, next_state_arr)
        next_over_capacity = (np_colsum(next_state_arr) > capacity_arr).any()
        accepted = accept_numba(u_current, u_next, temp) >= np.random.random()
        if accepted and not next_over_capacity:
            current_state_arr = next_state_arr
            u_current = u_next
        if u_current < min_unhappiness:
            best_state_arr = current_state_arr
            min_unhappiness = u_current
        temp *= 1 - cool_rate
        itercount += 1
        unhappiness_log = np.append(unhappiness_log, np.array([[u_current, min_unhappiness]],np.int32), axis=0)
    return (temp, min_unhappiness, current_state_arr, best_state_arr, unhappiness_log)
