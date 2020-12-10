from numpy.random import choice
# from functools import wraps
from typing import Iterator

from .hospital import Hospital, hospitals, hospital_weights, stack, altstack
# from .utils import underscorify

import numpy as np
import pandas as pd
import random
import re

ordinal_re = r"\d+(?:th|st|rd|nd)"

# stack strategy functions
# these need to be fixed somehow so they don't look retarded like they do now (global variables, redundant arguments)
# may need e.g. a separate class for HospitalList

# here's an idea that may fix it
# have a callable object that contains a preferencing strategy
# when called (takes an argument), it can check if it was initiated with a list or a function and return accordingly

def push_random_to_top(l):
    k = l[:]
    k.insert(0, k.pop(random.randint(0,len(k)-1)))
    return k

def push_weighted_random_to_top(l, w):
    k = l[:]
    # stupid, ugly fix but i don't really know what else to do
    origin = choice(len(k), 1, p=w)[0]
    k.insert(0, k.pop(origin))
    return k

def push_random_to_positions(l, *positions):
    k = l
    pairs = zip(positions, choice(len(k), len(positions)))
    for target, origin in pairs:
        k.insert(target, k.pop(origin))
    return k

def push_wt_random_to_positions(l, w, *positions):
    k = l
    pairs = zip(positions, choice(len(k), len(positions), p=w))
    for target, origin in pairs:
        k.insert(target, k.pop(origin))
    return k

class Hospitals(object):
    """
    Stores the hospitals and produces views of the hospitals that can be detranslated

    Hospitals are stored in order of their network numbers on HETI
    """
    def __init__(self, pair_count=3, stack_variants=7) -> None:
        self.hospitals, self.hospital_weights = hospitals, hospital_weights
        self.hospital_df = pd.DataFrame([{**h.as_dict(), "weight": hospital_weights[i]} for i, h in enumerate(hospitals)])
        self.dra_capacities = np.array(list(map(lambda h: h.dra, self.hospitals)))
        self.default_stack = stack
        self.preset_stacks = [stack, altstack]
        self._stack_rearrangement_pairsets = []
        self._rearrangement_pairset_used = []
        self._pair_count = pair_count
        self._stack_variants = stack_variants

    def _generate_stack_rearrangement(self) -> None:
        """
        Several assumptions:
        - Distribution of choices of rearrangements is random
        - Stack rearrangements only ever involve swapping one or more pairs of adjacent hospitals
        - Pairs cannot overlap (e.g. you cannot swap places 2-3 and 3-4)
        - There are never more than 5 swapped pairs in a stack rearrangement
        """
        pair_count = self._pair_count
        indices_remaining = set(range(len(self.default_stack)))
        pairs = []
        for i in range(pair_count):
            pair_first = random.choice(tuple(indices_remaining))
            pair_second = pair_first + 1 if pair_first != len(self.default_stack) - 1 else pair_first - 1
            pair = set([pair_first, pair_second])
            indices_remaining -= pair
            pairs.append(pair)
        self._stack_rearrangement_pairsets.append(pairs)

    def _generate_stack_rearrangements(self) -> None:
        count = self._stack_variants
        for i in range(count):
            self._generate_stack_rearrangement()
            self._rearrangement_pairset_used.append(False)
    
    def _get_rearranged_stack(self):
        if not self._stack_rearrangement_pairsets:
            self._generate_stack_rearrangements()
        pairs = random.choice(self._stack_rearrangement_pairsets)
        new_stack = self.default_stack.copy()
        for pair in pairs:
            i, j = pair
            temp = new_stack[i]
            new_stack[i] = new_stack[j]
            new_stack[j] = temp
        return new_stack

    def _use_strategy(self, strategy: str) -> "list[int]":
        """
        Maps strings to strategy functions. Returns a list of integers (positions)
        """
        s = strategy.lower().strip()
        stack_to_use = None
        hospital_sequence = list(range(len(self.hospitals)))
        if s == "stack":
            stack_to_use = self.default_stack
        elif s == "variant stack":
            stack_to_use = self._get_rearranged_stack()
        elif s == "random":
            stack_to_use = random.sample(hospital_sequence[:], len(hospital_sequence))
        elif s == "weighted random":
            stack_to_use = list(np.random.choice(hospital_sequence, len(hospital_sequence), p=self.hospital_weights, replace=False))
        elif s == "random with weighted random top":
            rearr = random.sample(hospital_sequence[:], len(hospital_sequence))
            stack_to_use = push_weighted_random_to_top(rearr, [self.hospital_weights[i] for i in rearr])
        elif s == "stack with random top":
            stack_to_use = push_random_to_top(self.default_stack)
        elif s == "stack with weighted random top":
            stack_to_use = push_weighted_random_to_top(self.default_stack, [self.hospital_weights[i] for i in self.default_stack])
        elif s == "variant stack with random top":
            stack_to_use = push_random_to_top(self._get_rearranged_stack())
        elif s == "variant stack with weighted random top":
            stack_rearr = self._get_rearranged_stack()
            stack_to_use = push_weighted_random_to_top(stack_rearr, [self.hospital_weights[i] for i in stack_rearr])
        # to deal with the possibility of people asking for random 12th, 14th, 2nd, etc.
        # i'll use regex to get the number out of the string
        elif re.match("stack with random (?:({0})(, )?)* and ({0})".format(ordinal_re), s):
            ordinals_found = [int(o[:-2])-1 for o in re.findall(ordinal_re, s)]
            stack_to_use = push_random_to_positions(self.default_stack, *ordinals_found)
        elif re.match("stack with weighted random (?:({0})(, )?)* and ({0})".format(ordinal_re), s):
            ordinals_found = [int(o[:-2])-1 for o in re.findall(ordinal_re, s)]
            stack_to_use = push_wt_random_to_positions(self.default_stack, [self.hospital_weights[i] for i in self.default_stack], *ordinals_found)
        elif re.match("variant stack with random (?:({0})(, )?)* and ({0})".format(ordinal_re), s):
            stack_rearr = self._get_rearranged_stack()
            ordinals_found = [int(o[:-2])-1 for o in re.findall(ordinal_re, s)]
            stack_to_use = push_random_to_positions(stack_rearr, *ordinals_found)
        elif re.match("variant stack with weighted random (?:({0})(, )?)* and ({0})".format(ordinal_re), s):
            stack_rearr = self._get_rearranged_stack()
            ordinals_found = [int(o[:-2])-1 for o in re.findall(ordinal_re, s)]
            stack_to_use = push_wt_random_to_positions(stack_rearr, [self.hospital_weights[i] for i in stack_rearr], *ordinals_found)
        else:
            raise NotImplementedError("Strategy '{strategy}' has not been implemented yet!".format(strategy=strategy))
        return stack_to_use


    def iterate_by_strategy(self, strategy: str) -> Iterator[Hospital]:
        """
        Returns an iterator (generator) over the hospitals, in order of strategy

        Similar to how a reducer would work in React Context/Redux, this uses string flags to create an iterator
        """
        stack_to_use = self._use_strategy(strategy)
        return (self.hospitals[h] for h in stack_to_use)

    def array_from_strategy(self, strategy: str) -> np.array:
        """
        Returns an np.array with the hospitals numbered in order of the applicant's strategy
        """
        stack_to_use = self._use_strategy(strategy)
        return np.array(stack_to_use)
