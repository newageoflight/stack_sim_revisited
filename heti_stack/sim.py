#!/usr/bin/env python3

"""
Allocation process simulation objects
"""

from abc import ABC, abstractmethod

from .sim_utils import cool_numba, one_array
from .applicant import ApplicantPool
from .base import category_counts
from .strategies import Hospitals
from .utils import POSITIVE_INFINITY

import matplotlib.pyplot as plt
import numpy as np
# import cupy as cp
import pandas as pd

plt.style.use("ggplot")
plt.rcParams["font.family"] = "Helvetica Neue"

# TODO: consider making an interactive data visualisation showing the sim doing the allocation
# basically, animate the bar charts and convergence graphs building up, as well as having a meter for how full the hospitals are
# 3 panels side by side, more or less
# cf. Kaggle for an example of how to do this
class Simulation(ABC):
    """
    Base simulation object, metaclass for the others
    """
    def __init__(self, starting_strategies: "list[tuple[str, float]]", pair_count=3, stack_variants=7, rounds=1) -> None:
        hospital_object = Hospitals(pair_count, stack_variants)
        self.starting_strategies = starting_strategies
        self.applicant_pool = ApplicantPool(starting_strategies, hospital_object)
        self.hospitals = hospital_object
        self._hospital_capacities = np.array(self.hospitals.hospital_df["capacity"])
        self._hospital_spots = np.array(self.hospitals.hospital_df["spots_remaining"])
        self.allocation_rounds = rounds
        self.preferences_matrix = np.vstack(self.applicant_pool.candidate_df["preferences"])
        self.allocation_matrix = np.zeros_like(self.preferences_matrix)

    def __repr__(self) -> str:
        return "{0}({1})".format(self.__class__.__name__, self.starting_strategies)

    def _dra_prefill(self) -> None:
        """
        Prefill DRA-eligible hospitals with candidates who have preferenced them first prior to allocation round
        """
        # first filter the applicant pool for anyone who indicated a dra hospital as first preference
        cat1_subdf = self.applicant_pool.candidate_df[self.applicant_pool.candidate_df["category"] == 0]
        cat1_indices = cat1_subdf.index
        dra_indices = np.where(self.hospitals.dra_capacities > 0)[0]
        dra_prefs = self.preferences_matrix[np.ix_(cat1_indices, dra_indices)]
        dra_firsts = dra_prefs == 0
        dra_firsts = dra_firsts.astype("int32")
        # allocate them to that hospital, capacity allowing
        # loop through each dra hospital
        for i in range(len(dra_indices)):
            preferenced_this_first = np.where(dra_firsts[:, i] == 1)[0]
            # sample however many will fit the dra capacity
            # or if there are fewer preferences than spots available, take everyone who preferenced it
            to_alloc = np.random.choice(preferenced_this_first,
                min(self.hospitals.dra_capacities[dra_indices[i]], len(preferenced_this_first)),
                replace=False)
            to_alloc = np.sort(to_alloc)
            self.allocation_matrix[to_alloc, dra_indices[i]] = 1
        # update the spots remaining count
        self._update_spots()
        # detranslate with the flag is_dra positive for applicants
        self._dra_detranslate()

    def _setup_initial_state(self) -> None:
        raise NotImplementedError

    def _update_spots(self) -> None:
        self._hospital_spots = self._hospital_capacities - self.allocation_matrix.sum(axis=0)

    @abstractmethod
    def run(self, dra_prefill=False) -> None:
        if dra_prefill:
            self._dra_prefill()
    
    def _dra_detranslate(self) -> None:
        """
        Detranslate/announce DRA preallocation results back to the applicant pool
        """
        allocation_col = np.argmax(self.allocation_matrix == 1, axis=1)
        got_allocated = np.any(self.allocation_matrix == 1, axis=1)
        self.applicant_pool.candidate_df["allocation"] = allocation_col
        self.applicant_pool.candidate_df["is_dra"] = got_allocated
        # no need to calculate preference values as they're all 0 (first)
        self.hospitals.hospital_df["spots_remaining"] = self._hospital_spots
        cdf = self.applicant_pool.candidate_df
        self.hospitals.hospital_df["filled_spots"] = [np.array(cdf[cdf["allocation"] == i]["uid"]) for i in range(len(self._hospital_spots))]

    def _detranslate(self) -> None:
        """
        Detranslate/announce the final allocation results back to the applicant pool
        """
        # First detranslate the allocations to the candidate df
        allocation_col = np.argmax(self.allocation_matrix == 1, axis=1)
        unallocated_bool = np.all(self.allocation_matrix == 0, axis=1)
        unallocated = np.arange(len(unallocated_bool))[unallocated_bool]
        allocation_col[unallocated] = -1

        self.applicant_pool.candidate_df["allocation"] = allocation_col
        # I haven't figured out a pure-numpy way to do this next line yet
        pref_vals = np.array([(self.preferences_matrix[i, p] if p >= 0 else -1) for i, p in enumerate(allocation_col)])
        self.applicant_pool.candidate_df["preference_number"] = pref_vals
        # Now propagate these changes to the hospitals
        self.hospitals.hospital_df["spots_remaining"] = self._hospital_spots
        # for convenience i'll make a new cdf with the candidate df
        cdf = self.applicant_pool.candidate_df
        self.hospitals.hospital_df["filled_spots"] = [np.array(cdf[cdf["allocation"] == i]["uid"]) for i in range(len(self._hospital_spots))]

    def unhappiness(self):
        return np.sum(self.allocation_matrix * self.preferences_matrix)


class CategoricalSimulation(Simulation):
    """
    Runs categorical allocation simulation
    """
    def run(self, dra_prefill=False) -> None:
        """
        Allocate applicants to hospitals via categorical allocation method
        """
        super().run(dra_prefill)
        hospital_count = len(self.hospitals.hospitals)
        for cat in range(len(category_counts)):
            cat_subdf = self.applicant_pool.candidate_df[self.applicant_pool.candidate_df["category"] == cat]
            cat_indices = cat_subdf.index
            # iterate over preference number
            for rank in range(hospital_count):
                # fill each hospital with everyone remaining who ranked it at this number
                for col_num in range(hospital_count):
                    # filter out anyone who's already been allocated
                    cat_allocs = self.allocation_matrix[cat_indices]
                    unallocated = np.all(cat_allocs == 0, axis=1)
                    unallocated_indices = cat_indices[unallocated]
                    # now do the allocation
                    cat_prefs = self.preferences_matrix[unallocated_indices]
                    # how many people preferenced this hospital at this rank?
                    cat_ranks = (cat_prefs == rank).astype("int32")
                    # now iterate through each hospital to check if it exceeds capacity or not
                    # sum the column - is it greater than the capacity?
                    ranked_this_hospital = cat_ranks[:, col_num]
                    ranked_this_indices = np.where(ranked_this_hospital == 1)[0]
                    total_prefs = np.sum(ranked_this_hospital)
                    alloc_rows = unallocated_indices[ranked_this_indices]
                    # print(unallocated_indices, alloc_rows, ranked_this_indices, self.preferences_matrix[alloc_rows])
                    to_set = cat_ranks[ranked_this_indices, col_num]
                    # if preferences exceeds remaining spots, take a random subsample
                    remaining_spots = self._hospital_spots[col_num]
                    # print(remaining_spots)
                    if total_prefs > remaining_spots:
                        subrows = np.sort(np.random.choice(len(alloc_rows), remaining_spots, replace=False))
                        alloc_rows = alloc_rows[subrows]
                        to_set = cat_ranks[ranked_this_indices[subrows], col_num]
                    self.allocation_matrix[alloc_rows, col_num] = to_set
                self._update_spots()
                # print(self._hospital_spots)

        self._detranslate()

class AnnealSimulation(Simulation):
    """
    Runs simulated annealing simulation
    """
    def __init__(self, starting_strategies: "list[tuple[str, float]]", pair_count=3, stack_variants=7, rounds=1, temp=10000.0, cool_rate=0.001, iterlimit=1000000) -> None:
        self.min_unhappiness = POSITIVE_INFINITY
        self.best_state = np.array([])
        self._temp = temp
        self._cooling_rate = cool_rate
        self._iterlimit = iterlimit
        self.unhappiness_df = pd.DataFrame(columns=["current_unhappiness", "min_unhappiness"])
        self.unhappiness_array = np.empty((0,2),dtype=np.int32)
        # self._backend = backend
        super().__init__(starting_strategies, pair_count, stack_variants, rounds)

    def _setup_initial_state(self) -> None:
        """
        Setup the initial state. 
        """
        for cat in range(len(category_counts)):
            # print("category", cat+1)
            # get all the unallocated applicants for this category first
            cat_subdf = self.applicant_pool.candidate_df[self.applicant_pool.candidate_df["category"] == cat]
            cat_indices = cat_subdf.index
            cat_allocs = self.allocation_matrix[cat_indices]
            unallocated = np.all(cat_allocs == 0, axis=1)
            cat_unallocated_indices = cat_indices[unallocated]
            cat_unallocated = self.allocation_matrix[cat_unallocated_indices]
            # to prevent overfilling we first generate a pre-allocation matrix
            # first generate a one-hot array for each spot in each hospital
            to_join = []
            for i in range(len(self._hospital_spots)):
                to_join.append(one_array(len(self._hospital_spots), i, v=self._hospital_spots[i]))
            prealloc_matrix = np.vstack(to_join)
            # randomise the order of the preallocation matrix
            np.random.shuffle(prealloc_matrix)
            # print(len(cat_unallocated), len(prealloc_matrix))
            if len(cat_unallocated) > len(prealloc_matrix):
                # too many candidates
                # take a random sample of the unallocated candidates
                subselection = np.random.choice(cat_unallocated_indices, size=len(prealloc_matrix), replace=False)
                self.allocation_matrix[subselection] = prealloc_matrix
            elif len(cat_unallocated) == len(prealloc_matrix):
                self.allocation_matrix[cat_unallocated_indices] = prealloc_matrix
            else:
                # not enough candidates
                # take a random sample of the preallocation matrix
                suballoc = np.random.choice(len(prealloc_matrix), size=len(cat_unallocated), replace=False)
                self.allocation_matrix[cat_unallocated_indices] = prealloc_matrix[suballoc]
            self._update_spots()

    def run(self, dra_prefill=False) -> None:
        super().run(dra_prefill)
        # backend = self._backend
        # setup initial state
        self._setup_initial_state()
        # run sim
        # itercount = 0
        # current state only accounts for placed candidates, so remove any unplaced candidates first
        placed_candidates_bool = np.any(self.allocation_matrix == 1, axis=1)
        placed_candidates = self.allocation_matrix[placed_candidates_bool]
        relevant_preferences = self.preferences_matrix[placed_candidates_bool]
        # add initial unhappiness stats to the array
        u_current = np.sum(placed_candidates * relevant_preferences)
        self.unhappiness_array = np.append(self.unhappiness_array, np.array([[u_current, u_current]], np.int32), axis=0)
        unhappiness_log = np.empty((0,2), np.int32)
        # print(placed_candidates, relevant_preferences)
        # if backend == "gpu":
        #     # move to gpu with cupy
        #     # try to do all operations on the gpu to avoid copying overhead
        #     current_state = cp.asarray(placed_candidates)
        #     pref_arr = cp.asarray(relevant_preferences)
        #     capacity_arr = cp.asarray(self._hospital_capacities)
        #     best_state = cp.asarray(self.best_state)
        #     unhappiness_log = cp.asarray(unhappiness_log)
        #     T = cp.asarray(self._temp)
        #     cool_rate = cp.asarray(self._cooling_rate)
        #     min_unhappiness = cp.asarray(self.min_unhappiness)
        #     while T >= 1e-8 and itercount < self._iterlimit:
        #         next_state = swap_two(current_state)
        #         u_current = cp.sum(pref_arr * current_state)
        #         u_next = cp.sum(pref_arr * next_state)
        #         accepted = accept(u_current, u_next, T) >= cp.random.random()
        #         next_over_capacity = (cp.sum(next_state, axis=0) > capacity_arr).any()
        #         if accepted and not next_over_capacity:
        #             current_state = next_state
        #             u_current = u_next
        #         if u_current < min_unhappiness:
        #             best_state = current_state
        #             min_unhappiness = u_current
        #         T *= 1 - cool_rate
        #         itercount += 1
        #         # the proper method would be concatenate but vstack works (and concat doesn't) so i'll just use that
        #         unhappiness_log = cp.vstack([unhappiness_log, cp.array([u_current, min_unhappiness], cp.int32)])

        #     # move the end results back to cpu memory via cp.asnumpy
        #     self.best_state = cp.asnumpy(best_state)
        #     unhappiness_log = cp.asnumpy(unhappiness_log)
        # if backend == "cpu":
            # if we can't use the GPU, use numba.jit
            # this is actually faster because the dataset is relatively small (1000s rather than 1000000s)
        self._temp, self.min_unhappiness, _, self.best_state, unhappiness_log = cool_numba(
            placed_candidates, relevant_preferences, self._hospital_capacities,
            self._temp, self._cooling_rate, self._iterlimit
        )
        self.unhappiness_array = np.append(self.unhappiness_array, unhappiness_log, axis=0)
        self.allocation_matrix[placed_candidates_bool] = self.best_state
        self._detranslate()

    def _detranslate(self):
        append_data = pd.DataFrame({"current_unhappiness": self.unhappiness_array[:,0],
            "min_unhappiness": self.unhappiness_array[:,1]})
        self.unhappiness_df = self.unhappiness_df.append(append_data, ignore_index=True)
        super()._detranslate()

    def plot_convergence(self):
        self.unhappiness_df.plot.line()
        plt.xlabel("Number of iterations")
        plt.ylabel("Global unhappiness")
        plt.title("Convergence")
        plt.tight_layout()
        plt.show()
        # plt.clf()
        # plt.cla()

class QBSimulation(Simulation):
    """
    Runs a simulation using the Queensland Ballot Algorithm
    """
    def run(self, dra_prefill=False):
        super().run(dra_prefill)
        # setup initial state
        non_dra_indices = np.where(np.all(self.allocation_matrix == 0, axis=1))[0]
        # iterate through categories
        for cat in range(len(category_counts)):
            total_remaining_spots = np.sum(self._hospital_spots)
            cat_subdf = self.applicant_pool.candidate_df[self.applicant_pool.candidate_df.category == cat]
            cat_indices = cat_subdf.index
            # get intersection of category and non-dra
            target_indices = np.intersect1d(non_dra_indices, cat_indices)
            # are there more targets than remaining spots?
            # remove anyone who can't participate if this is the case
            if len(target_indices) > total_remaining_spots:
                target_indices = np.random.choice(target_indices, total_remaining_spots)
            # apply algorithm to targets
            # first allocate everyone to their first preference
            target_firsts = (self.preferences_matrix[target_indices] == 0).astype(np.int32)
            self.allocation_matrix[target_indices] = target_firsts
            # update the spot counts
            self._update_spots()
            # print("Spots:", self._hospital_spots)
            # which hospitals are oversubscribed?
            oversubscribed = np.where(self._hospital_spots < 0)[0]
            # undersubscribed does not include full
            undersubscribed = np.where(self._hospital_spots > 0)[0]
            while len(oversubscribed) > 0:
                # choose a random target candidate from an oversubscribed hospital
                oversubscribed_hospital_places = self.allocation_matrix[np.ix_(target_indices, oversubscribed)]
                chosen_target = target_indices[np.random.choice(np.where(np.any(oversubscribed_hospital_places == 1, axis=1))[0], 1)[0]]
                # find their highest undersubscribed preference
                realloc_destination = np.argmin(self.preferences_matrix[chosen_target, undersubscribed])
                # print(chosen_target, realloc_destination)
                # print("Undersubscribed preferences:", self.preferences_matrix[chosen_target, undersubscribed])
                # reallocate them to that hospital
                self.allocation_matrix[chosen_target] = one_array(len(self._hospital_spots), undersubscribed[realloc_destination])
                # print("New allocation:", self.allocation_matrix[chosen_target])
                self._update_spots()
                # print("Spots:", self._hospital_spots)
                oversubscribed = np.where(self._hospital_spots < 0)[0]
                undersubscribed = np.where(self._hospital_spots > 0)[0]
            # print("Cat {0} finished".format(cat+1))
        self._detranslate()