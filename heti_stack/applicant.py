from functools import reduce
from operator import itemgetter

from IPython.display import display, HTML
from scipy import stats
from textwrap import wrap
from uuid import uuid4

from .base import category_counts
from .hospital import stack
from .utils import is_iterable, ordinal
from .strategies import Hospitals

import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd

plt.style.use("ggplot")
plt.rcParams["font.family"] = "Helvetica Neue"

hospital_generator = Hospitals()
hospital_list = hospital_generator.hospitals

# Applicant base class

class Applicant(object):
    """An applicant is assumed to have two properties that count in the algorithm:
    - Order of preferences
    - Category"""
    def __init__(self, strategy: str, category: int, hospitals_obj) -> None:
        self.uid = uuid4()
        self.strategy = strategy
        self.preferences = hospitals_obj.array_from_strategy(strategy)
        self.category = category
        self.allocation = None
        self.preference_number = None
        self.is_dra = False
    def __repr__(self) -> str:
        return self.__str__()
    def __str__(self) -> str:
        if not self.allocation:
            return "Unallocated Category {cat} applicant".format(cat=self.category+1)
        else:
            return "Category {cat} applicant allocated to {alloc} ({prefn}): {prefs}".format(
                cat=self.category+1,
                alloc=self.allocation,
                prefn=self.preference_number,
                prefs=self.preferences
            )
    def as_dict(self) -> dict:
        return {
            "uid": self.uid,
            "strategy": self.strategy,
            "preferences": self.preferences,
            "first_preference": self.preferences[0],
            "category": self.category,
            "allocation": self.allocation,
            "preference_number": self.preference_number,
            "is_dra": self.is_dra
        }
    def allocate(self, hospital, dra_prefill=False):
        self.allocation = hospital
        self.is_dra = dra_prefill
        self.preference_number = np.where(self.preferences == self.allocation)

class ApplicantPool(object):
    """
    Applicant pool container object.

    Does many of the functions in the original stack simulator.

    Applicants are stored in a dictionary but most of the work is done on a pandas df containing their uids

    Strategies with proportions are given as a dictionary containing:
    - The strategy name
    - The proportion of the applicants who will use that strategy
    """

    indices = [ordinal(n) for n in range(1, len(hospital_generator.hospitals)+1)]+["placed", "not_placed", "total"]
    candidate_cols = ["uid", "strategy", "preferences", "category", "allocation", "preference_number", "is_dra"]

    def __init__(self, strategies: "list[tuple[str, float]]", hospitals_obj) -> None:
        self._hospitals_obj = hospitals_obj
        self.candidates = dict()
        self.candidate_df = pd.DataFrame()
        self._strategies, self._strat_weights = [list(x) for x in list(zip(*strategies))]
        self._populate_candidates()

    def _populate_candidates(self) -> None:
        pre_df = []
        for i in range(len(category_counts)):
            cat, count = i, category_counts[i]
            for j in range(count):
                chosen_strategy = np.random.choice(self._strategies, p=self._strat_weights)
                new_candidate = Applicant(chosen_strategy, cat, self._hospitals_obj)
                self.candidates[new_candidate.uid] = new_candidate
                pre_df.append(new_candidate.as_dict())
        self.candidate_df = pd.DataFrame(pre_df)

    # Pool filtration functions, not used
    # TODO: delete these and create equivalent utility functions for np.ndarrays in sim.py

    @staticmethod
    def _candidate_filter(df, query: str) -> "pd.DataFrame":
        """
        Filter reducer for candidates similar to the strategy reducer

        Replaces the original filtration functions like got top 6, top 4, etc.
        """
        q = query.lower().strip()
        wanted_re = re.compile(r"wanted (?:a )?top (\d+) hospital")
        got_re = re.compile(r"got (?:a )?top (\d+) hospital")
        if wanted_re.match(q):
            matched = wanted_re.match(q)
            top_cap = int(matched.group(1))
            return df[df.first_preference.isin(stack[:top_cap])]
        elif got_re.match(q):
            matched = got_re.match(q)
            top_cap = int(matched.group(1))
            return df[df.allocation.isin(stack[:top_cap])]
        else:
            raise NotImplementedError("This filter hasn't been implemented yet!")

    def satisfied(self, rank, category=None, use_filter=None, exclude_dra=False):
        to_ret = self.candidate_df[(self.candidate_df["preference_number"] == rank) &
            (self.candidate_df["category"] == category if category != None else True) &
            (~self.candidate_df["is_dra"] if exclude_dra else True)]
        if use_filter:
            to_ret = self._candidate_filter(to_ret, use_filter)
        return to_ret

    def placed(self, category=None, use_filter=None, exclude_dra=False):
        to_ret = self.candidate_df[(self.candidate_df["allocation"] != -1) &
            (self.candidate_df["category"] == category if category != None else True) &
            (~self.candidate_df["is_dra"] if exclude_dra else True)]
        if use_filter:
            to_ret = self._candidate_filter(to_ret, use_filter)
        return to_ret

    def unplaced(self, category=None, use_filter=None, exclude_dra=False):
        to_ret = self.candidate_df[(self.candidate_df["allocation"] == -1) &
            (self.candidate_df["category"] == category if category != None else True) &
            (~self.candidate_df["is_dra"] if exclude_dra else True)]
        if use_filter:
            to_ret = self._candidate_filter(to_ret, use_filter)
        return to_ret

    def dra_only(self):
        return self.candidate_df[(self.candidate_df["is_dra"] == True)]

    def non_dra_only(self):
        return self.candidate_df[(self.candidate_df["is_dra"] == False)]

    def stratify_applicants_by_strategy(self):
        return self.candidate_df.set_index("strategy", append=True).reorder_levels([1,0]).sort_index()

    def plot_all_unseparated(self, use_filter=None, percentify=False, exclude_dra=False):
        cdf = self.placed(use_filter=use_filter, exclude_dra=exclude_dra)
        allgrouped = cdf.groupby(["preference_number"]).count()["uid"]
        if percentify:
            allgrouped *= 100. / len(cdf)
        allgrouped = self._make_plottable(allgrouped.to_frame())
        allgrouped.plot.bar(rot=30, legend=None)
        self._plot("Applicants who got their nth preference",
            "% of all placed applicants" if percentify else "Count",
            "Satisfied applicants (all)", percentify=percentify)

    def plot_all_separated(self, use_filter=None, percentify=False, exclude_dra=False):
        cdf = self.placed(use_filter=use_filter, exclude_dra=exclude_dra)
        subgrouped = cdf.groupby(["preference_number", "category"]).count()["uid"]
        if percentify:
            subgrouped *= 100. / len(cdf)
        unstacked = subgrouped.unstack()
        unstacked = self._make_plottable(unstacked)
        unstacked.plot.bar(rot=30)
        plt.legend(labels=unstacked.columns + 1)
        self._plot("Applicants who got their nth preference",
            "% of all placed applicants" if percentify else "Count",
            "Satisfied applicants (by category)", percentify=percentify)
    
    def plot_single_category(self, cat, use_filter=None, percentify=False, exclude_dra=False):
        cat_df = self.placed(category=cat, use_filter=use_filter, exclude_dra=exclude_dra)
        catgroup = cat_df.groupby(["preference_number"]).count()["uid"]
        if percentify:
            catgroup *= 100. / len(cat_df)
        catgroup = self._make_plottable(catgroup.to_frame())
        catgroup.plot.bar(rot=30, legend=None)
        self._plot("Applicants who got their nth preference",
            "% of placed applicants in category" if percentify else "Count",
            "Satisfied category {cat} applicants".format(cat=cat+1), percentify=percentify)
    
    def plot_every_category(self, use_filter=None, percentify=False, exclude_dra=False):
        for cat in self.placed(use_filter=use_filter)["category"].unique():
            self.plot_single_category(cat, use_filter, percentify, exclude_dra)

    @staticmethod
    def _plot(xlab, ylab, title, filename="", percentify=False):
        if not filename:
            filename = title
        if percentify:
            plt.yticks(np.arange(0, 101, 10))
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title('\n'.join(wrap(title, 60)))
        plt.tight_layout()
        plt.show()
        # plt.clf()
        # plt.cla()
        # plt.close('all')

    @staticmethod
    def _make_plottable(df):
        append_rows = pd.DataFrame(np.nan, index=np.setdiff1d(np.arange(15), np.array(df.index)), columns=df.columns)
        df = df.append(append_rows)
        df = df.sort_index()
        df = df.set_index(pd.Index([ordinal(n+1) for n in df.index]))
        return df

    def compare_two_subgroups(self, groups, use_filter=None, cat=None, percentify_plot=False, exclude_dra=False):
        """
        Run Mann-Whitney U test between two specified strategy subgroups
        Also allows for filters and composite subgroups i.e. adding >1 subgroup into one
        Should exclude unallocated applicants by default
        Both cols should be passed as lists
        stats.mannwhitneyu(a,b)
        """
        # TODO: implement compound groups i.e. merge two groups together and treat them as one to compare
        cdf = self.placed(category=cat, use_filter=use_filter, exclude_dra=exclude_dra)
        gb = cdf.groupby(["strategy"])
        new_df = self.make_groups(cdf, gb, groups)
        new_gb = new_df.groupby(["group"])
        to_compare = [new_gb.get_group(g)["preference_number"] for g in new_gb.groups]
        # needs to be graphed for a good visual comparison
        subgrouped = new_df.groupby(["preference_number", "group"]).count()["uid"]
        unstacked = subgrouped.unstack()
        avg_unhappiness = unstacked.mul(unstacked.index, axis=0, fill_value=0).sum()/unstacked.sum()
        display(avg_unhappiness.to_frame(name="Average unhappiness"))
        # fill the unstacked df with any missing values
        unstacked = self._make_plottable(unstacked)
        if percentify_plot:
            unstacked *= 100. / unstacked.sum()
        unstacked.plot.bar(rot=30)
        self._plot("Applicants who got their nth preference",
            "% of strategy subgroup" if percentify_plot else "Count",
            "Satisfied applicants (by group)", percentify=percentify_plot)
        return stats.mannwhitneyu(*to_compare)
    
    def compare_all_subgroups(self, use_filter=None, cat=None, percentify_plot=False, exclude_dra=False):
        """
        Runs Kruskal-Wallis H test between all strategy subgroups
        stats.kruskal(a,b,c)
        """
        cdf = self.placed(category=cat, use_filter=use_filter, exclude_dra=exclude_dra)
        gb = cdf.groupby(["strategy"])
        to_compare = [gb.get_group(g)["preference_number"] for g in gb.groups]
        # needs to be graphed for a good visual comparison
        subgrouped = cdf.groupby(["preference_number", "strategy"]).count()["uid"]
        unstacked = subgrouped.unstack()
        avg_unhappiness = unstacked.mul(unstacked.index, axis=0, fill_value=0).sum()/unstacked.sum()
        display(avg_unhappiness.to_frame(name="Average unhappiness"))
        # fill the unstacked df with any missing values
        append_rows = pd.DataFrame(np.nan, index=np.setdiff1d(np.arange(15), np.array(unstacked.index)), columns=unstacked.columns)
        unstacked = unstacked.append(append_rows)
        unstacked = unstacked.sort_index()
        unstacked = unstacked.set_index(pd.Index([ordinal(n+1) for n in unstacked.index]))
        if percentify_plot:
            unstacked *= 100. / unstacked.sum()
        unstacked.plot.bar(rot=30)
        self._plot("Applicants who got their nth preference",
            "% of strategy subgroup" if percentify_plot else "Count",
            "Satisfied applicants (by group)", percentify=percentify_plot)
        return stats.kruskal(*to_compare)

    def compare_two_firsts(self, groups, use_filter=None, cat=None, percentify=False, exclude_dra=False):
        """
        Runs chi-squared test between number of first preferences obtained in two specified strategy subgroups
        stats.chi2_contingency(chi2_table)
        """
        cdf = self.placed(category=cat, use_filter=use_filter, exclude_dra=exclude_dra)
        gb = cdf.groupby(["strategy"])
        new_df = self.make_groups(cdf, gb, groups)
        new_gb = new_df.groupby(["group"])
        to_compare = [(new_gb.get_group(g)["preference_number"] == 0).value_counts() for g in new_gb.groups]
        # not clean but it works
        contingency_table = np.array(list(map(lambda s: [s.get(True, 0), s.get(False, 0)], to_compare))).T
        contingency_df = pd.DataFrame(contingency_table, columns=[g for g in new_gb.groups], index=["Got first preference", "Did not get first preference"])
        if percentify:
            contingency_df_percents = contingency_df/contingency_df.sum()
            display(contingency_df_percents.style.format("{:.2%}"))
        else:
            display(contingency_df)
        return stats.chi2_contingency(contingency_table)

    def compare_all_firsts(self, use_filter=None, cat=None, percentify=False, exclude_dra=False):
        """
        Runs chi-squared test between number of first preferences obtained in all strategy subgroups
        stats.chi2_contingency(chi2_table)
        """
        cdf = self.placed(category=cat, use_filter=use_filter, exclude_dra=exclude_dra)
        gb = cdf.groupby(["strategy"])
        to_compare = [(gb.get_group(g)["preference_number"] == 0).value_counts() for g in gb.groups]
        # not clean but it works
        contingency_table = np.array(list(map(lambda s: [s.get(True, 0), s.get(False, 0)], to_compare))).T
        contingency_df = pd.DataFrame(contingency_table, columns=[g for g in gb.groups], index=["Got first preference", "Did not get first preference"])
        if percentify:
            contingency_df_percents = contingency_df/contingency_df.sum()
            display(contingency_df_percents.style.format("{:.2%}"))
        else:
            display(contingency_df)
        return stats.chi2_contingency(contingency_table)

    @staticmethod
    def make_groups(df, gb, groups):
        """
        Similar to pd.get_group but allows using lists to combine several groups
        """
        # first convert all groups to lists for safety
        groups_list = [[i] if type(i) != list else i for i in groups]
        df["group"] = ""
        for g in groups_list:
            group_name = "+".join(g)
            index = itemgetter(*g)(gb.groups)
            # either a tuple of indices or a single index
            if type(index) == tuple:
                index = reduce(lambda a, b: a.union(b), index)
            df.loc[index, "group"] = group_name
        return df
