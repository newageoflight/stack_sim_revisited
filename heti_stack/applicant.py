from scipy import stats
from textwrap import wrap
from uuid import uuid4

from .base import category_counts
from .hospital import stack
from .utils import ordinal
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
        wanted_re = re.compile(r"wanted (a )?top (\d+) hospital")
        got_re = re.compile(r"got (a )?top (\d+) hospital")
        if wanted_re.match(q):
            matched = wanted_re.match(q)
            top_cap = int(matched.group(1))
            return df[df.first_preference.isin(stack[:top_cap])]
        elif got_re.match(q):
            matched = wanted_re.match(q)
            top_cap = int(matched.group(1))
            return df[df.allocation.isin(stack[:top_cap])]
        else:
            raise NotImplementedError("This filter hasn't been implemented yet!")

    def satisfied(self, rank, category=None, use_filter=None):
        to_ret = self.candidate_df[(self.candidate_df["preference_number"] == rank) &
            (self.candidate_df["category"] == category if category != None else True)]
        if use_filter:
            to_ret = self._candidate_filter(to_ret, use_filter)
        return to_ret

    def placed(self, category=None, use_filter=None):
        to_ret = self.candidate_df[(self.candidate_df["allocation"] != -1) &
            (self.candidate_df["category"] == category if category != None else True)]
        if use_filter:
            to_ret = self._candidate_filter(to_ret, use_filter)
        return to_ret

    def unplaced(self, category=None, use_filter=None):
        to_ret = self.candidate_df[(self.candidate_df["allocation"] == -1) &
            (self.candidate_df["category"] == category if category != None else True)]
        if use_filter:
            to_ret = self._candidate_filter(to_ret, use_filter)
        return to_ret

    def dra_only(self):
        return self.candidate_df[(self.candidate_df["is_dra"] == True)]

    def non_dra_only(self):
        return self.candidate_df[(self.candidate_df["is_dra"] == False)]

    def stratify_applicants_by_strategy(self):
        return self.candidate_df.set_index("strategy", append=True).reorder_levels([1,0]).sort_index()

    def plot_all_unseparated(self, use_filter=None, percentify=False):
        cdf = self.placed(use_filter=use_filter)
        allgrouped = cdf.groupby(["preference_number"]).count()["uid"]
        if percentify:
            allgrouped *= 100. / len(cdf)
        allgrouped.plot.bar(rot=30)
        self._plot("Applicants who got their nth preference",
            "%" if percentify else "Count",
            "Satisfied applicants (all)", percentify=percentify)

    def plot_all_separated(self, use_filter=None, percentify=False):
        cdf = self.placed(use_filter=use_filter)
        subgrouped = cdf.groupby(["preference_number", "category"]).count()["uid"]
        if percentify:
            subgrouped *= 100. / len(cdf)
        unstacked = subgrouped.unstack()
        unstacked.plot.bar(rot=30)
        plt.legend(labels=[i+1 for i in range(len(unstacked.columns))])
        self._plot("Applicants who got their nth preference",
            "%" if percentify else "Count",
            "Satisfied applicants (by category)", percentify=percentify)
    
    def plot_single_category(self, cat, use_filter=None, percentify=False):
        cat_df = self.placed(category=cat, use_filter=use_filter)
        catgroup = cat_df.groupby(["preference_number"]).count()["uid"]
        if percentify:
            catgroup *= 100. / len(cat_df)
        catgroup.plot.bar(rot=30)
        self._plot("Applicants who got their nth preference",
            "%" if percentify else "Count",
            "Satisfied category {cat} applicants".format(cat=cat+1), percentify=percentify)
    
    def plot_every_category(self, use_filter=None, percentify=False):
        for cat in self.placed(use_filter=use_filter)["category"].unique():
            self.plot_single_category(cat, use_filter, percentify)

    @staticmethod
    def _plot(xlab, ylab, title, filename="", percentify=False):
        if not filename:
            filename = title
        plt.xticks(np.arange(15), [ordinal(n+1) for n in range(15)])
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

    def compare_two_subgroups(self, groups, use_filter=None, cat=None, percentify_plot=False):
        """
        Run Mann-Whitney U test between two specified strategy subgroups
        Also allows for filters and composite subgroups i.e. adding >1 subgroup into one
        Should exclude unallocated applicants by default
        Both cols should be passed as lists
        stats.mannwhitneyu(a,b)
        """
        cdf = self.placed(category=cat, use_filter=use_filter)
        if cat is not None:
            cdf = cdf[cdf["category"] == cat]
        gb = cdf.groupby(["strategy"])
        to_compare = [gb.get_group(g)["preference_number"] for g in groups]
        # needs to be graphed for a good visual comparison
        subgrouped = cdf.groupby(["strategy", "category"]).count()["uid"]
        if percentify_plot:
            subgrouped *= 100. / len(cdf)
        unstacked = subgrouped.unstack()[groups]
        unstacked.plot.bar(rot=30)
        plt.legend(labels=[g for g in gb.groups])
        self._plot("Applicants who got their nth preference",
            "%" if percentify_plot else "Count",
            "Satisfied applicants (by category)", percentify=percentify_plot)
        # make it so that each strategy gets their own bar
        return stats.mannwhitneyu(*to_compare)
    
    def compare_all_subgroups(self, use_filter=None, cat=None, percentify_plot=False):
        """
        Runs Kruskal-Wallis H test between all strategy subgroups
        stats.kruskal(a,b,c)
        """
        cdf = self.placed(category=cat, use_filter=use_filter)
        gb = cdf.groupby(["strategy"])
        to_compare = [gb.get_group(g)["preference_number"] for g in gb.groups]
        # needs to be graphed for a good visual comparison
        subgrouped = cdf.groupby(["strategy", "category"]).count()["uid"]
        if percentify_plot:
            subgrouped *= 100. / len(cdf)
        unstacked = subgrouped.unstack()
        unstacked.plot.bar(rot=30)
        plt.legend(labels=[g for g in gb.groups])
        self._plot("Applicants who got their nth preference",
            "%" if percentify_plot else "Count",
            "Satisfied applicants (by strategy)", percentify=percentify_plot)
        # each strategy gets their own bar
        # TODO: fix this, doesn't work as intended
        return stats.kruskal(*to_compare)

    def compare_two_firsts(self, groups, use_filter=None, cat=None):
        """
        Runs chi-squared test between number of first preferences obtained in two specified strategy subgroups
        stats.chi2_contingency(chi2_table)
        """
        cdf = self.placed(category=cat, use_filter=use_filter)
        if cat is not None:
            cdf = cdf[cdf["category"] == cat]
        gb = cdf.groupby(["strategy"])
        to_compare = [(gb.get_group(g)["preference_number"] == 0).value_counts() for g in groups]
        contingency_table = np.array(to_compare)[:, ::-1].T
        print(contingency_table)
        # TODO: make contingency table into pd.df for prettier printing
        return stats.chi2_contingency(contingency_table)

    def compare_all_firsts(self, use_filter=None, cat=None):
        """
        Runs chi-squared test between number of first preferences obtained in all strategy subgroups
        stats.chi2_contingency(chi2_table)
        """
        cdf = self.placed(category=cat, use_filter=use_filter)
        if cat is not None:
            cdf = cdf[cdf["category"] == cat]
        gb = cdf.groupby(["strategy"])
        to_compare = [(gb.get_group(g)["preference_number"] == 0).value_counts() for g in gb.groups]
        contingency_table = np.array(to_compare)[:, ::-1].T
        print(contingency_table)
        return stats.chi2_contingency(contingency_table)