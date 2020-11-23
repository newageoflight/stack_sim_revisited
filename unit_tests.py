# %%
from heti_stack.sim import CategoricalSimulation

import numpy as np

test_cat_sim = CategoricalSimulation([
    ("stack with random top", 0.25),
    ("variant stack", 0.25),
    ("random", 0.50)
])

# print(test_cat_sim.hospitals.hospital_df)
# print(stack)
%time test_cat_sim.run(dra_prefill=True)

alloc_arr = test_cat_sim.allocation_matrix
ones_per_row = np.count_nonzero(alloc_arr, axis=1)
passed = (ones_per_row <= 1).all()
if passed:
    print("Categorical simulation works properly")

cpool = test_cat_sim.applicant_pool
cpool.compare_two_subgroups([["stack with random top", "variant stack"], "random"])
# %%

#%%
from heti_stack.sim import AnnealSimulation

import numpy as np

test_anneal_gpu_sim = AnnealSimulation([
    ("stack", 0.50),
    ("random", 0.50)
], backend="gpu")

test_anneal_cpu_sim = AnnealSimulation([
    ("stack", 0.50),
    ("random", 0.50)
], backend="cpu")

print("testing gpu\n---")
%time test_anneal_gpu_sim.run()
print("\n\n")
print("testing cpu\n---")
%time test_anneal_cpu_sim.run()
print("\n\n")

test_anneal_gpu_sim.plot_convergence()
test_anneal_cpu_sim.plot_convergence()
print("If the graphs look correct, then the anneal sim works properly")
#%%

# %%
from heti_stack.sim import QBSimulation

sim = QBSimulation([
    ("stack", 1.0)
])

%time sim.run()

cpool = sim.applicant_pool
cdf = cpool.candidate_df
# %%

#%%
import pandas as pd

df = pd.DataFrame([("stack", 0.5), ("random", 0.5)], columns=["strategy", "percentage"])
df.style.format({"percentage": "{:.2%}"})
#%%
#%%
from heti_stack.tests import compare_unhappiness_for_multiple_sims

df = compare_unhappiness_for_multiple_sims([
    [("stack", 1.0)],
    [("random", 1.0)]
], "categorical", "anneal", "qb")
df.groupby(["starting_strategies", "algorithm"])
#%%
#%%
from heti_stack.tests import create_and_run_single_strategy_qb_simulation


create_and_run_single_strategy_qb_simulation("stack")
#%%