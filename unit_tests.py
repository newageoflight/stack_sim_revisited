# %%
from heti_stack.sim import CategoricalSimulation

import numpy as np

test_cat_sim = CategoricalSimulation([
    ("stack", 0.50),
    ("random", 0.50)
])

# print(test_cat_sim.hospitals.hospital_df)
# print(stack)
%time test_cat_sim.run()

alloc_arr = test_cat_sim.allocation_matrix
ones_per_row = np.count_nonzero(alloc_arr, axis=1)
passed = (ones_per_row <= 1).all()
if passed:
    print("Categorical simulation works properly")
# %%

#%%
from heti_stack.sim import AnnealSimulation

import numpy as np

test_anneal_sim = AnnealSimulation([
    ("stack", 0.50),
    ("random", 0.50)
], cool_rate=0.003)

%time test_anneal_sim.run()

test_anneal_sim.plot_convergence()
print("If the graph looks correct, then the anneal sim works properly")
#%%