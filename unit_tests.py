# %%
from heti_stack.sim import CategoricalSimulation

import numpy as np

test_cat_sim = CategoricalSimulation([
    ("stack with random top", 0.50),
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
