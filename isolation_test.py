# %%
from heti_stack.sim import CategoricalSimulation

import numpy as np

test_cat_sim = CategoricalSimulation([
    ("stack with random top", 0.50),
    ("random", 0.50)
])

# print(test_cat_sim.hospitals.hospital_df)
# print(stack)
test_cat_sim.run(dra_prefill=True)

alloc_arr = test_cat_sim.allocation_matrix
ones_per_row = np.count_nonzero(alloc_arr, axis=1)
passed = (ones_per_row <= 1).all()
if passed:
    print("Categorical simulation works properly")
# %%