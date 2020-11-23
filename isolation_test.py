# %%
from heti_stack.sim import QBSimulation

sim = QBSimulation([
    ("random with weighted random top", 0.1),
    ("variant stack", 0.3),
    ("stack", 0.6)
])

%time sim.run()

cpool = sim.applicant_pool
cdf = cpool.candidate_df
# %%