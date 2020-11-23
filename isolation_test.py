# %%
from heti_stack.tests import compare_two_groups_all_conditions, create_and_run_qb_sim


sim = create_and_run_qb_sim([
    ("random with weighted random top", 0.1),
    ("variant stack", 0.3),
    ("stack", 0.6)
])

compare_two_groups_all_conditions(sim, ["random with weighted random top", ["variant stack", "stack"]])
# %%