# %%
from heti_stack.sim import QBSimulation

sim = QBSimulation([
    ("stack", 0.50),
    ("random", 0.50)
])

%time sim.run()
# %%