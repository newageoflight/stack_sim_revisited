from .sim import AnnealSimulation, CategoricalSimulation

def create_and_run_sim(starting_strategies: "list[tuple[str, float]]", mode="anneal", **kwargs):
    """
    Create and run a simulation. Plus, generate plots and statistics afterwards.

    Basically a convenience macro for running a test from within a Jupyter notebook.
    """

    sim = None
    if mode == "anneal":
        sim = AnnealSimulation(starting_strategies, **kwargs)
    elif mode == "categorical":
        sim = CategoricalSimulation(starting_strategies, **kwargs)
    
    sim.run()

    if mode == "anneal":
        sim.plot_convergence()

    cpool = sim.applicant_pool
    cpool.plot_all_separated()
    cpool.plot_all_unseparated()
    cpool.plot_every_category()

    if len(starting_strategies) > 1:
        cpool.compare_all_subgroups()
        cpool.compare_all_firsts()