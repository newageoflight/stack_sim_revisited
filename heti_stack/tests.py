from .sim import AnnealSimulation, CategoricalSimulation

def create_and_run_sim(starting_strategies: "list[tuple[str, float]]", mode="anneal", percentify=True, **kwargs):
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
    cpool.plot_all_separated(percentify=percentify)
    cpool.plot_all_unseparated(percentify=percentify)
    cpool.plot_every_category(percentify=percentify)

    if len(starting_strategies) > 1:
        kruskal_stat, kruskal_p = cpool.compare_all_subgroups()
        print("Kruskal test statistic: {stat}\np value: {pval}".format(stat=kruskal_stat, pval=kruskal_p))
        chi2_stat, chi2_p, chi2_dof, _ = cpool.compare_all_firsts()
        print("Chi-squared test statistic: {stat} (with {dof} dof)\np value: {pval}".format(stat=chi2_stat, dof=chi2_dof, pval=chi2_p))