from IPython.display import display, HTML

from .sim import AnnealSimulation, CategoricalSimulation

def make_sim(starting_strategies: "list[tuple[str, float]]", mode="anneal", percentify=True, **kwargs):
    """
    Create and run simulation without generating stats and plots
    """
    sim = None
    if mode == "anneal":
        sim = AnnealSimulation(starting_strategies, **kwargs)
    elif mode == "categorical":
        sim = CategoricalSimulation(starting_strategies, **kwargs)
    
    sim.run()
    return sim

def create_and_run_sim(starting_strategies: "list[tuple[str, float]]", mode="anneal", percentify=True, **kwargs):
    """
    This function needs to be renamed, will do so once I rename everything in the ipynb
    Create and run a simulation. Plus, generate plots and statistics afterwards.

    Basically a convenience macro for running a test from within a Jupyter notebook.
    """
    sim = make_sim(starting_strategies, mode, percentify, **kwargs)
    generate_sim_stats(sim, mode, percentify)
    return sim

def generate_sim_stats(sim, mode="anneal", percentify=True):
    if mode == "anneal":
        sim.plot_convergence()

    cfilters = [None, "wanted top 4 hospital", "got top 4 hospital", "wanted top 6 hospital", "got top 6 hospital"]
    cpool = sim.applicant_pool

    for f in cfilters:
        display(HTML("<h5>Filter: {0}</h5>".format(f)))
        cpool.plot_all_separated(use_filter=f, percentify=percentify)
        cpool.plot_all_unseparated(use_filter=f, percentify=percentify)
        cpool.plot_every_category(use_filter=f, percentify=percentify)

        if len(sim.starting_strategies) > 1:
            display(HTML("<p>Kruskal test comparing overall happiness between strategy subgroups</p>"))
            kruskal_stat, kruskal_p = cpool.compare_all_subgroups(use_filter=f, percentify_plot=percentify)
            display(HTML("""
            <ul>
                <li>Kruskal test statistic: {stat}</li>
                <li><em>p</em> value: {pval}</li>
            </ul>""".format(stat=kruskal_stat, pval=kruskal_p if kruskal_p > 0.05 else "<strong>{0}</strong>".format(kruskal_p))))
            display(HTML("<p>Chi-squared test comparing first preferences obtained in strategy subgroups</p>"))
            chi2_stat, chi2_p, chi2_dof, _ = cpool.compare_all_firsts(use_filter=f, percentify=percentify)
            display(HTML("""
            <ul>
                <li>Chi-squared test statistic: {stat} (with {dof} dof)</li>
                <li><em>p</em> value: {pval}</li>
            </ul>""".format(stat=chi2_stat, dof=chi2_dof, pval=chi2_p if chi2_p > 0.05 else "<strong>{0}</strong>".format(chi2_p))))

def make_single_strategy_simulation(strategy: str, mode="anneal", percentify=True, **kwargs):
    starting_strategies = [(strategy, 1.0)]
    make_sim(starting_strategies, mode=mode, percentify=percentify, **kwargs)

def create_and_run_single_strategy_simulation(strategy: str, mode="anneal", percentify=True, **kwargs):
    starting_strategies = [(strategy, 1.0)]
    create_and_run_sim(starting_strategies, mode=mode, percentify=percentify, **kwargs)