from IPython.display import display, HTML

from .sim import AnnealSimulation, CategoricalSimulation, QBSimulation

def make_sim(starting_strategies: "list[tuple[str, float]]", mode="anneal", percentify=True, **kwargs):
    """
    Create and run simulation without generating stats and plots
    """
    sim = None
    if mode == "anneal":
        sim = AnnealSimulation(starting_strategies, **kwargs)
    elif mode == "categorical":
        sim = CategoricalSimulation(starting_strategies, **kwargs)
    elif mode == "qb":
        sim = QBSimulation(starting_strategies, **kwargs)
    
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

def create_and_run_qb_sim(starting_strategies: "list[tuple[str, float]]", percentify=True, **kwargs):
    return create_and_run_sim(starting_strategies, mode="qb", percentify=percentify, **kwargs)

def generate_sim_stats(sim, mode="anneal", percentify=True):
    if mode == "anneal":
        sim.plot_convergence()

    dra_toggle = [False, True]
    cfilters = [None, "wanted top 4 hospital", "got top 4 hospital", "wanted top 6 hospital", "got top 6 hospital"]
    cpool = sim.applicant_pool

    for d in dra_toggle:
        display(HTML("<h5>DRA {0}excluded</h5>".format("not " if not d else "")))
        for f in cfilters:
            display(HTML("<h6>Filter: {0}</h6>".format(f)))
            cpool.plot_all_separated(use_filter=f, percentify=percentify, exclude_dra=d)
            cpool.plot_all_unseparated(use_filter=f, percentify=percentify, exclude_dra=d)
            cpool.plot_every_category(use_filter=f, percentify=percentify, exclude_dra=d)

            if len(sim.starting_strategies) > 1:
                display(HTML("<p>Kruskal test comparing overall happiness between strategy subgroups</p>"))
                kruskal_stat, kruskal_p = cpool.compare_all_subgroups(use_filter=f, percentify_plot=percentify, exclude_dra=d)
                display(HTML("""
                <ul>
                    <li>Kruskal test statistic: {stat}</li>
                    <li><em>p</em> value: {pval}</li>
                </ul>""".format(stat=kruskal_stat, pval=kruskal_p if kruskal_p > 0.05 else "<strong>{0}</strong>".format(kruskal_p))))
                display(HTML("<p>Chi-squared test comparing first preferences obtained in strategy subgroups</p>"))
                chi2_stat, chi2_p, chi2_dof, _ = cpool.compare_all_firsts(use_filter=f, percentify=percentify, exclude_dra=d)
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

def compare_two_groups_util(sim, groups, use_filter=None, percentify=True, exclude_dra=False):
    """
    Utility for creating nice, formatted subgroup comparisons in a simulation where multiple subgroups exist
    """
    if len(sim.starting_strategies) > 1:
        cpool = sim.applicant_pool
        display(HTML("<p>Mann-Whitney U test comparing overall happiness between two strategy subgroups</p>"))
        mwu_stat, mwu_p = cpool.compare_two_subgroups(groups, use_filter=use_filter, percentify_plot=percentify, exclude_dra=exclude_dra)
        display(HTML("""
        <ul>
            <li>Mann-Whitney U test statistic: {stat}</li>
            <li><em>p</em> value: {pval}</li>
        </ul>""".format(stat=mwu_stat, pval=mwu_p if mwu_p > 0.05 else "<strong>{0}</strong>".format(mwu_p))))
        display(HTML("<p>Chi-squared test comparing first preferences obtained in two strategy subgroups</p>"))
        chi2_stat, chi2_p, chi2_dof, _ = cpool.compare_two_firsts(groups, use_filter=use_filter, percentify=percentify, exclude_dra=exclude_dra)
        display(HTML("""
        <ul>
            <li>Chi-squared test statistic: {stat} (with {dof} dof)</li>
            <li><em>p</em> value: {pval}</li>
        </ul>""".format(stat=chi2_stat, dof=chi2_dof, pval=chi2_p if chi2_p > 0.05 else "<strong>{0}</strong>".format(chi2_p))))
    else:
        raise Exception("This simulation doesn't have subgroups to compare")

def compare_two_groups_all_conditions(sim, groups, percentify=True):
    display(HTML("<h5>Compare groups: {0} vs {1}</h5>".format(*["+".join(e) if type(e) == list else e for e in groups])))
    dra_switches = [False, True]
    cfilters = [None, "wanted top 4 hospital", "got top 4 hospital", "wanted top 6 hospital", "got top 6 hospital"]
    for dra in dra_switches:
        display(HTML("<h5>DRA {0}excluded</h5>".format("not " if not dra else "")))
        for f in cfilters:
            display(HTML("<h6>Filter: {0}</h6>".format(f)))
            compare_two_groups_util(sim, groups, f, percentify=percentify, exclude_dra=dra)
            