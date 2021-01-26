import numpy as np

from swarmcg.shared import exceptions


def get_settings(ns):
    if ns.sim_type == "OPTIMAL":
        return _optimal(ns)
    elif ns.sim_type == "FAST":
        return _fast(ns)
    elif ns.sim_type == "TEST":
        return _test(ns)
    else:
        msg = f"Simulation type {ns.sim_type} is not valid. (OPTIMAL, FAST, TEST)"
        raise exceptions.InvalidArgument(msg)


def _optimal(ns):
    """OPTIMAL Simulation strategy
    Should be fine with any type of molecule, big or small,
    as long as the BI keeps yielding close enough results, which should be the case"""
    sim_types = {
        0: {'sim_duration': ns.sim_duration_short,
            'max_swarm_iter': int(round(6+np.sqrt(ns.cg_itp["nb_constraints"]+ns.cg_itp["nb_bonds"]+ns.cg_itp["nb_angles"]))),
            'max_swarm_iter_without_new_global_best': 6,
            'val_guess_fact': 1,
            'fct_guess_fact': 0.40},
        1: {'sim_duration': ns.sim_duration_short, 'max_swarm_iter': int(round(6+np.sqrt(ns.cg_itp["nb_angles"]+ns.cg_itp["nb_dihedrals"]))),
            'max_swarm_iter_without_new_global_best': 6,
            'val_guess_fact': 0.25,
            'fct_guess_fact': 0.30},
        2: {'sim_duration': ns.sim_duration_long,
            'max_swarm_iter': int(round(6+np.sqrt(ns.cg_itp["nb_constraints"]+ns.cg_itp["nb_bonds"]+ns.cg_itp["nb_angles"]+ns.cg_itp["nb_dihedrals"]))),
            'max_swarm_iter_without_new_global_best': 6,
            'val_guess_fact': 0.25,
            'fct_guess_fact': 0.20}
    }
    opti_cycles = [['constraint', 'bond', 'angle'], ['angle', 'dihedral'], ['constraint', 'bond', 'angle', 'dihedral']]
    sim_cycles = [0, 1, 2]  # simulations types
    return sim_types, opti_cycles, sim_cycles


def _fast(ns):
    """ Simulation strategy FAST -
    Suited for small molecules or rapid optimization"""
    sim_types = {
        0: {'sim_duration': 10, 'max_swarm_iter': 10, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 1, 'fct_guess_fact': 0.40},
        1: {'sim_duration': 10, 'max_swarm_iter': 10, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.30},
        2: {'sim_duration': 15, 'max_swarm_iter': 15, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.25}
    }
    opti_cycles = [['constraint', 'bond', 'angle'], ['dihedral'], ['constraint', 'bond', 'angle', 'dihedral']]
    sim_cycles = [0, 1, 2] # simulations types
    return sim_types, opti_cycles, sim_cycles


def _test(ns):
    """ Simulation strategy TEST -
    Suited for test"""
    sim_types = {
        0: {'sim_duration': 10, 'max_swarm_iter': 1, 'max_swarm_iter_without_new_global_best': 1, 'val_guess_fact': 1, 'fct_guess_fact': 0.40},
        1: {'sim_duration': 10, 'max_swarm_iter': 1, 'max_swarm_iter_without_new_global_best': 1, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.30},
        2: {'sim_duration': 15, 'max_swarm_iter': 1, 'max_swarm_iter_without_new_global_best': 1, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.25}
    }
    opti_cycles = [['constraint', 'bond', 'angle'], ['dihedral'], ['constraint', 'bond', 'angle', 'dihedral']]
    sim_cycles = [0, 1, 2] # simulations types
    return sim_types, opti_cycles, sim_cycles