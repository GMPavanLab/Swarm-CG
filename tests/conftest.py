import os
from types import SimpleNamespace

import pytest

import swarmcg
from swarmcg import config
from swarmcg.simulations.runner import ns_to_runner, SimulationStep
from swarmcg.simulations.simulation_steps import Minimisation, Equilibration, Production

TEST_DATA = "tests/data/"
ROOT_DIR = os.path.dirname(swarmcg.__file__)


@pytest.fixture(scope="module")
def ns_opt():

    parameters = {
        "exec_folder": "./MODEL_FOLDER",
        "exec_mode": 1,
        "aa_tpr_filename": f"{TEST_DATA}{config.metavar_aa_tpr}",
        "aa_traj_filename": f"{TEST_DATA}{config.metavar_aa_traj}",
        "cg_map_filename": f"{TEST_DATA}{config.metavar_cg_map}",
        "mapping_type": "COM",
        "cg_itp_filename": f"{TEST_DATA}{config.metavar_cg_itp}",
        "user_input": False,
        "gro_input_filename": f"{TEST_DATA}start_conf.gro",
        "top_input_filename": f"{TEST_DATA}system.top",
        "mdp_minimization_filename": f"{ROOT_DIR}/data/mini.mdp",
        "mdp_equi_filename": f"{ROOT_DIR}/data/equi.mdp",
        "mdp_md_filename": f"{ROOT_DIR}/data/md.mdp",
        "output_folder": ".",
        "out_dir": "",
        "gmx_path": config.gmx_path,
        "nb_threads": 0,
        "mpi_tasks": 0,
        "gpu_id": "",
        "gmx_args_str": "",
        "mini_maxwarn": 0,
        "sim_kill_delay": 60,
        "default_max_fct_bonds_opti": config.default_max_fct_bonds_opti,
        "default_max_fct_angles_opti_f1": config.default_max_fct_angles_opti_f1,
        "default_max_fct_angles_opti_f2": config.default_max_fct_angles_opti_f2,
        "default_abs_range_fct_dihedrals_opti_func_with_mult": config.default_abs_range_fct_dihedrals_opti_func_with_mult,
        "default_abs_range_fct_dihedrals_opti_func_without_mult": config.default_abs_range_fct_dihedrals_opti_func_without_mult,
        "sim_duration_short": 10,
        "sim_duration_long": 25,
        "bonds2angles_scoring_factor": config.bonds2angles_scoring_factor,
        "bw_constraints": config.bw_constraints,
        "bw_bonds": config.bw_bonds,
        "bw_angles": config.bw_angles,
        "bw_dihedrals": config.bw_dihedrals,
        "bonds_max_range": config.bonds_max_range,
        "aa_rg_offset": 0.0,
        "bonds_scaling": config.bonds_scaling,
        "bonds_scaling_str": config.bonds_scaling_str,
        "min_bonds_length": config.min_bonds_length,
        "temp": config.sim_temperature,
        "keep_all_sims": False,
        "v": True
    }

    # add value added in the optimisation process
    added = {
        "cg_itp_basename": parameters["cg_itp_filename"],
        "gro_input_basename": parameters["gro_input_filename"],
        "top_input_basename": parameters["top_input_filename"],
        "mdp_minimization_basename": parameters["mdp_minimization_filename"],
        "mdp_equi_basename": parameters["mdp_equi_filename"],
        "mdp_md_basename": parameters["mdp_md_filename"],
        "mismatch_order": False,
        "row_x_scaling": True,
        "row_y_scaling": True,
        "ncols_max": 0,
        "molname_in": None,
        "process_alive_time_sleep": 10,
        "process_alive_nb_cycles_dead": int(parameters["sim_kill_delay"] / 10),
        "bonds_rescaling_performed": False
    }
    parameters.update(added)

    def ns(**kwargs):
        for k, v in kwargs.items():
            parameters[k] = v
        return SimpleNamespace(**parameters)

    return ns


@pytest.fixture(scope="module")
def mini():
    return Minimisation(f"{ROOT_DIR}/data/mini.mdp")


@pytest.fixture(scope="module")
def equi():
    return Equilibration(f"{ROOT_DIR}/data/equi.mdp")


@pytest.fixture(scope="module")
def md():
    return Production(f"{ROOT_DIR}/data/md.mdp")


@pytest.fixture(scope="module")
def simstep_mini(ns_opt, mini):
    return ns_to_runner(ns_opt(), mini, f"{TEST_DATA}start_conf.gro")


@pytest.fixture(scope="module")
def simstep_equi(ns_opt, equi):
    return ns_to_runner(ns_opt(), equi, f"{TEST_DATA}mini.gro")


@pytest.fixture(scope="module")
def simstep_md(ns_opt, md):
    return ns_to_runner(ns_opt(), md, f"{TEST_DATA}equi.gro")
