from types import SimpleNamespace
import pytest

from swarmcg import config


@pytest.fixture(scope="session")
def ns_opt():
    parameters = {
        "exec_mode": 1,
        "aa_tpr_filename": config.metavar_aa_tpr,
        "aa_traj_filename": config.metavar_aa_traj,
        "cg_map_filename": config.metavar_cg_map,
        "mapping_type": "COM",
        "cg_itp_filename'": config.metavar_cg_itp,
        "user_input": "store_true",
        "gro_input_filename": "start_conf.gro",
        "top_input_filename": "system.top",
        "mdp_minimization_filename": "mini.mdp",
        "mdp_equi_filename": "equi.mdp",
        "mdp_md_filename": "'md.mdp",
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
        "min_bonds_length": config.help_min_bonds_length,
        "temp": config.sim_temperature,
        "keep_all_sims": False,
        "v": True
    }

    def ns(**kwargs):
        for k, v in kwargs.items():
            parameters[k] = v
        return SimpleNamespace(**parameters)

    return ns
