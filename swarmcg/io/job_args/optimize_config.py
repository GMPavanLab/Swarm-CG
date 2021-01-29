from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS

from swarmcg.shared import styling
from swarmcg.io.job_args import defaults


def get_optimize_args():

    print(styling.header_package("                    Module: CG model optimization\n"))

    formatter = lambda prog: RawTextHelpFormatter(prog, width=135, max_help_position=52)
    args_parser = ArgumentParser(
        description=styling.OPTIMISE_DESCR,
        formatter_class=formatter,
        add_help=False,
        usage=SUPPRESS
    )

    # TODO: handle trajectories for which no box informations are provided
    # TODO: explain what is modified in the MDP
    # TODO: explain module analyze_opti_moves.py can be used to monitor optimization at any point of the process
    # TODO: end the help message by a new frame with examples from the demo data

    req_args_header = styling.sep_close + "\n|                                     REQUIRED ARGUMENTS                                      |\n" + styling.sep_close
    opt_args_header = styling.sep_close + "\n|                                     OPTIONAL ARGUMENTS                                      |\n" + styling.sep_close
    bullet = " "

    optional_args0 = args_parser.add_argument_group(
    req_args_header + "\n\n" + bullet + "EXECUTION MODE")
    for arg in ["exec_mode", "sim_type"]:
        optional_args0.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    required_args = args_parser.add_argument_group(bullet + "REFERENCE AA MODEL")
    for arg in ["aa_tpr", "aa_traj", "cg_map", "mapping"]:
        required_args.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    sim_filenames_args = args_parser.add_argument_group(bullet + "CG MODEL OPTIMIZATION")
    for arg in ["cg_itp", "user_params", "cg_gro", "cg_top", "cg_mdp_mini", "cg_mdp_equi", "cg_mdp_md"]:
        sim_filenames_args.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args4 = args_parser.add_argument_group(opt_args_header + "\n\n" + bullet + "FILES HANDLING")
    for arg in ["in_dir", "out_dir"]:
        optional_args4.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args1 = args_parser.add_argument_group(bullet + "GROMACS SETTINGS")
    for arg in ["gmx", "nt", "mpi", "gpu_id", "gmx_args_str", "mini_maxwarn", "sim_kill_delay"]:
        optional_args1.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args6 = args_parser.add_argument_group(bullet + "CG MODEL FORCE CONSTANTS")
    for arg in ["max_fct_bonds_f1", "max_fct_angles_f1", "max_fct_angles_f2", "max_fct_dihedrals_f149", "max_fct_dihedrals_f2"]:
        optional_args6.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args5 = args_parser.add_argument_group(bullet + "CG MODEL SCORING")
    for arg in ["cg_time_short", "cg_time_long", "b2a_score_fact", "bw_constraints", "bw_bonds", "bw_angles", "bw_dihedrals", "bonds_max_range"]:
        optional_args5.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args2 = args_parser.add_argument_group(bullet + "CG MODEL SCALING")
    for arg in ["aa_rg_offset", "bonds_scaling", "bonds_scaling_str", "min_bonds_length"]:
        optional_args2.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args3 = args_parser.add_argument_group(bullet + "OTHERS")
    for arg in ["temp", "keep_all_sims", "bonds_scaling_str", "min_bonds_length"]:
        optional_args2.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args3.add_argument("-h", "--help", **defaults.help.args)
    optional_args3.add_argument("-v", "--verbose", **defaults.verbose.args)

    return args_parser