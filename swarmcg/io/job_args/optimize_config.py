from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS

from swarmcg.shared import styling
from swarmcg import config


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
    optional_args0.add_argument("-exec_mode", dest="exec_mode",
                                help="MODE 1: Tune both bonds/angles/dihedrals equilibrium values\n        and their force constants\nMODE 2: Tune only bonds/angles/dihedrals force constants\n        with FIXED equilibrium values from the prelim. CG ITP",
                                type=int, default=1, metavar=f"              (1)")

    required_args = args_parser.add_argument_group(bullet + "REFERENCE AA MODEL")
    required_args.add_argument("-aa_tpr", dest="aa_tpr_filename", help=config.help_aa_tpr, type=str,
                               default=config.metavar_aa_tpr,
                               metavar=f"      ({config.metavar_aa_tpr})")
    required_args.add_argument("-aa_traj", dest="aa_traj_filename", help=config.help_aa_traj,
                               type=str, default=config.metavar_aa_traj,
                               metavar=f"      ({config.metavar_aa_traj})")
    required_args.add_argument("-cg_map", dest="cg_map_filename", help=config.help_cg_map, type=str,
                               default=config.metavar_cg_map,
                               metavar=f"        ({config.metavar_cg_map})")
    required_args.add_argument("-mapping", dest="mapping_type", help=config.help_mapping_type, type=str,
                               default="COM", metavar=f"              (COM)")

    sim_filenames_args = args_parser.add_argument_group(bullet + "CG MODEL OPTIMIZATION")
    sim_filenames_args.add_argument("-cg_itp", dest="cg_itp_filename",
                                    help="ITP file of the CG model to optimize", type=str,
                                    default=config.metavar_cg_itp,
                                    metavar=f"      ({config.metavar_cg_itp})")
    sim_filenames_args.add_argument("-user_params", dest="user_input",
                                   help="If absent, only the BI is used as starting point for parametrization\nIf present, parameters in the input ITP files are considered\n",
                                   action="store_true", default=False)
    sim_filenames_args.add_argument("-cg_gro", dest="gro_input_filename",
                                    help="Starting GRO file used for iterative simulation\nWill be minimized and relaxed before each MD run",
                                    type=str, default="start_conf.gro",
                                    metavar=f"    (start_conf.gro)")
    sim_filenames_args.add_argument("-cg_top", dest="top_input_filename",
                                    help="TOP file used for iterative simulation", type=str,
                                    default="system.top", metavar=f"        (system.top)")
    sim_filenames_args.add_argument("-cg_mdp_mini", dest="mdp_minimization_filename",
                                    help="MDP file used for minimization runs", type=str,
                                    default="mini.mdp", metavar=f"     (mini.mdp)")
    sim_filenames_args.add_argument("-cg_mdp_equi", dest="mdp_equi_filename",
                                    help="MDP file used for equilibration runs", type=str,
                                    default="equi.mdp", metavar=f"     (equi.mdp)")
    sim_filenames_args.add_argument("-cg_mdp_md", dest="mdp_md_filename",
                                    help="MDP file used for the MD runs analyzed for optimization",
                                    type=str, default="md.mdp", metavar=f"         (md.mdp)")
    sim_filenames_args.add_argument('-sim_type', dest='sim_type',
                                    help='Simulation type setting',
                                    type=str, default='OPTIMAL', metavar='         (OPTIMAL)')

    optional_args4 = args_parser.add_argument_group(
    opt_args_header + "\n\n" + bullet + "FILES HANDLING")
    optional_args4.add_argument("-in_dir", dest="input_folder",
                                help="Additional prefix path used to find argument-provided files\nIf ambiguous, files found without prefix are preferred",
                                type=str, default=".", metavar=f"")
    optional_args4.add_argument("-out_dir", dest="output_folder",
                                help="Directory where to store all outputs of this program\nDefault -out_dir is named after timestamp",
                                type=str, default="", metavar=f"")

    optional_args1 = args_parser.add_argument_group(bullet + "GROMACS SETTINGS")
    optional_args1.add_argument("-gmx", dest="gmx_path", help=config.help_gmx_path, type=str,
                                default=config.gmx_path,
                                metavar=f"                  ({config.gmx_path})")
    optional_args1.add_argument("-nt", dest="nb_threads",
                                help="Nb of threads to use, forwarded to 'gmx mdrun -nt'", type=int,
                                default=0, metavar=f"                     (0)")
    optional_args1.add_argument("-mpi", dest="mpi_tasks",
                                help="Nb of mpi programs (X), triggers 'mpirun -np X gmx'", type=int,
                                default=0, metavar=f"")
    optional_args1.add_argument("-gpu_id", dest="gpu_id",
                                help="String (use quotes) space-separated list of GPU device IDs",
                                type=str, default="", metavar=f"")
    optional_args1.add_argument("-gmx_args_str", dest="gmx_args_str",
                                help="String (use quotes) of arguments to forward to gmx mdrun\nIf provided, arguments -nt and -gpu_id are ignored",
                               type=str, default="", metavar=f"")
    optional_args1.add_argument("-mini_maxwarn", dest="mini_maxwarn",
                                help="Max. number of warnings to ignore, forwarded to gmx\ngrompp -maxwarn at each minimization step",
                                type=int, default=1, metavar=f"           (1)")
    optional_args1.add_argument("-sim_kill_delay", dest="sim_kill_delay",
                                help="Time (s) after which to kill a simulation that has not been\nwriting into its log file, in case a simulation gets stuck",
                                type=int, default=60, metavar=f"        (60)")

    optional_args6 = args_parser.add_argument_group(bullet + "CG MODEL FORCE CONSTANTS")
    optional_args6.add_argument("-max_fct_bonds_f1", dest="default_max_fct_bonds_opti",
                                help=config.help_max_fct_bonds, type=float,
                                default=config.default_max_fct_bonds_opti,
                                metavar=f"   ({config.default_max_fct_bonds_opti})")
    optional_args6.add_argument("-max_fct_angles_f1", dest="default_max_fct_angles_opti_f1",
                                help=config.help_max_fct_angles_f1, type=float,
                                default=config.default_max_fct_angles_opti_f1,
                                metavar=f"   ({config.default_max_fct_angles_opti_f1})")
    optional_args6.add_argument("-max_fct_angles_f2", dest="default_max_fct_angles_opti_f2",
                                help=config.help_max_fct_angles_f2, type=float,
                                default=config.default_max_fct_angles_opti_f2,
                                metavar=f"   ({config.default_max_fct_angles_opti_f2})")
    optional_args6.add_argument("-max_fct_dihedrals_f149",
                                dest="default_abs_range_fct_dihedrals_opti_func_with_mult",
                                help=config.help_max_fct_dihedrals_with_mult, type=float,
                                default=config.default_abs_range_fct_dihedrals_opti_func_with_mult,
                                metavar=f"({config.default_abs_range_fct_dihedrals_opti_func_with_mult})")
    optional_args6.add_argument("-max_fct_dihedrals_f2",
                                dest="default_abs_range_fct_dihedrals_opti_func_without_mult",
                                help=config.help_max_fct_dihedrals_without_mult, type=float,
                                default=config.default_abs_range_fct_dihedrals_opti_func_without_mult,
                                metavar=f"({config.default_abs_range_fct_dihedrals_opti_func_without_mult})")

    optional_args5 = args_parser.add_argument_group(bullet + "CG MODEL SCORING")
    optional_args5.add_argument("-cg_time_short", dest="sim_duration_short",
                                help="Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycles 1 and 2, this will modify MDP file for the MD runs",
                                type=float, default=10, metavar=f"         (10)")
    optional_args5.add_argument("-cg_time_long", dest="sim_duration_long",
                                help="Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycle 3, this will modify MDP file for the MD runs",
                                type=float, default=25, metavar=f"          (25)")
    optional_args5.add_argument("-b2a_score_fact", dest="bonds2angles_scoring_factor",
                                help=config.help_bonds2angles_scoring_factor, type=float,
                                default=config.bonds2angles_scoring_factor,
                                metavar=f"       ({config.bonds2angles_scoring_factor})")
    optional_args5.add_argument("-bw_constraints", dest="bw_constraints",
                                help=config.help_bw_constraints, type=float,
                                default=config.bw_constraints,
                                metavar=f"     ({config.bw_constraints})")
    optional_args5.add_argument("-bw_bonds", dest="bw_bonds", help=config.help_bw_bonds, type=float,
                                default=config.bw_bonds,
                                metavar=f"            ({config.bw_bonds})")
    optional_args5.add_argument("-bw_angles", dest="bw_angles", help=config.help_bw_angles,
                                type=float, default=config.bw_angles,
                                metavar=f"            ({config.bw_angles})")
    optional_args5.add_argument("-bw_dihedrals", dest="bw_dihedrals", help=config.help_bw_dihedrals,
                                type=float, default=config.bw_dihedrals,
                                metavar=f"         ({config.bw_dihedrals})")
    optional_args5.add_argument("-bonds_max_range", dest="bonded_max_range",
                                help=config.help_bonds_max_range, type=float,
                                default=config.bonds_max_range,
                                metavar=f"       ({config.bonds_max_range})")

    optional_args2 = args_parser.add_argument_group(bullet + "CG MODEL SCALING")
    optional_args2.add_argument("-aa_rg_offset", dest="aa_rg_offset",
                                help="Radius of gyration offset (nm) to be applied to AA data\naccording to your potential bonds rescaling (for display only)",
                                type=float, default=0.00, metavar=f"        (0.00)")
    optional_args2.add_argument("-bonds_scaling", dest="bonds_scaling",
                                help=config.help_bonds_scaling, type=float,
                                default=config.bonds_scaling,
                                metavar=f"        ({config.bonds_scaling})")
    optional_args2.add_argument("-bonds_scaling_str", dest="bonds_scaling_str",
                                help=config.help_bonds_scaling_str, type=str,
                                default=config.bonds_scaling_str, metavar=f"")
    optional_args2.add_argument("-min_bonds_length", dest="min_bonds_length",
                                help=config.help_min_bonds_length, type=float,
                                default=config.min_bonds_length,
                                metavar=f"     ({config.min_bonds_length})")

    optional_args3 = args_parser.add_argument_group(bullet + "OTHERS")
    optional_args3.add_argument("-temp", dest="temp",
                                help="Temperature used to perform Boltzmann inversion (K)",
                                type=float, default=config.sim_temperature,
                                metavar=f"                 ({config.sim_temperature})")
    optional_args3.add_argument("-keep_all_sims", dest="keep_all_sims",
                                help="Store all gmx files for all simulations, may use disk space",
                                action="store_true", default=False)
    optional_args3.add_argument("-h", "--help", help="Show this help message and exit", action="help")
    optional_args3.add_argument("-v", "--verbose", dest="verbose", help=config.help_verbose,
                                action="store_true", default=False)

    return args_parser