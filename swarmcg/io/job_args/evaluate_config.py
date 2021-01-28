from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS

from swarmcg.shared import styling
from swarmcg import config


def get_evaluate_args():
    print(styling.header_package(
        "                Module: Model bonded terms assessment\n"))

    formatter = lambda prog: RawTextHelpFormatter(prog, width=135, max_help_position=52)
    args_parser = ArgumentParser(
        description=styling.EVALUATE_DESCR,
        formatter_class=formatter,
        add_help=False,
        usage=SUPPRESS
    )

    all_args_header = styling.sep_close + "\n|                                 REQUIRED/OPTIONAL ARGUMENTS                                 |\n" + styling.sep_close
    bullet = " "

    required_args = args_parser.add_argument_group(
        all_args_header + "\n\n" + bullet + "MODELS FILES")
    required_args.add_argument("-aa_tpr", dest="aa_tpr_filename", help=config.help_aa_tpr, type=str,
                               default=config.metavar_aa_tpr,
                               metavar=f"     ({config.metavar_aa_tpr})")
    required_args.add_argument("-aa_traj", dest="aa_traj_filename", help=config.help_aa_traj,
                               type=str, default=config.metavar_aa_traj,
                               metavar=f"     ({config.metavar_aa_traj})")
    required_args.add_argument("-cg_map", dest="cg_map_filename", help=config.help_cg_map, type=str,
                               default=config.metavar_cg_map,
                               metavar=f"       ({config.metavar_cg_map})")
    required_args.add_argument("-mapping", dest="mapping_type", help=config.help_mapping_type,
                               type=str,
                               default="COM", metavar="             (COM)")
    required_args.add_argument("-cg_itp", dest="cg_itp_filename",
                               help="ITP file of the CG model to evaluate", type=str,
                               default=config.metavar_cg_itp,
                               metavar=f"     ({config.metavar_cg_itp})")
    required_args.add_argument("-cg_tpr", dest="cg_tpr_filename",
                               help="TPR file of your CG simulation (omit for solo AA inspection)",
                               type=str, default=config.metavar_cg_tpr,
                               metavar=f"     ({config.metavar_cg_tpr})")
    required_args.add_argument("-cg_traj", dest="cg_traj_filename",
                               help="XTC file of your CG trajectory (omit for solo AA inspection)",
                               type=str, default=config.metavar_cg_traj,
                               metavar=f"     ({config.metavar_cg_traj})")

    optional_args = args_parser.add_argument_group(bullet + "CG MODEL SCALING")
    # optional_args.add_argument("-nb_threads", dest="nb_threads", help="number of threads to use", type=int, default=1, metavar="1") # TODO: does NOT work properly -- modif MDAnalysis code with OpenMP num_threads(n) in the pragma
    optional_args.add_argument("-bonds_scaling", dest="bonds_scaling",
                               help=config.help_bonds_scaling, type=float,
                               default=config.bonds_scaling,
                               metavar=f"       ({config.bonds_scaling})")
    optional_args.add_argument("-bonds_scaling_str", dest="bonds_scaling_str",
                               help=config.help_bonds_scaling_str, type=str,
                               default=config.bonds_scaling_str, metavar="")
    optional_args.add_argument("-min_bonds_length", dest="min_bonds_length",
                               help=config.help_min_bonds_length, type=float,
                               default=config.min_bonds_length,
                               metavar=f"    ({config.min_bonds_length})")
    optional_args.add_argument("-b2a_score_fact", dest="bonds2angles_scoring_factor",
                               help=config.help_bonds2angles_scoring_factor, type=float,
                               default=config.bonds2angles_scoring_factor,
                               metavar=f"      ({config.bonds2angles_scoring_factor})")

    graphical_args = args_parser.add_argument_group(bullet + "FIGURE DISPLAY")
    graphical_args.add_argument("-mismatch_ordering", dest="mismatch_order",
                                help="Enables ordering of bonds/angles/dihedrals by mismatch score\nbetween pairwise AA-mapped/CG distributions (can help diagnosis)",
                                default=False, action="store_true")
    graphical_args.add_argument("-bw_constraints", dest="bw_constraints",
                                help=config.help_bw_constraints, type=float,
                                default=config.bw_constraints,
                                metavar=f"    ({config.bw_constraints})")
    graphical_args.add_argument("-bw_bonds", dest="bw_bonds", help=config.help_bw_bonds, type=float,
                                default=config.bw_bonds,
                                metavar=f"           ({config.bw_bonds})")
    graphical_args.add_argument("-bw_angles", dest="bw_angles", help=config.help_bw_angles,
                                type=float, default=config.bw_angles,
                                metavar=f"           ({config.bw_angles})")
    graphical_args.add_argument("-bw_dihedrals", dest="bw_dihedrals", help=config.help_bw_dihedrals,
                                type=float, default=config.bw_dihedrals,
                                metavar=f"        ({config.bw_dihedrals})")
    graphical_args.add_argument("-disable_x_scaling", dest="row_x_scaling",
                                help="Disable auto-scaling of X axis across each row of the plot",
                                default=True, action="store_false")
    graphical_args.add_argument("-disable_y_scaling", dest="row_y_scaling",
                                help="Disable auto-scaling of Y axis across each row of the plot",
                                default=True, action="store_false")
    graphical_args.add_argument("-bonds_max_range", dest="bonded_max_range",
                                help=config.help_bonds_max_range, type=float,
                                default=config.bonds_max_range,
                                metavar=f"      ({config.bonds_max_range})")
    graphical_args.add_argument("-ncols", dest="ncols_max",
                                help="Max. nb of columns displayed in figure", type=int, default=0,
                                metavar="")  # TODO: make this a line return in plot instead of ignoring groups

    optional_args2 = args_parser.add_argument_group(bullet + "OTHERS")
    optional_args2.add_argument("-o", dest="plot_filename",
                                help="Filename for the output plot (extension/format can be one of:\neps, pdf, pgf, png, ps, raw, rgba, svg, svgz)",
                                type=str, default="distributions.png",
                                metavar=f"     (distributions.png)")
    optional_args2.add_argument("-h", "--help", action="help",
                                help="Show this help message and exit")
    optional_args2.add_argument("-v", "--verbose", dest="verbose", help=config.help_verbose,
                                action="store_true", default=False)
