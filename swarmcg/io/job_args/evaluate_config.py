from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS

from swarmcg.shared import styling
from swarmcg.io.job_args import defaults


def get_evaluate_args():

    print(styling.header_package("                Module: Model bonded terms assessment\n"))

    formatter = lambda prog: RawTextHelpFormatter(prog, width=135, max_help_position=52)
    args_parser = ArgumentParser(
        description=styling.EVALUATE_DESCR,
        formatter_class=formatter,
        add_help=False,
        usage=SUPPRESS
    )

    all_args_header = styling.sep_close + "\n|                                 REQUIRED/OPTIONAL ARGUMENTS                                 |\n" + styling.sep_close
    bullet = " "

    required_args = args_parser.add_argument_group(all_args_header + "\n\n" + bullet + "MODELS FILES")
    for arg in ["aa_tpr", "aa_traj", "cg_map", "mapping", "cg_itp", "cg_tpr", "cg_traj"]:
        print(arg)
        required_args.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args = args_parser.add_argument_group(bullet + "CG MODEL SCALING")
    # optional_args.add_argument("-nb_threads", dest="nb_threads", help="number of threads to use", type=int, default=1, metavar="1") # TODO: does NOT work properly -- modif MDAnalysis code with OpenMP num_threads(n) in the pragma
    for arg in ["bonds_scaling", "bonds_scaling_str", "min_bonds_length", "b2a_score_fact"]:
        print(arg)
        optional_args.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    graphical_args = args_parser.add_argument_group(bullet + "FIGURE DISPLAY")
    for arg in ["mismatch_ordering", "bw_constraints", "bw_bonds", "bw_angles", "bw_dihedrals",
                "disable_x_scaling", "disable_y_scaling", "bonds_max_range", "ncols"]:
        print(arg)
        graphical_args.add_argument(f"-{arg}", **getattr(defaults, arg).args)

    optional_args2 = args_parser.add_argument_group(bullet + "OTHERS")
    optional_args2.add_argument("-o_ev", **defaults.o_ev.args)
    optional_args2.add_argument("-h", "--help", **defaults.help.args)
    optional_args2.add_argument("-v", "--verbose",  **defaults.verbose.args)

    return args_parser