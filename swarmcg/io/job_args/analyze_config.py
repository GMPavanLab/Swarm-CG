from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS

from swarmcg.shared import styling


def get_analyze_args():

    print(styling.header_package("Module: Optimization run analysis\n"))

    formatter = lambda prog: RawTextHelpFormatter(prog, width=135, max_help_position=52)
    args_parser = ArgumentParser(
        description=styling.ANALYSE_DESCR,
        formatter_class=formatter,
        add_help=False,
        usage=SUPPRESS
    )

    args_header = styling.sep_close + "\n|                                         ARGUMENTS                                           |\n" + styling.sep_close
    bullet = " "

    required_args = args_parser.add_argument_group(args_header + "\n\n" + bullet + "INPUT/OUTPUT")
    required_args.add_argument("-opti_dir", dest="opti_dirname",
                               help="Directory created by module \"scg_optimize\" that contains all files\ngenerated during the optimization procedure",
                               type=str, metavar="")
    required_args.add_argument("-o", dest="plot_filename",
                               help="Filename for the output plot, produced in directory -opti_dir.\nExtension/format can be one of: eps, pdf, pgf, png, ps, raw, rgba,\nsvg, svgz",
                               type=str, default="opti_summary.png",
                               metavar="    (opti_summary.png)")

    optional_args = args_parser.add_argument_group(bullet + "OTHERS")
    optional_args.add_argument("-plot_scale", dest="plot_scale", help="Scale factor of the plot",
                               type=float, default=1.0, metavar="        (1.0)")
    optional_args.add_argument("-h", "--help", help="Show this help message and exit",
                               action="help")

    return args_parser