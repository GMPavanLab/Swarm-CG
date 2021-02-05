from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS

from swarmcg.shared import styling
from swarmcg.io.job_args import defaults


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
    required_args.add_argument("-opti_dir", **defaults.opti_dir.args)
    required_args.add_argument("-o", **defaults.o_an.args)

    optional_args = args_parser.add_argument_group(bullet + "OTHERS")
    optional_args.add_argument("-plot_scale", **defaults.plot_scale.args)
    optional_args.add_argument("-h", "-help", **defaults.help.args)

    return args_parser
