from .read import read_aa_traj, read_cg_itp_file, validate_cg_itp
from .write import write_cg_itp_file
from .job_args import get_optimize_args, get_analyze_args, get_evaluate_args


def read_xvg_col(xvg_file, col):
    """Read 1 column of xvg file and return as array column is 0-indexed"""
    with open(xvg_file, 'r') as fp:
        lines = [line.strip() for line in fp.read().split('\n')]
        data = []
        for line in lines:
            if not line.startswith(('#', '@')) and line != '':
                sp_lines = list(map(float, line.split()))
                data.append(sp_lines[col])
    return data