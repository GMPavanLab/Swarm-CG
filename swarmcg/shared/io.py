import contextlib
import sys


# print forced stdout enabled
def print_stdout_forced(*args, **kwargs):
    with contextlib.redirect_stdout(sys.__stdout__):
        print(*args, **kwargs, flush=True)


# read 1 column of xvg file and return as array
# column is 0-indexed
def read_xvg_col(xvg_file, col):
    with open(xvg_file, 'r') as fp:
        lines = [line.strip() for line in fp.read().split('\n')]
        data = []
        for line in lines:
            if not line.startswith(('#', '@')) and line != '':
                sp_lines = list(map(float, line.split()))
                data.append(sp_lines[col])
    return data
