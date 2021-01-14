import subprocess

from swarmcg.utils import print_stdout_forced


def gmx_args(ns, gmx_cmd, mpi=True):
    """Build gromacs command with arguments"""
    gmx_cmd = f"{ns.gmx_path} {gmx_cmd}"
    if ns.gmx_args_str != '':
        gmx_cmd = f"{gmx_cmd} {ns.gmx_args_str}"
    else:
        if ns.nb_threads > 0:
            gmx_cmd = f"{gmx_cmd} -nt {ns.nb_threads}"
        if len(ns.gpu_id) > 0:
            gmx_cmd = f"{gmx_cmd} -gpu_id {ns.gpu_id}"
    if mpi and ns.mpi_tasks > 1:
        gmx_cmd = f"mpirun -np {ns.mpi_tasks} {ns.gmx_cmd}"

    return gmx_cmd


def exec_gmx(gmx_cmd):
    """Execute gmx cmd and return only exit code"""
    with subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as gmx_process:
        gmx_out = gmx_process.communicate()[1].decode()
        gmx_process.kill()
    if gmx_process.returncode != 0:
        print_stdout_forced(
            'NON-ZERO EXIT CODE FOR COMMAND:', gmx_cmd, '\n\nCOMMAND OUTPUT:\n\n', gmx_out, '\n\n'
        )
    return gmx_process.returncode


def cmdline(command):
    """Execute command and return output"""
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
        success = True
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        success = False
    return success, output
