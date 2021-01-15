import time, subprocess, os, signal

import swarmcg.shared.exceptions as exceptions
import swarmcg.config as config
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


class SimulationStep:

    PREP_CMD = "{exec} grompp -c {gro} -f {mdp} -p {top} -o {md_output}"
    MD_CMD = "{exec} mdrun -deffnm {md_output} -nt {n_cpu} -gpu_id {gpu_id}"

    REQUIRED_FIELDS = ["exec", "gro", "mdp", "top", "md_output"]

    def __init__(self, sim_setup):
        self.sim_setup = sim_setup
        self.step_name = sim_setup.get("step_name")
        self._validate_init()

    def _validate_init(self):
        missing_args = ", ".join([i for i in SimulationStep.REQUIRED_FIELDS if i not in self.sim_setup.keys()])
        if missing_args:
            msg = (
                "The following arguments are missing: {missing_args}. Please check you input."
            )
            raise exceptions.InputArgumentError(msg)

    @property
    def swarmcg_flag(self):
        return self.sim_setup.get("swarmcg_flag")

    def _prepare_cmd(self, **kwargs):
        return SimulationStep.PREP_CMD.format(**{**self.sim_setup, **kwargs})

    def _run_cmd(self, aux_command, mpi=True):
        cmd = SimulationStep.MD_CMD.format(**self.sim_setup)
        if aux_command:
            cmd = f"{cmd} {aux_command}"
        threads = self.sim_setup.get("nb_threads")
        if threads > 0:
            cmd = f"{cmd} -nt {threads}"
        gpu = self.sim_setup.get("gpu_id")
        if gpu > 0:
            cmd = f"{cmd} -gpu_id {gpu}"
        mpi_tasks = self.sim_setup.get("mpi_tasks")
        if mpi and mpi_tasks > 1:
            cmd = f"mpirun -np {mpi_tasks} {cmd}"
        return cmd

    def _run_setup(self, exec_path):
        self.sim_setup.get("simulation_config").to_file(exec_path)

    def _run_prep(self, cmd):
        with subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as gmx_process:
            _ = gmx_process.communicate()[1].decode()
            gmx_process.kill()

        if gmx_process.returncode == 0:
            return self
        else:
            msg = (
                f"Gromacs grompp failed at MD {self.step_name} step, see its error message above. " 
                f"You may also want to check the parameters of the MDP file provided through "
                f"argument {self.swarmcg_flag}. If you think this is a bug, please consider opening "
                f"an issue on GitHub at {config.github_url}/issues."
            )
            raise exceptions.ComputationError(msg)

    def _run_md(self, cmd, monitor_file, keep_alive_n_cycles, seconds_between_checks):
        cycles_check, last_log_file_size = 0, 0
        _run_killed = False
        with subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, preexec_fn=os.setsid) as gmx_process:
            while gmx_process.poll() is None:  # while process is alive
                time.sleep(seconds_between_checks)
                cycles_check += 1

                if cycles_check % keep_alive_n_cycles == 0:

                    if os.path.isfile(monitor_file):
                        log_file_size = os.path.getsize(monitor_file)
                    else:
                        log_file_size = last_log_file_size

                    if log_file_size == last_log_file_size:
                        os.killpg(
                            os.getpgid(gmx_process.pid),
                            signal.SIGKILL
                        )  # kill all processes of process group
                        _run_killed = True
                    else:
                        last_log_file_size = log_file_size

            gmx_process.kill()

        if _run_killed:
            msg = (
                f"MD {self.step_name} run failed (unstable simulation was killed, with unstable "
                f"= NOT writing in log file for {keep_alive_n_cycles * seconds_between_checks} sec)"
            )
            raise exceptions.ComputationError(msg)

        else:
            return gmx_process.returncode

    def run(self, exec_path, aux_command):
        prep_cmd = self._prepare_cmd(exec_path)
        md_cmd = self._run_cmd(aux_command)
        return self._run_prep(prep_cmd)._run_md(md_cmd)


def ns_to_runner(ns, sim_config, prev_gro):
    return {
        "exec": ns.gmx_path,
        "gro": prev_gro,
        "mdp": ns.mdp_md_basename,
        "top": ns.top_input_basename,
        "gpu_id": ns.gpu_id,
        "mpi_tasks": ns.mpi_tasks,
        "nb_threads": ns.nb_threads,

        "swarmcg_flag": sim_config.swarmcg_flag,
        "step_name": sim_config.step_name,
        "md_output": sim_config.md_output,
        "simulation_config": sim_config
    }
