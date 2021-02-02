import time, subprocess, os, signal

import swarmcg.shared.exceptions as exceptions
import swarmcg.config as config
from swarmcg.utils import print_stdout_forced
from swarmcg.simulations.simulation_steps import select_class


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


class SimulationStep:

    PREP_CMD = "{exec} grompp -c {gro} -f {mdp} -p {top} -o {md_output} -maxwarn {maxwarn}"
    MD_CMD = "{exec} mdrun -deffnm {md_output}"

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

    @staticmethod
    def _validate_exec(exec):
        with open(os.devnull, 'w') as devnull:
            try:
                subprocess.call(exec, stdout=devnull, stderr=devnull)
            except OSError:
                msg = (
                    f"Cannot find GROMACS using alias {exec}, please provide "
                    f"the right GROMACS alias or path"
                )
                raise exceptions.ExecutableNotFound(msg)

    @property
    def swarmcg_flag(self):
        return self.sim_setup.get("swarmcg_flag")

    @property
    def output_gro(self):
        return f"{self.sim_setup.get('md_output')}.gro"

    def _prepare_cmd(self, **kwargs):
        return SimulationStep.PREP_CMD.format(**{**self.sim_setup, **kwargs})

    def _run_cmd(self, aux_command="", mpi=True):
        cmd = SimulationStep.MD_CMD.format(**self.sim_setup)
        if aux_command:
            cmd = f"{cmd} {aux_command}"

        threads = int(self.sim_setup.get("nb_threads"))
        if threads > 0:
            cmd = f"{cmd} -nt {threads}"

        gpu = self.sim_setup.get("gpu_id")
        if len(gpu) > 0:
            cmd = f"{cmd} -gpu_id {gpu}"

        mpi_tasks = int(self.sim_setup.get("mpi_tasks"))
        if mpi and mpi_tasks > 0:
            cmd = f"mpirun -np {mpi_tasks} {cmd}"
        return cmd

    def _run_setup(self, exec_path):
        sim_time = self.sim_setup.get("sim_duration")
        nb_frames = self.sim_setup.get("prod_nb_frames")
        self.sim_setup.get("simulation_config").modify_mdp(sim_time, nb_frames).to_file(exec_path)
        return self

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
                f"argument -{self.swarmcg_flag}. If you think this is a bug, please consider opening "
                f"an issue on GitHub at {config.github_url}/issues."
            )
            raise exceptions.ComputationError(msg)

    def _run_md(self, cmd):
        cycles_check, last_log_file_size = 0, 0
        _run_killed = False
        monitor_file = self.sim_setup.get("monitor_file")
        keep_alive_n_cycles = self.sim_setup.get("keep_alive_n_cycles")
        seconds_between_checks = self.sim_setup.get("seconds_between_checks")
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
            print_stdout_forced(msg)
        else:
            return gmx_process.returncode

    def run(self, exec_path, aux_command=""):
        prep_cmd = self._prepare_cmd()
        md_cmd = self._run_cmd(aux_command)
        return self._run_setup(exec_path)._run_prep(prep_cmd)._run_md(md_cmd)


def ns_to_runner(ns, sim_config, prev_gro):
    simulation_setup = {
        "exec": ns.gmx_path,
        "gro": prev_gro,
        "mdp": getattr(ns, sim_config.mdp_base_name),
        "top": ns.top_input_basename,

        "gpu_id": ns.gpu_id,
        "mpi_tasks": ns.mpi_tasks,
        "nb_threads": ns.nb_threads,
        "maxwarn": ns.mini_maxwarn,

        "swarmcg_flag": sim_config.swarmcg_flag,
        "step_name": sim_config.step_name,
        "md_output": sim_config.md_output,

        "monitor_file":  f"{sim_config.md_output}.log",
        "keep_alive_n_cycles": ns.process_alive_nb_cycles_dead,
        "seconds_between_checks": ns.process_alive_time_sleep,
        "simulation_config": sim_config,
    }
    if hasattr(ns, "prod_sim_time"):
        simulation_setup["sim_duration"] = getattr(ns, "prod_sim_time")
    if hasattr(ns, "prod_nb_frames"):
        simulation_setup["prod_nb_frames"] = getattr(ns, "prod_nb_frames")

    return simulation_setup


def generate_steps(ns):
    step_flags = ["mdp_minimization_basename", "mdp_equi_basename", "mdp_md_basename"]
    prev_gro = ns.gro_input_basename
    for step in step_flags:
        sim_config = select_class(step, ns)
        simulation_step = SimulationStep(ns_to_runner(ns, sim_config, prev_gro))
        prev_gro = simulation_step.output_gro
        yield simulation_step
