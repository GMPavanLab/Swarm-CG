import os, re

from swarmcg.shared import exceptions


class BaseSimulationConfig:

    REQUIRED_FIELDS = ["dt", "nsteps", "nstlog"]

    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise exceptions.MissingMdpFile(filename)
        else:
            self.sim_setup = BaseSimulationConfig.read_mdp(filename)

        self._validate_init()
        self.base_name = os.path.basename(filename)

    @staticmethod
    def read_mdp(filename):
        raw_content = open(filename, "r").readlines()
        CLEAN_PATTERN = "(.*[^;]);*(.*)"
        SUB_PATTERN = "[\n\t\s]*"
        SPLIT_PATTER = "(.*)=(.*)"
        f_clean = lambda x: re.sub(
            SUB_PATTERN, re.match(CLEAN_PATTERN, x).groups()[0], "", x
        )
        f_split = lambda x: re.match(SPLIT_PATTER, x).groups()
        cleaned = filter(None, map(f_clean, raw_content))
        split = map(f_split, cleaned)
        return {k: v for k, v in split}

    def to_string(self):
        output_string = ""
        for k, v in self.sim_setup.items():
            output_string += f"{k}\t\t = {str(v)}"
        return output_string

    def _validate_init(self):
        missing_args = ", ".join([i for i in type(self).REQUIRED_FIELDS
                                  if i not in self.sim_setup.keys()])
        if missing_args:
            msg = (
                "The following arguments are missing form mdp file for {self.name}: {missing_args}. "
                "Please check you input."
            )
            raise exceptions.MissformattedFile(msg)

    def modify_mdp(self, sim_time=None, nb_frames=1500, log_write_freq=5000,
                   energy_write_nb_frames_ratio=0.1):

        if sim_time is not None:
            new_nsteps = int(sim_time * 1000 / self.sim_setop["df"])
        else:
            new_nsteps = self.sim_setop["nsteps"]

        self.sim_setup["nsteps"] = new_nsteps
        self.sim_setup["nb_frames"] = nb_frames
        self.sim_setup["nstlog"] = log_write_freq
        self.sim_setup["nstvout"] = new_nsteps
        self.sim_setup["nstxout"] = new_nsteps
        self.sim_setup["nstfout"] = new_nsteps

        output_energy_freq = int(new_nsteps / nb_frames / energy_write_nb_frames_ratio)
        self.sim_setup["nstcalcenergy"] = output_energy_freq
        self.sim_setup["nstenergy"] = output_energy_freq
        self.sim_setup["nstxout-compressed "] = int(new_nsteps / nb_frames)

    def to_file(self, destination_path):
        open(os.join(destination_path, self.base_name), "w").writelines(self.to_string())


class Minimisation(BaseSimulationConfig):
    swarmcg_flag = "cg_sim_mdp_mini"
    step_name = "minimisation"
    md_output = "mini"


class Equilibration(BaseSimulationConfig):
    swarmcg_flag = "cg_sim_mdp_equi"
    step_name = "equilibration"
    md_output = "equi"


class Production(BaseSimulationConfig):
    swarmcg_flag = "cg_sim_mdp_md"
    step_name = "production"
    md_output = "md"


def select_class(flag, ns):
    filename = getattr(ns, flag)
    if "md" in flag:
        return Production(filename)
    elif "equi" in flag:
        return Equilibration(filename)
    elif "mini" in flag:
        return Minimisation(filename)
    else:
        ValueError(f"Flag {flag} does not correspond to any class.")
