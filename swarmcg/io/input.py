import os
from types import SimpleNamespace

import MDAnalysis as mda

from swarmcg.shared import exceptions


class BaseInput(SimpleNamespace):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_basename(self, name):
        return os.path.basename(getattr(self, name))

    @property
    def mda_backend(self):
        if mda.lib.distances.USED_OPENMP:  # if MDAnalysis was compiled with OpenMP support
            return "OpenMP"
        else:
            return "serial"

    @staticmethod
    def _file_path_exists(filename):
        return os.path.isfile(filename) or os.path.isdir(filename)


class OptInput(BaseInput):

    SIM_FILES_ARGS = [
        "aa_tpr_filename", "aa_traj_filename", "cg_map_filename", "cg_itp_filename",
        "gro_input_filename", "top_input_filename", "mdp_minimization_filename",
        "mdp_equi_filename", "mdp_md_filename"
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_validation()

    def input_validation(self):
        self._validate_output_folder()

    @property
    def molname_in(self):
        return None

    @property
    def process_alive_time_sleep(self):
        return 10

    @property
    def process_alive_nb_cycles_dead(self):
        return int(self.sim_kill_delay / self.process_alive_time_sleep)

    @property
    def bonds_rescaling_performed(self):
        return False

    @property
    def cg_itp_basename(self):
        return self._get_basename("cg_itp_filename")

    @property
    def gro_input_basename(self):
        return self._get_basename("gro_input_filename")

    @property
    def top_input_basename(self):
        return self._get_basename("top_input_filename")

    @property
    def mdp_minimization_basename(self):
        return self._get_basename("mdp_minimization_filename")

    @property
    def mdp_equi_basename(self):
        return self._get_basename("mdp_equi_filename")

    @property
    def mdp_md_basename(self):
        return self._get_basename("mdp_md_filename")

    def _validate_output_folder(self):
        """Avoid overwriting an output directory of a previous optimization run
        """
        if OptInput._file_path_exists(getattr(self, "exec_folder")):
            msg = (
                "Provided output folder already exists, please delete existing folder "
                "manually or provide another folder name."
            )
            raise exceptions.AvoidOverwritingFolder(msg)
        return self

    def _validate_filename(self, file_args):
        filename = getattr(self, file_args)
        if not OptInput._file_path_exists(filename):
            msg = (
                f"Cannot find input file {filename} store in attribute {file_args}."
            )
            raise FileNotFoundError(msg)
        return filename

    def _input_files(self, filenames):
        """Avoid overwriting an output directory of a previous optimization run
        """
        input_files = []
        for filename in filenames:
                input_files.append(self._validate_filename(filename))
        return input_files

    def simulation_filenames(self, all_files=True):
        if all_files:
            files = OptInput.SIM_FILES_ARGS[2:]
        else:
            files = OptInput.SIM_FILES_ARGS
        return self._input_files(files)
