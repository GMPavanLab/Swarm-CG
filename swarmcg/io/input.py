import os

import MDAnalysis as mda

from swarmcg.shared import exceptions


class BaseInput:

    def __init__(self, namespace):
        self._ns = namespace

    def __getattr__(self, name):
        if name not in vars(self):
            return getattr(self._ns, name)

    def _get_basename(self, name):
        return os.path.basename(getattr(self, name))

    @property
    def mda_backend(self):
        if mda.lib.distances.USED_OPENMP:  # if MDAnalysis was compiled with OpenMP support
            return "OpenMP"
        else:
            return "serial"

    def _file_path_exists(self, filename):
        return os.path.isfile(getattr(self, filename)) or os.path.isdir(getattr(self, filename))


class OptInput(BaseInput):

    def __init__(self, namespace):
        super().__init__(self, namespace)
        self._validate_output_folder()

    @property
    def molname_in(self):
        return None

    @property
    def mapping_type(self):
        return self._ns.mapping_type.upper()

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
        return self._get_basename(self.cg_itp_filename)

    @property
    def gro_input_basename(self):
        return self._get_basename(self.gro_input_filename)

    @property
    def top_input_basename(self):
        return self._get_basename(self.top_input_filename)

    @property
    def mdp_minimization_basename(self):
        return self._get_basename(self.mdp_minimization_filename)

    @property
    def mdp_equi_basename(self):
        return self._get_basename(self.mdp_equi_filename)

    @property
    def mdp_md_basename(self):
        return self._get_basename(self.mdp_md_filename)

    def _validate_output_folder(self):
        """Avoid overwriting an output directory of a previous optimization run
        """
        if not self._file_path_exists("exec_folder"):
            msg = (
                "Provided output folder already exists, please delete existing folder "
                "manually or provide another folder name."
            )
            raise exceptions.AvoidOverwritingFolder(msg)
