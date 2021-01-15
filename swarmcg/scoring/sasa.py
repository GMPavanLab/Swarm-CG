import os

import numpy as np

from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg.io import read_xvg_col
from swarmcg.simulations.runner import exec_gmx

PROBE_RADIUS = 0.26  # nm


def compute_SASA(ns, traj_type):
    """Compute average SASA"""
    # NOTE: currently this is just COM mappping via GMX to get the SASA, so it's approximative but that's OK
    #       this works with calls to GMX because only library MDTraj can compute SASA (not MDAnalysis)
    # TODO: MDA is working on it, keep an eye on this: https://github.com/MDAnalysis/mdanalysis/issues/2439
    if traj_type == 'AA':
        raise exceptions.InvalidArgument('Compute_SASA not implemented for AA atm')

    elif traj_type == 'AA_mapped':

        # NOTE: here we assume the VS all come after the real beads in the ITP [ atoms ] field
        #       we generate a new truncated TPR so that we can use GMX sasa, this is shit but no choice atm
        nb_beads_real = len(ns.real_beads_ids)

        # generate an cg_map.ndx file with the number of beads,
        # so we can call SASA on this group and we will have exactly the content we want
        ns.cg_ndx_filename = '../' + config.input_sim_files_dirname + '/cg_index.ndx'
        with open(ns.cg_ndx_filename, 'w') as fp:
            beads_ids_str = ' '.join(
                map(str, list(range(1, nb_beads_real + 1))))  # includes VS if present
            fp.write('[' + ns.cg_itp['moleculetype']['molname'] + ' ]\n' + beads_ids_str + '\n')

        # TODO: all these paths need to be fixed to allow for SASA calculation within evaluate_model.py
        #       that's why it's disabled atm

        ns.aa_traj_whole_filename = '../' + config.input_sim_files_dirname + '/aa_traj_whole.xtc'
        ns.aa_mapped_traj_whole_filename = '../' + config.input_sim_files_dirname + '/aa_mapped_traj_whole.xtc'
        ns.aa_mapped_sasa_filename = '../' + config.input_sim_files_dirname + '/aa_mapped_sasa.xvg'
        ns.aa_mapped_tpr_sasa_filename = '../' + config.input_sim_files_dirname + '/aa_mapped_tpr_sasa.tpr'

        non_zero_return_code = False

        # first make traj whole
        gmx_cmd = f'seq 0 1 | {ns.gmx_path} trjconv -s ../../{ns.aa_tpr_filename} -f ../../{ns.aa_traj_filename} -pbc mol -o {ns.aa_traj_whole_filename}'
        return_code = exec_gmx(gmx_cmd)
        if return_code != 0:
            non_zero_return_code = True

        # then map AA traj
        if not non_zero_return_code:
            gmx_cmd = f'seq 0 {nb_beads_real - 1} | {ns.gmx_path} traj -f {ns.aa_traj_whole_filename} -s ../../{ns.aa_tpr_filename} -oxt {ns.aa_mapped_traj_whole_filename} -n ../../{ns.cg_map_filename} -com -ng {nb_beads_real}'
            return_code = exec_gmx(gmx_cmd)
            if return_code != 0:
                non_zero_return_code = True

        # truncate the CG TPR to get only real beads
        if not non_zero_return_code:
            gmx_cmd = f'{ns.gmx_path} convert-tpr -s md.tpr -n {ns.cg_ndx_filename} -o {ns.aa_mapped_tpr_sasa_filename}'
            return_code = exec_gmx(gmx_cmd)
            if return_code != 0:
                non_zero_return_code = True

        # finally get sasa
        if not non_zero_return_code:
            gmx_cmd = f'{ns.gmx_path} sasa -s {ns.aa_mapped_tpr_sasa_filename} -f {ns.aa_mapped_traj_whole_filename} -n {ns.cg_ndx_filename} -surface 0 -o {ns.aa_mapped_sasa_filename} -probe {PROBE_RADIUS}'
            return_code = exec_gmx(gmx_cmd)
            if return_code != 0:
                non_zero_return_code = True

        if non_zero_return_code:
            msg = (
                "There were some errors while calculating SASA for AA-mapped trajectory.\n"
                "Please check the error messages displayed above."
            )
            raise exceptions.ComputationError(msg)
        else:
            sasa_aa_mapped_per_frame = read_xvg_col(ns.aa_mapped_sasa_filename, 1)
            ns.sasa_aa_mapped = round(np.mean(sasa_aa_mapped_per_frame), 2)
            ns.sasa_aa_mapped_std = round(np.std(sasa_aa_mapped_per_frame), 2)

    elif traj_type == 'CG':

        ns.cg_traj_whole_filename = 'md_whole.xtc'
        ns.cg_sasa_filename = 'cg_sasa.xvg'
        non_zero_return_code = False

        # first make traj whole
        gmx_cmd = f'seq 0 1 | {ns.gmx_path} trjconv -s {ns.cg_tpr_filename} -f {ns.cg_traj_filename} -pbc mol -o {ns.cg_traj_whole_filename}'
        return_code = exec_gmx(gmx_cmd)
        if return_code != 0:
            non_zero_return_code = True

        # then compute SASA
        if not non_zero_return_code:
            # surface to choose the index group, 2 is the molecule even when there are ions (0 and 1 are System and Others)
            gmx_cmd = f'{ns.gmx_path} sasa -s {ns.cg_tpr_filename} -f {ns.cg_traj_whole_filename} -n {ns.cg_ndx_filename} -surface 0 -o {ns.cg_sasa_filename} -probe {PROBE_RADIUS}'
            return_code = exec_gmx(gmx_cmd)
            if return_code != 0:
                non_zero_return_code = True

        if non_zero_return_code or not os.path.isfile(ns.cg_sasa_filename):  # extra security
            ns.sasa_cg, ns.sasa_cg_std = None, None
        else:
            sasa_cg_per_frame = read_xvg_col(ns.cg_sasa_filename, 1)
            ns.sasa_cg = round(np.mean(sasa_cg_per_frame), 2)
            ns.sasa_cg_std = round(np.std(sasa_cg_per_frame), 2)

    else:
        raise exceptions.ComputationError('Code error compute SASA')
