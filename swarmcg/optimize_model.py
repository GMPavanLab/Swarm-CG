# some numpy version have this ufunc warning at import + many packages call numpy and display annoying warnings
import warnings
warnings.filterwarnings("ignore")
import os, sys, shutil, subprocess, time, copy, contextlib
from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS
from shlex import quote as cmd_quote
from datetime import datetime

from fstpso import FuzzyPSO 
import numpy as np

import swarmcg.shared.styling
from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg import swarmCG as scg
from swarmcg.shared.styling import OPTIMISE_DESCR

warnings.resetwarnings()


def run(ns):

    from numpy import VisibleDeprecationWarning
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)  # filter MDAnalysis + numpy deprecation stuff that is annoying
    warnings.filterwarnings("ignore", category=ImportWarning)  # filter Matplotlib mpl_toolkits missing __init__ stuff

    # TODO: allow to feed a JSON file for cycles of optimization ?? this is more optional but useful for big stuff possibly
    # TODO: if using SASA through GMX SASA, ensure vdwradii.dat contains the MARTINI radii
    # TODO: give a warning when users specify a bond scaling without specifying an Rg offset !!!

    # TODO: AT OPTI CYCLE 2, FIND ANGLES THAT ARE TOO STEEP (CG) AND WHEN GENERATING THE NEW GUESSES, PUT 10-30-50-70% OF THE CURRENT BEST FORCE CONSTANT IN SEVERAL PARTICLES !!!!!!!!!

    # NOTE: gmx trjconv and sasa may produce bugs when using TPR produced with gromacs v5, only current solution seems to be implementing the SASA calculation using MDTraj

    #####################################
    # ARGUMENTS HANDLING / HELP DISPLAY #
    #####################################

    # namespace variables not directly linked to arguments for plotting or for global package interpretation
    ns.mismatch_order = False
    ns.row_x_scaling = True
    ns.row_y_scaling = True
    ns.ncols_max = 0  # 0 to display all
    # ns.atom_only = False
    ns.molname_in = None  # if None the first found using TPR atom ordering will be used
    ns.process_alive_time_sleep = 10  # nb of seconds between process alive check cycles
    ns.process_alive_nb_cycles_dead = int(ns.sim_kill_delay / ns.process_alive_time_sleep)  # nb of cycles without .log file bytes size changes to determine that the MD run is stuck
    ns.bonds_rescaling_performed = False  # for user information display

    # get basenames for simulation files
    ns.cg_itp_basename = os.path.basename(ns.cg_itp_filename)
    ns.gro_input_basename = os.path.basename(ns.gro_input_filename)
    ns.top_input_basename = os.path.basename(ns.top_input_filename)
    ns.mdp_minimization_basename = os.path.basename(ns.mdp_minimization_filename)
    ns.mdp_equi_basename = os.path.basename(ns.mdp_equi_filename)
    ns.mdp_md_basename = os.path.basename(ns.mdp_md_filename)


    ####################
    # ARGUMENTS CHECKS #
    ####################

    print()
    print(swarmcg.shared.styling.sep_close)
    print('| PRE-PROCESSING AND CONTROLS                                                                 |')
    print(swarmcg.shared.styling.sep_close)
    print()

    # TODO: check that at least 10-20% of the simulations of the 1st swarm iteration finished properly, otherwise lower all energies or tell the user he is not writting into the log file regularly enough
    # TODO: test this program with ITP files that contain all the different dihedral functions, angles functions, constraints etc
    # TODO: find some fuzzy logic to determine number of swarm iterations + take some large margin to ensure it will optimize correctly

    # avoid overwriting an output directory of a previous optimization run
    if os.path.isfile(ns.exec_folder) or os.path.isdir(ns.exec_folder):
        msg = (
            "Provided output folder already exists, please delete existing folder "
            "manually or provide another folder name."
        )
        raise exceptions.AvoidOverwritingFolder(msg)

    # check the mapping type
    ns.mapping_type = ns.mapping_type.upper()
    if ns.mapping_type != 'COM' and ns.mapping_type != 'COG':
        msg = "Mapping type provided via argument '-mapping' must be either COM or COG (Center of Mass or Center of Geometry)."
        raise exceptions.InputArgumentError(msg)

    # check that force constants limits make sense
    if ns.default_max_fct_bonds_opti <= 0:
        msg = f"Please provide a value > 0 for argument -max_fct_bonds_f1."
        raise exceptions.InputArgumentError(msg)
    if ns.default_max_fct_angles_opti_f1 <= 0:
        msg = f"Please provide a value > 0 for argument -max_fct_angles_opti_f1."
        raise exceptions.InputArgumentError(msg)
    if ns.default_max_fct_angles_opti_f2 <= 0:
        msg = f"Please provide a value > 0 for argument -max_fct_angles_opti_f2."
        raise exceptions.InputArgumentError(msg)

    # check if we can find files at user-provided location(s)
    # here the order of the args in the 2 lists below is important, be very careful if changing this or adding args
    arg_entries = vars(ns)  # dict view of the arguments namespace
    user_provided_filenames = ['aa_tpr_filename', 'aa_traj_filename', 'cg_map_filename', 'cg_itp_filename', 'gro_input_filename', 'top_input_filename', 'mdp_minimization_filename', 'mdp_equi_filename', 'mdp_md_filename']
    args_names = ['aa_tpr', 'aa_traj', 'cg_map', 'cg_itp', 'cg_gro', 'cg_top', 'cg_mdp_mini', 'cg_mdp_equi', 'cg_mdp_md']

    for i in range(len(user_provided_filenames)):
        filename_out_directory = arg_entries[user_provided_filenames[i]]

        if not os.path.isfile(filename_out_directory):

            # if an input folder is specified (because '.' is the default input_folder)
            if ns.input_folder != '.':
                filename_in_directory = ns.input_folder+'/'+arg_entries[user_provided_filenames[i]]
                if not os.path.isfile(filename_in_directory):
                    msg = (
                        'Cannot find file for argument -{} '
                        '(expected at location: {})'.format(args_names[i], filename_in_directory)
                    )
                    raise FileNotFoundError(msg)
                else:
                    arg_entries[user_provided_filenames[i]] = filename_in_directory

            else:
                msg = (
                    'Cannot find file for argument -{} '
                    '(expected at location: {})'.format(args_names[i], filename_out_directory)
                )
                raise FileNotFoundError(msg)

    # check that gromacs alias is correct
    with open(os.devnull, 'w') as devnull:
        try:
            subprocess.call(ns.gmx_path, stdout=devnull, stderr=devnull)
        except OSError:
            msg = (
                f"Cannot find GROMACS using alias {ns.gmx_path}, please provide "
                f"the right GROMACS alias or path"
            )
            raise exceptions.ExecutableNotFound(msg)

    # check that ITP filename for the model to optimize is indeed included in the TOP file of the simulation directory
    # then find all TOP includes for copying files for simulations at each iteration
    top_includes_filenames = []
    with open(arg_entries[user_provided_filenames[5]], 'r') as fp:
        all_top_lines = fp.read()
        if ns.cg_itp_basename not in all_top_lines:
            msg = "The CG ITP model filename you provided is not included in your TOP file."
            raise exceptions.MDSimulationInputError(msg)

        top_lines = all_top_lines.split('\n')
        top_lines = [top_line.strip().split(';')[0] for top_line in top_lines]  # the split removes comments
        for top_line in top_lines:
            if top_line.startswith('#include'):
                top_include = top_line.split()[1].replace('"', '').replace("'", '')  # remove potential single and double quotes around filenames
                arg_dirname = os.path.dirname(arg_entries[user_provided_filenames[5]])
                if arg_dirname == '':
                    arg_dirname = '.'
                top_includes_filenames.append(arg_dirname + '/' + top_include)

    # check gmx arguments conflicts
    if ns.gmx_args_str != '' and (ns.nb_threads != 0 or ns.gpu_id != ''):
        print(
            swarmcg.shared.styling.header_warning + 'Argument -gmx_args_str is provided together with one of arguments: -nb_threads, -gpu_id\nOnly argument -gmx_args_str will be used during this execution')

    # check bonds scaling arguments conflicts
    if (ns.bonds_scaling != config.bonds_scaling and ns.min_bonds_length != config.min_bonds_length) or (ns.bonds_scaling != config.bonds_scaling and ns.bonds_scaling_str != config.bonds_scaling_str) or (ns.min_bonds_length != config.min_bonds_length and ns.bonds_scaling_str != config.bonds_scaling_str):
        msg = (
            "Only one of arguments -bonds_scaling, -bonds_scaling_str and "
            "-min_bonds_length can be provided. Please check your parameters"
        )
        raise exceptions.InputArgumentError(msg)

    ##################
    # INITIALIZATION #
    ##################

    # scg.set_MDA_backend(ns)
    ns.mda_backend = 'serial'  # actually serial is faster because MDA is not properly parallelized atm

    # directory to write all files for current execution of optimizations routines
    os.mkdir(ns.exec_folder)
    os.mkdir(ns.exec_folder+'/.internal')
    os.mkdir(ns.exec_folder+'/'+config.distrib_plots_all_evals_dirname)
    os.mkdir(ns.exec_folder+'/'+config.log_files_all_evals_dirname)
    if ns.keep_all_sims:
        os.mkdir(ns.exec_folder+'/'+config.sim_files_all_evals_dirname)

    # prepare a directory to be copied at each iteration of the optimization, to run the new simulation
    os.mkdir(ns.exec_folder+'/'+config.input_sim_files_dirname)

    # get all TOP file includes copied into input simulation directory
    for top_include in top_includes_filenames:
        shutil.copy(top_include, ns.exec_folder + '/' + config.input_sim_files_dirname)

    # copy all other simulation files
    user_provided_sim_files = ['cg_itp_filename', 'gro_input_filename', 'top_input_filename', 'mdp_minimization_filename', 'mdp_equi_filename', 'mdp_md_filename']
    for sim_file in user_provided_sim_files:
        shutil.copy(arg_entries[sim_file], ns.exec_folder+'/'+config.input_sim_files_dirname)

    # modify the TOP file to adapt includes paths
    with open(ns.exec_folder+'/'+config.input_sim_files_dirname+'/'+ns.top_input_basename, 'r') as fp:
        all_top_lines = fp.read().split('\n')
    with open(ns.exec_folder+'/'+config.input_sim_files_dirname+'/'+ns.top_input_basename, 'w+') as fp:
        nb_includes = 0
        for i in range(len(all_top_lines)):
            if all_top_lines[i].startswith('#include'):
                all_top_lines[i] = '#include "'+os.path.basename(top_includes_filenames[nb_includes])+'"'
                nb_includes += 1
        fp.writelines('\n'.join(all_top_lines))

    ns.nb_eval = 0  # global count of evaluation steps
    ns.start_opti_ts = datetime.now().timestamp()
    ns.total_eval_time, ns.total_gmx_time, ns.total_model_eval_time = 0, 0, 0

    scg.create_bins_and_dist_matrices(ns)  # bins for EMD calculations
    scg.read_ndx_atoms2beads(ns)  # read mapping, get atoms accurences in beads
    scg.get_atoms_weights_in_beads(ns)  # get weights of atoms within beads

    scg.read_cg_itp_file(ns)  # load the ITP object and find out geoms grouping
    scg.process_scaling_str(ns)  # process the bonds scaling specified by user

    print()
    scg.read_aa_traj(ns)  # create universe and read traj
    scg.load_aa_data(ns)  # read atoms attributes
    scg.make_aa_traj_whole_for_selected_mols(ns)

    # for each CG bead, create atom groups for trajectory geoms calculation using mass and atom weights across beads
    scg.get_beads_MDA_atomgroups(ns)

    # get CG beads weights from ITP includes present in the TOP file
    # but do NOT erase the masses found in the ITP of the CG MODEL provided via arg -cg_itp
    for top_include in top_includes_filenames:

        with open(top_include, 'r') as fp:
            try:
                itp_lines = fp.read().split('\n')
                itp_lines = [itp_line.split(';')[0].strip() for itp_line in itp_lines]
            except UnicodeDecodeError:
                msg = "Cannot read CG ITP, it seems you provided a binary file."
                raise exceptions.MissformattedFile(msg)

            for itp_line in itp_lines:
                if itp_line != '':

                    try:  # look for beads and VS masses: try to find the format, this is exigent enough to be unique
                        sp_itp_line = itp_line.split()
                        b_type, b_mass, b_sitetype = sp_itp_line[0], float(sp_itp_line[1]), sp_itp_line[3]
                        if b_sitetype in ['A', 'V', 'D']:  # atom, virtual site, dummy (old virtual site)
                            for bead_id in range(len(ns.cg_itp['atoms'])):
                                if ns.cg_itp['atoms'][bead_id]['bead_type'] == b_type:
                                    if ns.cg_itp['atoms'][bead_id]['mass'] == None:
                                        ns.cg_itp['atoms'][bead_id]['mass'] = b_mass
                    except (IndexError, ValueError):
                        pass

    print('\nMapping the trajectory from AA to CG representation')
    scg.initialize_cg_traj(ns)
    scg.map_aa2cg_traj(ns)
    print()

    # touch results files to be appended to later
    with open(ns.exec_folder+'/'+config.opti_perf_recap_file, 'w') as fp:
        # TODO: print that file has been generated with Swarm-CG etc -- do this for basically all files
        fp.write(f'# nb constraints: {ns.nb_constraints}\n')
        fp.write(f'# nb bonds: {ns.nb_bonds}\n')
        fp.write(f'# nb angles: {ns.nb_angles}\n')
        fp.write(f'# nb dihedrals: {ns.nb_dihedrals}\n')
        fp.write('#\n')
        fp.write('# opti_cycle nb_eval fit_score_all fit_score_cstrs_bonds fit_score_angles fit_score_dihedrals eval_score Rg_AA_mapped Rg_CG parameters_set eval_time current_total_time\n')
    with open(ns.exec_folder+'/'+config.opti_pairwise_distances_file, 'w'):
        pass

    # set these to None to then check the variables have been filled (is not None), so we will do these calculations
    # one single time in function compare_models that is called at each iteration during optimization
    ns.gyr_aa_mapped, ns.gyr_aa_mapped_std = None, None
    ns.sasa_aa_mapped, ns.sasa_aa_mapped_std = None, None

    print('Calculating bonds, angles and dihedrals distributions in the reference AA-mapped model')

    ns.domains_val = {'constraint': [], 'bond': [], 'angle': [], 'dihedral': []}
    ns.data_BI = {'bond': [], 'angle': [], 'dihedral': []}  # store hists for BI, std and possibly some other stats

    # create all ref atom histograms to be used for pairwise distributions comparisons + find average geoms values as first guesses (without BI at this point)
    # get ref atom hists + find very first distances guesses for constraints groups
    for grp_constraint in range(ns.nb_constraints):

        constraint_avg, constraint_hist, constraint_values = scg.get_AA_bonds_distrib(ns, beads_ids=ns.cg_itp['constraint'][grp_constraint]['beads'], grp_type='constraint group', grp_nb=grp_constraint)
        if ns.exec_mode == 1:
            ns.cg_itp['constraint'][grp_constraint]['value'] = constraint_avg
        ns.cg_itp['constraint'][grp_constraint]['avg'] = constraint_avg
        ns.cg_itp['constraint'][grp_constraint]['hist'] = constraint_hist

        ns.domains_val['constraint'].append([round(np.min(constraint_values), 3), round(np.max(constraint_values), 3)])  # boundaries of equilibrium values
        print(f'  Constraint grp {grp_constraint+1} -- Average value: '+str(round(constraint_avg, 2))+' nm -- Initial equilibrium value: '+str(round(ns.cg_itp['constraint'][grp_constraint]['value'], 2))+' nm')

    # get ref atom hists + find very first distances and force constants guesses for bonds groups
    for grp_bond in range(ns.nb_bonds):

        bond_avg, bond_hist, bond_values = scg.get_AA_bonds_distrib(ns, beads_ids=ns.cg_itp['bond'][grp_bond]['beads'], grp_type='bond group', grp_nb=grp_bond)
        if ns.exec_mode == 1:
            ns.cg_itp['bond'][grp_bond]['value'] = bond_avg
        ns.cg_itp['bond'][grp_bond]['avg'] = bond_avg
        ns.cg_itp['bond'][grp_bond]['hist'] = bond_hist

        xmin, xmax = min(np.inf, ns.bins_bonds[np.min(np.nonzero(bond_hist))]), max(-np.inf, ns.bins_bonds[np.max(np.nonzero(bond_hist))+1])
        xmin, xmax = xmin-ns.bw_bonds, xmax+ns.bw_bonds
        ns.data_BI['bond'].append([np.histogram(bond_values, range=(xmin, xmax), bins=config.bi_nb_bins)[0], np.std(bond_values), np.mean(bond_values), (xmin, xmax)])

        ns.domains_val['bond'].append([round(np.min(bond_values), 3), round(np.max(bond_values), 3)])  # boundaries of equilibrium values
        print(f'  Bond grp {grp_bond+1} -- Average value: '+str(round(bond_avg, 2))+' nm -- Initial equilibrium value: '+str(round(ns.cg_itp['bond'][grp_bond]['value'], 2))+' nm')

    # get ref atom hists + find very first values and force constants guesses for angles groups
    for grp_angle in range(ns.nb_angles):

        angle_avg, angle_hist, angle_values_deg, angle_values_rad = scg.get_AA_angles_distrib(ns, beads_ids=ns.cg_itp['angle'][grp_angle]['beads'])
        if ns.exec_mode == 1:
            ns.cg_itp['angle'][grp_angle]['value'] = angle_avg
        ns.cg_itp['angle'][grp_angle]['avg'] = angle_avg
        ns.cg_itp['angle'][grp_angle]['hist'] = angle_hist

        xmin, xmax = min(np.inf, ns.bins_angles[np.min(np.nonzero(angle_hist))]), max(-np.inf, ns.bins_angles[np.max(np.nonzero(angle_hist))+1])
        xmin, xmax = xmin+ns.bw_angles/2, xmax-ns.bw_angles/2
        ns.data_BI['angle'].append([np.histogram(angle_values_rad, range=(np.deg2rad(xmin), np.deg2rad(xmax)), bins=config.bi_nb_bins)[0], np.std(angle_values_rad), (xmin, xmax)])

        ns.domains_val['angle'].append([round(np.min(angle_values_deg), 2), round(np.max(angle_values_deg), 2)])  # boundaries of equilibrium values
        print(f'  Angle grp {grp_angle+1} -- Average value: ' + str(round(angle_avg, 2)) + ' degrees -- Initial equilibrium value: '+str(round(ns.cg_itp['angle'][grp_angle]['value'], 2))+' degrees')

    # get ref atom hists + find very first values and force constants guesses for dihedrals groups
    for grp_dihedral in range(ns.nb_dihedrals):

        dihedral_avg, dihedral_hist, dihedral_values_deg, dihedral_values_rad = scg.get_AA_dihedrals_distrib(ns, beads_ids=ns.cg_itp['dihedral'][grp_dihedral]['beads'])
        if ns.exec_mode == 1:  # the dihedral equi value will be calculated from the BI fit, because for dihedrals it makes no sense to use the average
            ns.cg_itp['dihedral'][grp_dihedral]['value'] = dihedral_avg
        ns.cg_itp['dihedral'][grp_dihedral]['avg'] = dihedral_avg
        ns.cg_itp['dihedral'][grp_dihedral]['hist'] = dihedral_hist

        xmin, xmax = -180, 180
        ns.data_BI['dihedral'].append([np.histogram(dihedral_values_rad, range=(np.deg2rad(xmin), np.deg2rad(xmax)), bins=2 *config.bi_nb_bins)[0], np.std(dihedral_values_rad), np.mean(dihedral_values_rad), (xmin, xmax)])

        ns.domains_val['dihedral'].append([round(np.min(dihedral_values_deg), 2), round(np.max(dihedral_values_deg), 2)])  # boundaries of equilibrium values
        print(f'  Dihedral grp {grp_dihedral+1} -- Average value: ' + str(round(dihedral_avg, 2)) + ' degrees -- Initial equilibrium value: '+str(round(ns.cg_itp['dihedral'][grp_dihedral]['value'], 2))+' degrees')

    if not ns.bonds_rescaling_performed:
        print('  No bonds rescaling performed')

    # output png with all the reference distributions, so the user can check
    ns.atom_only = True
    ns.plot_filename = ns.exec_folder+'/'+config.ref_distrib_plots
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            scg.compare_models(ns, manual_mode=False)
    print()
    print('Plotted reference AA-mapped distributions (used as target during optimization) at location:\n ', ns.exec_folder+'/'+config.ref_distrib_plots)
    ns.atom_only = False


    ##################################
    # ITERATIVE OPTIMIZATION PROCESS #
    ##################################

    # parameters for each type of simulation during optimization cycles
    # sim duration (ns), max nb of SWARM iterations, max nb SWARM iterations without finding new global best, percentage applied for generating variations around initial guesses/values fed humanly
    # sim_type 0 is used for initialization exclusively + detecting too high force constants to lower them, no real optimization is expected from these runs

    # Settings: TEST / utlra-fast settings only for debugging -- DIHEDRALS APPLIED IN THE END EXCLUSIVELY
    # sim_types = {0: {'sim_duration': 0.3, 'max_swarm_iter': 1, 'max_swarm_iter_without_new_global_best': 1, 'val_guess_fact': 1.0, 'fct_guess_fact': 0.3},
    #        1: {'sim_duration': 0.3, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 2, 'val_guess_fact': 1.0, 'fct_guess_fact': 0.3},
    #        2: {'sim_duration': 0.3, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 2, 'val_guess_fact': 1.0, 'fct_guess_fact': 0.2},
    #        3: {'sim_duration': 0.3, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 2, 'val_guess_fact': 0.4, 'fct_guess_fact': 0.1}}
    # opti_cycles = [['constraint', 'bond', 'angle'], ['constraint', 'bond'], ['angle'], ['constraint', 'bond', 'angle'], ['dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
    # sim_cycles = [0, 1, 1, 2, 2, 3] # simulations types

    # Settings: TEST / utlra-fast settings only for debugging -- DIHEDRALS APPLIED IN THE END EXCLUSIVELY
    # sim_types = {0: {'sim_duration': 0.3, 'max_swarm_iter': 1, 'max_swarm_iter_without_new_global_best': 1, 'val_guess_fact': 1.0, 'fct_guess_fact': 0.3},
    #        1: {'sim_duration': 0.3, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 2, 'val_guess_fact': 1.0, 'fct_guess_fact': 0.3},
    #        2: {'sim_duration': 0.3, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 2, 'val_guess_fact': 1.0, 'fct_guess_fact': 0.2},
    #        3: {'sim_duration': 0.3, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 2, 'val_guess_fact': 0.4, 'fct_guess_fact': 0.1}}
    # opti_cycles = [['constraint', 'bond', 'angle'], ['constraint', 'bond', 'angle'], ['dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
    # sim_cycles = [0, 2, 2, 3] # simulations types

    # Settings: ROBUST / Suited for big molecules
    # sim_types = {0: {'sim_duration': 5, 'max_swarm_iter': 10, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 1, 'fct_guess_fact': 0.30},
    #        1: {'sim_duration': 8, 'max_swarm_iter': 10, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.25},
    #        2: {'sim_duration': 10, 'max_swarm_iter': 10, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.25},
    #        3: {'sim_duration': 15, 'max_swarm_iter': 20, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.25}}
    # opti_cycles = [['constraint', 'bond', 'angle'], ['constraint', 'bond'], ['angle'], ['constraint', 'bond', 'angle'], ['dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
    # sim_cycles = [0, 1, 1, 3, 2, 3] # simulations types

    # Strategy 1
    # Settings: FASTER / Suited for small molecules or rapid optimization
    # sim_types = {0: {'sim_duration': 10, 'max_swarm_iter': 10, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 1, 'fct_guess_fact': 0.40},
    #        1: {'sim_duration': 10, 'max_swarm_iter': 10, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.30},
    #        2: {'sim_duration': 15, 'max_swarm_iter': 15, 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.25}}
    # opti_cycles = [['constraint', 'bond', 'angle'], ['dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
    # sim_cycles = [0, 1, 2] # simulations types

    # previous best choice for versatile usage
    # Startegy 4
    # Settings: OPTIMAL / Should be fine with any type of molecule, big or small, as long as the BI keeps yielding close enough results, which should be the case
    # sim_types = {0: {'sim_duration': 10, 'max_swarm_iter': int(5+np.sqrt(ns.nb_constraints+ns.nb_bonds+ns.nb_angles)), 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 1, 'fct_guess_fact': 0.35},
    #        1: {'sim_duration': 10, 'max_swarm_iter': int(5+np.sqrt(ns.nb_angles+ns.nb_dihedrals)), 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.30},
    #        2: {'sim_duration': 10, 'max_swarm_iter': int(5+np.sqrt(ns.nb_constraints+ns.nb_bonds+ns.nb_angles+ns.nb_dihedrals)), 'max_swarm_iter_without_new_global_best': 5, 'val_guess_fact': 0.15, 'fct_guess_fact': 0.20}}
    # opti_cycles = [['constraint', 'bond', 'angle'], ['angle', 'dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
    # sim_cycles = [0, 1, 2] # simulations types

    # Startegy 5 -- Coupled to fewer particles
    # Settings: OPTIMAL / Should be fine with any type of molecule, big or small, as long as the BI keeps yielding close enough results, which should be the case
    sim_types = {0: {'sim_duration': ns.sim_duration_short, 'max_swarm_iter': int(round(6+np.sqrt(ns.nb_constraints+ns.nb_bonds+ns.nb_angles))), 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 1, 'fct_guess_fact': 0.40},
         1: {'sim_duration': ns.sim_duration_short, 'max_swarm_iter': int(round(6+np.sqrt(ns.nb_angles+ns.nb_dihedrals))), 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.30},
         2: {'sim_duration': ns.sim_duration_long, 'max_swarm_iter': int(round(6+np.sqrt(ns.nb_constraints+ns.nb_bonds+ns.nb_angles+ns.nb_dihedrals))), 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.20}}
    # opti_cycles = [['constraint', 'bond', 'angle'], ['angle', 'dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
    opti_cycles = [['constraint', 'bond', 'angle'], ['angle', 'dihedral'], ['constraint', 'bond', 'angle', 'dihedral']]
    sim_cycles = [0, 1, 2]  # simulations types

    # for tests
    # sim_types = {0: {'sim_duration': ns.sim_duration_short, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 1, 'fct_guess_fact': 0.40},
    #        1: {'sim_duration': ns.sim_duration_short, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.30},
    #        2: {'sim_duration': ns.sim_duration_long, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.20}}
    # opti_cycles = [['constraint', 'bond', 'angle', 'dihedral'], ['angle', 'dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
    # sim_cycles = [0, 1, 2] # simulations types

    # NOTE: currently, due to an issue in FST-PSO, number of swarm iterations performed is +2 when compared to the numbers we feed

    ns.opti_itp = copy.deepcopy(ns.cg_itp) # the ITP object that will be optimized stepwise, at the end of each optimization cycle (geom type wise)
    ns.eval_nb_geoms = {'constraint': 0, 'bond': 0, 'angle': 0, 'dihedral': 0}  # geoms to optimize at each step

    # remove dihedrals from cycles if CG ITP file does NOT contain dihedrals
    if ns.nb_dihedrals == 0:
        opti_cycles_cp, sim_cycles_cp = [], []
        nb_poped = 0
        for i in range(len(opti_cycles)):
            opti_cycles_cp.extend([[]])
            for j in range(len(opti_cycles[i])):
                if opti_cycles[i][j] != 'dihedral':
                    opti_cycles_cp[i-nb_poped].append(opti_cycles[i][j])
                if len(opti_cycles_cp[i-nb_poped]) == 0:
                    opti_cycles_cp.pop()
                    nb_poped += 1
                else:
                    sim_cycles_cp.extend([sim_cycles[i]])
        opti_cycles, sim_cycles = opti_cycles_cp, sim_cycles_cp

    # state variables for the cycles of optimization
    ns.performed_init_BI = {'bond': False, 'angle': False, 'dihedral': False}
    ns.opti_geoms_all = set(geom for opti_cycle_geoms in opti_cycles for geom in opti_cycle_geoms)
    ns.best_fitness = [np.inf, None]  # fitness_score, eval_step_best_score

    # storage for best independent set of parameters by geom, for initialization of a (few ?) special particle after 1st opti cycle
    ns.all_best_emd_dist_geoms = {'constraints': {}, 'bonds': {}, 'angles': {}, 'dihedrals': {}}
    ns.all_best_params_dist_geoms = {'constraints': {}, 'bonds': {}, 'angles': {}, 'dihedrals': {}}
    for i in range(ns.nb_constraints):
        ns.all_best_emd_dist_geoms['constraints'][i] = config.sim_crash_EMD_indep_score
        ns.all_best_params_dist_geoms['constraints'][i] = {}
    for i in range(ns.nb_bonds):
        ns.all_best_emd_dist_geoms['bonds'][i] = config.sim_crash_EMD_indep_score
        ns.all_best_params_dist_geoms['bonds'][i] = {}
    for i in range(ns.nb_angles):
        ns.all_best_emd_dist_geoms['angles'][i] = config.sim_crash_EMD_indep_score
        ns.all_best_params_dist_geoms['angles'][i] = {}
    for i in range(ns.nb_dihedrals):
        ns.all_best_emd_dist_geoms['dihedrals'][i] = config.sim_crash_EMD_indep_score
        ns.all_best_params_dist_geoms['dihedrals'][i] = {}

    #############################
    # START OPTIMIZATION CYCLES #
    #############################

    for i in range(len(opti_cycles)):

        ns.opti_cycle = {'nb_cycle': i+1, 'geoms': opti_cycles[i], 'nb_geoms': {'constraint': 0, 'bond': 0, 'angle': 0, 'dihedral': 0}}
        ns.out_itp = copy.deepcopy(ns.opti_itp) # input ITP copy, on which we might perform BI, and that is the object we will modify at each evaluation step to store the values from FST-PSO

        # model selection based on fitness + Rg during last optimization cycle
        # ns.all_rg_last_cycle, ns.all_fitness_last_cycle = np.array([]), np.array([])
        # ns.best_fitness_Rg_combined = 0 # id of the best model based on bonded fitness + Rg selection

        ns.prod_sim_time = sim_types[sim_cycles[i]]['sim_duration']
        ns.val_guess_fact = sim_types[sim_cycles[i]]['val_guess_fact']
        ns.fct_guess_fact = sim_types[sim_cycles[i]]['fct_guess_fact']
        ns.max_swarm_iter = sim_types[sim_cycles[i]]['max_swarm_iter']
        ns.max_swarm_iter_without_new_global_best = sim_types[sim_cycles[i]]['max_swarm_iter_without_new_global_best']

        # adapt number of geoms according to the optimization cycle
        geoms_display = []
        if 'constraint' in ns.opti_cycle['geoms'] or 'bond' in ns.opti_cycle['geoms']:
          geoms_display.append('constraints/bonds')
        if 'constraint' in ns.opti_cycle['geoms']:
          ns.opti_cycle['nb_geoms']['constraint'] = ns.nb_constraints
        if 'bond' in ns.opti_cycle['geoms']:
          ns.opti_cycle['nb_geoms']['bond'] = ns.nb_bonds
        if 'angle' in ns.opti_cycle['geoms']:
          ns.opti_cycle['nb_geoms']['angle'] = ns.nb_angles
          geoms_display.append('angles')
        if 'dihedral' in ns.opti_cycle['geoms']:
          ns.opti_cycle['nb_geoms']['dihedral'] = ns.nb_dihedrals
          geoms_display.append('dihedrals')
        geoms_display = ' & '.join(geoms_display)

        print()
        print(swarmcg.shared.styling.sep_close)
        print('| STARTING OPTIMIZATION CYCLE', ns.opti_cycle['nb_cycle'], '                                                              |')
        print('| Optimizing', geoms_display, ' '*(95-16-len(geoms_display)), '|')
        print(swarmcg.shared.styling.sep_close)

        # actual BI to get the initial guesses of force constants, for all selected geoms at this given optimization step
        # BI is performed:
        # -- exec_mode 1: all equilibrium values and force constants
        # -- exec_mode 2: equilibrium values are not touched for bonds, angles and dihedrals, but all their force constants are optimized
        scg.perform_BI(ns)  # performed on object ns.out_itp

        # build vector for search space boundaries + create variations around the BI initial guesses
        search_space_boundaries = scg.get_search_space_boundaries(ns)

        # ns.worst_fit_score = round(len(search_space_boundaries) * config.sim_crash_EMD_indep_score, 3)
        ns.worst_fit_score = round(\
          np.sqrt((ns.nb_constraints+ns.nb_bonds) * config.sim_crash_EMD_indep_score) + \
          np.sqrt(ns.nb_angles * config.sim_crash_EMD_indep_score) + \
          np.sqrt(ns.nb_dihedrals * config.sim_crash_EMD_indep_score) \
          , 3)
        # nb_particles = int(10 + 2*np.sqrt(len(search_space_boundaries)))  # formula used by FST-PSO to choose nb of particles, which defines the number of initial guesses we can use
        nb_particles = int(round(2 + np.sqrt(len(search_space_boundaries))))  # adapted to have less particles and fitted to our problems, which has good initial guesses and error driven initialization
        if nb_particles < 3:  # actually we probably should make this an open parameter soon (not the minimum, but the REQUIRED number of particles)
            nb_particles = 3
        # nb_particles = 2 # for tests
        initial_guess_list = scg.get_initial_guess_list(ns, nb_particles)

        # actual optimization
        with open(os.devnull, 'w') as devnull:
          with contextlib.redirect_stdout(devnull):
            FP = FuzzyPSO()
            FP.set_search_space(search_space_boundaries)
            FP.set_swarm_size(nb_particles)
            FP.set_fitness(fitness=scg.eval_function, arguments=ns, skip_test=True)
            result = FP.solve_with_fstpso(max_iter=ns.max_swarm_iter, initial_guess_list=initial_guess_list, max_iter_without_new_global_best=ns.max_swarm_iter_without_new_global_best)

        # update ITP object with the best solution using geoms considered at this given optimization step
        scg.update_cg_itp_obj(ns, parameters_set=result[0].X, update_type=2)

    # clean temporary copied directory with user's input files
    shutil.rmtree(ns.exec_folder+'/'+config.input_sim_files_dirname)

    # print some stats
    total_time_sec = datetime.now().timestamp() - ns.start_opti_ts
    total_time = round(total_time_sec / (60 * 60), 2)
    fitness_eval_time = round(ns.total_eval_time / (60 * 60), 2)
    init_time = round((total_time_sec - ns.total_eval_time) / (60 * 60), 2)
    ns.total_gmx_time = round(ns.total_gmx_time / (60 * 60), 2)
    ns.total_model_eval_time = round(ns.total_model_eval_time / (60 * 60), 2)
    print()
    print(swarmcg.shared.styling.sep_close)
    print('|  FINISHED PROPERLY                                                                          |')
    print(swarmcg.shared.styling.sep_close)
    print()
    print('Total nb of evaluation steps:', ns.nb_eval)
    print('Best model obtained at evaluation step number:', ns.best_fitness[1])
    print()
    print(f'Total execution time : {total_time} h')
    print(f'Initialization time  : {init_time} h ({round(init_time/total_time*100, 2)} %)')
    print(f'Simulations time     : {ns.total_gmx_time} h ({round(ns.total_gmx_time/total_time*100, 2)} %)')
    print(f'Models scoring time  : {ns.total_model_eval_time} h ({round(ns.total_model_eval_time/total_time*100, 2)} %)')
    print()


def main():

    print(
    swarmcg.shared.styling.header_package('                    Module: CG model optimization\n'))

    formatter = lambda prog: RawTextHelpFormatter(prog, width=135, max_help_position=52)
    args_parser = ArgumentParser(
    description=OPTIMISE_DESCR,
    formatter_class=formatter,
    add_help=False,
    usage=SUPPRESS
    )

    # TODO: handle trajectories for which no box informations are provided
    # TODO: explain what is modified in the MDP
    # TODO: explain module analyze_opti_moves.py can be used to monitor optimization at any point of the process
    # TODO: end the help message by a new frame with examples from the demo data

    req_args_header = swarmcg.shared.styling.sep_close + '\n|                                     REQUIRED ARGUMENTS                                      |\n' + swarmcg.shared.styling.sep_close
    opt_args_header = swarmcg.shared.styling.sep_close + '\n|                                     OPTIONAL ARGUMENTS                                      |\n' + swarmcg.shared.styling.sep_close
    bullet = ' '

    optional_args0 = args_parser.add_argument_group(
    req_args_header + '\n\n' + bullet + 'EXECUTION MODE')
    optional_args0.add_argument('-exec_mode', dest='exec_mode',
                              help='MODE 1: Tune both bonds/angles/dihedrals equilibrium values\n        and their force constants\nMODE 2: Tune only bonds/angles/dihedrals force constants\n        with FIXED equilibrium values from the prelim. CG ITP',
                              type=int, default=1, metavar='              (1)')

    required_args = args_parser.add_argument_group(bullet + 'REFERENCE AA MODEL')
    required_args.add_argument('-aa_tpr', dest='aa_tpr_filename', help=config.help_aa_tpr, type=str,
                             default=config.metavar_aa_tpr,
                             metavar='      ' + scg.par_wrap(config.metavar_aa_tpr))
    required_args.add_argument('-aa_traj', dest='aa_traj_filename', help=config.help_aa_traj,
                             type=str, default=config.metavar_aa_traj,
                             metavar='      ' + scg.par_wrap(config.metavar_aa_traj))
    required_args.add_argument('-cg_map', dest='cg_map_filename', help=config.help_cg_map, type=str,
                             default=config.metavar_cg_map,
                             metavar='        ' + scg.par_wrap(config.metavar_cg_map))
    required_args.add_argument('-mapping', dest='mapping_type', help=config.help_mapping_type, type=str,
                             default='COM', metavar='              (COM)')

    sim_filenames_args = args_parser.add_argument_group(bullet + 'CG MODEL OPTIMIZATION')
    sim_filenames_args.add_argument('-cg_itp', dest='cg_itp_filename',
                                  help='ITP file of the CG model to optimize', type=str,
                                  default=config.metavar_cg_itp,
                                  metavar='      ' + scg.par_wrap(config.metavar_cg_itp))
    sim_filenames_args.add_argument('-user_params', dest='user_input',
                                help='If absent, only the BI is used as starting point for parametrization\nIf present, parameters in the input ITP files are considered\n',
                                action='store_true', default=False)
    sim_filenames_args.add_argument('-cg_gro', dest='gro_input_filename',
                                  help='Starting GRO file used for iterative simulation\nWill be minimized and relaxed before each MD run',
                                  type=str, default='start_conf.gro',
                                  metavar='    (start_conf.gro)')
    sim_filenames_args.add_argument('-cg_top', dest='top_input_filename',
                                  help='TOP file used for iterative simulation', type=str,
                                  default='system.top', metavar='        (system.top)')
    sim_filenames_args.add_argument('-cg_mdp_mini', dest='mdp_minimization_filename',
                                  help='MDP file used for minimization runs', type=str,
                                  default='mini.mdp', metavar='     (mini.mdp)')
    sim_filenames_args.add_argument('-cg_mdp_equi', dest='mdp_equi_filename',
                                  help='MDP file used for equilibration runs', type=str,
                                  default='equi.mdp', metavar='     (equi.mdp)')
    sim_filenames_args.add_argument('-cg_mdp_md', dest='mdp_md_filename',
                                  help='MDP file used for the MD runs analyzed for optimization',
                                  type=str, default='md.mdp', metavar='         (md.mdp)')

    optional_args4 = args_parser.add_argument_group(
    opt_args_header + '\n\n' + bullet + 'FILES HANDLING')
    optional_args4.add_argument('-in_dir', dest='input_folder',
                              help='Additional prefix path used to find argument-provided files\nIf ambiguous, files found without prefix are preferred',
                              type=str, default='.', metavar='')
    optional_args4.add_argument('-out_dir', dest='output_folder',
                              help='Directory where to store all outputs of this program\nDefault -out_dir is named after timestamp',
                              type=str, default='', metavar='')

    optional_args1 = args_parser.add_argument_group(bullet + 'GROMACS SETTINGS')
    optional_args1.add_argument('-gmx', dest='gmx_path', help=config.help_gmx_path, type=str,
                              default=config.gmx_path,
                              metavar='                  ' + scg.par_wrap(config.gmx_path))
    optional_args1.add_argument('-nt', dest='nb_threads',
                              help="Nb of threads to use, forwarded to 'gmx mdrun -nt'", type=int,
                              default=0, metavar='                     (0)')
    optional_args1.add_argument('-mpi', dest='mpi_tasks',
                                help="Nb of mpi programs (X), triggers 'mpirun -np X gmx'", type=int,
                                default=0, metavar='')
    optional_args1.add_argument('-gpu_id', dest='gpu_id',
                              help='String (use quotes) space-separated list of GPU device IDs',
                              type=str, default='', metavar='')
    optional_args1.add_argument('-gmx_args_str', dest='gmx_args_str',
                              help='String (use quotes) of arguments to forward to gmx mdrun\nIf provided, arguments -nt and -gpu_id are ignored',
                              type=str, default='', metavar='')
    optional_args1.add_argument('-mini_maxwarn', dest='mini_maxwarn',
                              help='Max. number of warnings to ignore, forwarded to gmx\ngrompp -maxwarn at each minimization step',
                              type=int, default=1, metavar='           (1)')
    optional_args1.add_argument('-sim_kill_delay', dest='sim_kill_delay',
                              help='Time (s) after which to kill a simulation that has not been\nwriting into its log file, in case a simulation gets stuck',
                              type=int, default=60, metavar='        (60)')

    optional_args6 = args_parser.add_argument_group(bullet + 'CG MODEL FORCE CONSTANTS')
    optional_args6.add_argument('-max_fct_bonds_f1', dest='default_max_fct_bonds_opti',
                              help=config.help_max_fct_bonds, type=float,
                              default=config.default_max_fct_bonds_opti,
                              metavar='   ' + scg.par_wrap(config.default_max_fct_bonds_opti))
    optional_args6.add_argument('-max_fct_angles_f1', dest='default_max_fct_angles_opti_f1',
                              help=config.help_max_fct_angles_f1, type=float,
                              default=config.default_max_fct_angles_opti_f1,
                              metavar='   ' + scg.par_wrap(config.default_max_fct_angles_opti_f1))
    optional_args6.add_argument('-max_fct_angles_f2', dest='default_max_fct_angles_opti_f2',
                              help=config.help_max_fct_angles_f2, type=float,
                              default=config.default_max_fct_angles_opti_f2,
                              metavar='   ' + scg.par_wrap(config.default_max_fct_angles_opti_f2))
    optional_args6.add_argument('-max_fct_dihedrals_f149',
                              dest='default_abs_range_fct_dihedrals_opti_func_with_mult',
                              help=config.help_max_fct_dihedrals_with_mult, type=float,
                              default=config.default_abs_range_fct_dihedrals_opti_func_with_mult,
                              metavar='' + scg.par_wrap(
                                config.default_abs_range_fct_dihedrals_opti_func_with_mult))
    optional_args6.add_argument('-max_fct_dihedrals_f2',
                              dest='default_abs_range_fct_dihedrals_opti_func_without_mult',
                              help=config.help_max_fct_dihedrals_without_mult, type=float,
                              default=config.default_abs_range_fct_dihedrals_opti_func_without_mult,
                              metavar='' + scg.par_wrap(
                                config.default_abs_range_fct_dihedrals_opti_func_without_mult))

    optional_args5 = args_parser.add_argument_group(bullet + 'CG MODEL SCORING')
    optional_args5.add_argument('-cg_time_short', dest='sim_duration_short',
                              help='Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycles 1 and 2, this will modify MDP file for the MD runs',
                              type=float, default=10, metavar='         (10)')
    optional_args5.add_argument('-cg_time_long', dest='sim_duration_long',
                              help='Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycle 3, this will modify MDP file for the MD runs',
                              type=float, default=25, metavar='          (25)')
    optional_args5.add_argument('-b2a_score_fact', dest='bonds2angles_scoring_factor',
                              help=config.help_bonds2angles_scoring_factor, type=float,
                              default=config.bonds2angles_scoring_factor,
                              metavar='       ' + scg.par_wrap(config.bonds2angles_scoring_factor))
    optional_args5.add_argument('-bw_constraints', dest='bw_constraints',
                              help=config.help_bw_constraints, type=float,
                              default=config.bw_constraints,
                              metavar='     ' + scg.par_wrap(config.bw_constraints))
    optional_args5.add_argument('-bw_bonds', dest='bw_bonds', help=config.help_bw_bonds, type=float,
                              default=config.bw_bonds,
                              metavar='            ' + scg.par_wrap(config.bw_bonds))
    optional_args5.add_argument('-bw_angles', dest='bw_angles', help=config.help_bw_angles,
                              type=float, default=config.bw_angles,
                              metavar='            ' + scg.par_wrap(config.bw_angles))
    optional_args5.add_argument('-bw_dihedrals', dest='bw_dihedrals', help=config.help_bw_dihedrals,
                              type=float, default=config.bw_dihedrals,
                              metavar='         ' + scg.par_wrap(config.bw_dihedrals))
    optional_args5.add_argument('-bonds_max_range', dest='bonded_max_range',
                              help=config.help_bonds_max_range, type=float,
                              default=config.bonds_max_range,
                              metavar='       ' + scg.par_wrap(config.bonds_max_range))

    optional_args2 = args_parser.add_argument_group(bullet + 'CG MODEL SCALING')
    optional_args2.add_argument('-aa_rg_offset', dest='aa_rg_offset',
                              help='Radius of gyration offset (nm) to be applied to AA data\naccording to your potential bonds rescaling (for display only)',
                              type=float, default=0.00, metavar='        ' + scg.par_wrap('0.00'))
    optional_args2.add_argument('-bonds_scaling', dest='bonds_scaling',
                              help=config.help_bonds_scaling, type=float,
                              default=config.bonds_scaling,
                              metavar='        ' + scg.par_wrap(config.bonds_scaling))
    optional_args2.add_argument('-bonds_scaling_str', dest='bonds_scaling_str',
                              help=config.help_bonds_scaling_str, type=str,
                              default=config.bonds_scaling_str, metavar='')
    optional_args2.add_argument('-min_bonds_length', dest='min_bonds_length',
                              help=config.help_min_bonds_length, type=float,
                              default=config.min_bonds_length,
                              metavar='     ' + scg.par_wrap(config.min_bonds_length))

    optional_args3 = args_parser.add_argument_group(bullet + 'OTHERS')
    optional_args3.add_argument('-temp', dest='temp',
                              help='Temperature used to perform Boltzmann inversion (K)',
                              type=float, default=config.sim_temperature,
                              metavar='                 ' + scg.par_wrap(config.sim_temperature))
    optional_args3.add_argument('-keep_all_sims', dest='keep_all_sims',
                              help='Store all gmx files for all simulations, may use disk space',
                              action='store_true', default=False)
    optional_args3.add_argument('-h', '--help', help='Show this help message and exit', action='help')
    optional_args3.add_argument('-v', '--verbose', dest='verbose', help=config.help_verbose,
                              action='store_true', default=False)

    # display help if script was called without arguments
    if len(sys.argv) == 1:
        args_parser.print_help()
        sys.exit()

    # arguments handling, display command line if help or no arguments provided
    # argcomplete.autocomplete(parser)
    ns = args_parser.parse_args()

    # do NOT display the stack by default
    if not ns.verbose:
        sys.tracebacklimit = 0

    input_cmdline = ' '.join(map(cmd_quote, sys.argv))
    ns.exec_folder = time.strftime(
    "MODEL_OPTI__STARTED_%d-%m-%Y_%Hh%Mm%Ss")  # default folder name for all files of this optimization run, in case none is provided
    if ns.output_folder != '':
        ns.exec_folder = ns.output_folder

    print('Working directory:', os.getcwd())
    print('Command line:', input_cmdline)
    print('Results directory:', ns.exec_folder)

    run(ns)

if __name__ == "__main__":
    main()
