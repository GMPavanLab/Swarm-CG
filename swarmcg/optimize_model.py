import warnings

# some numpy version have this ufunc warning at import + many packages call numpy and display annoying warnings
warnings.filterwarnings("ignore")
import os, sys, re, shutil, subprocess, time, copy, contextlib
from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS
from shlex import quote as cmd_quote
from fstpso import FuzzyPSO 
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
import MDAnalysis as mda
from . import config
from . import swarmCG as scg
warnings.resetwarnings()


def main():

  from numpy import VisibleDeprecationWarning
  warnings.filterwarnings("ignore", category=VisibleDeprecationWarning) # filter MDAnalysis + numpy deprecation stuff that is annoying

  # TODO: allow to feed a JSON file or DICT-like string for which bonds group to rescale for AA
  # TODO: allow to feed a JSON file for cycles of optimization ?? this is more optional but useful for big stuff possibly
  # TODO: if using SASA through GMX SASA, ensure vdwradii.dat contains the MARTINI radii
  # TODO: give a warning when users specify a bond scaling without specifying an Rg offset !!!

  # TODO: AT OPTI CYCLE 2, FIND ANGLES THAT ARE TOO STEEP (CG) AND WHEN GENERATING THE NEW GUESSES, PUT 10-30-50-70% OF THE CURRENT BEST FORCE CONSTANT IN SEVERAL PARTICLES !!!!!!!!!

  # NOTE: gmx trjconv and sasa may produce bugs when using TPR produced with gromacs v5, only current solution seems to be implementing the SASA calculation using MDTraj


  #####################################
  # ARGUMENTS HANDLING / HELP DISPLAY #
  #####################################

  print(scg.header_package('                    Module: CG model optimization\n'))

  args_parser = ArgumentParser(description='''\
This module automatically optimizes the bonded parameters of a CG model to best match the bonds,
angles and dihedrals distributions of a reference AA model. Different sets of bonded parameters
are explored via swarm optimization (FST-PSO) and iterative CG simulations. Bonded parameters are
evaluated for the matching they produce between AA and CG distributions via a scoring function
relying on the Earth Movers' Distance (EMD/Wasserstein). The process is designed to execute in
4-24h on a standard desktop machine, according to hardware, molecule size and simulations setup. 

This module has 2 optimization modes:

  (1) TUNE BOTH BONDS LENGTHS, ANGLES/DIHEDRALS VALUES AND THEIR FORCE CONSTANTS. First uses
      Boltzmann Inversion to estimate bonds lengths, angles/dihedrals values and their force
      constants, then runs optimization to best fit the reference AA-mapped distributions.

  (2) TUNE ONLY FORCE CONSTANTS FOR ANGLES/DIHEDRALS VALUES AND ALL PARAMETERS FOR BONDS.
      Equilibrium values of angles/dihedrals provided in the preliminary  CG ITP model are
      conserved while optimization best fits reference AA-mapped distributions.

Independently of parameters, the expected input is:

  (1) Atomistic trajectory of the molecule   (gromacs binary TPR + trajectory files XTC TRR)
  (2) Mapping file, atoms to CG beads        (gromacs NDX format)
  (3) CG model ITP file to be optimized      (group identical bonds/angles/dihedrals, see below)
  (4) CG simulation files                    (initial configuration GRO + system TOP + MDP files)

You can prepare a directory using default input filenames, then provide only argument -in_dir.
If -in_dir is provided, all filenames provided as arguments will also be searched for within
this directory. Demonstration data are available at '''+config.github_url+'''.

Arguments allows to specify scaling of the AA bonds used as reference to optimize the CG model.
An image displaying all AA reference distributions will be created at the very beginning of the
optimization process. You can check it to make sure scaling is conform to your expectations.

The CG model preliminary ITP file follows the standard ITP format, with one subtlety. The file
can include groups of bonds, angles and dihedrals that will be considered identical. Their
distributions will be averaged within groups. This is important to obtain reliable results for
symmetrical molecules. Groups can be formed using empty line(s) or comment(s), like this:

  [ angles ]

  ; i     j     k    funct   angle  force.c.
  ; grp 1
    5     6    10        1     150       40      ; NOTE 1: force constants can be set to 0
    9     8    11        1     150       40      ;         in the prelim. model to optimize
  ; grp 2
    1     6    10        2     120        0      ; NOTE 2: either comment(s) or empty line(s)
    4     8    11        2     120        0      ;         separate groups of bonds/ang/dihe.

The AA trajectory is mapped on-the-fly using file from argument -cg_map, which uses gromacs NDX
file format. Periodic boundary conditions are handled internally if the input AA trajectory
contains box dimensions.''', formatter_class=lambda prog: RawTextHelpFormatter(prog, width=135, max_help_position=52), add_help=False, usage=SUPPRESS)

  # TODO: handle trajectories for which no box informations are provided
  # TODO: explain what is modified in the MDP
  # TODO: explain module analyze_opti_moves.py can be used to monitor optimization at any point of the process
  # TODO: end the help message by a new frame with examples from the demo data

  req_args_header = config.sep_close+'\n|                                     REQUIRED ARGUMENTS                                      |\n'+config.sep_close
  opt_args_header = config.sep_close+'\n|                                     OPTIONAL ARGUMENTS                                      |\n'+config.sep_close
  # bullet = '❭'
  # bullet = '★'
  # bullet = '|'
  bullet = ' '

  optional_args0 = args_parser.add_argument_group(req_args_header+'\n\n'+bullet+'EXECUTION MODE')
  optional_args0.add_argument('-exec_mode', dest='exec_mode', help='MODE 1: Tune both bonds lengths, angles/dihedrals values\n        and their force constants\nMODE 2: Like MODE 1 but angles/dihedrals values in the prelim.\n        CG model ITP are conserved during optimization', type=int, default=1, metavar='              (1)')

  required_args = args_parser.add_argument_group(bullet+'REFERENCE AA MODEL')
  required_args.add_argument('-aa_tpr', dest='aa_tpr_filename', help=config.help_aa_tpr, type=str, default=config.metavar_aa_tpr, metavar='      '+scg.par_wrap(config.metavar_aa_tpr))
  required_args.add_argument('-aa_traj', dest='aa_traj_filename', help=config.help_aa_traj, type=str, default=config.metavar_aa_traj, metavar='      '+scg.par_wrap(config.metavar_aa_traj))
  required_args.add_argument('-cg_map', dest='cg_map_filename', help=config.help_cg_map, type=str, default=config.metavar_cg_map, metavar='        '+scg.par_wrap(config.metavar_cg_map))

  sim_filenames_args = args_parser.add_argument_group(bullet+'CG MODEL OPTIMIZATION')
  sim_filenames_args.add_argument('-cg_itp', dest='cg_itp_filename', help='ITP file of the CG model to optimize', type=str, default=config.metavar_cg_itp, metavar='      '+scg.par_wrap(config.metavar_cg_itp))
  sim_filenames_args.add_argument('-cg_gro', dest='gro_input_filename', help='Starting GRO file used for iterative simulation\nWill be minimized and relaxed before each MD run', type=str, default='start_conf.gro', metavar='    (start_conf.gro)')
  sim_filenames_args.add_argument('-cg_top', dest='top_input_filename', help='TOP file used for iterative simulation', type=str, default='system.top', metavar='        (system.top)')
  sim_filenames_args.add_argument('-cg_mdp_mini', dest='mdp_minimization_filename', help='MDP file used for minimization runs', type=str, default='mini.mdp', metavar='     (mini.mdp)')
  sim_filenames_args.add_argument('-cg_mdp_equi', dest='mdp_equi_filename', help='MDP file used for equilibration runs', type=str, default='equi.mdp', metavar='     (equi.mdp)')
  sim_filenames_args.add_argument('-cg_mdp_md', dest='mdp_md_filename', help='MDP file used for the MD runs analyzed for optimization', type=str, default='md.mdp', metavar='         (md.mdp)')

  optional_args4 = args_parser.add_argument_group(opt_args_header+'\n\n'+bullet+'FILES HANDLING')
  optional_args4.add_argument('-in_dir', dest='input_folder', help='Additional prefix path used to find argument-provided files\nIf ambiguous, files found without prefix are preferred', type=str, default='.', metavar='')
  optional_args4.add_argument('-out_dir', dest='output_folder', help='Directory where to store all outputs of this program\nDefault -out_dir is named after timestamp', type=str, default='', metavar='')

  optional_args1 = args_parser.add_argument_group(bullet+'GROMACS SETTINGS')
  optional_args1.add_argument('-gmx', dest='gmx_path', help=config.help_gmx_path, type=str, default=config.gmx_path, metavar='                  '+scg.par_wrap(config.gmx_path))
  optional_args1.add_argument('-nt', dest='nb_threads', help='Number of threads to use, forwarded to gmx mdrun -nt', type=int, default=0, metavar='                     (0)')
  optional_args1.add_argument('-gpu_id', dest='gpu_id', help='String (use quotes) space-separated list of GPU device IDs', type=str, default='', metavar='')
  optional_args1.add_argument('-gmx_args_str', dest='gmx_args_str', help='String (use quotes) of arguments to forward to gmx mdrun\nIf provided, arguments -nt and -gpu_id are ignored', type=str, default='', metavar='')
  optional_args1.add_argument('-mini_maxwarn', dest='mini_maxwarn', help='Max. number of warnings to ignore, forwarded to gmx\ngrompp -maxwarn at each minimization step', type=int, default=1, metavar='           (1)')
  optional_args1.add_argument('-sim_kill_delay', dest='sim_kill_delay', help='Time (s) after which to kill a simulation that has not been\nwriting into its log file, in case a simulation gets stuck', type=int, default=60, metavar='        (60)')

  optional_args2 = args_parser.add_argument_group(bullet+'CG MODEL SCALING')
  optional_args2.add_argument('-aa_rg_offset', dest='aa_rg_offset', help='Radius of gyration offset (nm) to be applied to AA data\naccording to your potential bonds rescaling (for display only)', type=float, default=0.00, metavar='        '+scg.par_wrap('0.00'))
  optional_args2.add_argument('-bonds_scaling', dest='bonds_scaling', help=config.help_bonds_scaling, type=float, default=config.bonds_scaling, metavar='        '+scg.par_wrap(config.bonds_scaling))
  optional_args2.add_argument('-bonds_scaling_str', dest='bonds_scaling_str', help=config.help_bonds_scaling_str, type=str, default=config.bonds_scaling_str, metavar='')
  optional_args2.add_argument('-min_bonds_length', dest='min_bonds_length', help=config.help_min_bonds_length, type=float, default=config.min_bonds_length, metavar='     '+scg.par_wrap(config.min_bonds_length))

  optional_args5 = args_parser.add_argument_group(bullet+'CG MODEL SCORING')
  optional_args5.add_argument('-cg_time_short', dest='sim_duration_short', help='Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycles 1 and 2, this will modify MDP file for the MD runs', type=float, default=10, metavar='         (10)')
  optional_args5.add_argument('-cg_time_long', dest='sim_duration_long', help='Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycle 3, this will modify MDP file for the MD runs', type=float, default=25, metavar='          (25)')
  optional_args5.add_argument('-b2a_score_fact', dest='bonds2angles_scoring_factor', help=config.help_bonds2angles_scoring_factor, type=float, default=config.bonds2angles_scoring_factor, metavar='       '+scg.par_wrap(config.bonds2angles_scoring_factor))
  optional_args5.add_argument('-bw_constraints', dest='bw_constraints', help=config.help_bw_constraints, type=float, default=config.bw_constraints, metavar='     '+scg.par_wrap(config.bw_constraints))
  optional_args5.add_argument('-bw_bonds', dest='bw_bonds', help=config.help_bw_bonds, type=float, default=config.bw_bonds, metavar='            '+scg.par_wrap(config.bw_bonds))
  optional_args5.add_argument('-bw_angles', dest='bw_angles', help=config.help_bw_angles, type=float, default=config.bw_angles, metavar='              '+scg.par_wrap(config.bw_angles))
  optional_args5.add_argument('-bw_dihedrals', dest='bw_dihedrals', help=config.help_bw_dihedrals, type=float, default=config.bw_dihedrals, metavar='           '+scg.par_wrap(config.bw_dihedrals))
  optional_args5.add_argument('-bonds_max_range', dest='bonded_max_range', help=config.help_bonds_max_range, type=float, default=config.bonds_max_range, metavar='        '+scg.par_wrap(config.bonds_max_range))

  optional_args6 = args_parser.add_argument_group(bullet+'CG MODEL FORCE CONSTANTS')
  optional_args6.add_argument('-max_fct_bonds_f1', dest='default_max_fct_bonds_opti', help=config.help_max_fct_bonds, type=float, default=config.default_max_fct_bonds_opti, metavar='   '+scg.par_wrap(config.default_max_fct_bonds_opti))
  optional_args6.add_argument('-max_fct_angles_f1', dest='default_max_fct_angles_opti_f1', help=config.help_max_fct_angles_f1, type=float, default=config.default_max_fct_angles_opti_f1, metavar='   '+scg.par_wrap(config.default_max_fct_angles_opti_f1))
  optional_args6.add_argument('-max_fct_angles_f2', dest='default_max_fct_angles_opti_f2', help=config.help_max_fct_angles_f2, type=float, default=config.default_max_fct_angles_opti_f2, metavar='   '+scg.par_wrap(config.default_max_fct_angles_opti_f2))
  optional_args6.add_argument('-max_fct_dihedrals_f149', dest='default_abs_range_fct_dihedrals_opti_func_with_mult', help=config.help_max_fct_dihedrals_with_mult, type=float, default=config.default_abs_range_fct_dihedrals_opti_func_with_mult, metavar=''+scg.par_wrap(config.default_abs_range_fct_dihedrals_opti_func_with_mult))
  optional_args6.add_argument('-max_fct_dihedrals_f2', dest='default_max_fct_dihedrals_opti_func_without_mult', help=config.help_max_fct_dihedrals_without_mult, type=float, default=config.default_max_fct_dihedrals_opti_func_without_mult, metavar=''+scg.par_wrap(config.default_max_fct_dihedrals_opti_func_without_mult))

  optional_args3 = args_parser.add_argument_group(bullet+'OTHERS')
  optional_args3.add_argument('-temp', dest='temp', help='Temperature used to perform Boltzmann inversion (K)', type=float, default=config.sim_temperature, metavar='                 '+scg.par_wrap(config.sim_temperature))
  optional_args3.add_argument('-keep_all_sims', dest='keep_all_sims', help='Store all gmx files for all simulations, may use disk space', action='store_true', default=False)
  optional_args3.add_argument('-h', '--help', help='Show this help message and exit', action='help')
  optional_args3.add_argument('-v', '--verbose', dest='verbose', help=config.help_verbose, action='store_true', default=False)

  # display help if script was called without arguments
  if len(sys.argv) == 1:
      args_parser.print_help()
      sys.exit()

  # arguments handling, display command line if help or no arguments provided
  # argcomplete.autocomplete(parser)
  ns = args_parser.parse_args()
  input_cmdline = ' '.join(map(cmd_quote, sys.argv))
  ns.exec_folder = time.strftime("MODEL_OPTI__STARTED_%d-%m-%Y_%Hh%Mm%Ss") # default folder name for all files of this optimization run, in case none is provided
  if ns.output_folder != '':
  	ns.exec_folder = ns.output_folder
  print('Working directory:', os.getcwd())
  print('Command line:', input_cmdline)
  print('Results directory:', ns.exec_folder)

  # namespace variables not directly linked to arguments for plotting or for global package interpretation
  ns.mismatch_order = False
  ns.row_x_scaling = True
  ns.row_y_scaling = True
  ns.ncols_max = 0 # 0 to display all
  # ns.atom_only = False
  ns.molname_in = None # if None the first found using TPR atom ordering will be used
  ns.process_alive_time_sleep = 10 # nb of seconds between process alive check cycles
  ns.process_alive_nb_cycles_dead = int(ns.sim_kill_delay / ns.process_alive_time_sleep) # nb of cycles without .log file bytes size changes to determine that the MD run is stuck
  ns.bonds_rescaling_performed = False # for user information display

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
  print()
  print(config.sep_close)
  print('| PRE-PROCESSING AND CONTROLS                                                                 |')
  print(config.sep_close)
  # print()

  # TODO: check that at least 10-20% of the simulations of the 1st swarm iteration finished properly, otherwise lower all energies or tell the user he is not writting into the log file regularly enough
  # TODO: test this program with ITP files that contain all the different dihedral functions, angles functions, constraints etc
  # TODO: find some fuzzy logic to determine number of swarm iterations + take some large margin to ensure it will optimize correctly

  # avoid overwriting an output directory of a previous optimization run
  if os.path.isfile(ns.exec_folder) or os.path.isdir(ns.exec_folder):
    sys.exit(config.header_error+'Provided output folder already exists, please delete existing folder manually or provide another folder name.')

  # check if we can find files at user-provided location(s)
  arg_entries = vars(ns) # dict view of the arguments namespace
  user_provided_filenames = ['aa_tpr_filename', 'aa_traj_filename', 'cg_map_filename', 'cg_itp_filename', 'gro_input_filename', 'top_input_filename', 'mdp_minimization_filename', 'mdp_equi_filename', 'mdp_md_filename']
  args_names = ['aa_tpr', 'aa_traj', 'cg_map', 'cg_itp', 'cg_sim_gro', 'cg_sim_top', 'cg_sim_mdp_mini', 'cg_sim_mdp_equi', 'cg_sim_mdp_md']

  for i in range(len(user_provided_filenames)):
    arg_entry = user_provided_filenames[i]
    if not os.path.isfile(arg_entries[arg_entry]):
      data_folder_path = ns.input_folder+'/'+arg_entries[arg_entry]
      if ns.input_folder != '.' and os.path.isfile(data_folder_path):
        arg_entries[arg_entry] = data_folder_path
      else:
        if ns.input_folder == '':
          data_folder_path = arg_entries[arg_entry]
        sys.exit(config.header_error+'Cannot find file for argument -'+args_names[i]+' (expected at location: '+data_folder_path+')')

  # check that gromacs alias is correct
  with open(os.devnull, 'w') as devnull:
    try:
      subprocess.call(ns.gmx_path, stdout=devnull, stderr=devnull)
    except OSError:
      sys.exit(config.header_error+'Cannot find GROMACS using alias \''+ns.gmx_path+'\', please provide the right GROMACS alias or path')

  # check that ITP filename for the model to optimize is indeed included in the TOP file of the simulation directory
  # then find all TOP includes for copying files for simulations at each iteration
  top_includes_filenames = []
  with open(ns.top_input_filename, 'r') as fp:
    all_top_lines = fp.read()
    if ns.cg_itp_basename not in all_top_lines:
      sys.exit(config.header_error+'The CG ITP model filename you provided is not included in your TOP file')

    top_lines = all_top_lines.split('\n')
    top_lines = [top_line.strip().split(';')[0] for top_line in top_lines] # split for comments
    for top_line in top_lines:
      if top_line.startswith('#include'):
        top_include = top_line.split()[1].replace('"', '').replace("'", '') # remove potential single and double quotes
        top_includes_filenames.append(top_include)
    # TODO: VERIFY THE PRESENCE OF ALLLLLLLLL TOP FILES, NOT ONLY THE CG MODEL'S

  # check gmx arguments conflicts
  if ns.gmx_args_str != '' and (ns.nb_threads != 0 or ns.gpu_id != ''):
  	print(config.header_warning+'Argument -gmx_args_str is provided together with one of arguments: -nb_threads, -gpu_id\nOnly argument -gmx_args_str will be used during this execution')

  # check bonds scaling arguments conflicts
  if (ns.bonds_scaling != config.bonds_scaling and ns.min_bonds_length != config.min_bonds_length) or (ns.bonds_scaling != config.bonds_scaling and ns.bonds_scaling_str != config.bonds_scaling_str) or (ns.min_bonds_length != config.min_bonds_length and ns.bonds_scaling_str != config.bonds_scaling_str):
  	sys.exit(config.header_error+'Only one of arguments -bonds_scaling, -bonds_scaling_str and -min_bonds_length can be provided\nPlease check your parameters')
  # if ns.bonds_scaling < 1:
  # 	sys.exit(config.header_error+'Bonds scaling factor is inferior to 1, please check your parameters')


  ##################
  # INITIALIZATION #
  ##################

  scg.set_MDA_backend(ns)
  ns.mda_backend = 'serial' # clusters execution

  # directory to write all files for current execution of optimizations routines
  os.mkdir(ns.exec_folder)
  os.mkdir(ns.exec_folder+'/.internal')
  os.mkdir(ns.exec_folder+'/'+config.distrib_plots_all_evals_dirname)
  os.mkdir(ns.exec_folder+'/'+config.log_files_all_evals_dirname)
  if ns.keep_all_sims:
  	os.mkdir(ns.exec_folder+'/'+config.sim_files_all_evals_dirname)

  # prepare a directory to be copied at each iteration of the optimization, to run the new simulation
  os.mkdir(ns.exec_folder+'/'+config.input_sim_files_dirname)
  user_provided_sim_files = ['cg_itp_filename', 'gro_input_filename', 'top_input_filename', 'mdp_minimization_filename', 'mdp_equi_filename', 'mdp_md_filename']

  for sim_file in user_provided_sim_files:
  	shutil.copy(arg_entries[sim_file], ns.exec_folder+'/'+config.input_sim_files_dirname)

  # get all TOP file includes copied into input simulation directory
  top_include_dirbase = os.path.dirname(arg_entries['top_input_filename'])
  for top_include in top_includes_filenames:
  	# shutil.copy(top_include_dirbase+'/'+top_include, ns.exec_folder+'/'+config.input_sim_files_dirname) # PROBLEM LUCA
    # shutil.copy(ns.input_folder+'/'+top_include_dirbase+'/'+top_include, ns.exec_folder+'/'+config.input_sim_files_dirname) # PROBLEM WITH PIP

    # print(ns.input_folder, top_include_dirbase, top_include)
    shutil.copy(ns.input_folder+'/'+top_include, ns.exec_folder+'/'+config.input_sim_files_dirname)

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

  ns.nb_eval = 0 # global count of evaluation steps
  ns.start_opti_ts = datetime.now().timestamp()
  ns.total_eval_time, ns.total_gmx_time, ns.total_model_eval_time = 0, 0, 0

  scg.create_bins_and_dist_matrices(ns) # bins for EMD calculations
  scg.read_ndx_atoms2beads(ns) # read mapping, get atoms accurences in beads
  scg.get_atoms_weights_in_beads(ns) # get weights of atoms within beads

  print()

  # read starting CG ITP file
  with open(ns.cg_itp_filename, 'r') as fp:
  	itp_lines = fp.read().split('\n')
  	itp_lines = [itp_line.strip() for itp_line in itp_lines]
  	scg.read_cg_itp_file(ns, itp_lines) # loads ITP object that contains our reference atomistic data -- won't ever be modified during execution

  # touch results files to be appended to later
  with open(ns.exec_folder+'/'+config.opti_perf_recap_file, 'w') as fp:
  	# TODO: print that file has been generated with Opti-CG etc -- do this for basically all files
  	# TODO: add some info on the opti cycles ??
  	fp.write('# nb constraints: '+str(ns.nb_constraints)+'\n')
  	fp.write('# nb bonds: '+str(ns.nb_bonds)+'\n')
  	fp.write('# nb angles: '+str(ns.nb_angles)+'\n')
  	fp.write('# nb dihedrals: '+str(ns.nb_dihedrals)+'\n')
  	fp.write('#\n')
  	fp.write('# opti_cycle nb_eval fit_score_all fit_score_cstrs_bonds fit_score_angles fit_score_dihedrals eval_score Rg_AA_mapped Rg_CG parameters_set eval_time current_total_time\n')
  with open(ns.exec_folder+'/'+config.opti_pairwise_distances_file, 'w'):
  	pass

  # process specific bonds scaling string, if provided
  ns.bonds_scaling_specific = None
  if ns.bonds_scaling_str != config.bonds_scaling_str:
    sp_str = ns.bonds_scaling_str.split()
    if len(sp_str) % 2 != 0:
      sys.exit(config.header_error+'Cannot interpret argument -bonds_scaling_str as provided: \''+ns.bonds_scaling_str+'\'\nPlease check your parameters, or help for an example')
    ns.bonds_scaling_specific = dict()
    i = 0
    try:
      while i < len(sp_str):
        geom_id = sp_str[i][1:]
        if sp_str[i][0].upper() == 'C':
          if int(geom_id) > ns.nb_constraints:
            sys.exit(config.header_error+'Cannot interpret argument -bonds_scaling_str as provided: \''+ns.bonds_scaling_str+'\'\nA constraint group id exceeds the number of constraints groups defined in the input CG ITP file\nPlease check your parameters, or help for an example')
          if not 'C'+geom_id in ns.bonds_scaling_specific:
            if float(sp_str[i+1]) < 0:
              sys.exit(config.header_error+'Cannot interpret argument -bonds_scaling_str as provided: \''+ns.bonds_scaling_str+'\'\nYou cannot provide negative values for average distribution length\nPlease check your parameters, or help for an example')
            ns.bonds_scaling_specific['C'+geom_id] = float(sp_str[i+1])
          else:
            sys.exit(config.header_error+'Cannot interpret argument -bonds_scaling_str as provided: \''+ns.bonds_scaling_str+'\'\nA constraint group id is provided multiple times (id: '+str(geom_id)+')\nPlease check your parameters, or help for an example')
        elif sp_str[i][0].upper() == 'B':
          if int(geom_id) > ns.nb_bonds:
            sys.exit(config.header_error+'Cannot interpret argument -bonds_scaling_str as provided: \''+ns.bonds_scaling_str+'\'\nA bond group id exceeds the number of bonds groups defined in the input CG ITP file\nPlease check your parameters, or help for an example')
          if not 'B'+geom_id in ns.bonds_scaling_specific:
            if float(sp_str[i+1]) < 0:
              sys.exit(config.header_error+'Cannot interpret argument -bonds_scaling_str as provided: \''+ns.bonds_scaling_str+'\'\nYou cannot provide negative values for average distribution length\nPlease check your parameters, or help for an example')
            ns.bonds_scaling_specific['B'+geom_id] = float(sp_str[i+1])
          else:
            sys.exit(config.header_error+'Cannot interpret argument -bonds_scaling_str as provided: \''+ns.bonds_scaling_str+'\'\nA bond group id is provided multiple times (id: '+str(geom_id)+')\nPlease check your parameters, or help for an example')
        i += 2
    except ValueError:
      sys.exit(config.header_error+'Cannot interpret argument -bonds_scaling_str as provided: \''+ns.bonds_scaling_str+'\'\nPlease check your parameters, or help for an example')

  # read atom mapped trajectory + find domains boundaries for values ranges (NOT the force constants, for which it is config/user defined already)
  print()
  scg.read_aa_traj(ns)
  scg.load_aa_data(ns)
  scg.make_aa_traj_whole_for_selected_mols(ns)

  print('\nCalculating bonds, angles and dihedrals distributions for reference AA-mapped model')

  # for each CG bead, create atom groups for trajectory geoms calculation using mass and atom weights across beads
  scg.get_beads_MDA_atomgroups(ns)

  ns.gyr_aa_mapped, ns.gyr_aa_mapped_std = None, None # will be computed one single time with model evaluation script
  ns.sasa_aa_mapped, ns.sasa_aa_mapped_std = None, None # will be computed one single time with model evaluation script

  ns.domains_val = {'constraint': [], 'bond': [], 'angle': [], 'dihedral': []}
  ns.data_BI = {'bond': [], 'angle': [], 'dihedral': []} # store hists for BI, std and possibly some other stats

  # create all ref atom histograms to be used for pairwise distributions comparisons + find average geoms values as first guesses (without BI at this point)
  # get ref atom hists + find very first distances guesses for constraints groups
  for grp_constraint in range(ns.nb_constraints):

    constraint_avg, constraint_hist, constraint_values = scg.get_AA_bonds_distrib(ns, beads_ids=ns.cg_itp['constraint'][grp_constraint]['beads'], grp_type='constraint group', grp_nb=grp_constraint)
    # if ns.exec_mode == 1:
    ns.cg_itp['constraint'][grp_constraint]['value'] = constraint_avg
    ns.cg_itp['constraint'][grp_constraint]['avg'] = constraint_avg
    ns.cg_itp['constraint'][grp_constraint]['hist'] = constraint_hist

    ns.domains_val['constraint'].append([round(np.min(constraint_values), 3), round(np.max(constraint_values), 3)])

  # get ref atom hists + find very first distances and force constants guesses for bonds groups
  for grp_bond in range(ns.nb_bonds):

    bond_avg, bond_hist, bond_values = scg.get_AA_bonds_distrib(ns, beads_ids=ns.cg_itp['bond'][grp_bond]['beads'], grp_type='bond group', grp_nb=grp_bond)
    # if ns.exec_mode == 1:
    ns.cg_itp['bond'][grp_bond]['value'] = bond_avg
    ns.cg_itp['bond'][grp_bond]['avg'] = bond_avg
    ns.cg_itp['bond'][grp_bond]['hist'] = bond_hist

    xmin, xmax = min(np.inf, ns.bins_bonds[np.min(np.nonzero(bond_hist))]), max(-np.inf, ns.bins_bonds[np.max(np.nonzero(bond_hist))+1])
    xmin, xmax = xmin-ns.bw_bonds, xmax+ns.bw_bonds	
    ns.data_BI['bond'].append([np.histogram(bond_values, range=(xmin, xmax), bins=config.bi_nb_bins)[0], np.std(bond_values), np.mean(bond_values), (xmin, xmax)])

    ns.domains_val['bond'].append([round(np.min(bond_values), 3), round(np.max(bond_values), 3)]) # boundaries of force constats during optimization

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

    ns.domains_val['angle'].append([round(np.min(angle_values_deg), 2), round(np.max(angle_values_deg), 2)]) # boundaries of force constats during optimization

  # get ref atom hists + find very first values and force constants guesses for dihedrals groups
  for grp_dihedral in range(ns.nb_dihedrals):

    dihedral_avg, dihedral_hist, dihedral_values_deg, dihedral_values_rad = scg.get_AA_dihedrals_distrib(ns, beads_ids=ns.cg_itp['dihedral'][grp_dihedral]['beads'])
    if ns.exec_mode == 1: # the angle value for dihedral will be calculated from the BI fit, because for dihedrals it makes no sense to use the average
      ns.cg_itp['dihedral'][grp_dihedral]['value'] = dihedral_avg
    ns.cg_itp['dihedral'][grp_dihedral]['avg'] = dihedral_avg
    ns.cg_itp['dihedral'][grp_dihedral]['hist'] = dihedral_hist

    xmin, xmax = -180, 180
    ns.data_BI['dihedral'].append([np.histogram(dihedral_values_rad, range=(np.deg2rad(xmin), np.deg2rad(xmax)), bins=2*config.bi_nb_bins)[0], np.std(dihedral_values_rad), np.mean(dihedral_values_rad), (xmin, xmax)])

    ns.domains_val['dihedral'].append([round(np.min(dihedral_values_deg), 2), round(np.max(dihedral_values_deg), 2)]) # boundaries of force constats during optimization

  if not ns.bonds_rescaling_performed:
    print('  No bonds rescaling performed')

  # output png with all the reference distributions, so the user can check
  ns.atom_only = True
  ns.plot_filename = ns.exec_folder+'/'+config.ref_distrib_plots
  with contextlib.redirect_stdout(open(os.devnull, 'w')) as devnull:
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

  # THIS IS THE CURRENT CHOICE
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
  opti_cycles = [['constraint', 'bond', 'angle'], ['angle', 'dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
  sim_cycles = [0, 1, 2] # simulations types

  # for tests
  # sim_types = {0: {'sim_duration': ns.sim_duration_short, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 1, 'fct_guess_fact': 0.40},
  #        1: {'sim_duration': ns.sim_duration_short, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.30},
  #        2: {'sim_duration': ns.sim_duration_long, 'max_swarm_iter': 2, 'max_swarm_iter_without_new_global_best': 6, 'val_guess_fact': 0.25, 'fct_guess_fact': 0.20}}
  # opti_cycles = [['constraint', 'bond', 'angle'], ['angle', 'dihedral'], ['constraint', 'bond', 'angle', 'dihedral']] # optimization cycles to perform with given geom objects
  # sim_cycles = [0, 1, 2] # simulations types

  # NOTE: currently, due to an issue in FST-PSO, number of swarm iterations performed is +2 when compared to the numbers we feed

  ns.opti_itp = copy.deepcopy(ns.cg_itp) # the ITP object that will be optimized stepwise, at the end of each optimization cycle (geom type wise)
  ns.eval_nb_geoms = {'constraint': 0, 'bond': 0, 'angle': 0, 'dihedral': 0} # geoms to optimize at each step

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
  # print(opti_cycles)

  # state variables for the cycles of optimization
  ns.performed_init_BI = {'bond': False, 'angle': False, 'dihedral': False}
  ns.opti_geoms_all = set(geom for opti_cycle_geoms in opti_cycles for geom in opti_cycle_geoms)
  ns.best_fitness = [np.inf, None] # fitness_score, eval_step_best_score

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
    print()
    print(config.sep_close)
    print('| STARTING OPTIMIZATION CYCLE', ns.opti_cycle['nb_cycle'], '                                                              |')
    print('| Optimizing', geoms_display, ' '*(95-16-len(geoms_display)), '|')
    print(config.sep_close)

  	# actual BI to get the initial guesses of force constants, for all selected geoms at this given optimization step
    # BI is performed:
    # -- exec_mode 1: all values and force constants
    # -- exec_mode 2: values are not touched for angles and dihedrals, but all force constants are estimated
    scg.perform_BI(ns) # performed on object ns.out_itp

    # build vector for search space boundaries + create variations around the BI initial guesses
    search_space_boundaries = scg.get_search_space_boundaries(ns)
    # ns.worst_fit_score = round(len(search_space_boundaries) * config.sim_crash_EMD_indep_score, 3)
    ns.worst_fit_score = round(\
      np.sqrt((ns.nb_constraints+ns.nb_bonds) * config.sim_crash_EMD_indep_score) + \
      np.sqrt(ns.nb_angles * config.sim_crash_EMD_indep_score) + \
      np.sqrt(ns.nb_dihedrals * config.sim_crash_EMD_indep_score) \
      , 3)
    # nb_particles = int(10 + 2*np.sqrt(len(search_space_boundaries))) # formula used by FST-PSO to choose nb of particles, which defines the number of initial guesses we can use
    nb_particles = int(round(2 + np.sqrt(len(search_space_boundaries)))) # adapted to have less particles and fitted to our problems, which has good initial guesses and error driven initialization
    # nb_particles = 2 # for tests
    initial_guess_list = scg.get_initial_guess_list(ns, nb_particles)

  	# actual optimization
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
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
  print(config.sep_close)
  print('  FINISHED PROPERLY')
  print(config.sep_close)
  print()
  print('Total nb of evaluation steps:', ns.nb_eval)
  print('Best model obtained at evaluation step number:', ns.best_fitness[1])
  print()
  print('Total execution time :', total_time, 'h')
  print('Initialization time  :', init_time, 'h ('+str(round(init_time/total_time*100, 2))+' %)')
  print('Simulations time     :', ns.total_gmx_time, 'h ('+str(round(ns.total_gmx_time/total_time*100, 2))+' %)')
  print('Models scoring time  :', ns.total_model_eval_time, 'h ('+str(round(ns.total_model_eval_time/total_time*100, 2))+' %)')
  print()





