# some numpy version have this ufunc warning at import + many packages call numpy and display annoying warnings
import warnings

warnings.filterwarnings("ignore")
import os, sys
from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS
from shlex import quote as cmd_quote
import numpy as np

import matplotlib

import swarmcg.shared.styling
from swarmcg import swarmCG as scg
from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg.shared.styling import EVALUATE_DESCR

warnings.resetwarnings()
matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation


def run(ns):

	print()
	print(swarmcg.shared.styling.sep_close)
	print('| PRE-PROCESSING                                                                              |')
	print(swarmcg.shared.styling.sep_close)
	print()

	from numpy import VisibleDeprecationWarning
	warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)  # filter MDAnalysis + numpy deprecation stuff that is annoying

	# TODO: make it possible to feed a delta/offset for Rg in case the model has bonds scaling ?

	# get basenames for simulation files
	ns.cg_itp_basename = os.path.basename(ns.cg_itp_filename)

	# NOTE: some arguments exist only in the scope of optimization (optimize_model.py) or only in the scope of model
	#       evaluation (evaluate_mode.py), but still need to be defined here -> Change this to something less messy
	ns.molname_in = None
	ns.gyr_aa_mapped, ns.gyr_aa_mapped_std = None, None
	ns.sasa_aa_mapped, ns.sasa_aa_mapped_std = None, None
	ns.aa_rg_offset = 0  # TODO: allow an argument more in evaluate_model, like in optimiwe_model, for adding an offset to Rg
	ns.user_input = False
	ns.default_max_fct_bonds_opti = np.inf
	ns.default_max_fct_angles_opti_f1 = np.inf
	ns.default_max_fct_angles_opti_f2 = np.inf
	ns.default_abs_range_fct_dihedrals_opti_func_with_mult = np.inf
	ns.default_abs_range_fct_dihedrals_opti_func_without_mult = np.inf

	# scg.set_MDA_backend(ns)
	ns.mda_backend = 'serial'  # actually serial is faster because MDA is not properly parallelized atm

	if not os.path.isfile(ns.aa_tpr_filename):
		msg = (
			f"Cannot find topology file of the atomistic simulation at location: {ns.aa_tpr_filename}\n"
			f"(TPR or other portable topology formats supported by MDAnalysis)"
		)
		raise exceptions.MissingCoordinateFile(msg)
	if not os.path.isfile(ns.aa_traj_filename):
		msg = (
			f"Cannot find trajectory file of the atomistic simulation at location: {ns.aa_traj_filename}\n"
			f"(XTC, TRR, or other trajectory formats supported by MDAnalysis)"
		)
		raise exceptions.MissingTrajectoryFile(msg)

	if not os.path.isfile(ns.cg_map_filename):
		msg = (
			f"Cannot find CG beads mapping file at location: {ns.cg_map_filename}\n"
			f"(NDX-like file format)"
		)
		raise exceptions.MissingIndexFile(msg)

	if not os.path.isfile(ns.cg_itp_filename):
		msg = f"Cannot find ITP file of the CG model at location: {ns.cg_itp_filename}"
		raise exceptions.MissingItpFile(msg)

	# check bonds scaling arguments conflicts
	if (ns.bonds_scaling != config.bonds_scaling and ns.min_bonds_length != config.min_bonds_length) or (ns.bonds_scaling != config.bonds_scaling and ns.bonds_scaling_str != config.bonds_scaling_str) or (ns.min_bonds_length != config.min_bonds_length and ns.bonds_scaling_str != config.bonds_scaling_str):
		msg = (
			"Only one of arguments -bonds_scaling, -bonds_scaling_str and -min_bonds_length "
			"can be provided. Please check your parameters"
		)
		raise exceptions.InputArgumentError(msg)

	# check the mapping type
	ns.mapping_type = ns.mapping_type.upper()
	if ns.mapping_type != 'COM' and ns.mapping_type != 'COG':
		msg = "Mapping type provided via argument '-mapping' must be either COM or COG (Center of Mass or Center of Geometry)."
		raise exceptions.InputArgumentError(msg)

	# display parameters for function compare_models
	if not os.path.isfile(ns.cg_tpr_filename) or not os.path.isfile(ns.cg_traj_filename):
		# switch to atomistic mapping inspection exclusively (= do NOT plot the CG distributions)
		print('Could not find file(s) for either CG topology or trajectory')
		print('  Going for inspection of AA-mapped distributions exclusively')
		print()
		ns.atom_only = True
	else:
		ns.atom_only = False

	try:
		if not ns.plot_filename.split('.')[-1] in ['eps', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz']:
			ns.plot_filename = ns.plot_filename+'.png'
	except IndexError as e:
		ns.plot_filename = ns.plot_filename+'.png'

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

	print('\nMapping the trajectory from AA to CG representation')
	scg.initialize_cg_traj(ns)
	scg.map_aa2cg_traj(ns)
	print()

	scg.compare_models(ns, manual_mode=True, calc_sasa=False)


def main():

	print(swarmcg.shared.styling.header_package(
		'                Module: Model bonded terms assessment\n'))

	formatter = lambda prog: RawTextHelpFormatter(prog, width=135, max_help_position=52)
	args_parser = ArgumentParser(
		description=EVALUATE_DESCR,
		formatter_class=formatter,
		add_help=False,
		usage=SUPPRESS
	)

	all_args_header = swarmcg.shared.styling.sep_close + '\n|                                 REQUIRED/OPTIONAL ARGUMENTS                                 |\n' + swarmcg.shared.styling.sep_close
	bullet = ' '

	required_args = args_parser.add_argument_group(
		all_args_header + '\n\n' + bullet + 'MODELS FILES')
	required_args.add_argument('-aa_tpr', dest='aa_tpr_filename', help=config.help_aa_tpr, type=str,
							   default=config.metavar_aa_tpr,
							   metavar='     ' + scg.par_wrap(config.metavar_aa_tpr))
	required_args.add_argument('-aa_traj', dest='aa_traj_filename', help=config.help_aa_traj,
							   type=str, default=config.metavar_aa_traj,
							   metavar='     ' + scg.par_wrap(config.metavar_aa_traj))
	required_args.add_argument('-cg_map', dest='cg_map_filename', help=config.help_cg_map, type=str,
							   default=config.metavar_cg_map,
							   metavar='       ' + scg.par_wrap(config.metavar_cg_map))
	required_args.add_argument('-mapping', dest='mapping_type', help=config.help_mapping_type, type=str,
							   default='COM', metavar='             (COM)')
	required_args.add_argument('-cg_itp', dest='cg_itp_filename',
							   help='ITP file of the CG model to evaluate', type=str,
							   default=config.metavar_cg_itp,
							   metavar='     ' + scg.par_wrap(config.metavar_cg_itp))
	required_args.add_argument('-cg_tpr', dest='cg_tpr_filename',
							   help='TPR file of your CG simulation (omit for solo AA inspection)',
							   type=str, default=config.metavar_cg_tpr,
							   metavar='     ' + scg.par_wrap(config.metavar_cg_tpr))
	required_args.add_argument('-cg_traj', dest='cg_traj_filename',
							   help='XTC file of your CG trajectory (omit for solo AA inspection)',
							   type=str, default=config.metavar_cg_traj,
							   metavar='     ' + scg.par_wrap(config.metavar_cg_traj))

	optional_args = args_parser.add_argument_group(bullet + 'CG MODEL SCALING')
	# optional_args.add_argument('-nb_threads', dest='nb_threads', help='number of threads to use', type=int, default=1, metavar='1') # TODO: does NOT work properly -- modif MDAnalysis code with OpenMP num_threads(n) in the pragma
	optional_args.add_argument('-bonds_scaling', dest='bonds_scaling',
							   help=config.help_bonds_scaling, type=float,
							   default=config.bonds_scaling,
							   metavar='       ' + scg.par_wrap(config.bonds_scaling))
	optional_args.add_argument('-bonds_scaling_str', dest='bonds_scaling_str',
							   help=config.help_bonds_scaling_str, type=str,
							   default=config.bonds_scaling_str, metavar='')
	optional_args.add_argument('-min_bonds_length', dest='min_bonds_length',
							   help=config.help_min_bonds_length, type=float,
							   default=config.min_bonds_length,
							   metavar='    ' + scg.par_wrap(config.min_bonds_length))
	optional_args.add_argument('-b2a_score_fact', dest='bonds2angles_scoring_factor',
							   help=config.help_bonds2angles_scoring_factor, type=float,
							   default=config.bonds2angles_scoring_factor,
							   metavar='      ' + scg.par_wrap(config.bonds2angles_scoring_factor))

	graphical_args = args_parser.add_argument_group(bullet + 'FIGURE DISPLAY')
	graphical_args.add_argument('-mismatch_ordering', dest='mismatch_order',
								help='Enables ordering of bonds/angles/dihedrals by mismatch score\nbetween pairwise AA-mapped/CG distributions (can help diagnosis)',
								default=False, action='store_true')
	graphical_args.add_argument('-bw_constraints', dest='bw_constraints',
								help=config.help_bw_constraints, type=float,
								default=config.bw_constraints,
								metavar='    ' + scg.par_wrap(config.bw_constraints))
	graphical_args.add_argument('-bw_bonds', dest='bw_bonds', help=config.help_bw_bonds, type=float,
								default=config.bw_bonds,
								metavar='           ' + scg.par_wrap(config.bw_bonds))
	graphical_args.add_argument('-bw_angles', dest='bw_angles', help=config.help_bw_angles,
								type=float, default=config.bw_angles,
								metavar='           ' + scg.par_wrap(config.bw_angles))
	graphical_args.add_argument('-bw_dihedrals', dest='bw_dihedrals', help=config.help_bw_dihedrals,
								type=float, default=config.bw_dihedrals,
								metavar='        ' + scg.par_wrap(config.bw_dihedrals))
	graphical_args.add_argument('-disable_x_scaling', dest='row_x_scaling',
								help='Disable auto-scaling of X axis across each row of the plot',
								default=True, action='store_false')
	graphical_args.add_argument('-disable_y_scaling', dest='row_y_scaling',
								help='Disable auto-scaling of Y axis across each row of the plot',
								default=True, action='store_false')
	graphical_args.add_argument('-bonds_max_range', dest='bonded_max_range',
								help=config.help_bonds_max_range, type=float,
								default=config.bonds_max_range,
								metavar='      ' + scg.par_wrap(config.bonds_max_range))
	graphical_args.add_argument('-ncols', dest='ncols_max',
								help='Max. nb of columns displayed in figure', type=int, default=0,
								metavar='')  # TODO: make this a line return in plot instead of ignoring groups

	optional_args2 = args_parser.add_argument_group(bullet + 'OTHERS')
	optional_args2.add_argument('-o', dest='plot_filename',
								help='Filename for the output plot (extension/format can be one of:\neps, pdf, pgf, png, ps, raw, rgba, svg, svgz)',
								type=str, default='distributions.png', metavar='     (distributions.png)')
	optional_args2.add_argument('-h', '--help', action='help',
								help='Show this help message and exit')
	optional_args2.add_argument('-v', '--verbose', dest='verbose', help=config.help_verbose,
								action='store_true', default=False)

	# arguments handling, display command line if help or no arguments provided
	ns = args_parser.parse_args()
	input_cmdline = ' '.join(map(cmd_quote, sys.argv))
	print('Working directory:', os.getcwd())
	print('Command line:', input_cmdline)

	run(ns)

if __name__ == "__main__":
	main()
