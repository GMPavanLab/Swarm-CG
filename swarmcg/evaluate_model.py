# some numpy version have this ufunc warning at import + many packages call numpy and display annoying warnings
import warnings

warnings.filterwarnings("ignore")
import os, sys
from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS
from shlex import quote as cmd_quote

import matplotlib

import swarmcg.shared.styling
from swarmcg import swarmCG as scg
from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg.shared.styling import EVALUATE_DESCR

warnings.resetwarnings()
matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation


def run(ns):

	from numpy import VisibleDeprecationWarning
	warnings.filterwarnings("ignore", category=VisibleDeprecationWarning) # filter MDAnalysis + numpy deprecation stuff that is annoying

	# TODO: make it possible to feed a delta for Rg in case the model has scaling ?

	ns.molname_in = None # TODO: arguments that exist only in the scope of optimization (useless for manual model evaluation) -- but this could be modified to be allowed to evaluate models in mixed membranes, averaging distribs for given molecule name only
	ns.gyr_aa_mapped, ns.gyr_aa_mapped_std = None, None
	# ns.sasa_aa_mapped, ns.sasa_aa_mapped_std = None, None
	ns.aa_rg_offset = 0

	scg.set_MDA_backend(ns)

	# TODO: add missing checks -- if some are missing
	# TODO: factorize all checks and put them in global lib
	if not os.path.isfile(ns.aa_tpr_filename):
		msg = (
			"Cannot find coordinate file of the atomistic simulation"
			"(GRO, PDB, or other trajectory formats supported by MDAnalysis)"
		)
		raise exceptions.MissingCoordinateFile(msg)
	if not os.path.isfile(ns.aa_traj_filename):
		msg = (
			"Cannot find trajectory file of the atomistic simulation"
			"(XTC, TRR, or other trajectory formats supported by MDAnalysis)"
		)
		raise exceptions.MissingTrajectoryFile(msg)

	if not os.path.isfile(ns.cg_map_filename):
		msg = "Cannot find CG beads mapping file (NDX-like file format)"
		raise exceptions.MissingIndexFile(msg)

	if not os.path.isfile(ns.cg_itp_filename):
		msg = "Cannot find ITP file of the CG model"
		raise exceptions.MissingItpFile(msg)

	# check bonds scaling arguments conflicts
	if (ns.bonds_scaling != config.bonds_scaling and ns.min_bonds_length != config.min_bonds_length) or (ns.bonds_scaling != config.bonds_scaling and ns.bonds_scaling_str != config.bonds_scaling_str) or (ns.min_bonds_length != config.min_bonds_length and ns.bonds_scaling_str != config.bonds_scaling_str):
		msg = (
			"Only one of arguments -bonds_scaling, -bonds_scaling_str and -min_bonds_length "
			"can be provided. Please check your parameters"
		)
		raise exceptions.InputArgumentError(msg)

	print()
	print(swarmcg.shared.styling.sep_close)
	print('| PRE-PROCESSING                                                                              |')
	print(swarmcg.shared.styling.sep_close)
	print()

	# display parameters for function compare_models
	if not os.path.isfile(ns.cg_tpr_filename) or not os.path.isfile(ns.cg_traj_filename):
		# switch to atomistic mapping inspection exclusively (= do NOT use real CG distributions)
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

	scg.create_bins_and_dist_matrices(ns)
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
	# required_args.add_argument('-figmolname', dest='figmolname', help='TODO REMOVE', type=str, required=True) # TODO: remove, this was just for figures

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
	# ONLY FOR PAPER FIGURES
	# optional_args.add_argument('-datamol', dest='datamol', help='Save bonded score and Rg values for each frame across simulation', type=str, default='MOL_EXEC_MODE')

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
								metavar='             ' + scg.par_wrap(config.bw_angles))
	graphical_args.add_argument('-bw_dihedrals', dest='bw_dihedrals', help=config.help_bw_dihedrals,
								type=float, default=config.bw_dihedrals,
								metavar='          ' + scg.par_wrap(config.bw_dihedrals))
	graphical_args.add_argument('-disable_x_scaling', dest='row_x_scaling',
								help='Disable auto-scaling of X axis across each row of the plot',
								default=True, action='store_false')
	graphical_args.add_argument('-disable_y_scaling', dest='row_y_scaling',
								help='Disable auto-scaling of Y axis across each row of the plot',
								default=True, action='store_false')
	graphical_args.add_argument('-bonds_max_range', dest='bonded_max_range',
								help=config.help_bonds_max_range, type=float,
								default=config.bonds_max_range,
								metavar='       ' + scg.par_wrap(config.bonds_max_range))
	graphical_args.add_argument('-ncols', dest='ncols_max',
								help='Max. nb of columns displayed in figure', type=int, default=0,
								metavar='')  # TODO: make this a line return in plot instead of ignoring groups

	optional_args2 = args_parser.add_argument_group(bullet + 'OTHERS')
	optional_args2.add_argument('-o', dest='plot_filename',
								help='Filename for the output plot (extension/format can be one of:\neps, pdf, pgf, png, ps, raw, rgba, svg, svgz)',
								type=str, default='distributions.png', metavar='distributions.png')
	optional_args2.add_argument('-h', '--help', action='help',
								help='Show this help message and exit')
	optional_args2.add_argument('-v', '--verbose', dest='verbose', help=config.help_verbose,
								action='store_true', default=False)

	# display help if script was called without arguments
	if len(sys.argv) == 1:
		args_parser.print_help()
		sys.exit()

	# arguments handling, display command line if help or no arguments provided
	ns = args_parser.parse_args()
	input_cmdline = ' '.join(map(cmd_quote, sys.argv))
	print('Working directory:', os.getcwd())
	print('Command line:', input_cmdline)

	run(ns)