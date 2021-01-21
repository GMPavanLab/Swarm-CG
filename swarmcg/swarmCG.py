import re
import collections

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MDAnalysis as mda
from pyemd import emd
from scipy.optimize import curve_fit

import swarmcg.scoring as scores
import swarmcg.simulations.vs_functions as vsf
from swarmcg import config
from swarmcg.shared import math_utils, styling, exceptions, catch_warnings
from swarmcg.utils import draw_float
from swarmcg.simulations.potentials import (gmx_bonds_func_1, gmx_angles_func_1, gmx_angles_func_2,
	gmx_dihedrals_func_1, gmx_dihedrals_func_2)

matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation

# TODO: When provided trajectory file does NOT contain PBC infos (box position and size for each frame, which are present in XTC format for example), we want to stil accept the provided trajectory format (if accepted by MDAnalysis) but we automatically disable the handling of PBC by the code


def par_wrap(string):
	return f'({string})'


# read one or more molecules from the AA TPR and trajectory
def load_aa_data(ns):

	ns.all_atoms = dict() # atom centered connectivity + atom type + heavy atom boolean + bead(s) to which the atom belongs (can belong to multiple beads depending on mapping)
	ns.all_aa_mols = [] # atom groups for each molecule of interest, in case we use several and average the distributions across many molecules, as we would do for membranes analysis

	if ns.molname_in == None:

		molname_atom_group = ns.aa_universe.atoms[0].fragment  # select the AA connected graph for the first moltype found in TPR
		ns.all_aa_mols.append(molname_atom_group)

		# atoms and their attributes
		for i in range(len(molname_atom_group)):

			atom_id = ns.aa_universe.atoms[i].id
			atom_type = ns.aa_universe.atoms[i].type[0]  # this was to check for hydrogens but we don't need it atm
			atom_charge = ns.aa_universe.atoms[i].charge
			atom_heavy = True
			if atom_type[0].upper() == 'H':
				atom_heavy = False

			ns.all_atoms[atom_id] = {'conn': set(), 'atom_type': atom_type, 'atom_charge': atom_charge, 'heavy': atom_heavy, 'beads_ids': set(), 'beads_types': set(), 'residue_names': set()}
			# print(ns.aa_universe.atoms[i].id, ns.aa_universe.atoms[i])

	# TODO: allow reading multiple instances of a molecule to build the reference distributions,
	#       for extended usage with NOT just one flexible molecule in solvent
	else:
		pass

	# print(ns.aa_universe.atoms[1000].segindex, ns.aa_universe.atoms[134].resindex, ns.aa_universe.atoms[134].molnum)

	# for seg in ns.aa_universe.segments:

	# 	print(seg.segid)
	# 	sel = ns.aa_universe.select_atoms("segid "+str(seg.segid))
	# 	print(sel.atoms)

	# print(ns.aa_universe.atoms[0].segid)

	# sel = ns.aa_universe.select_atoms("moltype SOL")
	# for atom in sel.atoms:
	# 	print(atom.id)
	# 	print("  ", sel.atoms[atom.id].fragment)

	# TODO: print this charge, if it is not null then we need to check for Q-type beads and for the 2 Q-types that have no defined charge value, raise a warning to tell the user he has to edit the file manually
	# net_charge = molname_atom_group.total_charge()
	# print('Net charge of the reference all atom model:', round(net_charge, 4))


# load CG beads from NDX-like file
def read_ndx_atoms2beads(ns):

	with open(ns.cg_map_filename, 'r') as fp:

		ndx_lines = fp.read().split('\n')
		ndx_lines = [ndx_line.strip().split(';')[0] for ndx_line in ndx_lines]  # split for comments

		ns.atoms_occ_total = collections.Counter()
		ns.all_beads = dict()  # atoms id mapped to each bead
		bead_id = 0
		current_section = 'Beginning of file'

		for i in range(len(ndx_lines)):
			ndx_line = ndx_lines[i]
			if ndx_line != '':

				if bool(re.search('\[.*\]', ndx_line)):
					current_section = ndx_line
					ns.all_beads[bead_id] = {'atoms_id': [], 'section': current_section, 'line_nb': i+1}
					current_bead_id = bead_id
					bead_id += 1

				else:
					try:
						bead_atoms_id = [int(atom_id)-1 for atom_id in ndx_line.split()]  # retrieve indexing from 0 for atoms IDs for MDAnalysis
						ns.all_beads[current_bead_id]['atoms_id'].extend(bead_atoms_id)  # all atoms included in current bead

						for atom_id in bead_atoms_id:  # bead to which each atom belongs (one atom can belong to multiple beads if there is split-mapping)
							ns.atoms_occ_total[atom_id] += 1

					except NameError:
						msg = (
							"The CG beads mapping (NDX) file does NOT seem to contain CG beads "
							"sections.\nPlease verify the input mapping. The expected format is "
							"Gromacs NDX."
						)
						raise exceptions.MissformattedFile(msg)

					except ValueError:  # non-integer atom ID provided
						msg = (
							f"Incorrect reading of the sections content in the CG beads mapping "
							f"(NDX) file.\nFound non-integer values for some IDs at line "
							f"{str(i + 1)} under section {current_section}."
						)
						raise exceptions.MissformattedFile(msg)

	for bead_id in ns.all_beads:
		if len(ns.all_beads[bead_id]['atoms_id']) == 0:
			msg = (
				f"The ITP file contains an empty section named {ns.all_beads[bead_id]['section']} starting at line {ns.all_beads[bead_id]['line_nb']}."
				f"Empty sections are NOT allowed, please fill or delete it."
			)
			raise exceptions.MissformattedFile(msg)


# calculate weight ratio of atom ID in given CG bead
# this is for splitting atom weight in case an atom is mapped to several CG beads
def get_atoms_weights_in_beads(ns):

	ns.atom_w = dict()
	if ns.verbose:
		print('Calculating atoms weights ratio within mapped CG beads')
	for bead_id in ns.all_beads:
		# print('Weighting bead_id', bead_id)
		ns.atom_w[bead_id] = dict()
		beads_atoms_counts = collections.Counter(ns.all_beads[bead_id]['atoms_id'])
		for atom_id in beads_atoms_counts:
			ns.atom_w[bead_id][atom_id] = round(beads_atoms_counts[atom_id] / ns.atoms_occ_total[atom_id], 3)
			if ns.verbose and ns.mapping_type == 'COM':
				print('  CG bead ID', bead_id+1, '-- Atom ID', atom_id+1, 'has weight ratio =', ns.atom_w[bead_id][atom_id])
	if ns.verbose:
		print()


# for each CG bead, create atom groups for trajectory geoms calculation using mass and atom weights across beads
def get_beads_MDA_atomgroups(ns):

	ns.mda_beads_atom_grps, ns.mda_weights_atom_grps = dict(), dict()
	for bead_id in ns.atom_w:
		try:
			# print('Created bead_id', bead_id, 'using atoms', [atom_id for atom_id in ns.atom_w[bead_id]])
			if ns.mapping_type == 'COM':
				ns.mda_beads_atom_grps[bead_id] = mda.AtomGroup([atom_id for atom_id in ns.atom_w[bead_id]], ns.aa_universe)
				ns.mda_weights_atom_grps[bead_id] = np.array([ns.atom_w[bead_id][atom_id]*ns.aa_universe.atoms[atom_id].mass for atom_id in ns.atom_w[bead_id]])
			elif ns.mapping_type == 'COG':
				ns.mda_beads_atom_grps[bead_id] = mda.AtomGroup([atom_id for atom_id in ns.atom_w[bead_id]], ns.aa_universe)
				ns.mda_weights_atom_grps[bead_id] = np.array([1 for _ in ns.atom_w[bead_id]])

		except IndexError as e:
			msg = (
				f"An ID present in your mapping (NDX) file could not be found in the AA trajectory. "
				f"Please check your mapping (NDX) file.\nSee the error below to understand which "
				f"ID (here 0-indexed) could not be found:\n\n{str(e)}"
			)
			raise exceptions.MissformattedFile(msg)

# update coarse-grain ITP
def update_cg_itp_obj(ns, parameters_set, update_type):

	if update_type == 1:  # intermediary
		itp_obj = ns.out_itp
	elif update_type == 2:  # cycles optimized
		itp_obj = ns.opti_itp
	else:
		msg = (
			f"Code error in function update_cg_itp_obj.\nPlease consider opening an issue on GitHub "
			f"at {config.github_url}."
		)
		raise exceptions.InvalidArgument(msg)

	for i in range(ns.opti_cycle['nb_geoms']['constraint']):
		if ns.exec_mode == 1:
			itp_obj['constraint'][i]['value'] = round(parameters_set[i], 3)  # constraint - distance

	for i in range(ns.opti_cycle['nb_geoms']['bond']):
		if ns.exec_mode == 1:
			itp_obj['bond'][i]['value'] = round(parameters_set[ns.opti_cycle['nb_geoms']['constraint'] + i], 3)  # bond - distance
			itp_obj['bond'][i]['fct'] = round(parameters_set[ns.opti_cycle['nb_geoms']['constraint'] + ns.opti_cycle['nb_geoms']['bond'] + i], 3)  # bond - force constant
		else:
			itp_obj['bond'][i]['fct'] = round(parameters_set[i], 3)  # bond - force constant

	for i in range(ns.opti_cycle['nb_geoms']['angle']):
		if ns.exec_mode == 1:
			itp_obj['angle'][i]['value'] = round(parameters_set[ns.opti_cycle['nb_geoms']['constraint'] + 2 * ns.opti_cycle['nb_geoms']['bond'] + i], 2)  # angle - value
			itp_obj['angle'][i]['fct'] = round(parameters_set[ns.opti_cycle['nb_geoms']['constraint'] + 2 * ns.opti_cycle['nb_geoms']['bond'] + ns.opti_cycle['nb_geoms']['angle'] + i], 2)  # angle - force constant
		else:
			itp_obj['angle'][i]['fct'] = round(parameters_set[ns.opti_cycle['nb_geoms']['bond'] + i], 2)  # angle - force constant

	for i in range(ns.opti_cycle['nb_geoms']['dihedral']):
		if ns.exec_mode == 1:
			itp_obj['dihedral'][i]['value'] = round(parameters_set[ns.opti_cycle['nb_geoms']['constraint'] + 2 * ns.opti_cycle['nb_geoms']['bond'] + 2 * ns.opti_cycle['nb_geoms']['angle'] + i], 2)  # dihedral - value
			itp_obj['dihedral'][i]['fct'] = round(parameters_set[ns.opti_cycle['nb_geoms']['constraint'] + 2 * ns.opti_cycle['nb_geoms']['bond'] + 2 * ns.opti_cycle['nb_geoms']['angle'] + ns.opti_cycle['nb_geoms']['dihedral'] + i], 2) # dihedral - force constant
		else:
			itp_obj['dihedral'][i]['fct'] = round(parameters_set[ns.opti_cycle['nb_geoms']['bond'] + ns.opti_cycle['nb_geoms']['angle'] + i], 2)  # dihedral - force constant


# set dimensions of the search space according to the type of optimization (= geom type(s) to optimize)
def get_search_space_boundaries(ns):

	search_space_boundaries = []

	if ns.opti_cycle['nb_geoms']['constraint'] > 0:
		if ns.exec_mode == 1:
			search_space_boundaries.extend(ns.domains_val['constraint'])  # constraints equilibrium values

	if ns.opti_cycle['nb_geoms']['bond'] > 0:
		if ns.exec_mode == 1:
			search_space_boundaries.extend(ns.domains_val['bond'])  # bonds equilibrium values
		search_space_boundaries.extend([[0, ns.default_max_fct_bonds_opti]] * ns.opti_cycle['nb_geoms']['bond'])  # bonds force constants

	if ns.opti_cycle['nb_geoms']['angle'] > 0:
		if ns.exec_mode == 1:
			search_space_boundaries.extend(ns.domains_val['angle'])  # angles equilibrium values

		for grp_angle in range(ns.opti_cycle['nb_geoms']['angle']):  # angles force constants
			if ns.cg_itp['angle'][grp_angle]['func'] == 1:
				search_space_boundaries.extend([[0, ns.default_max_fct_angles_opti_f1]])
			elif ns.cg_itp['angle'][grp_angle]['func'] == 2:
				search_space_boundaries.extend([[0, ns.default_max_fct_angles_opti_f2]])

	if ns.opti_cycle['nb_geoms']['dihedral'] > 0:
		if ns.exec_mode == 1:
			search_space_boundaries.extend(ns.domains_val['dihedral'])  # dihedrals equilibrium values

		for grp_dihedral in range(ns.opti_cycle['nb_geoms']['dihedral']):  # dihedrals force constants
			if ns.cg_itp['dihedral'][grp_dihedral]['func'] == 2:
				search_space_boundaries.extend([[-ns.default_abs_range_fct_dihedrals_opti_func_without_mult, ns.default_abs_range_fct_dihedrals_opti_func_without_mult]])
			elif ns.cg_itp['dihedral'][grp_dihedral]['func'] in config.dihedral_func_with_mult:
				search_space_boundaries.extend([[-ns.default_abs_range_fct_dihedrals_opti_func_with_mult, ns.default_abs_range_fct_dihedrals_opti_func_with_mult]])

	return search_space_boundaries


# build initial guesses for particles initialization, as variations around parameters obtained via Boltzmann inversion (BI)
# this is done in an iterative fashion:
#   1st read atom mapped traj constraints/bonds and perform BI to obtain the 1st set of parameters and then find variations in this function
#   2nd read angles from the best constraints/bonds-only optimized model, perform BI and do the ratio with BI of the atom mapped traj to add only the required amount of energy and obtain 1st set of parameters
#   3rd do dihedrals the similarly, using BI ratio
def get_initial_guess_list(ns, nb_particles):

	initial_guess_list = []  # array of arrays (inner arrays are the values used for particles initialization)

	# the first particle is initialized as EXACTLY the values of the current CG ITP object (or BI in exec_mode 1)
	# except if force constants are outside of the searchable domain defined for optimization
	# for bonds lengths and angles/dihedrals values, we perform no checks
	input_guess = []

	if ns.exec_mode == 1:
		for i in range(ns.opti_cycle['nb_geoms']['constraint']):
			input_guess.append(min(max(ns.out_itp['constraint'][i]['value'], ns.domains_val['constraint'][i][0]), ns.domains_val['constraint'][i][1]))  # constraints equilibrium values

		for i in range(ns.opti_cycle['nb_geoms']['bond']):
			input_guess.append(min(max(ns.out_itp['bond'][i]['value'], ns.domains_val['bond'][i][0]), ns.domains_val['bond'][i][1]))  # bonds equilibrium values

	for i in range(ns.opti_cycle['nb_geoms']['bond']):
		input_guess.append(min(max(ns.out_itp['bond'][i]['fct'], 0), ns.default_max_fct_bonds_opti))  # bonds force constants

	if ns.exec_mode == 1:
		for i in range(ns.opti_cycle['nb_geoms']['angle']):
			input_guess.append(min(max(ns.out_itp['angle'][i]['value'], ns.domains_val['angle'][i][0]), ns.domains_val['angle'][i][1]))  # angles equilibrium values

	for i in range(ns.opti_cycle['nb_geoms']['angle']):
		if ns.cg_itp['angle'][i]['func'] == 1:
			input_guess.append(min(max(ns.out_itp['angle'][i]['fct'], 0), ns.default_max_fct_angles_opti_f1))  # angles force constants
		elif ns.cg_itp['angle'][i]['func'] == 2:
			input_guess.append(min(max(ns.out_itp['angle'][i]['fct'], 0), ns.default_max_fct_angles_opti_f2))  # angles force constants

	if ns.exec_mode == 1:
		for i in range(ns.opti_cycle['nb_geoms']['dihedral']):
			input_guess.append(min(max(ns.out_itp['dihedral'][i]['value'], ns.domains_val['dihedral'][i][0]), ns.domains_val['dihedral'][i][1]))  # dihedrals equilibrium values

	for i in range(ns.opti_cycle['nb_geoms']['dihedral']):
		if ns.cg_itp['dihedral'][i]['func'] == 2:
			input_guess.append(min(max(ns.out_itp['dihedral'][i]['fct'], -ns.default_abs_range_fct_dihedrals_opti_func_without_mult), ns.default_abs_range_fct_dihedrals_opti_func_without_mult))  # dihedrals force constants
		else:
			input_guess.append(min(max(ns.out_itp['dihedral'][i]['fct'], -ns.default_abs_range_fct_dihedrals_opti_func_with_mult), ns.default_abs_range_fct_dihedrals_opti_func_with_mult))  # dihedrals force constants

	initial_guess_list.append(input_guess)
	num_particle_random_start = 1  # first particle is DBI

	# The second particle is initialized either:
	# (1) Using best EMD score for each geom and the parameters that yielded these EMD scores. This is independant
	# of exec_mode, because we use only previously selected parameters for this particle. If yet no independant best
	# is recorded for a given geom (dihedrals in fact), values are taken from best optimized model until now.
	# (2) If we are in opti cycle 1 and -user_params is provided, then this particle is instead
	# initialized as the users parameters.
	if ns.opti_cycle['nb_cycle'] > 1:

		num_particle_random_start += 1
		input_guess = []

		# constraints equilibrium values
		if ns.exec_mode == 1:
			for i in range(ns.opti_cycle['nb_geoms']['constraint']):
				if ns.all_best_emd_dist_geoms['constraints'][i] != config.sim_crash_EMD_indep_score:
					input_guess.append(ns.all_best_params_dist_geoms['constraints'][i]['params'][0])
				else:
					input_guess.append(min(max(ns.out_itp['constraint'][i]['value'], ns.domains_val['constraint'][i][0]), ns.domains_val['constraint'][i][1]))

		# bonds equilibrium values
		if ns.exec_mode == 1:
			for i in range(ns.opti_cycle['nb_geoms']['bond']):
				if ns.all_best_emd_dist_geoms['bonds'][i] != config.sim_crash_EMD_indep_score:
					input_guess.append(ns.all_best_params_dist_geoms['bonds'][i]['params'][0])
				else:
					input_guess.append(min(max(ns.out_itp['bond'][i]['value'], ns.domains_val['bond'][i][0]), ns.domains_val['bond'][i][1]))
		# bonds force constants
		for i in range(ns.opti_cycle['nb_geoms']['bond']):
			if ns.all_best_emd_dist_geoms['bonds'][i] != config.sim_crash_EMD_indep_score:
				input_guess.append(ns.all_best_params_dist_geoms['bonds'][i]['params'][1])
			else:
				input_guess.append(min(max(ns.out_itp['bond'][i]['fct'], 0), ns.default_max_fct_bonds_opti))

		# angles equilibrium values
		if ns.exec_mode == 1:
			for i in range(ns.opti_cycle['nb_geoms']['angle']):
				if ns.all_best_emd_dist_geoms['angles'][i] != config.sim_crash_EMD_indep_score:
					input_guess.append(ns.all_best_params_dist_geoms['angles'][i]['params'][0])
				else:
					input_guess.append(min(max(ns.out_itp['angle'][i]['value'], ns.domains_val['angle'][i][0]), ns.domains_val['angle'][i][1]))
		# angles force constants
		for i in range(ns.opti_cycle['nb_geoms']['angle']):
			if ns.all_best_emd_dist_geoms['angles'][i] != config.sim_crash_EMD_indep_score:
				input_guess.append(ns.all_best_params_dist_geoms['angles'][i]['params'][1])
			else:
				if ns.cg_itp['angle'][i]['func'] == 1:
					input_guess.append(min(max(ns.out_itp['angle'][i]['fct'], 0), ns.default_max_fct_angles_opti_f1))
				elif ns.cg_itp['angle'][i]['func'] == 2:
					input_guess.append(min(max(ns.out_itp['angle'][i]['fct'], 0), ns.default_max_fct_angles_opti_f2))

		# dihedrals equilibrium values
		if ns.exec_mode == 1:
			for i in range(ns.opti_cycle['nb_geoms']['dihedral']):
				if ns.all_best_emd_dist_geoms['dihedrals'][i] != config.sim_crash_EMD_indep_score:
					input_guess.append(ns.all_best_params_dist_geoms['dihedrals'][i]['params'][0])
				else:
					input_guess.append(min(max(ns.out_itp['dihedral'][i]['value'], ns.domains_val['dihedral'][i][0]), ns.domains_val['dihedral'][i][1]))
		# dihedrals force constants
		for i in range(ns.opti_cycle['nb_geoms']['dihedral']):
			if ns.all_best_emd_dist_geoms['dihedrals'][i] != config.sim_crash_EMD_indep_score:
				input_guess.append(ns.all_best_params_dist_geoms['dihedrals'][i]['params'][1])
			else:
				if ns.cg_itp['dihedral'][i]['func'] == 2:
					input_guess.append(
						min(max(ns.out_itp['dihedral'][i]['fct'], -ns.default_abs_range_fct_dihedrals_opti_func_without_mult), ns.default_abs_range_fct_dihedrals_opti_func_without_mult))
				else:
					input_guess.append(min(
						max(ns.out_itp['dihedral'][i]['fct'], -ns.default_abs_range_fct_dihedrals_opti_func_with_mult), ns.default_abs_range_fct_dihedrals_opti_func_with_mult))

		initial_guess_list.append(input_guess)

	# optionally second particle is initialized as input parameters ONLY at start of opti cycle 1
	elif ns.user_input:

		num_particle_random_start += 1
		input_guess = []

		# constraints equilibrium values
		if ns.exec_mode == 1:
			for i in range(ns.opti_cycle['nb_geoms']['constraint']):
				input_guess.append(min(max(ns.out_itp['constraint'][i]['value_user'], ns.domains_val['constraint'][i][0]), ns.domains_val['constraint'][i][1]))

		# bonds equilibrium values
		if ns.exec_mode == 1:
			for i in range(ns.opti_cycle['nb_geoms']['bond']):
				input_guess.append(min(max(ns.out_itp['bond'][i]['value_user'], ns.domains_val['bond'][i][0]), ns.domains_val['bond'][i][1]))

		# bonds force constants
		for i in range(ns.opti_cycle['nb_geoms']['bond']):
			input_guess.append(min(max(ns.out_itp['bond'][i]['fct_user'], 0), ns.default_max_fct_bonds_opti))

		# angles equilibrium values
		if ns.exec_mode == 1:
			for i in range(ns.opti_cycle['nb_geoms']['angle']):
				input_guess.append(min(max(ns.out_itp['angle'][i]['value_user'], ns.domains_val['angle'][i][0]), ns.domains_val['angle'][i][1]))

		# angles force constants
		for i in range(ns.opti_cycle['nb_geoms']['angle']):
			if ns.cg_itp['angle'][i]['func'] == 1:
				input_guess.append(min(max(ns.out_itp['angle'][i]['fct_user'], 0), ns.default_max_fct_angles_opti_f1))
			elif ns.cg_itp['angle'][i]['func'] == 2:
				input_guess.append(min(max(ns.out_itp['angle'][i]['fct_user'], 0), ns.default_max_fct_angles_opti_f2))

		# dihedrals equilibrium values
		if ns.exec_mode == 1:
			for i in range(ns.opti_cycle['nb_geoms']['dihedral']):
				input_guess.append(min(max(ns.out_itp['dihedral'][i]['value_user'], ns.domains_val['dihedral'][i][0]), ns.domains_val['dihedral'][i][1]))
		# dihedrals force constants
		for i in range(ns.opti_cycle['nb_geoms']['dihedral']):
			if ns.cg_itp['dihedral'][i]['func'] == 2:
				input_guess.append(
					min(max(ns.out_itp['dihedral'][i]['fct_user'], -ns.default_abs_range_fct_dihedrals_opti_func_without_mult), ns.default_abs_range_fct_dihedrals_opti_func_without_mult))
			else:
				input_guess.append(min(
					max(ns.out_itp['dihedral'][i]['fct_user'], -ns.default_abs_range_fct_dihedrals_opti_func_with_mult), ns.default_abs_range_fct_dihedrals_opti_func_with_mult))

		initial_guess_list.append(input_guess)

	# for the other particles we generate variations of the input CG ITP, still within defined boundaries for optimization
	# boundaries are defined:
	#   - for constraints/bonds length and angles/dihedrals values, according to atomistic mapped trajectory and maximum searchable
	#   - for force constants, according to default or user provided maximal ranges (see config file for defaults)
	for i in range(num_particle_random_start, nb_particles):
		init_guess = []

		# constraints equilibrium values
		if ns.exec_mode == 1:
			for j in range(ns.opti_cycle['nb_geoms']['constraint']):
				try:
					emd_err_fact = max(1, ns.all_emd_dist_geoms['constraints'][j]/2)
				except:
					emd_err_fact = 1
				draw_low = max(ns.out_itp['constraint'][j]['value']-config.bond_dist_guess_variation*ns.val_guess_fact*emd_err_fact, ns.domains_val['constraint'][j][0])
				draw_high = min(ns.out_itp['constraint'][j]['value']+config.bond_dist_guess_variation*ns.val_guess_fact*emd_err_fact, ns.domains_val['constraint'][j][1])
				init_guess.append(draw_float(draw_low, draw_high, 3))

		# bonds equilibrium values
		if ns.exec_mode == 1:
			for j in range(ns.opti_cycle['nb_geoms']['bond']):
				try:
					emd_err_fact = max(1, ns.all_emd_dist_geoms['bonds'][j]/2)
				except:
					emd_err_fact = 1
				draw_low = max(ns.out_itp['bond'][j]['value']-config.bond_dist_guess_variation*ns.val_guess_fact*emd_err_fact, ns.domains_val['bond'][j][0])
				draw_high = min(ns.out_itp['bond'][j]['value']+config.bond_dist_guess_variation*ns.val_guess_fact*emd_err_fact, ns.domains_val['bond'][j][1])
				init_guess.append(draw_float(draw_low, draw_high, 3))
				# print('Particle', i+1, '-- BOND', j+1, '-- VALUE RANGE', draw_low, draw_high)

		# bonds force constants
		for j in range(ns.opti_cycle['nb_geoms']['bond']):
			try:
				emd_err_fact = max(1, ns.all_emd_dist_geoms['bonds'][j]/2)
			except:
				emd_err_fact = 1
			draw_low = max(min(ns.out_itp['bond'][j]['fct']*(1-ns.fct_guess_fact*emd_err_fact), ns.out_itp['bond'][j]['fct']-config.fct_guess_min_flat_diff_bonds), 0)
			draw_high = min(max(ns.out_itp['bond'][j]['fct']*(1+ns.fct_guess_fact*emd_err_fact), ns.out_itp['bond'][j]['fct']+config.fct_guess_min_flat_diff_bonds), ns.default_max_fct_bonds_opti)
			init_guess.append(draw_float(draw_low, draw_high, 3))
			# print('Particle', i+1, '-- BOND', j+1, '-- FCT RANGE', draw_low, draw_high)

		# angles equilibrium values
		if ns.exec_mode == 1:
			for j in range(ns.opti_cycle['nb_geoms']['angle']):
				try:
					emd_err_fact = max(1, ns.all_emd_dist_geoms['angles'][j]/2)
				except:
					emd_err_fact = 1
				draw_low = max(ns.out_itp['angle'][j]['value']-config.angle_value_guess_variation*ns.val_guess_fact*emd_err_fact, ns.domains_val['angle'][j][0])
				draw_high = min(ns.out_itp['angle'][j]['value']+config.angle_value_guess_variation*ns.val_guess_fact*emd_err_fact, ns.domains_val['angle'][j][1])
				init_guess.append(draw_float(draw_low, draw_high, 3))
				# print('Particle', i+1, '-- ANGLE', j+1, '-- VALUE RANGE', draw_low, draw_high)

		# angles force constants
		for j in range(ns.opti_cycle['nb_geoms']['angle']):
			try:
				emd_err_fact = max(1, ns.all_emd_dist_geoms['angles'][j]/2)
			except:
				emd_err_fact = 1
			draw_low = max(min(ns.out_itp['angle'][j]['fct']*(1-ns.fct_guess_fact*emd_err_fact), ns.out_itp['angle'][j]['fct']-config.fct_guess_min_flat_diff_angles), 0)
			if ns.cg_itp['angle'][j]['func'] == 1:
				draw_high = min(max(ns.out_itp['angle'][j]['fct']*(1+ns.fct_guess_fact*emd_err_fact), ns.out_itp['angle'][j]['fct']+config.fct_guess_min_flat_diff_angles), ns.default_max_fct_angles_opti_f1)
			elif ns.cg_itp['angle'][j]['func'] == 2:
				draw_high = min(max(ns.out_itp['angle'][j]['fct']*(1+ns.fct_guess_fact*emd_err_fact), ns.out_itp['angle'][j]['fct']+config.fct_guess_min_flat_diff_angles), ns.default_max_fct_angles_opti_f2)
			init_guess.append(draw_float(draw_low, draw_high, 3))
			# print('Particle', i+1, '-- ANGLE', j+1, '-- FCT RANGE', draw_low, draw_high)

		# dihedrals equilibrium values
		if ns.exec_mode == 1:
			for j in range(ns.opti_cycle['nb_geoms']['dihedral']):
				try:
					emd_err_fact = max(1, ns.all_emd_dist_geoms['dihedrals'][j]/5)
				except:
					emd_err_fact = 1
				draw_low = max(ns.out_itp['dihedral'][j]['value']-config.dihedral_value_guess_variation*ns.val_guess_fact*emd_err_fact, ns.domains_val['dihedral'][j][0])
				draw_high = min(ns.out_itp['dihedral'][j]['value']+config.dihedral_value_guess_variation*ns.val_guess_fact*emd_err_fact, ns.domains_val['dihedral'][j][1])
				init_guess.append(draw_float(draw_low, draw_high, 3))
				# print('Particle', i+1, '-- DIHEDRAL', j+1, '-- VALUE RANGE', draw_low, draw_high)

		# dihedrals force constants
		for j in range(ns.opti_cycle['nb_geoms']['dihedral']):

			try:
				emd_err_fact = max(1, ns.all_emd_dist_geoms['dihedrals'][j]/5)
			except:
				emd_err_fact = 1

			# here force constants can be negative, proceed accordingly
			if ns.out_itp['dihedral'][j]['fct'] > 0:  # if positive
				# initial variations range
				draw_low = ns.out_itp['dihedral'][j]['fct']*(1-ns.fct_guess_fact*emd_err_fact)
				draw_high = ns.out_itp['dihedral'][j]['fct']*(1+ns.fct_guess_fact*emd_err_fact)
			else:
				# initial variations range
				draw_low = ns.out_itp['dihedral'][j]['fct']*(1+ns.fct_guess_fact*emd_err_fact)
				draw_high = ns.out_itp['dihedral'][j]['fct']*(1-ns.fct_guess_fact*emd_err_fact)

			# make sure the minimal variation range is enforced + stay within defined boundaries
			if ns.cg_itp['dihedral'][j]['func'] == 2:
				draw_low = max(min(draw_low, ns.out_itp['dihedral'][j]['fct']-config.fct_guess_min_flat_diff_dihedrals_without_mult), ns.default_abs_range_fct_dihedrals_opti_func_without_mult)
				draw_high = min(max(draw_high, ns.out_itp['dihedral'][j]['fct']+config.fct_guess_min_flat_diff_dihedrals_without_mult), ns.default_abs_range_fct_dihedrals_opti_func_without_mult)
			else:
				draw_low = max(min(draw_low, ns.out_itp['dihedral'][j]['fct']-config.fct_guess_min_flat_diff_dihedrals_with_mult), -ns.default_abs_range_fct_dihedrals_opti_func_with_mult)
				draw_high = min(max(draw_high, ns.out_itp['dihedral'][j]['fct']+config.fct_guess_min_flat_diff_dihedrals_with_mult), ns.default_abs_range_fct_dihedrals_opti_func_with_mult)
			init_guess.append(draw_float(draw_low, draw_high, 3))
			# print('Particle', i+1, '-- DIHEDRAL', j+1, '-- FCT RANGE', draw_low, draw_high)

		initial_guess_list.append(init_guess)  # register new particle, built during this loop

	return initial_guess_list


def initialize_cg_traj(ns):

	masses = np.array([val['mass'] for val in ns.cg_itp['atoms']])
	names = np.array([val['atom'] for val in ns.cg_itp['atoms']])
	resnames = np.array([val['residue'] for val in ns.cg_itp['atoms']])
	resid = np.array([val['resnr'] for val in ns.cg_itp['atoms']])
	nr = len(set([val['resnr'] for val in ns.cg_itp['atoms']]))

	ns.aa2cg_universe = mda.Universe.empty(len(ns.cg_itp['atoms']), n_residues=nr, atom_resindex=resid, n_segments=1, residue_segindex=np.ones(nr), trajectory=True)
	ns.aa2cg_universe.add_TopologyAttr('masses')
	ns.aa2cg_universe._topology.masses.values = np.array(masses)
	ns.aa2cg_universe.add_TopologyAttr('names')
	ns.aa2cg_universe._topology.names.values = names
	ns.aa2cg_universe.add_TopologyAttr('resnames')
	ns.aa2cg_universe._topology.resnames.values = resnames


def map_aa2cg_traj(ns):

	if ns.mapping_type == 'COM':
		print('  Interpretation: Center of Mass (COM)')
	elif ns.mapping_type == 'COG':
		print('  Interpretation: Center of Geometry (COG)')

	# regular beads are mapped using center of mass of groups of atoms
	coord = np.empty((len(ns.aa_universe.trajectory), len(ns.cg_itp['atoms']), 3))
	for bead_id in range(len(ns.cg_itp['atoms'])):
		if not ns.cg_itp['atoms'][bead_id]['bead_type'].startswith('v'):  # bead is NOT a virtual site
			traj = np.empty((len(ns.aa_universe.trajectory), 3))
			for ts in ns.aa_universe.trajectory:
				traj[ts.frame] = ns.mda_beads_atom_grps[bead_id].center(
					ns.mda_weights_atom_grps[bead_id], pbc=None, compound='group'
				)  # no need for PBC handling, trajectories were made wholes for the molecule
			coord[:, bead_id, :] = traj

	ns.aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)

	# virtual sites are mapped using previously defined regular beads positions and appropriate virtual sites functions
	# it is also possible to use a VS for defining another VS position, if the VS used for definition is defined before
	# no need to check if the functions used for VS definition are correct here, this has been done already
	for bead_id in range(len(ns.cg_itp['atoms'])):
		if ns.cg_itp['atoms'][bead_id]['bead_type'].startswith('v'):

			traj = np.empty((len(ns.aa2cg_universe.trajectory), 3))

			if ns.cg_itp['atoms'][bead_id]['vs_type'] == 2:
				vs_def_beads_ids = ns.cg_itp['virtual_sites2'][bead_id]['vs_def_beads_ids']
				vs_params = ns.cg_itp['virtual_sites2'][bead_id]['vs_params']

				if ns.cg_itp['virtual_sites2'][bead_id]['func'] == 1:
					vsf.vs2_func_1(ns, traj, vs_def_beads_ids, vs_params)
				elif ns.cg_itp['virtual_sites2'][bead_id]['func'] == 2:
					vsf.vs2_func_2(ns, traj, vs_def_beads_ids, vs_params)

			if ns.cg_itp['atoms'][bead_id]['vs_type'] == 3:
				vs_def_beads_ids = ns.cg_itp['virtual_sites3'][bead_id]['vs_def_beads_ids']
				vs_params = ns.cg_itp['virtual_sites3'][bead_id]['vs_params']

				if ns.cg_itp['virtual_sites3'][bead_id]['func'] == 1:
					vsf.vs3_func_1(ns, traj, vs_def_beads_ids, vs_params)
				elif ns.cg_itp['virtual_sites3'][bead_id]['func'] == 2:
					vsf.vs3_func_2(ns, traj, vs_def_beads_ids, vs_params)
				elif ns.cg_itp['virtual_sites3'][bead_id]['func'] == 3:
					vsf.vs3_func_3(ns, traj, vs_def_beads_ids, vs_params)
				elif ns.cg_itp['virtual_sites3'][bead_id]['func'] == 4:
					vsf.vs3_func_4(ns, traj, vs_def_beads_ids, vs_params)

			# here it's normal there is only function 2, that's the only one that exists in gromacs for some reason
			if ns.cg_itp['atoms'][bead_id]['vs_type'] == 4:
				vs_def_beads_ids = ns.cg_itp['virtual_sites4'][bead_id]['vs_def_beads_ids']
				vs_params = ns.cg_itp['virtual_sites4'][bead_id]['vs_params']

				if ns.cg_itp['virtual_sites4'][bead_id]['func'] == 2:
					vsf.vs4_func_2(ns, traj, vs_def_beads_ids, vs_params)

			if ns.cg_itp['atoms'][bead_id]['vs_type'] == 'n':
				vs_def_beads_ids = ns.cg_itp['virtual_sitesn'][bead_id]['vs_def_beads_ids']
				vs_params = ns.cg_itp['virtual_sitesn'][bead_id]['vs_params']

				if ns.cg_itp['virtual_sitesn'][bead_id]['func'] == 1:
					vsf.vsn_func_1(ns, traj, vs_def_beads_ids)
				elif ns.cg_itp['virtual_sitesn'][bead_id]['func'] == 2:
					vsf.vsn_func_2(ns, traj, vs_def_beads_ids, bead_id)
				elif ns.cg_itp['virtual_sitesn'][bead_id]['func'] == 3:
					vsf.vsn_func_3(ns, traj, vs_def_beads_ids, vs_params)

			coord[:, bead_id, :] = traj

	ns.aa2cg_universe.load_new(coord, format=mda.coordinates.memory.MemoryReader)


# use selected whole molecules as MDA atomgroups and make their coordinates whole, inplace, across the complete tAA rajectory
def make_aa_traj_whole_for_selected_mols(ns):
	
	# TODO: add an option to NOT read the PBC in case user would feed a trajectory that is already unwrapped for
	#       molecule and their trajectory does NOT contain box dimensions (universe.dimensions)
	#       (this was an issue I encountered with Davide B3T traj GRO)
	for _ in ns.aa_universe.trajectory:
		for aa_mol in ns.all_aa_mols:
			mda.lib.mdamath.make_whole(aa_mol, inplace=True)


@catch_warnings(RuntimeWarning)  # ignore the warning "divide by 0 encountered in true_divide" while calculating sigma
def perform_BI(ns):
	"""Update ITP force constants with Boltzmann inversion for selected geoms at this
	given optimization step"""
	# NOTE: currently all of these are just BI, not BI to completion using only required ADDITIONAL amount of energy, which might make a difference when we perform the BI after bonds+angles optimization cycles
	# TODO: refactorize BI in separate function to be used during both model_prep and at start of model_opti
	# TODO: other dihedrals functions
	# TODO: If the first opti run of BI fails, lower force constants by 10% and retry, again and again until it works, or tell the user something is very wrong after 20 tries with 50% of the force constants that all did NOT work

	if not ns.performed_init_BI['bond'] and ns.opti_cycle['nb_geoms']['bond'] > 0:

		if ns.verbose:
			print()
			print('Performing Direct Boltzmann Inversion (DBI) to estimate bonds force constants')

		for grp_bond in range(ns.opti_cycle['nb_geoms']['bond']):

			hists_geoms_bi, std_grp_bond, avg_grp_bond, bi_xrange = ns.data_BI['bond'][grp_bond]
			hist_geoms_modif = hists_geoms_bi**2 * (max(hists_geoms_bi) / max(hists_geoms_bi**2))

			nb_passes = 3
			alpha = 0.55
			for _ in range(nb_passes):
				hist_geoms_modif = math_utils.ewma(hist_geoms_modif, alpha, int(config.bi_nb_bins / 10))

			y = -config.kB * ns.temp * np.log(hist_geoms_modif + 1)
			x = np.linspace(bi_xrange[0], bi_xrange[1], config.bi_nb_bins, endpoint=True)
			k = config.kB * ns.temp / std_grp_bond / std_grp_bond * 100 / 2

			params_guess = [k, avg_grp_bond*10, min(y)]  # multiply for amgstrom for BI

			# calculate derivative to use as sigma for fitting
			y_forward_shift = collections.deque(y)
			y_forward_shift.rotate(3)
			deriv = abs(y - y_forward_shift)
			deriv = collections.deque(deriv)
			deriv.rotate(-3)

			nb_passes = 5
			for _ in range(nb_passes):
				deriv = math_utils.sma(deriv, int(config.bi_nb_bins / 5))

			deriv *= np.sqrt(y/min(y))
			deriv = 1/deriv
			sigma = np.where(y < max(y), deriv, np.inf)

			popt, pcov = curve_fit(gmx_bonds_func_1, x * 10, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)  # multiply for amgstrom for BI

			# here we just update the force constant, bond length is already set to the average of distribution
			ns.out_itp['bond'][grp_bond]['fct'] = popt[0]*100

			# stay within limits in case user requires low force constants
			if not 0 <= ns.out_itp['bond'][grp_bond]['fct'] <= min(config.default_max_fct_bonds_bi, ns.default_max_fct_bonds_opti):
				ns.out_itp['bond'][grp_bond]['fct'] = min(config.default_max_fct_bonds_bi, ns.default_max_fct_bonds_opti) / 2

			if ns.verbose:
				print('  Bond group', grp_bond+1, 'estimated force constant:', round(ns.out_itp['bond'][grp_bond]['fct'], 2))

		ns.performed_init_BI['bond'] = True

	if not ns.performed_init_BI['angle'] and ns.opti_cycle['nb_geoms']['angle'] > 0:

		if ns.verbose:
			print()
			print('Performing Direct Boltzmann Inversion (DBI) to estimate angles force constants')

		for grp_angle in range(ns.opti_cycle['nb_geoms']['angle']):

			hists_geoms_bi, std_rad_grp_angle, bi_xrange = ns.data_BI['angle'][grp_angle]
			y = -config.kB * ns.temp * np.log(hists_geoms_bi + 1)
			x = np.linspace(np.deg2rad(bi_xrange[0]), np.deg2rad(bi_xrange[1]), config.bi_nb_bins, endpoint=True)
			k = config.kB * ns.temp / std_rad_grp_angle / std_rad_grp_angle * 100 / 2

			sigma = np.where(y < max(y), 0.1, np.inf)  # this is definitely better when angles have bimodal distributions

			# use appropriate angle function
			func = ns.cg_itp['angle'][grp_angle]['func']

			if func == 1:
				params_guess = [k, std_rad_grp_angle, min(y)]
				popt, pcov = curve_fit(gmx_angles_func_1, x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)
				popt[0] = abs(popt[0])  # just to be safe, in case the fit yielded negative fct values but this is very unlikely since we provide good starting parameters for the fit

			elif func == 2:
				params_guess = [max(y)-min(y), std_rad_grp_angle, min(y)]
				try:
					popt, pcov = curve_fit(gmx_angles_func_2, x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)
					if popt[0] < 0:  # correct the negative force constant that can result from the fit of stiff angles at values close to 180
						popt[0] = config.default_max_fct_angles_bi * 0.8 # stiff is most probably max fct value, so get close to it
					elif bi_xrange[1] == 180 - ns.bw_angles/2:
						popt[0] += 10
				except RuntimeError:  # curve fit did not converge
					popt[0] = 30

			# here we just update the force constant, angle value is already set to the average of distribution
			ns.out_itp['angle'][grp_angle]['fct'] = popt[0]

			# stay within limits in case user requires low force constants
			if func == 1:
				if not 0 <= ns.out_itp['angle'][grp_angle]['fct'] <= min(config.default_max_fct_angles_bi, ns.default_max_fct_angles_opti_f1):
					ns.out_itp['angle'][grp_angle]['fct'] = min(config.default_max_fct_angles_bi, ns.default_max_fct_angles_opti_f1) / 2
			elif func == 2:
				if not 0 <= ns.out_itp['angle'][grp_angle]['fct'] <= min(config.default_max_fct_angles_bi, ns.default_max_fct_angles_opti_f2):
					ns.out_itp['angle'][grp_angle]['fct'] = min(config.default_max_fct_angles_bi, ns.default_max_fct_angles_opti_f2) / 2

			if ns.verbose:
				print('  Angle group', grp_angle+1, 'estimated force constant:', round(ns.out_itp['angle'][grp_angle]['fct'], 2))

		ns.performed_init_BI['angle'] = True

	if not ns.performed_init_BI['dihedral'] and ns.opti_cycle['nb_geoms']['dihedral'] > 0:

		if ns.verbose:
			print()
			print('Performing Direct Boltzmann Inversion (DBI) to estimate dihedrals force constants')

		for grp_dihedral in range(ns.opti_cycle['nb_geoms']['dihedral']):

			# TODO: clearly the initial fit of dihedrals could be done better -- initial guesses are pretty bad

			hists_geoms_bi, std_rad_grp_dihedral, avg_rad_grp_dihedral, bi_xrange = ns.data_BI['dihedral'][grp_dihedral]
			y = -config.kB * ns.temp * np.log(hists_geoms_bi + 1)
			x = np.linspace(np.deg2rad(bi_xrange[0]), np.deg2rad(bi_xrange[1]), 2*config.bi_nb_bins, endpoint=True)
			k = config.kB * ns.temp / std_rad_grp_dihedral / std_rad_grp_dihedral

			sigma = np.where(y < max(y), 0.1, np.inf)

			# use appropriate dihedral function
			func = ns.cg_itp['dihedral'][grp_dihedral]['func']

			if ns.exec_mode == 2:  # in Mode 2, make the fit according to the equilibrium value provided by user
				avg_rad_grp_dihedral = np.deg2rad(ns.cg_itp['dihedral'][grp_dihedral]['value_user'])
				# NOTE: this could trigger some convergence issue of scipy's curve_fit

			if func in config.dihedral_func_with_mult:
				multiplicity = ns.cg_itp['dihedral'][grp_dihedral]['mult']  # multiplicity stays the same as in input CG ITP, it's only during model_prep that we could compare between different multiplicities
				params_guess = [max(y)-min(y), avg_rad_grp_dihedral, min(y)]
				popt, pcov = curve_fit(gmx_dihedrals_func_1(mult=multiplicity), x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)

			elif func == 2:
				params_guess = [k, avg_rad_grp_dihedral, min(y)]
				popt, pcov = curve_fit(gmx_dihedrals_func_2, x, y, p0=params_guess, sigma=sigma, maxfev=99999, absolute_sigma=False)

			if ns.exec_mode == 1:  # in Mode 1, use the fitted value as equilibrium value (but stay within range)
				# ns.out_itp['dihedral'][grp_dihedral]['value'] = max(min(np.rad2deg(popt[1]), ns.domains_val['dihedral'][grp_dihedral][1]), ns.domains_val['dihedral'][grp_dihedral][0])
				ns.out_itp['dihedral'][grp_dihedral]['value'] = np.rad2deg(popt[1])  # we will apply limits of equilibrium values later

			print('  Dihedral group', grp_dihedral+1, 'estimated force constant BEFORE MODIFIER:', round(popt[0], 2))
			ns.out_itp['dihedral'][grp_dihedral]['fct'] = popt[0]

			# stay within limits in case user requires low force constants
			if func in config.dihedral_func_with_mult:
				if not max(-config.default_abs_range_fct_dihedrals_bi_func_with_mult, -ns.default_abs_range_fct_dihedrals_opti_func_with_mult) <= ns.out_itp['dihedral'][grp_dihedral]['fct'] <= min(-config.default_abs_range_fct_dihedrals_bi_func_with_mult, -ns.default_abs_range_fct_dihedrals_opti_func_with_mult):
					ns.out_itp['dihedral'][grp_dihedral]['fct'] = np.sign(ns.out_itp['dihedral'][grp_dihedral]['fct']) * min(config.default_abs_range_fct_dihedrals_bi_func_with_mult, ns.default_abs_range_fct_dihedrals_opti_func_with_mult) / 2
			else:
				if not max(-config.default_abs_range_fct_dihedrals_bi_func_without_mult, -ns.default_abs_range_fct_dihedrals_opti_func_without_mult) <= ns.out_itp['dihedral'][grp_dihedral]['fct'] <= min(-config.default_abs_range_fct_dihedrals_bi_func_without_mult, -ns.default_abs_range_fct_dihedrals_opti_func_without_mult):
					ns.out_itp['dihedral'][grp_dihedral]['fct'] = np.sign(ns.out_itp['dihedral'][grp_dihedral]['fct']) * min(config.default_abs_range_fct_dihedrals_bi_func_without_mult, ns.default_abs_range_fct_dihedrals_opti_func_without_mult) / 2

			if ns.verbose:
				print('  Dihedral group', grp_dihedral+1, 'estimated force constant:', round(ns.out_itp['dihedral'][grp_dihedral]['fct'], 2))

		ns.performed_init_BI['dihedral'] = True


def process_scaling_str(ns):

	# process specific bonds scaling string, if provided
	ns.bonds_scaling_specific = None
	if ns.bonds_scaling_str != config.bonds_scaling_str:
		sp_str = ns.bonds_scaling_str.split()
		if len(sp_str) % 2 != 0:
			msg = (
				f"Cannot interpret argument -bonds_scaling_str as provided: {ns.bonds_scaling_str}.\n"
				f"Please check your parameters, or the help (-h) for an example."
			)
			raise exceptions.InvalidArgument(msg)

		ns.bonds_scaling_specific = dict()
		i = 0
		try:
			while i < len(sp_str):
				geom_id = sp_str[i][1:]
				if sp_str[i][0].upper() == 'C':
					if int(geom_id) > ns.cg_itp["nb_constraints"]:
						info = "A constraint group id exceeds the number of constraints groups defined in the input CG ITP file."
						raise exceptions.InvalidArgument('bonds_scaling_str', ns.bonds_scaling_str, info)
					if not 'C' + geom_id in ns.bonds_scaling_specific:
						if float(sp_str[i + 1]) < 0:
							info = "You cannot provide negative values for average distribution length."
							raise exceptions.InvalidArgument('bonds_scaling_str', ns.bonds_scaling_str, info)
						ns.bonds_scaling_specific['C' + geom_id] = float(sp_str[i + 1])
					else:
						info = f"A constraint group id is provided multiple times (id: {geom_id})"
						raise exceptions.InvalidArgument('bonds_scaling_str', ns.bonds_scaling_str, info)
				elif sp_str[i][0].upper() == 'B':
					if int(geom_id) > ns.cg_itp["nb_bonds"]:
						info = "A bond group id exceeds the number of bonds groups defined in the input CG ITP file."
						raise exceptions.InvalidArgument('bonds_scaling_str', ns.bonds_scaling_str, info)
					if not 'B' + geom_id in ns.bonds_scaling_specific:
						if float(sp_str[i + 1]) < 0:
							info = "You cannot provide negative values for average distribution length."
							raise exceptions.InvalidArgument('bonds_scaling_str', ns.bonds_scaling_str, info)
						ns.bonds_scaling_specific['B' + geom_id] = float(sp_str[i + 1])
					else:
						info = f"A bond group id is provided multiple times (id: {geom_id})"
						raise exceptions.InvalidArgument('bonds_scaling_str', ns.bonds_scaling_str, info)
				i += 2
		except ValueError:
			raise exceptions.InvalidArgument('bonds_scaling_str', ns.bonds_scaling_str)


# compare 2 models -- atomistic and CG models with plotting
def compare_models(ns, manual_mode=True, ignore_dihedrals=False, calc_sasa=False, record_best_indep_params=False):

	# graphical parameters
	plt.rcParams['grid.color'] = 'k' # plt grid appearance settings
	plt.rcParams['grid.linestyle'] = ':'
	plt.rcParams['grid.linewidth'] = 0.5

	row_wise_ranges = {}
	row_wise_ranges['max_range_constraints'], row_wise_ranges['max_range_bonds'], row_wise_ranges['max_range_angles'], row_wise_ranges['max_range_dihedrals'] = 0, 0, 0, 0

	if ns.atom_only:
		scores.compute_Rg(ns, traj_type='AA')
		print('Radius of gyration (AA reference, NOT CG-mapped):', ns.gyr_aa, 'nm')

	# proceed with CG data
	if not ns.atom_only:

		print('Reading CG trajectory')
		ns.cg_universe = mda.Universe(ns.cg_tpr_filename, ns.cg_traj_filename, in_memory=True, refresh_offsets=True, guess_bonds=False)
		print('  Found', len(ns.cg_universe.trajectory), 'frames')

		if manual_mode:
			# here we read the CG beads masses + actualize the mapped trajectory object
			for bead_id in range(len(ns.cg_itp['atoms'])):
				ns.cg_itp['atoms'][bead_id]['mass'] = ns.cg_universe.atoms[bead_id].mass
			masses = np.array([val['mass'] for val in ns.cg_itp['atoms']])
			ns.aa2cg_universe._topology.masses.values = np.array(masses)

		# create fake bonds in the CG MDA universe, that will be used only for making the molecule whole
		# we make bonds between each VS and their beads definition, so we retrieve the connectivity
		# iteratively towards the real CG beads, that are all connected
		if len(ns.cg_itp["vs_beads_ids"]) > 0:
			fake_bonds = []
			for vs_type in ['2', '3', '4', 'n']:
				try:
					for bead_id in ns.cg_itp['virtual_sites'+vs_type]:
						for vs_def_bead_id in ns.cg_itp['virtual_sites'+vs_type][bead_id]['vs_def_beads_ids']:
							fake_bonds.append([bead_id, vs_def_bead_id])
				except (IndexError, ValueError):
					pass
			ns.cg_universe.add_bonds(fake_bonds, guessed=False)

		# select the whole molecule as an MDA atomgroup and make its coordinates whole, inplace, across the complete trajectory
		ag_mol = mda.AtomGroup([bead_id for bead_id in range(len(ns.cg_itp['atoms']))], ns.cg_universe)
		for _ in ns.cg_universe.trajectory:
			mda.lib.mdamath.make_whole(ag_mol, inplace=True)

		# this requires CG data for mapping -- especially, masses are taken from the CG TPR but the CG ITP is also used atm
		if ns.gyr_aa_mapped == None:
			scores.compute_Rg(ns, traj_type='AA_mapped')
			print()
			print('Radius of gyration (AA reference, CG-mapped, no bonds scaling):', ns.gyr_aa_mapped, '+/-', ns.gyr_aa_mapped_std, 'nm')

		scores.compute_Rg(ns, traj_type='CG')
		print('Radius of gyration (CG model):', ns.gyr_cg, '+/-', ns.gyr_cg_std, 'nm')

		if calc_sasa:

			if ns.sasa_aa_mapped == None:
				scores.compute_SASA(ns, traj_type='AA_mapped')

			scores.compute_SASA(ns, traj_type='CG')
			print()

			# this line checks that gmx trjconv could read the md.xtc trajectory from the opti
			# this is to catch bugged simulation that actually finished and produced the files,
			# but the .gro is a 2D bugged file for example, or trjactory is unreadable by gmx
			if ns.sasa_cg == None:
				return 0, 0, 0, 0, 0, None  # ns.sasa_cg == None will be checked in eval_function and worst score will be attributed

	print()
	print(styling.sep_close, flush=True)
	print('| SCORING AND PLOTTING                                                                        |', flush=True)
	print(styling.sep_close, flush=True)
	print()

	# bonded_calc_time = datetime.now().timestamp()

	# constraints
	print('Processing constraints ...', flush=True)
	diff_ordered_grp_constraints = list(range(ns.cg_itp["nb_constraints"]))
	avg_diff_grp_constraints, row_wise_ranges['constraints'] = [], {}
	constraints = {}

	for grp_constraint in range(ns.cg_itp["nb_constraints"]):

		constraints[grp_constraint] = {'AA': {'x': [], 'y': []}, 'CG': {'x': [], 'y': []}}

		if manual_mode:
			constraint_avg, constraint_hist, _ = scores.get_AA_bonds_distrib(ns, beads_ids=ns.cg_itp['constraint'][grp_constraint]['beads'], grp_type='constraints group', grp_nb=grp_constraint)
			constraints[grp_constraint]['AA']['avg'] = constraint_avg
			constraints[grp_constraint]['AA']['hist'] = constraint_hist
		else:  # use atomistic reference that was loaded by the optimization routines
			constraints[grp_constraint]['AA']['avg'] = ns.cg_itp['constraint'][grp_constraint]['avg']
			constraints[grp_constraint]['AA']['hist'] = ns.cg_itp['constraint'][grp_constraint]['hist']

		for i in range(1, len(constraints[grp_constraint]['AA']['hist'])-1):
			if constraints[grp_constraint]['AA']['hist'][i-1] > 0 or constraints[grp_constraint]['AA']['hist'][i] > 0 or constraints[grp_constraint]['AA']['hist'][i+1] > 0:
				constraints[grp_constraint]['AA']['x'].append(np.mean(ns.bins_constraints[i:i+2]))
				constraints[grp_constraint]['AA']['y'].append(constraints[grp_constraint]['AA']['hist'][i])

		if not ns.atom_only:
			try:
				constraint_avg, constraint_hist, _ = scores.get_CG_bonds_distrib(ns, beads_ids=ns.cg_itp['constraint'][grp_constraint]['beads'], grp_type='constraint')
				constraints[grp_constraint]['CG']['avg'] = constraint_avg
				constraints[grp_constraint]['CG']['hist'] = constraint_hist

				for i in range(1, len(constraint_hist)-1):
					if constraint_hist[i-1] > 0 or constraint_hist[i] > 0 or constraint_hist[i+1] > 0: # TODO: find real min/max correctly, currently this code is garbage (here or nearby) and not robust to changes in bandwidth, in particular for small bandwidths
						constraints[grp_constraint]['CG']['x'].append(np.mean(ns.bins_constraints[i:i+2]))
						constraints[grp_constraint]['CG']['y'].append(constraint_hist[i])

				domain_min = min(constraints[grp_constraint]['AA']['x'][0], constraints[grp_constraint]['CG']['x'][0])
				domain_max = max(constraints[grp_constraint]['AA']['x'][-1], constraints[grp_constraint]['CG']['x'][-1])
				avg_diff_grp_constraints.append(emd(constraints[grp_constraint]['AA']['hist'], constraints[grp_constraint]['CG']['hist'], ns.bins_constraints_dist_matrix) * ns.bonds2angles_scoring_factor)
			except IndexError:
				msg = (
					f"Most probably because you have bonds or constraints that "
					f"exceed {ns.bonded_max_range} nm.\nIncrease bins range for bonds and "
					f"constraints and retry!\nSee argument -bonds_max_range."
				)
				raise ValueError(msg)
		else:
			avg_diff_grp_constraints.append(constraints[grp_constraint]['AA']['avg'])

		if ns.row_x_scaling:
			if ns.atom_only:
				row_wise_ranges['constraints'][grp_constraint] = [constraints[grp_constraint]['AA']['x'][0], constraints[grp_constraint]['AA']['x'][-1]]
			else:
				row_wise_ranges['constraints'][grp_constraint] = [domain_min, domain_max]
			if row_wise_ranges['constraints'][grp_constraint][1] - row_wise_ranges['constraints'][grp_constraint][0] > row_wise_ranges['max_range_constraints']:
				row_wise_ranges['max_range_constraints'] = row_wise_ranges['constraints'][grp_constraint][1] - row_wise_ranges['constraints'][grp_constraint][0]

	# constraint groups ordered by mean difference between atomistic-mapped and CG models
	if ns.mismatch_order and not ns.atom_only:
		diff_ordered_grp_constraints = [x for _, x in sorted(zip(avg_diff_grp_constraints, diff_ordered_grp_constraints), key=lambda pair: pair[0], reverse=True)]

	# bonds
	print('Processing bonds ...', flush=True)
	diff_ordered_grp_bonds = list(range(ns.cg_itp["nb_bonds"]))
	avg_diff_grp_bonds, row_wise_ranges['bonds'] = [], {}
	bonds = {}

	for grp_bond in range(ns.cg_itp["nb_bonds"]):

		bonds[grp_bond] = {'AA': {'x': [], 'y': []}, 'CG': {'x': [], 'y': []}}

		if manual_mode:
			bond_avg, bond_hist, _ = scores.get_AA_bonds_distrib(ns, beads_ids=ns.cg_itp['bond'][grp_bond]['beads'], grp_type='bonds group', grp_nb=grp_bond)
			bonds[grp_bond]['AA']['avg'] = bond_avg
			bonds[grp_bond]['AA']['hist'] = bond_hist
		else:  # use atomistic reference that was loaded by the optimization routines
			bonds[grp_bond]['AA']['avg'] = ns.cg_itp['bond'][grp_bond]['avg']
			bonds[grp_bond]['AA']['hist'] = ns.cg_itp['bond'][grp_bond]['hist']

		for i in range(1, len(bonds[grp_bond]['AA']['hist'])-1):
			if bonds[grp_bond]['AA']['hist'][i-1] > 0 or bonds[grp_bond]['AA']['hist'][i] > 0 or bonds[grp_bond]['AA']['hist'][i+1] > 0:
				bonds[grp_bond]['AA']['x'].append(np.mean(ns.bins_bonds[i:i+2]))
				bonds[grp_bond]['AA']['y'].append(bonds[grp_bond]['AA']['hist'][i])

		if not ns.atom_only:
			try:
				bond_avg, bond_hist, _ = scores.get_CG_bonds_distrib(ns, beads_ids=ns.cg_itp['bond'][grp_bond]['beads'], grp_type='bond')
				bonds[grp_bond]['CG']['avg'] = bond_avg
				bonds[grp_bond]['CG']['hist'] = bond_hist

				for i in range(1, len(bond_hist)-1):
					if bond_hist[i-1] > 0 or bond_hist[i] > 0 or bond_hist[i+1] > 0:
						bonds[grp_bond]['CG']['x'].append(np.mean(ns.bins_bonds[i:i+2]))
						bonds[grp_bond]['CG']['y'].append(bond_hist[i])

				domain_min = min(bonds[grp_bond]['AA']['x'][0], bonds[grp_bond]['CG']['x'][0])
				domain_max = max(bonds[grp_bond]['AA']['x'][-1], bonds[grp_bond]['CG']['x'][-1])
				avg_diff_grp_bonds.append(emd(bonds[grp_bond]['AA']['hist'], bonds[grp_bond]['CG']['hist'], ns.bins_bonds_dist_matrix) * ns.bonds2angles_scoring_factor)
			except IndexError:
				msg = (
					f"Most probably because you have bonds or constraints that "
					f"exceed {ns.bonded_max_range} nm.\nIncrease bins range for bonds and "
					f"constraints and retry!\nSee argument -bonds_max_range."
				)
				raise ValueError(msg)
		else:
			avg_diff_grp_bonds.append(bonds[grp_bond]['AA']['avg'])

		if ns.row_x_scaling:
			if ns.atom_only:
				row_wise_ranges['bonds'][grp_bond] = [bonds[grp_bond]['AA']['x'][0], bonds[grp_bond]['AA']['x'][-1]]
			else:
				row_wise_ranges['bonds'][grp_bond] = [domain_min, domain_max]
			if row_wise_ranges['bonds'][grp_bond][1] - row_wise_ranges['bonds'][grp_bond][0] > row_wise_ranges['max_range_bonds']:
				row_wise_ranges['max_range_bonds'] = row_wise_ranges['bonds'][grp_bond][1] - row_wise_ranges['bonds'][grp_bond][0]

	# bond groups ordered by mean difference between atomistic-mapped and CG models
	if ns.mismatch_order and not ns.atom_only:
		diff_ordered_grp_bonds = [x for _, x in sorted(zip(avg_diff_grp_bonds, diff_ordered_grp_bonds), key=lambda pair: pair[0], reverse=True)]

	# angles
	print('Processing angles ...', flush=True)
	diff_ordered_grp_angles = list(range(ns.cg_itp["nb_angles"]))
	avg_diff_grp_angles, row_wise_ranges['angles'] = [], {}
	angles = {}

	for grp_angle in range(ns.cg_itp["nb_angles"]):

		angles[grp_angle] = {'AA': {'x': [], 'y': []}, 'CG': {'x': [], 'y': []}}

		if manual_mode:
			angle_avg, angle_hist, _, _ = scores.get_AA_angles_distrib(ns, beads_ids=ns.cg_itp['angle'][grp_angle]['beads'])
			angles[grp_angle]['AA']['avg'] = angle_avg
			angles[grp_angle]['AA']['hist'] = angle_hist
		else:  # use atomistic reference that was loaded by the optimization routines
			angles[grp_angle]['AA']['avg'] = ns.cg_itp['angle'][grp_angle]['avg']
			angles[grp_angle]['AA']['hist'] = ns.cg_itp['angle'][grp_angle]['hist']

		for i in range(1, len(angles[grp_angle]['AA']['hist'])-1):
			if angles[grp_angle]['AA']['hist'][i-1] > 0 or angles[grp_angle]['AA']['hist'][i] > 0 or angles[grp_angle]['AA']['hist'][i+1] > 0:
				angles[grp_angle]['AA']['x'].append(np.mean(ns.bins_angles[i:i+2]))
				angles[grp_angle]['AA']['y'].append(angles[grp_angle]['AA']['hist'][i])

		if not ns.atom_only:
			angle_avg, angle_hist, _, _ = scores.get_CG_angles_distrib(ns, beads_ids=ns.cg_itp['angle'][grp_angle]['beads'])
			angles[grp_angle]['CG']['avg'] = angle_avg
			angles[grp_angle]['CG']['hist'] = angle_hist

			for i in range(1, len(angle_hist)-1):
				if angle_hist[i-1] > 0 or angle_hist[i] > 0 or angle_hist[i+1] > 0:
					angles[grp_angle]['CG']['x'].append(np.mean(ns.bins_angles[i:i+2]))
					angles[grp_angle]['CG']['y'].append(angle_hist[i])

			domain_min = min(angles[grp_angle]['AA']['x'][0], angles[grp_angle]['CG']['x'][0])
			domain_max = max(angles[grp_angle]['AA']['x'][-1], angles[grp_angle]['CG']['x'][-1])
			avg_diff_grp_angles.append(emd(angles[grp_angle]['AA']['hist'], angles[grp_angle]['CG']['hist'], ns.bins_angles_dist_matrix))
		else:
			avg_diff_grp_angles.append(angles[grp_angle]['AA']['avg'])

		if ns.row_x_scaling:
			if ns.atom_only:
				row_wise_ranges['angles'][grp_angle] = [angles[grp_angle]['AA']['x'][0], angles[grp_angle]['AA']['x'][-1]]
			else:
				row_wise_ranges['angles'][grp_angle] = [domain_min, domain_max]
			if row_wise_ranges['angles'][grp_angle][1] - row_wise_ranges['angles'][grp_angle][0] > row_wise_ranges['max_range_angles']:
				row_wise_ranges['max_range_angles'] = row_wise_ranges['angles'][grp_angle][1] - row_wise_ranges['angles'][grp_angle][0]

	# angle groups ordered by mean difference between atomistic-mapped and CG models
	if ns.mismatch_order and not ns.atom_only:
		diff_ordered_grp_angles = [x for _, x in sorted(zip(avg_diff_grp_angles, diff_ordered_grp_angles), key=lambda pair: pair[0], reverse=True)]

	# dihedrals
	print('Processing dihedrals ...', flush=True)
	diff_ordered_grp_dihedrals = list(range(ns.cg_itp["nb_dihedrals"]))
	avg_diff_grp_dihedrals, row_wise_ranges['dihedrals'] = [], {}
	dihedrals = {}

	for grp_dihedral in range(ns.cg_itp["nb_dihedrals"]):
		
		dihedrals[grp_dihedral] = {'AA': {'x': [], 'y': []}, 'CG': {'x': [], 'y': []}}

		if manual_mode:
			dihedral_avg, dihedral_hist, _, _ = scores.get_AA_dihedrals_distrib(ns, beads_ids=ns.cg_itp['dihedral'][grp_dihedral]['beads'])
			dihedrals[grp_dihedral]['AA']['avg'] = dihedral_avg
			dihedrals[grp_dihedral]['AA']['hist'] = dihedral_hist
		else:  # use atomistic reference that was loaded by the optimization routines
			dihedrals[grp_dihedral]['AA']['avg'] = ns.cg_itp['dihedral'][grp_dihedral]['avg']
			dihedrals[grp_dihedral]['AA']['hist'] = ns.cg_itp['dihedral'][grp_dihedral]['hist']

		for i in range(1, len(dihedrals[grp_dihedral]['AA']['hist'])-1):
			if dihedrals[grp_dihedral]['AA']['hist'][i-1] > 0 or dihedrals[grp_dihedral]['AA']['hist'][i] > 0 or dihedrals[grp_dihedral]['AA']['hist'][i+1] > 0:
				dihedrals[grp_dihedral]['AA']['x'].append(np.mean(ns.bins_dihedrals[i:i+2]))
				dihedrals[grp_dihedral]['AA']['y'].append(dihedrals[grp_dihedral]['AA']['hist'][i])

		if not ns.atom_only:
			dihedral_avg, dihedral_hist, _, _ = scores.get_CG_dihedrals_distrib(ns, beads_ids=ns.cg_itp['dihedral'][grp_dihedral]['beads'])
			dihedrals[grp_dihedral]['CG']['avg'] = dihedral_avg
			dihedrals[grp_dihedral]['CG']['hist'] = dihedral_hist

			for i in range(1, len(dihedral_hist)-1):
				if dihedral_hist[i-1] > 0 or dihedral_hist[i] > 0 or dihedral_hist[i+1] > 0:
					dihedrals[grp_dihedral]['CG']['x'].append(np.mean(ns.bins_dihedrals[i:i+2]))
					dihedrals[grp_dihedral]['CG']['y'].append(dihedral_hist[i])

			domain_min = min(dihedrals[grp_dihedral]['AA']['x'][0], dihedrals[grp_dihedral]['CG']['x'][0])
			domain_max = max(dihedrals[grp_dihedral]['AA']['x'][-1], dihedrals[grp_dihedral]['CG']['x'][-1])
			avg_diff_grp_dihedrals.append(emd(dihedrals[grp_dihedral]['AA']['hist'], dihedrals[grp_dihedral]['CG']['hist'], ns.bins_dihedrals_dist_matrix))
		else:
			avg_diff_grp_dihedrals.append(dihedrals[grp_dihedral]['AA']['avg'])

		if ns.row_x_scaling:
			if ns.atom_only:
				row_wise_ranges['dihedrals'][grp_dihedral] = [dihedrals[grp_dihedral]['AA']['x'][0], dihedrals[grp_dihedral]['AA']['x'][-1]]
			else:
				row_wise_ranges['dihedrals'][grp_dihedral] = [domain_min, domain_max]
			if row_wise_ranges['dihedrals'][grp_dihedral][1] - row_wise_ranges['dihedrals'][grp_dihedral][0] > row_wise_ranges['max_range_dihedrals']:
				row_wise_ranges['max_range_dihedrals'] = row_wise_ranges['dihedrals'][grp_dihedral][1] - row_wise_ranges['dihedrals'][grp_dihedral][0]

	# dihedral groups ordered by mean difference between atomistic-mapped and CG models
	if ns.mismatch_order and not ns.atom_only:
		diff_ordered_grp_dihedrals = [x for _, x in sorted(zip(avg_diff_grp_dihedrals, diff_ordered_grp_dihedrals), key=lambda pair: pair[0], reverse=True)]

	# bonded_calc_time = datetime.now().timestamp() - bonded_calc_time
	# print('Time for reference distributions calculation:', round(bonded_calc_time / 60, 2), 'min')

	###############################
	# DISPLAY DISTRIBUTIONS PLOTS #
	###############################

	larger_group = max(ns.cg_itp["nb_constraints"], ns.cg_itp["nb_bonds"], ns.cg_itp["nb_angles"], ns.cg_itp["nb_dihedrals"])
	nrow, nrows, ncols = -1, 4, min(ns.ncols_max, larger_group)
	if ns.ncols_max == 0:
		ncols = larger_group
	if larger_group > ncols:
		hidden_cols = larger_group - ncols
		if ns.atom_only:
			print(f'Displaying max {ncols} distributions per row using the CG ITP file ordering of distributions groups ({hidden_cols} more are hidden)')
		else:
			if not ns.mismatch_order:
				print(f'{styling.header_warning}Displaying max {ncols} distributions groups per row and this can be MISLEADING because ordering by pairwise AA-mapped vs. CG distributions mismatch is DISABLED ({hidden_cols} more are hidden)')
			else:
				print(f'Displaying max {ncols} distributions groups per row ordered by pairwise AA-mapped vs. CG distributions difference ({hidden_cols} more are hidden)')
	else:
		print()
		if not ns.mismatch_order:
			print('Distributions groups will be displayed using the CG ITP file groups ordering')
		else:
			print('Distributions groups will be displayed using ranked mismatch score between pairwise AA-mapped and CG distributions')
	nrows -= sum([ns.cg_itp["nb_constraints"] == 0, ns.cg_itp["nb_bonds"] == 0, ns.cg_itp["nb_angles"] == 0, ns.cg_itp["nb_dihedrals"] == 0])

	# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3), squeeze=False)
	# this fucking line was responsible of the big memory leak (figures were not closing) so I let this here for memory
	fig = plt.figure(figsize=(ncols*3, nrows*3))
	ax = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)

	# record the min/max y for each geom type
	constraints_min_y, bonds_min_y, angles_min_y, dihedrals_min_y = 10, 10, 10, 10
	constraints_max_y, bonds_max_y, angles_max_y, dihedrals_max_y = 0, 0, 0, 0

	# constraints
	if ns.cg_itp["nb_constraints"] != 0:
		print()
		nrow += 1
		for i in range(ncols):
			if i < ns.cg_itp["nb_constraints"]:
				grp_constraint = diff_ordered_grp_constraints[i]

				if config.use_hists:
					ax[nrow][i].step(constraints[grp_constraint]['AA']['x'], constraints[grp_constraint]['AA']['y'], label='AA-mapped', color=config.atom_color, where='mid', alpha=config.line_alpha)
					ax[nrow][i].fill_between(constraints[grp_constraint]['AA']['x'], constraints[grp_constraint]['AA']['y'], color=config.atom_color, step='mid', alpha=config.fill_alpha)
				else:
					ax[nrow][i].plot(constraints[grp_constraint]['AA']['x'], constraints[grp_constraint]['AA']['y'], label='AA-mapped', color=config.atom_color, alpha=config.line_alpha)
					ax[nrow][i].fill_between(constraints[grp_constraint]['AA']['x'], constraints[grp_constraint]['AA']['y'], color=config.atom_color, alpha=config.fill_alpha)
				ax[nrow][i].plot(constraints[grp_constraint]['AA']['avg'], 0, color=config.atom_color, marker='D')

				if not ns.atom_only:
					ax[nrow][i].set_title(f'Constraint grp {grp_constraint + 1} - EMD Δ {round(avg_diff_grp_constraints[grp_constraint], 3)}')
					if config.use_hists:
						ax[nrow][i].step(constraints[grp_constraint]['CG']['x'], constraints[grp_constraint]['CG']['y'], label='CG', color=config.cg_color, where='mid', alpha=config.line_alpha)
						ax[nrow][i].fill_between(constraints[grp_constraint]['CG']['x'], constraints[grp_constraint]['CG']['y'], color=config.cg_color, step='mid', alpha=config.fill_alpha)
					else:
						ax[nrow][i].plot(constraints[grp_constraint]['CG']['x'], constraints[grp_constraint]['CG']['y'], label='CG', color=config.cg_color, alpha=config.line_alpha)
						ax[nrow][i].fill_between(constraints[grp_constraint]['CG']['x'], constraints[grp_constraint]['CG']['y'], color=config.cg_color, alpha=config.fill_alpha)
					ax[nrow][i].plot(constraints[grp_constraint]['CG']['avg'], 0, color=config.cg_color, marker='D')
					print(f"Constraint {grp_constraint + 1} -- AA Avg: {round(constraints[grp_constraint]['AA']['avg'], 3)} nm -- CG Avg: {round(constraints[grp_constraint]['CG']['avg'], 3)}")
				else:
					ax[nrow][i].set_title(f'Constraint grp {grp_constraint+1} - Avg {round(avg_diff_grp_constraints[grp_constraint], 3)} nm')
					print(f"Constraint {grp_constraint + 1} -- AA Avg: {round(constraints[grp_constraint]['AA']['avg'], 3)}")
				ax[nrow][i].grid(zorder=0.5)
				if ns.row_x_scaling:
					ax[nrow][i].set_xlim(np.mean(row_wise_ranges['constraints'][grp_constraint])-row_wise_ranges['max_range_constraints']/2*1.1, np.mean(row_wise_ranges['constraints'][grp_constraint])+row_wise_ranges['max_range_constraints']/2*1.1)
				if i % 2 == 0:
					ax[nrow][i].legend(loc='upper left')
				if ax[nrow][i].get_ylim()[0] < constraints_min_y:
					constraints_min_y = ax[nrow][i].get_ylim()[0]
				if ax[nrow][i].get_ylim()[1] > constraints_max_y:
					constraints_max_y = ax[nrow][i].get_ylim()[1]

			else:
				ax[nrow][i].set_visible(False)

	# bonds
	if ns.cg_itp["nb_bonds"] != 0:
		print()
		nrow += 1
		for i in range(ncols):
			if i < ns.cg_itp["nb_bonds"]:
				grp_bond = diff_ordered_grp_bonds[i]

				if config.use_hists:
					ax[nrow][i].step(bonds[grp_bond]['AA']['x'], bonds[grp_bond]['AA']['y'], label='AA-mapped', color=config.atom_color, where='mid', alpha=config.line_alpha)
					ax[nrow][i].fill_between(bonds[grp_bond]['AA']['x'], bonds[grp_bond]['AA']['y'], color=config.atom_color, step='mid', alpha=config.fill_alpha)
				else:
					ax[nrow][i].plot(bonds[grp_bond]['AA']['x'], bonds[grp_bond]['AA']['y'], label='AA-mapped', color=config.atom_color, alpha=config.line_alpha)
					ax[nrow][i].fill_between(bonds[grp_bond]['AA']['x'], bonds[grp_bond]['AA']['y'], color=config.atom_color, alpha=config.fill_alpha)
				ax[nrow][i].plot(bonds[grp_bond]['AA']['avg'], 0, color=config.atom_color, marker='D')

				if not ns.atom_only:
					ax[nrow][i].set_title(f'Bond grp {grp_bond+1} - EMD Δ {round(avg_diff_grp_bonds[grp_bond], 3)}')
					if config.use_hists:
						ax[nrow][i].step(bonds[grp_bond]['CG']['x'], bonds[grp_bond]['CG']['y'], label='CG', color=config.cg_color, where='mid', alpha=config.line_alpha)
						ax[nrow][i].fill_between(bonds[grp_bond]['CG']['x'], bonds[grp_bond]['CG']['y'], color=config.cg_color, step='mid', alpha=config.fill_alpha)
					else:
						ax[nrow][i].plot(bonds[grp_bond]['CG']['x'], bonds[grp_bond]['CG']['y'], label='CG', color=config.cg_color, alpha=config.line_alpha)
						ax[nrow][i].fill_between(bonds[grp_bond]['CG']['x'], bonds[grp_bond]['CG']['y'], color=config.cg_color, alpha=config.fill_alpha)
					ax[nrow][i].plot(bonds[grp_bond]['CG']['avg'], 0, color=config.cg_color, marker='D')
					print(f"Bond {grp_bond + 1} -- AA Avg: {round(bonds[grp_bond]['AA']['avg'], 3)} nm -- CG Avg: {round(bonds[grp_bond]['CG']['avg'], 3)} nm")
				else:
					ax[nrow][i].set_title(f"Bond grp {grp_bond+1} - Avg {round(avg_diff_grp_bonds[grp_bond], 3)} nm")
					print(f"Bond {grp_bond+1} -- AA Avg: {round(bonds[grp_bond]['AA']['avg'], 3)}")
				ax[nrow][i].grid(zorder=0.5)
				if ns.row_x_scaling:
					ax[nrow][i].set_xlim(np.mean(row_wise_ranges['bonds'][grp_bond])-row_wise_ranges['max_range_bonds']/2*1.1, np.mean(row_wise_ranges['bonds'][grp_bond])+row_wise_ranges['max_range_bonds']/2*1.1)
				if i % 2 == 0:
					ax[nrow][i].legend(loc='upper left')
				if ax[nrow][i].get_ylim()[0] < bonds_min_y:
					bonds_min_y = ax[nrow][i].get_ylim()[0]
				if ax[nrow][i].get_ylim()[1] > bonds_max_y:
					bonds_max_y = ax[nrow][i].get_ylim()[1]

			else:
				ax[nrow][i].set_visible(False)

	# angles
	if ns.cg_itp["nb_angles"] != 0:
		print()
		nrow += 1
		for i in range(ncols):
			if i < ns.cg_itp["nb_angles"]:
				grp_angle = diff_ordered_grp_angles[i]

				if config.use_hists:
					ax[nrow][i].step(angles[grp_angle]['AA']['x'], angles[grp_angle]['AA']['y'], label='AA-mapped', color=config.atom_color, where='mid', alpha=config.line_alpha)
					ax[nrow][i].fill_between(angles[grp_angle]['AA']['x'], angles[grp_angle]['AA']['y'], color=config.atom_color, step='mid', alpha=config.fill_alpha)
				else:
					ax[nrow][i].plot(angles[grp_angle]['AA']['x'], angles[grp_angle]['AA']['y'], label='AA-mapped', color=config.atom_color, alpha=config.line_alpha)
					ax[nrow][i].fill_between(angles[grp_angle]['AA']['x'], angles[grp_angle]['AA']['y'], color=config.atom_color, alpha=config.fill_alpha)
				ax[nrow][i].plot(angles[grp_angle]['AA']['avg'], 0, color=config.atom_color, marker='D')

				if not ns.atom_only:
					ax[nrow][i].set_title(f'Angle grp {grp_angle+1} - EMD Δ {round(avg_diff_grp_angles[grp_angle], 3)}')
					if config.use_hists:
						ax[nrow][i].step(angles[grp_angle]['CG']['x'], angles[grp_angle]['CG']['y'], label='CG', color=config.cg_color, where='mid', alpha=config.line_alpha)
						ax[nrow][i].fill_between(angles[grp_angle]['CG']['x'], angles[grp_angle]['CG']['y'], color=config.cg_color, step='mid', alpha=config.fill_alpha)
					else:
						ax[nrow][i].plot(angles[grp_angle]['CG']['x'], angles[grp_angle]['CG']['y'], label='CG', color=config.cg_color, alpha=config.line_alpha)
						ax[nrow][i].fill_between(angles[grp_angle]['CG']['x'], angles[grp_angle]['CG']['y'], color=config.cg_color, alpha=config.fill_alpha)
					ax[nrow][i].plot(angles[grp_angle]['CG']['avg'], 0, color=config.cg_color, marker='D')
					print(f"Angle {grp_angle+1} -- AA Avg: {round(angles[grp_angle]['AA']['avg'], 1)}° -- CG Avg: {round(angles[grp_angle]['CG']['avg'], 1)}°")
				else:
					ax[nrow][i].set_title(f"Angle grp {grp_angle+1} - Avg {round(avg_diff_grp_angles[grp_angle], 1)}°")
					print(f"Angle {grp_angle+1} -- AA Avg: {round(angles[grp_angle]['AA']['avg'], 1)}")
				ax[nrow][i].grid(zorder=0.5)
				if ns.row_x_scaling:
					ax[nrow][i].set_xlim(np.mean(row_wise_ranges['angles'][grp_angle])-row_wise_ranges['max_range_angles']/2*1.1, np.mean(row_wise_ranges['angles'][grp_angle])+row_wise_ranges['max_range_angles']/2*1.1)
				if i % 2 == 0:
					ax[nrow][i].legend(loc='upper left')
				if ax[nrow][i].get_ylim()[0] < angles_min_y:
					angles_min_y = ax[nrow][i].get_ylim()[0]
				if ax[nrow][i].get_ylim()[1] > angles_max_y:
					angles_max_y = ax[nrow][i].get_ylim()[1]

			else:
				ax[nrow][i].set_visible(False)

	# dihedrals
	if ns.cg_itp["nb_dihedrals"] != 0:
		print()
		nrow += 1
		for i in range(ncols):
			if i < ns.cg_itp["nb_dihedrals"]:
				grp_dihedral = diff_ordered_grp_dihedrals[i]

				if config.use_hists:
					ax[nrow][i].step(dihedrals[grp_dihedral]['AA']['x'], dihedrals[grp_dihedral]['AA']['y'], label='AA-mapped', color=config.atom_color, where='mid', alpha=config.line_alpha)
					ax[nrow][i].fill_between(dihedrals[grp_dihedral]['AA']['x'], dihedrals[grp_dihedral]['AA']['y'], color=config.atom_color, step='mid', alpha=config.fill_alpha)
				else:
					ax[nrow][i].plot(dihedrals[grp_dihedral]['AA']['x'], dihedrals[grp_dihedral]['AA']['y'], label='AA-mapped', color=config.atom_color, alpha=config.line_alpha)
					ax[nrow][i].fill_between(dihedrals[grp_dihedral]['AA']['x'], dihedrals[grp_dihedral]['AA']['y'], color=config.atom_color, alpha=config.fill_alpha)
				ax[nrow][i].plot(dihedrals[grp_dihedral]['AA']['avg'], 0, color=config.atom_color, marker='D')

				if not ns.atom_only:
					ax[nrow][i].set_title(f'Dihedral grp {grp_dihedral+1} - EMD Δ {round(avg_diff_grp_dihedrals[grp_dihedral], 3)}')
					if config.use_hists:
						ax[nrow][i].step(dihedrals[grp_dihedral]['CG']['x'], dihedrals[grp_dihedral]['CG']['y'], label='CG', color=config.cg_color, where='mid', alpha=config.line_alpha)
						ax[nrow][i].fill_between(dihedrals[grp_dihedral]['CG']['x'], dihedrals[grp_dihedral]['CG']['y'], color=config.cg_color, step='mid', alpha=config.fill_alpha)
					else:
						ax[nrow][i].plot(dihedrals[grp_dihedral]['CG']['x'], dihedrals[grp_dihedral]['CG']['y'], label='CG', color=config.cg_color, alpha=config.line_alpha)
						ax[nrow][i].fill_between(dihedrals[grp_dihedral]['CG']['x'], dihedrals[grp_dihedral]['CG']['y'], color=config.cg_color, alpha=config.fill_alpha)
					ax[nrow][i].plot(dihedrals[grp_dihedral]['CG']['avg'], 0, color=config.cg_color, marker='D')
					print(f"Dihedral {grp_dihedral+1} -- AA Avg: {round(dihedrals[grp_dihedral]['AA']['avg'], 1)}° -- CG Avg: {round(dihedrals[grp_dihedral]['CG']['avg'], 1)}°")
				else:
					ax[nrow][i].set_title(f'Dihedral grp {grp_dihedral+1} - Avg {round(avg_diff_grp_dihedrals[grp_dihedral], 1)}°')
					print(f"Dihedral {grp_dihedral+1} -- AA Avg: {round(dihedrals[grp_dihedral]['AA']['avg'], 1)}")
				ax[nrow][i].grid(zorder=0.5)
				if ns.row_x_scaling:
					ax[nrow][i].set_xlim(np.mean(row_wise_ranges['dihedrals'][grp_dihedral])-row_wise_ranges['max_range_dihedrals']/2*1.1, np.mean(row_wise_ranges['dihedrals'][grp_dihedral])+row_wise_ranges['max_range_dihedrals']/2*1.1)
				if i % 2 == 0:
					ax[nrow][i].legend(loc='upper left')
				if ax[nrow][i].get_ylim()[0] < dihedrals_min_y:
					dihedrals_min_y = ax[nrow][i].get_ylim()[0]
				if ax[nrow][i].get_ylim()[1] > dihedrals_max_y:
					dihedrals_max_y = ax[nrow][i].get_ylim()[1]

			else:
				ax[nrow][i].set_visible(False)

	# now we have all the ylims, so make them all consistent
	if ns.row_y_scaling:
		nrow = -1
		if ns.cg_itp["nb_constraints"] != 0:
			nrow += 1
			for i in range(ns.cg_itp["nb_constraints"]):
				ax[nrow][i].set_ylim(bottom=constraints_min_y, top=constraints_max_y)
		if ns.cg_itp["nb_bonds"] != 0:
			nrow += 1
			for i in range(ns.cg_itp["nb_bonds"]):
				ax[nrow][i].set_ylim(bottom=bonds_min_y, top=bonds_max_y)
		if ns.cg_itp["nb_angles"] != 0:
			nrow += 1
			for i in range(ns.cg_itp["nb_angles"]):
				ax[nrow][i].set_ylim(bottom=angles_min_y, top=angles_max_y)
		if ns.cg_itp["nb_dihedrals"] != 0:
			nrow += 1
			for i in range(ns.cg_itp["nb_dihedrals"]):
				ax[nrow][i].set_ylim(bottom=dihedrals_min_y, top=dihedrals_max_y)

	# calculate global fitness score and contributions from each geom type
	all_dist_pairwise = '' # for global optimization plotting
	all_emd_dist_geoms = {'constraints': [], 'bonds': [], 'angles': [], 'dihedrals': []}

	if not ns.atom_only:
		fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = 0, 0, 0, 0

		for i in range(ns.cg_itp["nb_constraints"]):
			dist_pairwise = avg_diff_grp_constraints[diff_ordered_grp_constraints[i]]
			all_dist_pairwise += str(dist_pairwise)+' '
			all_emd_dist_geoms['constraints'].append(dist_pairwise)

			# keep track of independent best parameters
			if record_best_indep_params:
				if dist_pairwise < ns.all_best_emd_dist_geoms['constraints'][i]:
					ns.all_best_emd_dist_geoms['constraints'][i] = dist_pairwise
					ns.all_best_params_dist_geoms['constraints'][i]['params'] = [ns.out_itp['constraint'][i]['value']]

			dist_pairwise = dist_pairwise ** 2
			fit_score_constraints_bonds += dist_pairwise

		for i in range(ns.cg_itp["nb_bonds"]):
			dist_pairwise = avg_diff_grp_bonds[diff_ordered_grp_bonds[i]]
			all_dist_pairwise += str(dist_pairwise)+' '
			all_emd_dist_geoms['bonds'].append(dist_pairwise)

			# keep track of independent best parameters
			if record_best_indep_params:
				if dist_pairwise < ns.all_best_emd_dist_geoms['bonds'][i]:
					ns.all_best_emd_dist_geoms['bonds'][i] = dist_pairwise
					ns.all_best_params_dist_geoms['bonds'][i]['params'] = [ns.out_itp['bond'][i]['value'], ns.out_itp['bond'][i]['fct']]

			dist_pairwise = dist_pairwise ** 2
			fit_score_constraints_bonds += dist_pairwise

		for i in range(ns.cg_itp["nb_angles"]):
			dist_pairwise = avg_diff_grp_angles[diff_ordered_grp_angles[i]]
			all_dist_pairwise += str(dist_pairwise)+' '
			all_emd_dist_geoms['angles'].append(dist_pairwise)

			# keep track of independent best parameters
			if record_best_indep_params:
				if dist_pairwise < ns.all_best_emd_dist_geoms['angles'][i]:
					ns.all_best_emd_dist_geoms['angles'][i] = dist_pairwise
					ns.all_best_params_dist_geoms['angles'][i]['params'] = [ns.out_itp['angle'][i]['value'], ns.out_itp['angle'][i]['fct']]

			dist_pairwise = dist_pairwise ** 2
			fit_score_angles += dist_pairwise

		# dihedrals_dist_pairwise = 0
		for i in range(ns.cg_itp["nb_dihedrals"]):
			dist_pairwise = avg_diff_grp_dihedrals[diff_ordered_grp_dihedrals[i]]
			all_dist_pairwise += str(dist_pairwise)+' '
			all_emd_dist_geoms['dihedrals'].append(dist_pairwise)

			# keep track of independent best parameters
			if record_best_indep_params and not ignore_dihedrals:
				if dist_pairwise < ns.all_best_emd_dist_geoms['dihedrals'][i]:
					ns.all_best_emd_dist_geoms['dihedrals'][i] = dist_pairwise
					ns.all_best_params_dist_geoms['dihedrals'][i]['params'] = [ns.out_itp['dihedral'][i]['value'], ns.out_itp['dihedral'][i]['fct']]

			dist_pairwise = dist_pairwise ** 2
			fit_score_dihedrals += dist_pairwise

		fit_score_constraints_bonds = np.sqrt(fit_score_constraints_bonds)
		fit_score_angles = np.sqrt(fit_score_angles)
		fit_score_dihedrals = np.sqrt(fit_score_dihedrals)

		fit_score_total = fit_score_constraints_bonds + fit_score_angles + fit_score_dihedrals

		fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = round(fit_score_total, 3), round(fit_score_constraints_bonds, 3), round(fit_score_angles, 3), round(fit_score_dihedrals, 3)
		all_dist_pairwise += '\n'
		print()
		print('Using bonds to angles/dihedrals (C) scoring constant:', ns.bonds2angles_scoring_factor)
		print()
		print('Global fitness score:', fit_score_total, '(lower is better)', flush=True)
		print('  Bonds/Constraints constribution to fitness score:', fit_score_constraints_bonds, flush=True)
		print('  Angles constribution to fitness score:', fit_score_angles, flush=True)
		print('  Dihedrals constribution to fitness score:', fit_score_dihedrals, flush=True)

		plt.tight_layout(rect=[0, 0, 1, 0.9])
		eval_score = fit_score_total
		if ignore_dihedrals and ns.cg_itp["nb_dihedrals"] > 0:
			eval_score -= fit_score_dihedrals
		sup_title = f'FITNESS SCORE\nTotal: {round(eval_score, 3)} -- Constraints/Bonds: {fit_score_constraints_bonds} -- Angles: {fit_score_angles} -- Dihedrals: {fit_score_dihedrals}'
		if ignore_dihedrals and ns.cg_itp["nb_dihedrals"] > 0:
			sup_title += ' (ignored)'
		plt.suptitle(sup_title)
	else:
		plt.tight_layout()

	# here we close everything we can close because there was a memory leak from plotting
	plt.savefig(ns.plot_filename)
	plt.close(fig)
	print()
	print('Distributions plot written at location:\n ', ns.plot_filename, flush=True)
	print()

	if not manual_mode and not ns.atom_only:
		return fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, all_dist_pairwise, all_emd_dist_geoms
	else:
		return




