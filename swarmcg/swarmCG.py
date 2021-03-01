# some numpy version have this ufunc warning at import + many packages call numpy and display annoying warnings
import warnings
warnings.filterwarnings("ignore")
import sys, re, random, os, shutil, subprocess, signal, time, contextlib
import warnings, collections
from datetime import datetime

# matplotlib new version has some problems with incorrectly uninstalled files at version upgrade and display a lot of warnings
# also some numpy version have this ufunc warning at import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MDAnalysis as mda
from pyemd import emd
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

from swarmcg import config
from swarmcg.shared import utils, styling
from swarmcg.shared import exceptions
from swarmcg.simulations.potentials import (gmx_bonds_func_1, gmx_angles_func_1, gmx_angles_func_2,
	gmx_dihedrals_func_1, gmx_dihedrals_func_2)
import swarmcg.simulations.vs_functions as vsf

matplotlib.use('AGG')  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation
warnings.resetwarnings()

# TODO: When provided trajectory file does NOT contain PBC infos (box position and size for each frame, which are present in XTC format for example), we want to stil accept the provided trajectory format (if accepted by MDAnalysis) but we automatically disable the handling of PBC by the code


# cast object as string, enclose by parentheses and return a string -- for arguments display in help
def par_wrap(string):
	return f'({string})'


# set MDAnalysis backend and number of threads
# NOTE: this is not used because MDA is not properly parallelized, in fact with OpenMP backend it's slower than in serial
def set_MDA_backend(ns):

	if mda.lib.distances.USED_OPENMP:  # if MDAnalysis was compiled with OpenMP support
		ns.mda_backend = 'OpenMP'
	else:
		ns.mda_backend = 'serial'


# draw random float between given range and apply rounding to given digit
def draw_float(low, high, dg_rnd):

	return round(random.uniform(low, high), dg_rnd) # low and high included


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


# check if functions present in CG ITP file can be used by this program, if not we throw an error
# authorized functions are defined in config.py (we switch them on in config.py once we have tested them)
def verify_handled_functions(geom, func, line_nb):

	try:
		func = int(func)
	except (ValueError, IndexError):
		msg = (
			f"Unexpected error while reading CG ITP file at line {line_nb}, please check this file."
		)
		raise exceptions.MissformattedFile(msg)

	if func not in config.handled_functions[geom]:
		functions_str = ", ".join(map(str, config.handled_functions[geom]))
		if functions_str == '':
			functions_str = 'None'
		msg = (
			f"Error while reading {geom} function in CG ITP file at line {line_nb}.\n"
			f"This potential function is not implemented in Swarm-CG at the moment.\n"
			f"Please use one of these {geom} potential functions: {functions_str}.\n\n"
			f"If you feel this is an important missing feature, please feel free to\n"
			f"open an issue on github at {config.github_url}/issues."
		)
		raise exceptions.MissformattedFile(msg)

	return func


# TODO: the 3 next functions below (section_switch, vs_error_control, read_cg_itp_file) could be isolated in a sort of
#       "topology reader" class, and next we would include the formats from other MD engines

# sections switch for reading ITP sections
def section_switch(section_read, section_active):

	for section_current in section_read:
		section_read[section_current] = False
	if section_active is not None:
		section_read[section_active] = True


# vs_type in [2, 3, 4, n], then they each have specific functions to define their positions
def vs_error_control(ns, bead_id, vs_type, func, line_nb, vs_def_beads_ids=None):

	if bead_id >= len(ns.cg_itp['atoms']):
		msg = (
			f"A virtual site is defined for ID {bead_id + 1}, while this ID exceeds the number of atoms"
			f" defined in the CG ITP file."
		)
		raise exceptions.MissformattedFile(msg)

	if vs_def_beads_ids is not None:
		for bid in vs_def_beads_ids:
			if bid >= len(ns.cg_itp['atoms']):
				msg = (
					f"The definition of virtual site ID {bead_id + 1} makes use of ID {bid + 1}, while this ID exceeds"
					f" the number of atoms defined in the CG ITP file."
				)
				raise exceptions.MissformattedFile(msg)

	if not ns.cg_itp['atoms'][bead_id]['bead_type'].startswith('v'):
		msg = (
			f"CG bead number {bead_id + 1} is referenced to as a virtual site, but its bead type"
			f" does NOT start with letter 'v'."
		)
		raise exceptions.MissformattedFile(msg)

	vs_type_str = f'virtual_sites{vs_type}'
	func = verify_handled_functions(vs_type_str, func, line_nb)

	return func


# read coarse-grain ITP
def read_cg_itp_file(ns):

	print('Reading Coarse-Grained (CG) ITP file')
	ns.cg_itp = {'moleculetype': {'molname': '', 'nrexcl': 0}, 'atoms': [], 'constraint': [], 'bond': [], 'angle': [], 'dihedral': [], 'virtual_sites2': {}, 'virtual_sites3': {}, 'virtual_sites4': {}, 'virtual_sitesn': {}, 'exclusion': []}
	ns.real_beads_ids, ns.vs_beads_ids = [], []
	ns.nb_constraints, ns.nb_bonds, ns.nb_angles, ns.nb_dihedrals = -1, -1, -1, -1

	with open(ns.cg_itp_filename, 'r') as fp:
		try:
			itp_lines = fp.read().split('\n')
			itp_lines = [itp_line.split(';')[0].strip() for itp_line in itp_lines]
		except UnicodeDecodeError:
			msg = "Cannot read CG ITP, it seems you provided a binary file."
			raise exceptions.MissformattedFile(msg)

	section_read = {
		'moleculetype': False,
		'atom': False,
		'constraint': False,
		'bond': False,
		'angle': False,
		'dihedral': False,
		'vs_2': False,
		'vs_3': False,
		'vs_4': False,
		'vs_n': False,
		'exclusion': False
	}

	def msg_force_boundaries(line, min_fct, max_fct, str_arg):
		msg = (
			f'You activated the option to take into account the parameters provided via\n'
			f'the ITP file, but the input force constant provided at line {line} is outside\n'
			f'of the range of authorized values: [{min_fct}, {max_fct}].\n'
			f'Please either modify the argument {str_arg} or modify the force\n'
			f'constant in your ITP file.'
		)
		return msg

	for i in range(len(itp_lines)):
		itp_line = itp_lines[i]
		if itp_line != '':

			if bool(re.search('\[.*moleculetype.*\]', itp_line)):
				section_switch(section_read, 'moleculetype')
			elif bool(re.search('\[.*atoms.*\]', itp_line)):
				section_switch(section_read, 'atom')
			elif bool(re.search('\[.*constraint.*\]', itp_line)):
				section_switch(section_read, 'constraint')
			elif bool(re.search('\[.*bond.*\]', itp_line)):
				section_switch(section_read, 'bond')
			elif bool(re.search('\[.*angle.*\]', itp_line)):
				section_switch(section_read, 'angle')
			elif bool(re.search('\[.*dihedral.*\]', itp_line)):
				section_switch(section_read, 'dihedral')
			elif bool(re.search('\[.*virtual_sites2.*\]', itp_line)):
				section_switch(section_read, 'vs_2')
			elif bool(re.search('\[.*virtual_sites3.*\]', itp_line)):
				section_switch(section_read, 'vs_3')
			elif bool(re.search('\[.*virtual_sites4.*\]', itp_line)):
				section_switch(section_read, 'vs_4')
			elif bool(re.search('\[.*virtual_sitesn.*\]', itp_line)):
				section_switch(section_read, 'vs_n')
			elif bool(re.search('\[.*exclusion.*\]', itp_line)):
				section_switch(section_read, 'exclusion')
			elif bool(re.search('\[.*\]', itp_line)):  # all other sections
				section_switch(section_read, None)

			else:
				sp_itp_line = itp_line.split()

				if section_read['moleculetype']:

					ns.cg_itp['moleculetype']['molname'], ns.cg_itp['moleculetype']['nrexcl'] = sp_itp_line[0], int(sp_itp_line[1])

				elif section_read['atom']:

					# TODO: test what happens if there are VS in the middle of real CG beads in the [ atoms ] section
					#       because most probably this won't be OK -- Not sure who does this though, but seems possible

					if len(sp_itp_line) == 7:
						bead_id, bead_type, resnr, residue, atom, cgnr, charge = sp_itp_line[:7]
						mass = None

						# In case the masses are ABSENT in the ITP file (probably the most normal case with
						# MARTINI usage), then we will read the CG masses from the TPR file to avoid having
						# to look into TOP and potentially multiple ITP files:
						#
						#  - from evaluate_model.py this means a TPR has been provided already, if the user is not using
						#    the script for exclusive AA distributions inspection (in which case we don't need the masses
						#    at all because we will use mapped/splitted weights of the atoms into CG beads exclusively anyway)
						#  - from optimize_model.py all the ITP included in the TOP file are read to find
						#    appropriate masses

					elif len(sp_itp_line) == 8:
						bead_id, bead_type, resnr, residue, atom, cgnr, charge, mass = sp_itp_line[:8]
						mass = float(mass)
					else:
						msg = (
						"The atom description from the input itp file: \n\n {} \n\n"
						"does not contain the correct number of fields. Please insert "
						"the following information: \n\n  bead_id, bead_type, resnr, "
						"residue, atom, cgnr, charge, [mass] \n\n".format(itp_line)
						)
						raise exceptions.MissformattedFile(msg)

					# discriminate between real beads and virtual sites
					if bead_type.startswith('v'):
						ns.vs_beads_ids.append(int(bead_id) - 1)
					else:
						ns.real_beads_ids.append(int(bead_id) - 1)

					# assignment of the variables value
					ns.cg_itp['atoms'].append({'bead_id': int(bead_id) - 1, 'bead_type': bead_type, 'resnr': int(resnr), 'residue': residue,'atom': atom, 'cgnr': int(cgnr), 'charge': float(charge), 'mass': mass, 'vs_type': None})
					# here there is still MASS and VS_TYPE that are subject to later modification

					if not len(ns.cg_itp['atoms']) == int(bead_id):
						msg = (
							f"Swarm-CG handles .itp files with atoms indexed consecutively starting from 1.\n"
							f"The bead numbered {bead_id + 1} does not follow this formatting."
						)
						raise exceptions.MissformattedFile(msg)

				elif section_read['constraint']:

					# beginning of a new group
					if itp_lines[i-1] == '' or itp_lines[i-1].startswith(';') or bool(re.search('\[.*constraint.*\]', itp_lines[i-1])):
						ns.nb_constraints += 1
						if itp_lines[i-1].startswith('; constraint type'):
							geom_type = itp_lines[i-1].split()[3]  # if the current CG ITP was generated with our package

						else:
							geom_type = str(len(ns.cg_itp['constraint'])+1)
						ns.cg_itp['constraint'].append({'geom_type': geom_type, 'beads': [], 'func': [], 'value': [], 'value_user': []})  # initialize storage for this new group

					try:
						ns.cg_itp['constraint'][ns.nb_constraints]['beads'].append([int(bead_id)-1 for bead_id in sp_itp_line[0:2]])  # retrieve indexing from 0 for CG beads IDS for MDAnalysis
					except ValueError:
						msg = (
							"Incorrect reading of the CG ITP file within [constraints] section.\n"
							"Please check this file."
						)
						raise exceptions.MissformattedFile(msg)

					func = verify_handled_functions('constraint', sp_itp_line[2], i+1)
					ns.cg_itp['constraint'][ns.nb_constraints]['func'].append(func)
					ns.cg_itp['constraint'][ns.nb_constraints]['value'].append(float(sp_itp_line[3]))
					ns.cg_itp['constraint'][ns.nb_constraints]['value_user'].append(float(sp_itp_line[3]))

				elif section_read['bond']:

					# beginning of a new group
					if itp_lines[i-1] == '' or itp_lines[i-1].startswith(';') or bool(re.search('\[.*bond.*\]', itp_lines[i-1])):
						ns.nb_bonds += 1
						if itp_lines[i-1].startswith('; bond type'):
							geom_type = itp_lines[i-1].split()[3]  # if the current CG ITP was generated with our package
						else:
							geom_type = str(len(ns.cg_itp['bond'])+1)
						ns.cg_itp['bond'].append({'geom_type': geom_type, 'beads': [], 'func': [], 'value': [], 'value_user': [], 'fct': [], 'fct_user': []})  # initialize storage for this new group

					try:
						ns.cg_itp['bond'][ns.nb_bonds]['beads'].append([int(bead_id)-1 for bead_id in sp_itp_line[0:2]])  # retrieve indexing from 0 for CG beads IDS for MDAnalysis
					except ValueError:
						msg = (
							"Incorrect reading of the CG ITP file within [bonds] section.\n"
							"Please check this file."
						)
						raise exceptions.MissformattedFile(msg)

					func = verify_handled_functions('bond', sp_itp_line[2], i+1)
					ns.cg_itp['bond'][ns.nb_bonds]['func'].append(func)
					ns.cg_itp['bond'][ns.nb_bonds]['value'].append(float(sp_itp_line[3]))
					ns.cg_itp['bond'][ns.nb_bonds]['value_user'].append(float(sp_itp_line[3]))
					ns.cg_itp['bond'][ns.nb_bonds]['fct'].append(float(sp_itp_line[4]))
					ns.cg_itp['bond'][ns.nb_bonds]['fct_user'].append(float(sp_itp_line[4]))

					if ns.user_input and not 0 <= float(sp_itp_line[4]) <= ns.default_max_fct_bonds_opti:
						raise exceptions.MissformattedFile(msg_force_boundaries(i+1, 0, ns.default_max_fct_bonds_opti, '-max_fct_bonds_f1'))

				elif section_read['angle']:

					# beginning of a new group
					if itp_lines[i-1] == '' or itp_lines[i-1].startswith(';') or bool(re.search('\[.*angle.*\]', itp_lines[i-1])):
						ns.nb_angles += 1
						if itp_lines[i-1].startswith('; angle type'):
							geom_type = itp_lines[i-1].split()[3]  # if the current CG ITP was generated with our package
						else:
							geom_type = str(len(ns.cg_itp['angle'])+1)
						ns.cg_itp['angle'].append({'geom_type': geom_type, 'beads': [], 'func': [], 'value': [], 'value_user': [], 'fct': [], 'fct_user': []})  # initialize storage for this new group

					try:
						ns.cg_itp['angle'][ns.nb_angles]['beads'].append([int(bead_id)-1 for bead_id in sp_itp_line[0:3]])  # retrieve indexing from 0 for CG beads IDS for MDAnalysis
					except ValueError:
						msg = (
							"Incorrect reading of the CG ITP file within [angles] section.\n"
							"Please check this file."
						)
						raise exceptions.MissformattedFile(msg)

					func = verify_handled_functions('angle', sp_itp_line[3], i+1)
					ns.cg_itp['angle'][ns.nb_angles]['func'].append(func)
					ns.cg_itp['angle'][ns.nb_angles]['value'].append(float(sp_itp_line[4]))
					ns.cg_itp['angle'][ns.nb_angles]['value_user'].append(float(sp_itp_line[4]))
					ns.cg_itp['angle'][ns.nb_angles]['fct'].append(float(sp_itp_line[5]))
					ns.cg_itp['angle'][ns.nb_angles]['fct_user'].append(float(sp_itp_line[5]))

					if ns.user_input:
						if func == 1 and not 0 <= float(sp_itp_line[5]) <= ns.default_max_fct_angles_opti_f1:
							raise exceptions.MissformattedFile(msg_force_boundaries(i+1, 0, ns.default_max_fct_angles_opti_f1, '-max_fct_angles_f1'))
						elif func == 2 and not 0 <= float(sp_itp_line[5]) <= ns.default_max_fct_angles_opti_f2:
							raise exceptions.MissformattedFile(msg_force_boundaries(i+1, 0, ns.default_max_fct_angles_opti_f2, '-max_fct_angles_f2'))

				elif section_read['dihedral']:

					# beginning of a new group
					if itp_lines[i-1] == '' or itp_lines[i-1].startswith(';') or bool(re.search('\[.*dihedral.*\]', itp_lines[i-1])):
						ns.nb_dihedrals += 1
						if itp_lines[i-1].startswith('; dihedral type'):
							geom_type = itp_lines[i-1].split()[3]  # if the current CG ITP was generated with our package
						else:
							geom_type = str(len(ns.cg_itp['dihedral'])+1)
						ns.cg_itp['dihedral'].append({'geom_type': geom_type, 'beads': [], 'func': [], 'value': [], 'value_user': [], 'fct': [], 'fct_user': [], 'mult': []})  # initialize storage for this new group

					try:
						ns.cg_itp['dihedral'][ns.nb_dihedrals]['beads'].append([int(bead_id)-1 for bead_id in sp_itp_line[0:4]])  # retrieve indexing from 0 for CG beads IDS for MDAnalysis
					except ValueError:
						msg = (
							"Incorrect reading of the CG ITP file within [dihedrals] section.\n"
							"Please check this file."
						)
						raise exceptions.MissformattedFile(msg)

					func = verify_handled_functions('dihedral', sp_itp_line[4], i+1)
					ns.cg_itp['dihedral'][ns.nb_dihedrals]['func'].append(func)
					ns.cg_itp['dihedral'][ns.nb_dihedrals]['value'].append(float(sp_itp_line[5]))  # issue happens here for functions that are not handled
					ns.cg_itp['dihedral'][ns.nb_dihedrals]['value_user'].append(float(sp_itp_line[5]))
					ns.cg_itp['dihedral'][ns.nb_dihedrals]['fct'].append(float(sp_itp_line[6]))
					ns.cg_itp['dihedral'][ns.nb_dihedrals]['fct_user'].append(float(sp_itp_line[6]))

					if ns.user_input:
						if func in config.dihedral_func_with_mult and not -ns.default_abs_range_fct_dihedrals_opti_func_with_mult <= float(sp_itp_line[6]) <= ns.default_abs_range_fct_dihedrals_opti_func_with_mult:
							raise exceptions.MissformattedFile(msg_force_boundaries(i+1, -ns.default_abs_range_fct_dihedrals_opti_func_with_mult, ns.default_abs_range_fct_dihedrals_opti_func_with_mult, '-max_fct_dihedrals_f149'))
						elif func == 2 and not -ns.default_abs_range_fct_dihedrals_opti_func_without_mult <= float(sp_itp_line[6]) <= ns.default_abs_range_fct_dihedrals_opti_func_without_mult:
							raise exceptions.MissformattedFile(msg_force_boundaries(i+1, -ns.default_abs_range_fct_dihedrals_opti_func_without_mult, ns.default_abs_range_fct_dihedrals_opti_func_without_mult, '-max_fct_dihedrals_f2'))

					# handle multiplicity if function assumes multiplicity
					if func in config.dihedral_func_with_mult:
						try:
							ns.cg_itp['dihedral'][ns.nb_dihedrals]['mult'].append(int(sp_itp_line[7]))
						except (IndexError, ValueError):  # incorrect read of multiplicity
							msg = f"Incorrect read of multiplicity in dihedral with potential function {func} at ITP line {i+1}."
							raise exceptions.MissformattedFile(msg)
					else:  # no multiplicity parameter is expected
						ns.cg_itp['dihedral'][ns.nb_dihedrals]['mult'].append(None)

				elif section_read['vs_2']:

					vs_type = 2
					bead_id = int(sp_itp_line[0])-1
					vs_def_beads_ids = [int(bid)-1 for bid in sp_itp_line[1:3]]
					func = sp_itp_line[3]  # will be casted to int in the verification below (for factorizing checks)
					func = vs_error_control(ns, bead_id, vs_type, func, i + 1, vs_def_beads_ids)  # i is the line number
					vs_params = float(sp_itp_line[4])
					ns.cg_itp['atoms'][bead_id]['vs_type'] = vs_type
					ns.cg_itp['virtual_sites2'][bead_id] = {'bead_id': bead_id, 'func': func, 'vs_def_beads_ids': vs_def_beads_ids, 'vs_params': vs_params}

				elif section_read['vs_3']:

					vs_type = 3
					bead_id = int(sp_itp_line[0])-1
					vs_def_beads_ids = [int(bid)-1 for bid in sp_itp_line[1:4]]
					func = sp_itp_line[4]  # will be casted to int in the verification below (for factorizing checks)
					func = vs_error_control(ns, bead_id, vs_type, func, i + 1, vs_def_beads_ids)  # i is the line number
					if func in [1, 2, 3]:
						vs_params = [float(param) for param in sp_itp_line[5:7]]
					elif func == 4:
						vs_params = [float(param) for param in sp_itp_line[5:8]]
					ns.cg_itp['atoms'][bead_id]['vs_type'] = vs_type
					ns.cg_itp['virtual_sites3'][bead_id] = {'bead_id': bead_id, 'func': func, 'vs_def_beads_ids': vs_def_beads_ids, 'vs_params': vs_params}

				elif section_read['vs_4']:

					vs_type = 4
					bead_id = int(sp_itp_line[0]) - 1
					vs_def_beads_ids = [int(bid) - 1 for bid in sp_itp_line[1:5]]
					func = sp_itp_line[5]  # will be casted to int in the verification below (for factorizing checks)
					func = vs_error_control(ns, bead_id, vs_type, func, i + 1, vs_def_beads_ids)  # i is the line number
					vs_params = [float(param) for param in sp_itp_line[6:9]]
					ns.cg_itp['atoms'][bead_id]['vs_type'] = vs_type
					ns.cg_itp['virtual_sites4'][bead_id] = {'bead_id': bead_id, 'func': func, 'vs_def_beads_ids': vs_def_beads_ids, 'vs_params': vs_params}

				elif section_read['vs_n']:

					vs_type = 'n'
					bead_id = int(sp_itp_line[0])-1
					func = sp_itp_line[1]  # will be casted to int in verification below (for factorizing checks)
					# here we do the check in 2 steps, because the reading of beads_ids depends on the function
					func = vs_error_control(ns, bead_id, vs_type, func, i + 1, vs_def_beads_ids=None)  # i is the line number
					if func == 3:
						vs_def_beads_ids = [int(sp_itp_line[2:][i])-1 for i in range(0, len(sp_itp_line[2:]), 2)]
						vs_params = [float(sp_itp_line[2:][i]) for i in range(1, len(sp_itp_line[2:]), 2)]
					else:
						vs_def_beads_ids = [int(bid) - 1 for bid in sp_itp_line[2:]]
						vs_params = None
					func = vs_error_control(ns, bead_id, vs_type, func, i + 1, vs_def_beads_ids)  # i is the line number
					ns.cg_itp['atoms'][bead_id]['vs_type'] = vs_type
					ns.cg_itp['virtual_sitesn'][bead_id] = {'bead_id': bead_id, 'func': func, 'vs_def_beads_ids': vs_def_beads_ids, 'vs_params': vs_params}

				elif section_read['exclusion']:

					ns.cg_itp['exclusion'].append([int(bead_id)-1 for bead_id in sp_itp_line])

	# error handling, verify that funct, value and fct are all identical within the group, as they should be, and reduce arrays to single elements
	# TODO: make these messages more clear and CORRECT for the dihedral function handling -- also explain this is the current Opti.CG implementation, function 9 might come in next version
	# TODO: check what kind of error or processing is done when a correct line is duplicated within a group ?? probably it goes on in a bad way

	def msg(geom, grp_geom):
		str_msg = (
			f"In the provided CG ITP file {geom}s have been grouped, but {geom}s group "
			f"{str(grp_geom + 1)} holds lines that have different parameters.\nParameters should be "
			f"identical within a group, only CG beads IDs should differ.\n"
			f"Please correct the CG ITP file and separate groups using a blank or commented line."
		)
		return str_msg

	for geom in ['constraint']:  # constraints only
		for grp_geom in range(len(ns.cg_itp[geom])):
			for var in ['func', 'value', 'value_user']:
				var_set = set(ns.cg_itp[geom][grp_geom][var])
				if len(var_set) == 1:
					ns.cg_itp[geom][grp_geom][var] = var_set.pop()
				else:
					raise exceptions.MissformattedFile(msg(geom, grp_geom))

	for geom in ['bond', 'angle']:  # bonds and angles only
		for grp_geom in range(len(ns.cg_itp[geom])):
			for var in ['func', 'value', 'value_user', 'fct', 'fct_user']:
				var_set = set(ns.cg_itp[geom][grp_geom][var])
				if len(var_set) == 1:
					ns.cg_itp[geom][grp_geom][var] = var_set.pop()
				else:
					raise exceptions.MissformattedFile(msg(geom, grp_geom))

	for geom in ['dihedral']:  # dihedrals only
		for grp_geom in range(len(ns.cg_itp[geom])):
			for var in ['func', 'value', 'value_user', 'fct', 'fct_user']:
				var_set = set(ns.cg_itp[geom][grp_geom][var])
				if len(var_set) == 1:
					ns.cg_itp[geom][grp_geom][var] = var_set.pop()
				else:
					raise exceptions.MissformattedFile(msg(geom, grp_geom))

			for var in ['mult']:
				var_set = set(ns.cg_itp[geom][grp_geom][var])
				if len(var_set) == 1:
					ns.cg_itp[geom][grp_geom][var] = var_set.pop()
				else:
					raise exceptions.MissformattedFile(msg(geom, grp_geom))

	# verify we have as many real CG beads (i.e. NOT virtual sites) in the ITP than in the mapping file
	if len(ns.real_beads_ids) != len(ns.all_beads):
		msg = (
			"The CG beads mapping (NDX) file does NOT include as many CG beads as the ITP file.\n"
			"Please check the NDX and ITP files you provided."
		)
		raise exceptions.MissformattedFile(msg)

	ns.nb_constraints += 1
	ns.nb_bonds += 1
	ns.nb_angles += 1
	ns.nb_dihedrals += 1
	print(f'  Found {len(ns.real_beads_ids)} beads')
	print(f'  Found {len(ns.vs_beads_ids)} virtual sites')
	print(f'  Found {ns.nb_constraints} constraints groups')
	print(f'  Found {ns.nb_bonds} bonds groups')
	print(f'  Found {ns.nb_angles} angles groups')
	print(f'  Found {ns.nb_dihedrals} dihedrals groups')


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


# compute average radius of gyration
def compute_Rg(ns, traj_type):

	if traj_type == 'AA':

		gyr_aa = np.empty(len(ns.aa_universe.trajectory))
		for ts in ns.aa_universe.trajectory:
			gyr_aa[ts.frame] = ns.aa_universe.atoms[:len(ns.all_atoms)].radius_of_gyration(pbc=None, backend=ns.mda_backend)
		ns.gyr_aa = round(np.average(gyr_aa) / 10, 3)  # retrieve nm
		ns.gyr_aa_std = round(np.std(gyr_aa) / 10, 3)  # retrieve nm

	elif traj_type == 'AA_mapped':

		gyr_aa_mapped = np.empty(len(ns.aa_universe.trajectory))
		for ts in ns.aa2cg_universe.trajectory:
			gyr_aa_mapped[ts.frame] = ns.aa2cg_universe.atoms[:len(ns.cg_itp['atoms'])].radius_of_gyration(pbc=None, backend=ns.mda_backend)
		ns.gyr_aa_mapped = round(np.average(gyr_aa_mapped) / 10 + ns.aa_rg_offset, 3)  # retrieve nm
		ns.gyr_aa_mapped_std = round(np.std(gyr_aa_mapped) / 10, 3)  # retrieve nm

	elif traj_type == 'CG':

		gyr_cg = np.empty(len(ns.cg_universe.trajectory))
		for ts in ns.cg_universe.trajectory:
			gyr_cg[ts.frame] = ns.cg_universe.atoms[:len(ns.cg_itp['atoms'])].radius_of_gyration(pbc=None, backend=ns.mda_backend)
		ns.gyr_cg = round(np.average(gyr_cg) / 10, 3)  # retrieve nm
		ns.gyr_cg_std = round(np.std(gyr_cg) / 10, 3)  # retrieve nm

	else:
		raise RuntimeError('Unexpected error in function: compute_Rg')


# read 1 column of xvg file and return as array
# column is 0-indexed
def read_xvg_col(xvg_file, col):
	with open(xvg_file, 'r') as fp:
		lines = [line.strip() for line in fp.read().split('\n')]
		data = []
		for line in lines:
			if not line.startswith(('#', '@')) and line != '':
				sp_lines = list(map(float, line.split()))
				data.append(sp_lines[col])
	return data


# execute gmx cmd and return only exit code
def exec_gmx(gmx_cmd):
	with subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as gmx_process:
		gmx_out = gmx_process.communicate()[1].decode()
		gmx_process.kill()
	if gmx_process.returncode != 0:
		print_stdout_forced('NON-ZERO EXIT CODE FOR COMMAND:', gmx_cmd, '\n\nCOMMAND OUTPUT:\n\n', gmx_out, '\n\n')
	return gmx_process.returncode


# compute average SASA
# NOTE: currently this is just COM mappping via GMX to get the SASA, so it's approximative but that's OK
#       this works with calls to GMX because only library MDTraj can compute SASA (not MDAnalysis)
# TODO: MDA is working on it, keep an eye on this: https://github.com/MDAnalysis/mdanalysis/issues/2439
def compute_SASA(ns, traj_type):

	probe_radius = 0.26  # nm

	if traj_type == 'AA':
		raise exceptions.InvalidArgument('Compute_SASA not implemented for AA atm')

	elif traj_type == 'AA_mapped':

		# NOTE: here we assume the VS all come after the real beads in the ITP [ atoms ] field
		#       we generate a new truncated TPR so that we can use GMX sasa, this is shit but no choice atm
		nb_beads_real = len(ns.real_beads_ids)

		# generate an cg_map.ndx file with the number of beads,
		# so we can call SASA on this group and we will have exactly the content we want
		ns.cg_ndx_filename = '../'+config.input_sim_files_dirname+'/cg_index.ndx'
		with open(ns.cg_ndx_filename, 'w') as fp:
			beads_ids_str = ' '.join(map(str, list(range(1, nb_beads_real+1))))  # includes VS if present
			fp.write('[' + ns.cg_itp['moleculetype']['molname'] + ' ]\n' + beads_ids_str + '\n')

		# TODO: all these paths need to be fixed to allow for SASA calculation within evaluate_model.py
		#       that's why it's disabled atm

		ns.aa_traj_whole_filename = '../'+config.input_sim_files_dirname+'/aa_traj_whole.xtc'
		ns.aa_mapped_traj_whole_filename = '../'+config.input_sim_files_dirname+'/aa_mapped_traj_whole.xtc'
		ns.aa_mapped_sasa_filename = '../'+config.input_sim_files_dirname+'/aa_mapped_sasa.xvg'
		ns.aa_mapped_tpr_sasa_filename = '../'+config.input_sim_files_dirname+'/aa_mapped_tpr_sasa.tpr'

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
			gmx_cmd = f'{ns.gmx_path} sasa -s {ns.aa_mapped_tpr_sasa_filename} -f {ns.aa_mapped_traj_whole_filename} -n {ns.cg_ndx_filename} -surface 0 -o {ns.aa_mapped_sasa_filename} -probe {probe_radius}'
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
			gmx_cmd = f'{ns.gmx_path} sasa -s {ns.cg_tpr_filename} -f {ns.cg_traj_whole_filename} -n {ns.cg_ndx_filename} -surface 0 -o {ns.cg_sasa_filename} -probe {probe_radius}'
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


# print coarse-grain ITP
# here we have a switch for print_sections because we might want to optimize constraints/bonds/angles/dihedrals
# separately, so we can leave some out with the switch and they will be optimized later
def write_cg_itp_file(itp_obj, out_path_itp, print_sections=['constraint', 'bond', 'angle', 'dihedral', 'exclusion']):

	with open(out_path_itp, 'w') as fp:

		fp.write('[ moleculetype ]\n')
		fp.write('; molname        nrexcl\n')
		fp.write('{0:<4} {1:>13}\n'.format(itp_obj['moleculetype']['molname'], itp_obj['moleculetype']['nrexcl']))

		fp.write('\n\n[ atoms ]\n')
		fp.write('; id type resnr residue   atom  cgnr    charge     mass\n\n')

		for i in range(len(itp_obj['atoms'])):
			# if the ITP did NOT contain masses, they are set at 0 in this field during ITP reading
			if itp_obj['atoms'][i]['mass'] is not None:
				fp.write('{0:<4} {1:>4}    {6:>2}  {2:>6} {3:>6}  {4:<4} {5:9.5f}     {7:<5.2f}\n'.format(
					itp_obj['atoms'][i]['bead_id']+1, itp_obj['atoms'][i]['bead_type'],
					itp_obj['atoms'][i]['residue'], itp_obj['atoms'][i]['atom'], i+1, itp_obj['atoms'][i]['charge'],
					itp_obj['atoms'][i]['resnr'], itp_obj['atoms'][i]['mass']))
			else:
				fp.write('{0:<4} {1:>4}    {6:>2}  {2:>6} {3:>6}  {4:<4} {5:9.5f}\n'.format(
					itp_obj['atoms'][i]['bead_id'] + 1, itp_obj['atoms'][i]['bead_type'],
					itp_obj['atoms'][i]['residue'], itp_obj['atoms'][i]['atom'], i + 1, itp_obj['atoms'][i]['charge'],
					itp_obj['atoms'][i]['resnr']))

		if 'constraint' in print_sections and 'constraint' in itp_obj and len(itp_obj['constraint']) > 0:
			fp.write('\n\n[ constraints ]\n')
			fp.write(';   i     j   funct   length\n')

			for j in range(len(itp_obj['constraint'])):

				constraint_type = itp_obj['constraint'][j]['geom_type']
				fp.write('\n; constraint type '+constraint_type+'\n')
				grp_val = itp_obj['constraint'][j]['value']

				for i in range(len(itp_obj['constraint'][j]['beads'])):
					fp.write('{beads[0]:>5} {beads[1]:>5} {0:>7} {1:8.3f}      ; {2}\n'.format(
						itp_obj['constraint'][j]['func'], grp_val, constraint_type,
						beads=[bead_id+1 for bead_id in itp_obj['constraint'][j]['beads'][i]]))

		if 'bond' in print_sections and 'bond' in itp_obj and len(itp_obj['bond']) > 0:
			fp.write('\n\n[ bonds ]\n')
			fp.write(';   i     j   funct   length   force.c.\n')

			for j in range(len(itp_obj['bond'])):

				bond_type = itp_obj['bond'][j]['geom_type']
				fp.write('\n; bond type '+bond_type+'\n')
				grp_val, grp_fct = itp_obj['bond'][j]['value'], itp_obj['bond'][j]['fct']

				for i in range(len(itp_obj['bond'][j]['beads'])):
					fp.write('{beads[0]:>5} {beads[1]:>5} {0:>7} {1:8.3f}  {2:7.2f}           ; {3}\n'.format(
						itp_obj['bond'][j]['func'], grp_val, grp_fct, bond_type,
						beads=[bead_id+1 for bead_id in itp_obj['bond'][j]['beads'][i]]))

		if 'angle' in print_sections and 'angle' in itp_obj and len(itp_obj['angle']) > 0:
			fp.write('\n\n[ angles ]\n')
			fp.write(';   i     j     k   funct     angle   force.c.\n')

			for j in range(len(itp_obj['angle'])):

				angle_type = itp_obj['angle'][j]['geom_type']
				fp.write('\n; angle type '+angle_type+'\n')
				grp_val, grp_fct = itp_obj['angle'][j]['value'], itp_obj['angle'][j]['fct']

				for i in range(len(itp_obj['angle'][j]['beads'])):
					fp.write('{beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {0:>7} {1:9.2f}   {2:7.2f}           ; {3}\n'.format(
						itp_obj['angle'][j]['func'], grp_val, grp_fct, angle_type,
						beads=[bead_id+1 for bead_id in itp_obj['angle'][j]['beads'][i]]))

		if 'dihedral' in print_sections and 'dihedral' in itp_obj and len(itp_obj['dihedral']) > 0:
			fp.write('\n\n[ dihedrals ]\n')
			fp.write(';   i     j     k     l   funct     dihedral   force.c.   mult.\n')

			for j in range(len(itp_obj['dihedral'])):

				dihedral_type = itp_obj['dihedral'][j]['geom_type']
				fp.write('\n; dihedral type '+dihedral_type+'\n')
				grp_val, grp_fct = itp_obj['dihedral'][j]['value'], itp_obj['dihedral'][j]['fct']

				for i in range(len(itp_obj['dihedral'][j]['beads'])):

					# handle writing of multiplicity
					multiplicity = itp_obj['dihedral'][j]['mult']
					if multiplicity == None:
						multiplicity = ''

					fp.write('{beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {beads[3]:>5} {0:>7}    {1:9.2f} {2:7.2f}       {4}     ; {3}\n'.format(
						itp_obj['dihedral'][j]['func'], grp_val, grp_fct, dihedral_type, multiplicity,
						beads=[bead_id+1 for bead_id in itp_obj['dihedral'][j]['beads'][i]]))

		# here starts 4 almost identical blocks, that differ only by vs_2, vs_3, vs_4, vs_n
		# but we could still need to write several of these sections (careful if factorizing this)
		if len(itp_obj['virtual_sites2']) > 0:
			fp.write('\n\n[ virtual_sites2 ]\n')
			fp.write(';  vs     i     j  func   param\n')
			for bead_id in itp_obj['virtual_sites2']:
				fp.write('{0:>5} {beads[0]:>5} {beads[1]:>5} {1:>5}   {2}\n'.format(
					str(itp_obj['virtual_sites2'][bead_id]['bead_id'] + 1),
					str(itp_obj['virtual_sites2'][bead_id]['func']),
					itp_obj['virtual_sites2'][bead_id]['vs_params'],
					beads=[bid+1 for bid in itp_obj['virtual_sites2'][bead_id]['vs_def_beads_ids']])
				)

		if len(itp_obj['virtual_sites3']) > 0:
			fp.write('\n\n[ virtual_sites3 ]\n')
			fp.write(';  vs     i     j     k  func   params\n')
			for bead_id in itp_obj['virtual_sites3']:
				fp.write('{0:>5} {beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {1:>5}   {2}\n'.format(
					str(itp_obj['virtual_sites3'][bead_id]['bead_id'] + 1),
					str(itp_obj['virtual_sites3'][bead_id]['func']),
					'  '.join(map(str, itp_obj['virtual_sites3'][bead_id]['vs_params'])),
					beads=[bid+1 for bid in itp_obj['virtual_sites3'][bead_id]['vs_def_beads_ids']])
				)

		if len(itp_obj['virtual_sites4']) > 0:
			fp.write('\n\n[ virtual_sites4 ]\n')
			fp.write(';  vs     i     j     k     l  func   params\n')
			for bead_id in itp_obj['virtual_sites4']:
				fp.write('{0:>5} {beads[0]:>5} {beads[1]:>5} {beads[2]:>5} {beads[3]:>5} {1:>5}   {2}\n'.format(
					str(itp_obj['virtual_sites4'][bead_id]['bead_id'] + 1),
					str(itp_obj['virtual_sites4'][bead_id]['func']),
					'  '.join(map(str, itp_obj['virtual_sites4'][bead_id]['vs_params'])),
					beads=[bid+1 for bid in itp_obj['virtual_sites4'][bead_id]['vs_def_beads_ids']])
				)

		if len(itp_obj['virtual_sitesn']) > 0:
			fp.write('\n\n[ virtual_sitesn ]\n')
			fp.write(';  vs  func     def\n')
			for bead_id in itp_obj['virtual_sitesn']:
				params = []
				if itp_obj['virtual_sitesn'][bead_id]['func'] == 3:
					for i in range(len(itp_obj['virtual_sitesn'][bead_id]['vs_def_beads_ids'])):
						params.append('{} {}'.format(itp_obj['virtual_sitesn'][bead_id]['vs_def_beads_ids'][i]+1, itp_obj['virtual_sitesn'][bead_id]['vs_params'][i]))
					params = '  '.join(params)
				else:
					params = ' '.join(['{:>4}'.format(bid + 1) for bid in itp_obj['virtual_sitesn'][bead_id]['vs_def_beads_ids']])
				fp.write('{:>5} {:>5}   {}\n'.format(
					itp_obj['virtual_sitesn'][bead_id]['bead_id'] + 1,
					itp_obj['virtual_sitesn'][bead_id]['func'],
					params)
				)

		if 'exclusion' in print_sections and 'exclusion' in itp_obj and len(itp_obj['exclusion']) > 0:
			fp.write('\n\n[ exclusions ]\n')
			fp.write(';   i     j\n\n')

			for j in range(len(itp_obj['exclusion'])):
				fp.write(('{:>4} '*len(itp_obj['exclusion'][j])+'\n').format(*[bead_id+1 for bead_id in itp_obj['exclusion'][j]]))

		fp.write('\n\n')


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


# read atomistic trajectory
def read_aa_traj(ns):

	print('Reading All Atom (AA) trajectory')
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=ImportWarning) # ignore warning: "bootstrap.py:219: ImportWarning: can't resolve package from __spec__ or __package__, falling back on __name__ and __path__"
		ns.aa_universe = mda.Universe(ns.aa_tpr_filename, ns.aa_traj_filename, in_memory=True, refresh_offsets=True, guess_bonds=False)  # setting guess_bonds=False disables angles, dihedrals and improper_dihedrals guessing, which is activated by default in some MDA versions
	print('  Found', len(ns.aa_universe.trajectory), 'frames')


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


# build gromacs command with arguments
def gmx_args(ns, gmx_cmd, mpi=True):

	gmx_cmd = f"{ns.gmx_path} {gmx_cmd}"
	if ns.gmx_args_str != '':
		gmx_cmd = f"{gmx_cmd} {ns.gmx_args_str}"
	else:
		if ns.nb_threads > 0:
			gmx_cmd = f"{gmx_cmd} -nt {ns.nb_threads}"
		if len(ns.gpu_id) > 0:
			gmx_cmd = f"{gmx_cmd} -gpu_id {ns.gpu_id}"
	if mpi and ns.mpi_tasks > 1:
		gmx_cmd = f"mpirun -np {ns.mpi_tasks} {ns.gmx_cmd}"

	return gmx_cmd


# get bins and distance matrix for pairwise distributions comparison using Earth Mover's Distance (EMD)
def create_bins_and_dist_matrices(ns, constraints=True):

	# bins for histogram distributions of bonds/angles
	if constraints:
		ns.bins_constraints = np.arange(0, ns.bonded_max_range+ns.bw_constraints, ns.bw_constraints)
	ns.bins_bonds = np.arange(0, ns.bonded_max_range+ns.bw_bonds, ns.bw_bonds)
	ns.bins_angles = np.arange(0, 180+2*ns.bw_angles, ns.bw_angles)  # one more bin for angle/dihedral because we are later using a strict inferior for bins definitions
	ns.bins_dihedrals = np.arange(-180, 180+2*ns.bw_dihedrals, ns.bw_dihedrals)

	# bins distance for Earth Mover's Distance (EMD) to calculate histograms similarity
	if constraints:
		bins_constraints_reshape = np.array(ns.bins_constraints).reshape(-1,1)
		ns.bins_constraints_dist_matrix = cdist(bins_constraints_reshape, bins_constraints_reshape)
	bins_bonds_reshape = np.array(ns.bins_bonds).reshape(-1,1)
	ns.bins_bonds_dist_matrix = cdist(bins_bonds_reshape, bins_bonds_reshape)
	bins_angles_reshape = np.array(ns.bins_angles).reshape(-1,1)
	ns.bins_angles_dist_matrix = cdist(bins_angles_reshape, bins_angles_reshape)
	bins_dihedrals_reshape = np.array(ns.bins_dihedrals).reshape(-1,1)
	bins_dihedrals_dist_matrix = cdist(bins_dihedrals_reshape, bins_dihedrals_reshape)  # 'classical' distance matrix
	ns.bins_dihedrals_dist_matrix = np.where(bins_dihedrals_dist_matrix > max(bins_dihedrals_dist_matrix[0])/2, max(bins_dihedrals_dist_matrix[0])-bins_dihedrals_dist_matrix, bins_dihedrals_dist_matrix) # periodic distance matrix


# calculate bonds distribution from AA trajectory
def get_AA_bonds_distrib(ns, beads_ids, grp_type, grp_nb):

	bond_values = np.empty(len(ns.aa2cg_universe.trajectory) * len(beads_ids))
	frame_values = np.empty(len(beads_ids))
	bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)

	for ts in ns.aa2cg_universe.trajectory:
		for i in range(len(beads_ids)):

			bead_id_1, bead_id_2 = beads_ids[i]
			bead_pos_1[i] = ns.aa2cg_universe.atoms[bead_id_1].position
			bead_pos_2[i] = ns.aa2cg_universe.atoms[bead_id_2].position

		mda.lib.distances.calc_bonds(bead_pos_1, bead_pos_2, backend=ns.mda_backend, box=None, result=frame_values)
		bond_values[len(beads_ids)*ts.frame:len(beads_ids)*(ts.frame+1)] = frame_values / 10  # retrieved nm

	bond_avg_init = round(np.average(bond_values), 3)

	# NOTE: for rescaling we first take the average of the group, then we rescale
	#       this means if a bond group has a bimodal distribution, the rescale distribution is still bimodal

	# rescale all bonds length if argument -bonds_scaling is provided
	if ns.bonds_scaling != config.bonds_scaling:
		bond_values = [bond_length * ns.bonds_scaling for bond_length in bond_values]
		bond_avg_final = round(np.average(bond_values), 3)
		ns.bonds_rescaling_performed = True
		print('  Ref. AA-mapped distrib. rescaled to avg', bond_avg_final, 'nm for', grp_type, grp_nb+1, '(initially', bond_avg_init, 'nm)')

	# or shift distributions for bonds that are too short for direct CG mapping (according to argument -min_bonds_length) 
	elif bond_avg_init < ns.min_bonds_length:
		bond_rescale_factor = ns.min_bonds_length / bond_avg_init
		bond_values = [bond_length * bond_rescale_factor for bond_length in bond_values]
		bond_avg_final = round(np.average(bond_values), 3)
		ns.bonds_rescaling_performed = True
		print('  Ref. AA-mapped distrib. rescaled to avg', bond_avg_final, 'nm for', grp_type, grp_nb+1, '(initially', bond_avg_init, 'nm)')

	# or if specific lengths were provided for constraints and/or bonds
	elif ns.bonds_scaling_specific is not None:

		if grp_type.startswith('constraint'):
			geom_id_full = f'C{grp_nb+1}'
		if grp_type.startswith('bond'):
			geom_id_full = f'B{grp_nb+1}'

		if (geom_id_full.startswith('C') and geom_id_full in ns.bonds_scaling_specific) or (geom_id_full.startswith('B') and geom_id_full in ns.bonds_scaling_specific):
			bond_rescale_factor = ns.bonds_scaling_specific[geom_id_full] / bond_avg_init
			bond_values = [bond_length * bond_rescale_factor for bond_length in bond_values]
			bond_avg_final = round(np.average(bond_values), 3)
			ns.bonds_rescaling_performed = True
			print('  Ref. AA-mapped distrib. rescaled to avg', bond_avg_final, 'nm for', grp_type, grp_nb+1, '(initially', bond_avg_init, 'nm)')
		else:
			bond_avg_final = bond_avg_init

	else:
		bond_avg_final = bond_avg_init

	# or alternatively, do not rescale these bonds but add specific exclusion rules, OR JUST SUGGEST USER TO CHECK THIS
	# exclusions storage format: ns.cg_itp['exclusion'].append([int(bead_id)-1 for bead_id in sp_itp_line[0:2]])

	if grp_type.startswith('constraint'):
		bond_hist = np.histogram(bond_values, ns.bins_constraints, density=True)[0]*ns.bw_constraints  # retrieve 1-sum densities
	if grp_type.startswith('bond'):
		bond_hist = np.histogram(bond_values, ns.bins_bonds, density=True)[0]*ns.bw_bonds  # retrieve 1-sum densities

	return bond_avg_final, bond_hist, bond_values


# calculate angles distribution from AA trajectory
def get_AA_angles_distrib(ns, beads_ids):

	angle_values_rad = np.empty(len(ns.aa2cg_universe.trajectory) * len(beads_ids))
	frame_values = np.empty(len(beads_ids))
	bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)

	for ts in ns.aa2cg_universe.trajectory:
		for i in range(len(beads_ids)):

			bead_id_1, bead_id_2, bead_id_3 = beads_ids[i]
			bead_pos_1[i] = ns.aa2cg_universe.atoms[bead_id_1].position
			bead_pos_2[i] = ns.aa2cg_universe.atoms[bead_id_2].position
			bead_pos_3[i] = ns.aa2cg_universe.atoms[bead_id_3].position

		mda.lib.distances.calc_angles(bead_pos_1, bead_pos_2, bead_pos_3, backend=ns.mda_backend, box=None, result=frame_values)
		angle_values_rad[len(beads_ids)*ts.frame:len(beads_ids)*(ts.frame+1)] = frame_values

	angle_values_deg = np.rad2deg(angle_values_rad)
	angle_avg = round(np.mean(angle_values_deg), 3)
	angle_hist = np.histogram(angle_values_deg, ns.bins_angles, density=True)[0]*ns.bw_angles  # retrieve 1-sum densities

	return angle_avg, angle_hist, angle_values_deg, angle_values_rad


# calculate dihedrals distribution from AA trajectory
def get_AA_dihedrals_distrib(ns, beads_ids):

	dihedral_values_rad = np.empty(len(ns.aa2cg_universe.trajectory) * len(beads_ids))
	frame_values = np.empty(len(beads_ids))
	bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_4 = np.empty((len(beads_ids), 3), dtype=np.float32)

	for ts in ns.aa2cg_universe.trajectory:
		for i in range(len(beads_ids)):

			bead_id_1, bead_id_2, bead_id_3, bead_id_4 = beads_ids[i]
			bead_pos_1[i] = ns.aa2cg_universe.atoms[bead_id_1].position
			bead_pos_2[i] = ns.aa2cg_universe.atoms[bead_id_2].position
			bead_pos_3[i] = ns.aa2cg_universe.atoms[bead_id_3].position
			bead_pos_4[i] = ns.aa2cg_universe.atoms[bead_id_4].position

		mda.lib.distances.calc_dihedrals(bead_pos_1, bead_pos_2, bead_pos_3, bead_pos_4, backend=ns.mda_backend, box=None, result=frame_values)
		dihedral_values_rad[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

	dihedral_values_deg = np.rad2deg(dihedral_values_rad)
	dihedral_avg = round(np.mean(dihedral_values_deg), 3)
	dihedral_hist = np.histogram(dihedral_values_deg, ns.bins_dihedrals, density=True)[0]*ns.bw_dihedrals  # retrieve 1-sum densities

	return dihedral_avg, dihedral_hist, dihedral_values_deg, dihedral_values_rad


# calculate bonds distribution from CG trajectory
def get_CG_bonds_distrib(ns, beads_ids, grp_type):

	bond_values = np.empty(len(ns.cg_universe.trajectory) * len(beads_ids))
	frame_values = np.empty(len(beads_ids))
	bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)

	for ts in ns.cg_universe.trajectory:  # no need for PBC handling, trajectories were made wholes for the molecule
		for i in range(len(beads_ids)):

			bead_id_1, bead_id_2 = beads_ids[i]
			bead_pos_1[i] = ns.cg_universe.atoms[bead_id_1].position
			bead_pos_2[i] = ns.cg_universe.atoms[bead_id_2].position

		mda.lib.distances.calc_bonds(bead_pos_1, bead_pos_2, backend=ns.mda_backend, box=None, result=frame_values)
		bond_values[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values / 10  # retrieved nm

	bond_avg = round(np.mean(bond_values), 3)
	if grp_type == 'constraint':
		bond_hist = np.histogram(bond_values, ns.bins_constraints, density=True)[0]*ns.bw_constraints  # retrieve 1-sum densities
	if grp_type == 'bond':
		bond_hist = np.histogram(bond_values, ns.bins_bonds, density=True)[0]*ns.bw_bonds  # retrieve 1-sum densities

	return bond_avg, bond_hist, bond_values


# calculate angles using MDAnalysis
def get_CG_angles_distrib(ns, beads_ids):

	angle_values_rad = np.empty(len(ns.cg_universe.trajectory) * len(beads_ids))
	frame_values = np.empty(len(beads_ids))
	bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)

	for ts in ns.cg_universe.trajectory:  # no need for PBC handling, trajectories were made wholes for the molecule
		for i in range(len(beads_ids)):

			bead_id_1, bead_id_2, bead_id_3 = beads_ids[i]
			bead_pos_1[i] = ns.cg_universe.atoms[bead_id_1].position
			bead_pos_2[i] = ns.cg_universe.atoms[bead_id_2].position
			bead_pos_3[i] = ns.cg_universe.atoms[bead_id_3].position

		mda.lib.distances.calc_angles(bead_pos_1, bead_pos_2, bead_pos_3, backend=ns.mda_backend, box=None, result=frame_values)
		angle_values_rad[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

	angle_values_deg = np.rad2deg(angle_values_rad)

	# get group average and histogram non-null values for comparison and display
	angle_avg = round(np.mean(angle_values_deg), 3)
	angle_hist = np.histogram(angle_values_deg, ns.bins_angles, density=True)[0]*ns.bw_angles  # retrieve 1-sum densities

	return angle_avg, angle_hist, angle_values_deg, angle_values_rad


# calculate dihedrals using MDAnalysis
def get_CG_dihedrals_distrib(ns, beads_ids):

	dihedral_values_rad = np.empty(len(ns.cg_universe.trajectory) * len(beads_ids))
	frame_values = np.empty(len(beads_ids))
	bead_pos_1 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_2 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_3 = np.empty((len(beads_ids), 3), dtype=np.float32)
	bead_pos_4 = np.empty((len(beads_ids), 3), dtype=np.float32)

	for ts in ns.cg_universe.trajectory:  # no need for PBC handling, trajectories were made wholes for the molecule
		for i in range(len(beads_ids)):

			bead_id_1, bead_id_2, bead_id_3, bead_id_4 = beads_ids[i]
			bead_pos_1[i] = ns.cg_universe.atoms[bead_id_1].position
			bead_pos_2[i] = ns.cg_universe.atoms[bead_id_2].position
			bead_pos_3[i] = ns.cg_universe.atoms[bead_id_3].position
			bead_pos_4[i] = ns.cg_universe.atoms[bead_id_4].position

		mda.lib.distances.calc_dihedrals(bead_pos_1, bead_pos_2, bead_pos_3, bead_pos_4, backend=ns.mda_backend, box=None, result=frame_values)
		dihedral_values_rad[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values

	dihedral_values_deg = np.rad2deg(dihedral_values_rad)

	# get group average and histogram non-null values for comparison and display
	dihedral_avg = round(np.mean(dihedral_values_deg), 3)
	dihedral_hist = np.histogram(dihedral_values_deg, ns.bins_dihedrals, density=True)[0]*ns.bw_dihedrals  # retrieve 1-sum densities

	return dihedral_avg, dihedral_hist, dihedral_values_deg, dihedral_values_rad


# update ITP force constants with Boltzmann inversion for selected geoms at this given optimization step
def perform_BI(ns):
	
	# NOTE: currently all of these are just BI, not BI to completion using only required ADDITIONAL amount of energy, which might make a difference when we perform the BI after bonds+angles optimization cycles
	# TODO: refactorize BI in separate function to be used during both model_prep and at start of model_opti
	# TODO: other dihedrals functions
	# TODO: If the first opti run of BI fails, lower force constants by 10% and retry, again and again until it works, or tell the user something is very wrong after 20 tries with 50% of the force constants that all did NOT work 

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore the warning "divide by 0 encountered in true_divide" while calculating sigma

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
					hist_geoms_modif = utils.ewma(hist_geoms_modif, alpha, int(config.bi_nb_bins/10))

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
					deriv = utils.sma(deriv, int(config.bi_nb_bins/5))

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
					if int(geom_id) > ns.nb_constraints:
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
					if int(geom_id) > ns.nb_bonds:
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
		compute_Rg(ns, traj_type='AA')
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
		if len(ns.vs_beads_ids) > 0:
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
			compute_Rg(ns, traj_type='AA_mapped')
			print()
			print('Radius of gyration (AA reference, CG-mapped, no bonds scaling):', ns.gyr_aa_mapped, '+/-', ns.gyr_aa_mapped_std, 'nm')

		compute_Rg(ns, traj_type='CG')
		print('Radius of gyration (CG model):', ns.gyr_cg, '+/-', ns.gyr_cg_std, 'nm')

		if calc_sasa:

			if ns.sasa_aa_mapped == None:
				compute_SASA(ns, traj_type='AA_mapped')

			compute_SASA(ns, traj_type='CG')
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
	diff_ordered_grp_constraints = list(range(ns.nb_constraints))
	avg_diff_grp_constraints, row_wise_ranges['constraints'] = [], {}
	constraints = {}

	for grp_constraint in range(ns.nb_constraints):

		constraints[grp_constraint] = {'AA': {'x': [], 'y': []}, 'CG': {'x': [], 'y': []}}

		if manual_mode:
			constraint_avg, constraint_hist, _ = get_AA_bonds_distrib(ns, beads_ids=ns.cg_itp['constraint'][grp_constraint]['beads'], grp_type='constraints group', grp_nb=grp_constraint)
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
				constraint_avg, constraint_hist, _ = get_CG_bonds_distrib(ns, beads_ids=ns.cg_itp['constraint'][grp_constraint]['beads'], grp_type='constraint')
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
	diff_ordered_grp_bonds = list(range(ns.nb_bonds))
	avg_diff_grp_bonds, row_wise_ranges['bonds'] = [], {}
	bonds = {}

	for grp_bond in range(ns.nb_bonds):

		bonds[grp_bond] = {'AA': {'x': [], 'y': []}, 'CG': {'x': [], 'y': []}}

		if manual_mode:
			bond_avg, bond_hist, _ = get_AA_bonds_distrib(ns, beads_ids=ns.cg_itp['bond'][grp_bond]['beads'], grp_type='bonds group', grp_nb=grp_bond)
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
				bond_avg, bond_hist, _ = get_CG_bonds_distrib(ns, beads_ids=ns.cg_itp['bond'][grp_bond]['beads'], grp_type='bond')
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
	diff_ordered_grp_angles = list(range(ns.nb_angles))
	avg_diff_grp_angles, row_wise_ranges['angles'] = [], {}
	angles = {}

	for grp_angle in range(ns.nb_angles):

		angles[grp_angle] = {'AA': {'x': [], 'y': []}, 'CG': {'x': [], 'y': []}}

		if manual_mode:
			angle_avg, angle_hist, _, _ = get_AA_angles_distrib(ns, beads_ids=ns.cg_itp['angle'][grp_angle]['beads'])
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
			angle_avg, angle_hist, _, _ = get_CG_angles_distrib(ns, beads_ids=ns.cg_itp['angle'][grp_angle]['beads'])
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
	diff_ordered_grp_dihedrals = list(range(ns.nb_dihedrals))
	avg_diff_grp_dihedrals, row_wise_ranges['dihedrals'] = [], {}
	dihedrals = {}

	for grp_dihedral in range(ns.nb_dihedrals):
		
		dihedrals[grp_dihedral] = {'AA': {'x': [], 'y': []}, 'CG': {'x': [], 'y': []}}

		if manual_mode:
			dihedral_avg, dihedral_hist, _, _ = get_AA_dihedrals_distrib(ns, beads_ids=ns.cg_itp['dihedral'][grp_dihedral]['beads'])
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
			dihedral_avg, dihedral_hist, _, _ = get_CG_dihedrals_distrib(ns, beads_ids=ns.cg_itp['dihedral'][grp_dihedral]['beads'])
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

	larger_group = max(ns.nb_constraints, ns.nb_bonds, ns.nb_angles, ns.nb_dihedrals)
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
	nrows -= sum([ns.nb_constraints == 0, ns.nb_bonds == 0, ns.nb_angles == 0, ns.nb_dihedrals == 0])

	# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3), squeeze=False)
	# this fucking line was responsible of the big memory leak (figures were not closing) so I let this here for memory
	fig = plt.figure(figsize=(ncols*3, nrows*3))
	ax = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)

	# record the min/max y for each geom type
	constraints_min_y, bonds_min_y, angles_min_y, dihedrals_min_y = 10, 10, 10, 10
	constraints_max_y, bonds_max_y, angles_max_y, dihedrals_max_y = 0, 0, 0, 0

	# constraints
	if ns.nb_constraints != 0:
		print()
		nrow += 1
		for i in range(ncols):
			if i < ns.nb_constraints:
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
	if ns.nb_bonds != 0:
		print()
		nrow += 1
		for i in range(ncols):
			if i < ns.nb_bonds:
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
	if ns.nb_angles != 0:
		print()
		nrow += 1
		for i in range(ncols):
			if i < ns.nb_angles:
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
	if ns.nb_dihedrals != 0:
		print()
		nrow += 1
		for i in range(ncols):
			if i < ns.nb_dihedrals:
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
		if ns.nb_constraints != 0:
			nrow += 1
			for i in range(ns.nb_constraints):
				ax[nrow][i].set_ylim(bottom=constraints_min_y, top=constraints_max_y)
		if ns.nb_bonds != 0:
			nrow += 1
			for i in range(ns.nb_bonds):
				ax[nrow][i].set_ylim(bottom=bonds_min_y, top=bonds_max_y)
		if ns.nb_angles != 0:
			nrow += 1
			for i in range(ns.nb_angles):
				ax[nrow][i].set_ylim(bottom=angles_min_y, top=angles_max_y)
		if ns.nb_dihedrals != 0:
			nrow += 1
			for i in range(ns.nb_dihedrals):
				ax[nrow][i].set_ylim(bottom=dihedrals_min_y, top=dihedrals_max_y)

	# calculate global fitness score and contributions from each geom type
	all_dist_pairwise = '' # for global optimization plotting
	all_emd_dist_geoms = {'constraints': [], 'bonds': [], 'angles': [], 'dihedrals': []}

	if not ns.atom_only:
		fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = 0, 0, 0, 0

		for i in range(ns.nb_constraints):
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

		for i in range(ns.nb_bonds):
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

		for i in range(ns.nb_angles):
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
		for i in range(ns.nb_dihedrals):
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
		if ignore_dihedrals and ns.nb_dihedrals > 0:
			eval_score -= fit_score_dihedrals
		sup_title = f'FITNESS SCORE\nTotal: {round(eval_score, 3)} -- Constraints/Bonds: {fit_score_constraints_bonds} -- Angles: {fit_score_angles} -- Dihedrals: {fit_score_dihedrals}'
		if ignore_dihedrals and ns.nb_dihedrals > 0:
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


# modify MDP file to adjust simulation length or other parameters
def modify_mdp(mdp_filename, sim_time=None, nb_frames=1500, log_write_freq=5000, energy_write_nb_frames_ratio=0.1):

	# TODO: this gives an incorrect number of frames in some cases
	# TODO: this whole function is really shit, but atm the MDP is user provided so we cannot use placeholders + not sure what kind of mistakes can be made in MDP files that are provided

	# read input
	with open(mdp_filename, 'r') as fp:
		mdp_lines_in = fp.read().split('\n')
		mdp_lines = [mdp_line.split(';')[0].strip() for mdp_line in mdp_lines_in] # split for comments

	dt_line, nsteps_line = -1, -1 # line at which we have found dt and nsteps entries
	nstlog_line = -1 # line at which we have found nstlog entry
	nstxout_line, nstvout_line, nstfout_line = -1, -1, -1 # lines at which we have found nstxout, nstvout and nstfout entries
	nstcalcenergy_line, nstenergy_line = -1, -1 # lines are which we have found nstcalcenergy and nstenergy
	nstxout_compressed_line = -1 # line at which we have found nstxout-compressed entry

	for i in range(len(mdp_lines)):
		mdp_line = mdp_lines[i]

		if mdp_line.startswith('dt'):
			sp_dt_line = mdp_line.split('=')
			if sp_dt_line[0].strip() == 'dt': # discard other lines that could start with 'dt'
				dt_line = i
				dt = float(sp_dt_line[1].strip())

		elif mdp_line.startswith('nsteps'):
			sp_nsteps_line = mdp_line.split('=')
			if sp_nsteps_line[0].strip() == 'nsteps': # discard other lines that could start with 'nsteps'
				nsteps_line = i
				nsteps = int(sp_nsteps_line[1].strip())

		elif mdp_line.startswith('nstlog'):
			sp_nstlog_line = mdp_line.split('=')
			if sp_nstlog_line[0].strip() == 'nstlog': # discard other lines that could start with 'nstlog'
				nstlog_line = i

		elif mdp_line.startswith('nstxout') and not mdp_line.startswith('nstxout-compressed'):
			sp_nstxout_line = mdp_line.split('=')
			if sp_nstxout_line[0].strip() == 'nstxout': # discard other lines that could start with 'nstxout'
				nstxout_line = i

		elif mdp_line.startswith('nstvout'):
			sp_nstvout_line = mdp_line.split('=')
			if sp_nstvout_line[0].strip() == 'nstvout': # discard other lines that could start with 'nstvout'
				nstvout_line = i

		elif mdp_line.startswith('nstfout'):
			sp_nstfout_line = mdp_line.split('=')
			if sp_nstfout_line[0].strip() == 'nstfout': # discard other lines that could start with 'nstfout'
				nstfout_line = i

		elif mdp_line.startswith('nstcalcenergy'):
			sp_nstcalcenergy_line = mdp_line.split('=')
			if sp_nstcalcenergy_line[0].strip() == 'nstcalcenergy': # discard other lines that could start with 'nstcalcenergy'
				nstcalcenergy_line = i

		elif mdp_line.startswith('nstenergy'):
			sp_nstenergy_line = mdp_line.split('=')
			if sp_nstenergy_line[0].strip() == 'nstenergy': # discard other lines that could start with 'nstenergy'
				nstenergy_line = i

		elif mdp_line.startswith('nstxout-compressed') or mdp_line.startswith('nstxtcout'):
			sp_nstxout_compressed_line = mdp_line.split('=')
			nstxout_compressed_line = i

	# adjust simulation time according to timestep
	if sim_time is not None:
		if dt_line != -1 and nsteps_line != -1:
			nsteps = int(sim_time*1000 / dt)
			mdp_lines_in[nsteps_line] = f'{sp_nsteps_line[0]}= {nsteps}    ; automatically modified by Swarm-CG'
		else:
			msg = "The provided MD MDP file does not contain one of these entries: dt, nsteps."
			raise exceptions.MissformattedFile(msg)

	# force writting to the log file every given nb of steps, to make sure simulations won't be killed for
	# insufficient writting to the log file (which we use to check for simulations that are stuck/bugged)
	if nstlog_line != -1:
		nstlog = log_write_freq
		mdp_lines_in[nstlog_line] = f'{sp_nstlog_line[0]}= {nstlog}    ; automatically modified by Swarm-CG'
	else:
		msg = "The provided MD MDP file does not contain one of these entries: nstlog."
		raise exceptions.MissformattedFile(msg)

	# force NOT writting coordinates data, as this can only slow the simulation and we don't need it
	nstxout = nsteps
	if nstxout_line != -1:
		mdp_lines_in[nstxout_line] = f'{sp_nstxout_line[0]}= {nstxout}    ; automatically modified by Swarm-CG'
	else:
		mdp_lines_in.append(f'nstxout = {nstxout}    ; automatically added by Swarm-CG')

	# force NOT writting velocities data, as this can only slow the simulation and we don't need it
	nstvout = nsteps
	if nstvout_line != -1:
		mdp_lines_in[nstvout_line] = f'{sp_nstvout_line[0]}= {nstvout}    ; automatically modified by Swarm-CG'
	else:
		mdp_lines_in.append(f'nstvout = {nstvout}    ; automatically added by Swarm-CG')

	# force NOT writting forces data, as this can only slow the simulation and we don't need it
	nstfout = nsteps
	if nstfout_line != -1:
		mdp_lines_in[nstfout_line] = f'{sp_nstfout_line[0]}= {nstfout}    ; automatically modified by Swarm-CG'
	else:
		mdp_lines_in.append(f'nstfout = {nstfout}    ; automatically added by Swarm-CG')

	# force calculating and writing frames at given frequency, to not slow down
	# the simulation too much but still allow for energy analysis
	nstcalcenergy = int(nsteps / nb_frames / energy_write_nb_frames_ratio)
	nstenergy = nstcalcenergy
	if nstcalcenergy_line != -1:
		mdp_lines_in[nstcalcenergy_line] = f'{sp_nstcalcenergy_line[0]}= {nstcalcenergy}    ; automatically modified by Swarm-CG'
	else:
		mdp_lines_in.append(f'nstcalcenergy = {nstcalcenergy}    ; automatically added by Swarm-CG')
	if nstenergy_line != -1:
		mdp_lines_in[nstenergy_line] = f'{sp_nstenergy_line[0]}= {nstenergy}    ; automatically modified by Swarm-CG'
	else:
		mdp_lines_in.append(f'nstenergy = {nstenergy}    ; automatically added by Swarm-CG')

	# force writting compressed frames at given frequency, so that we obtain the desired number of frames for each CG simulation/evaluation step
	nstxout_compressed = int(nsteps / nb_frames)
	if nstxout_compressed_line != -1:
		mdp_lines_in[nstxout_compressed_line] = f'{sp_nstxout_compressed_line[0]}= {nstxout_compressed}    ; automatically modified by Swarm-CG'
	else:
		# sys.exit(config.header_error+'The provided MD MDP file does not contain one of these entries: nstxout-compressed')
		mdp_lines_in.append(f'nstxout-compressed = {nstxout_compressed}    ; automatically added by Swarm-CG')

	# write output
	with open(mdp_filename, 'w') as fp:
		for mdp_line in mdp_lines_in:
			fp.write(mdp_line+'\n')


# execute command and return output
def cmdline(command):

	try:
		output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode()
		success = True 
	except subprocess.CalledProcessError as e:
		output = e.output.decode()
		success = False

	return success, output


# print forced stdout enabled
def print_stdout_forced(*args, **kwargs):

	with contextlib.redirect_stdout(sys.__stdout__):
		print(*args, **kwargs, flush=True)


# evaluation function to be optimized using FST-PSO
def eval_function(parameters_set, ns):

	ns.nb_eval += 1
	start_eval_ts = datetime.now().timestamp()

	print_stdout_forced()
	# TODO: this should use logging
	print_stdout_forced(
		f"Starting iteration {ns.nb_eval} at {time.strftime('%H:%M:%S')} on {time.strftime('%d-%m-%Y')}"
	)

	# enter the execution directory
	os.chdir(ns.exec_folder)

	# create new directory for new parameters evaluation
	current_eval_dir = f'{config.iteration_sim_files_dirname}_eval_step_{ns.nb_eval}'
	shutil.copytree(config.input_sim_files_dirname, current_eval_dir)

	# create a modified CG ITP file with parameters according to current evaluation type
	update_cg_itp_obj(ns, parameters_set=parameters_set, update_type=1)
	out_path_itp = f'{config.iteration_sim_files_dirname}_eval_step_{ns.nb_eval}/{ns.cg_itp_basename}'
	if ns.opti_cycle['nb_geoms']['dihedral'] == 0:
		print_sections = ['constraint', 'bond', 'angle', 'exclusion']
	else:
		print_sections = ['constraint', 'bond', 'angle', 'dihedral', 'exclusion']
	write_cg_itp_file(ns.out_itp, out_path_itp, print_sections=print_sections)

	# enter current evaluation directory and stay there until all sims are finished or failed
	os.chdir(current_eval_dir)

	# run simulation with new parameters
	start_gmx_ts = datetime.now().timestamp()
	mini_killed, equi_killed, md_run_killed, new_best_fit = False, False, False, False

	# start from final conformation of previous best scored model -- that should improve precision of BP, especially using mode 2
	# previous_best_final_conf = '../'+config.best_fitted_model_dirname+'/md.gro'
	# if os.path.isfile(previous_best_final_conf):
	# 	ns.gro_input_basename = previous_best_final_conf

	# grompp -- minimization
	gmx_cmd = f'{ns.gmx_path} grompp -c {ns.gro_input_basename} -p {ns.top_input_basename} -f {ns.mdp_minimization_basename} -o mini -maxwarn {ns.mini_maxwarn}'
	with subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as gmx_process:
		gmx_out = gmx_process.communicate()[1].decode()
		gmx_process.kill()

	if gmx_process.returncode == 0:
		# mdrun -- minimization
		gmx_cmd = gmx_args(ns, 'mdrun -deffnm mini', mpi=False)
		with subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid) as gmx_process:  # create a process group for the minimization run

			# check if minimization run is stuck because of instabilities
			cycles_check = 0
			last_log_file_size = 0
			while gmx_process.poll() is None:  # while process is alive
				time.sleep(ns.process_alive_time_sleep)
				cycles_check += 1

				if cycles_check % ns.process_alive_nb_cycles_dead == 0:  # every minute or so, kill process if we determine it is stuck because the .log file's bytes size has not changed
					if os.path.isfile(current_eval_dir+'/mini.log'):
						log_file_size = os.path.getsize(current_eval_dir+'/mini.log')  # get size of .log file in bytes, as a mean of detecting the minimization run is stuck
					else:
						log_file_size = last_log_file_size  # minimization is stuck if the process was not able to create log file at start
					if log_file_size == last_log_file_size:  # minimization is stuck if the process is not writing to log file anymore
						os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)  # kill all processes of process group
						mini_killed = True
					else:
						last_log_file_size = log_file_size
			gmx_process.kill()

	else:
		msg = (
			f"{gmx_out}\n\n"
			f"Gromacs grompp failed at MD minimization step, see its error message above.\n"
			f"You may also want to check the parameters of the MDP file provided through\n"
			f"argument -cg_sim_mdp_mini. If you think this is a bug, please consider opening\n"
			f"an issue on GitHub at {config.github_url}/issues."
		)
		raise exceptions.ComputationError(msg)

	# if minimization finished properly, we just check for the .gro file printed in the end
	if os.path.isfile('mini.gro'):

		# grompp -- EQUI
		gmx_cmd = ns.gmx_path+' grompp -c mini.gro -p '+ns.top_input_basename+' -f '+ns.mdp_equi_basename+' -o equi'
		with subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as gmx_process:
			gmx_out = gmx_process.communicate()[1].decode()
			gmx_process.kill()

		if gmx_process.returncode == 0:
			# mdrun -- EQUI
			gmx_cmd = gmx_args(ns, 'mdrun -deffnm equi')
			with subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid) as gmx_process:  # create a process group for the EQUI run

				# check if EQUI run is stuck because of instabilities
				cycles_check = 0
				last_log_file_size = 0
				while gmx_process.poll() is None:  # while process is alive
					time.sleep(ns.process_alive_time_sleep)
					cycles_check += 1

					if cycles_check % ns.process_alive_nb_cycles_dead == 0:  # every minute or so, kill process if we determine it is stuck because the .log file's bytes size has not changed
						if os.path.isfile(current_eval_dir+'/equi.log'):
							log_file_size = os.path.getsize(current_eval_dir+'/equi.log')  # get size of .log file in bytes, as a mean of detecting the EQUI run is stuck
						else:
							log_file_size = last_log_file_size  # EQUI is stuck if the process was not able to create log file at start
						if log_file_size == last_log_file_size:  # EQUI is stuck if the process is not writing to log file anymore
							os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)  # kill all processes of process group
							equi_killed = True
						else:
							last_log_file_size = log_file_size
				gmx_process.kill()

		else:
			msg = (
				f"{gmx_out}\n\n"
				f"Gromacs grompp failed at MD equilibration step, see its error message above.\n"
				f"You may also want to check the parameters of the MDP file provided through\n"
				f"argument -cg_sim_mdp_equi. If you think this is a bug, please consider opening\n"
				f"an issue on GitHub at {config.github_url}/issues."
			)
			raise exceptions.ComputationError(msg)

		# if EQUI finished properly, we just check for the .gro file printed in the end
		if os.path.isfile('equi.gro'):

			# adapt duration of the simulation
			modify_mdp(mdp_filename=ns.mdp_md_basename, sim_time=ns.prod_sim_time)

			# grompp -- MD
			gmx_cmd = ns.gmx_path+' grompp -c equi.gro -p '+ns.top_input_basename+' -f '+ns.mdp_md_basename+' -o md'
			with subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as gmx_process:
				gmx_out = gmx_process.communicate()[1].decode()
				gmx_process.kill()

			if gmx_process.returncode == 0:
				# mdrun -- MD
				gmx_cmd = gmx_args(ns, 'mdrun -deffnm md')
				with subprocess.Popen([gmx_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid) as gmx_process:  # create a process group for the MD run

					# check if MD run is stuck because of instabilities
					cycles_check = 0
					last_log_file_size = 0
					while gmx_process.poll() is None:  # while process is alive
						time.sleep(ns.process_alive_time_sleep)
						cycles_check += 1

						if cycles_check % ns.process_alive_nb_cycles_dead == 0:  # every minute or so, kill process if we determine it is stuck because the .log file's bytes size has not changed
							if os.path.isfile('md.log'):
								log_file_size = os.path.getsize('md.log')  # get size of .log file in bytes, as a mean of detecting the MD run is stuck
							else:
								log_file_size = last_log_file_size  # MD run is stuck if the process was not able to create log file at start
							if log_file_size == last_log_file_size:  # MD run is stuck if the process is not writing to log file anymore
								os.killpg(os.getpgid(gmx_process.pid), signal.SIGKILL)  # kill all processes of process group
								md_run_killed = True
							else:
								last_log_file_size = log_file_size
					gmx_process.kill()

			else:
				msg = (
					f"{gmx_out}\n\n"
					f"Gromacs grompp failed at MD production step, see its error message above.\n"
					f"You may also want to check the parameters of the MDP file provided through\n"
					f"argument -cg_sim_mdp_md. If you think this is a bug, please consider opening\n"
					f"an issue on GitHub at {config.github_url}/issues."
				)
				raise exceptions.ComputationError(msg)

			# to verify if MD run finished properly, we check for the .gro file printed in the end
			if os.path.isfile('md.gro'):

				# get distributions and evaluate fitness
				ns.cg_tpr_filename = 'md.tpr'
				ns.cg_traj_filename = 'md.xtc'
				ns.plot_filename = 'distributions.png'
				ns.total_gmx_time += datetime.now().timestamp() - start_gmx_ts

				start_model_eval_ts = datetime.now().timestamp()
				ignore_dihedrals = False
				if ns.opti_cycle['nb_geoms']['dihedral'] == 0:
					ignore_dihedrals = True
				fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, all_dist_pairwise, all_emd_dist_geoms = compare_models(ns, manual_mode=False, ignore_dihedrals=ignore_dihedrals, calc_sasa=True, record_best_indep_params=True)
				ns.total_model_eval_time += datetime.now().timestamp() - start_model_eval_ts

				# if gmx sasa failed to compute, it's most likely because there were inconsistent shifts across PBC in the trajectory = failed run
				if ns.sasa_cg is not None:

					# store the distributions for each evaluation step
					shutil.move('distributions.png', f'../{config.distrib_plots_all_evals_dirname}/distributions_eval_step_{ns.nb_eval}.png')

					eval_score = 0
					if 'constraint' in ns.opti_cycle['geoms'] and 'bond' in ns.opti_cycle['geoms']:
						eval_score += fit_score_constraints_bonds
					if 'angle' in ns.opti_cycle['geoms']:
						eval_score += fit_score_angles
					if 'dihedral' in ns.opti_cycle['geoms']:
						eval_score += fit_score_dihedrals

					global_score = 0
					if 'constraint' in ns.opti_geoms_all and 'bond' in ns.opti_geoms_all:
						global_score += fit_score_constraints_bonds
					if 'angle' in ns.opti_geoms_all:
						global_score += fit_score_angles
					if 'dihedral' in ns.opti_geoms_all:
						global_score += fit_score_dihedrals

					# model selection based only on bonded parametrization score
					regular_eval = True
					if regular_eval:
						if global_score < ns.best_fitness[0]:
							new_best_fit = True
							ns.best_fitness = global_score, ns.nb_eval
							ns.all_emd_dist_geoms = all_emd_dist_geoms

				else:
					print_stdout_forced("  MD run failed (molecule exploded)")
					eval_score, fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = [ns.worst_fit_score] * 5
					ns.gyr_cg, ns.gyr_cg_std, ns.sasa_cg, ns.sasa_cg_std = None, None, None, None
					ns.total_gmx_time += datetime.now().timestamp() - start_gmx_ts

			else:
				if md_run_killed:
					print_stdout_forced(
						f"  MD run failed (unstable simulation was killed, with unstable "
						f"= NOT writing in log file for {str(ns.sim_kill_delay)} sec)"
					)
				else:
					print_stdout_forced('  MD run failed (simulation process terminated with error)')
				eval_score, fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = [ns.worst_fit_score] * 5
				ns.gyr_cg, ns.gyr_cg_std, ns.sasa_cg, ns.sasa_cg_std = None, None, None, None
				ns.total_gmx_time += datetime.now().timestamp() - start_gmx_ts

		else:
			if equi_killed:
				print_stdout_forced(
					f"  Equilibration run failed (unstable simulation was killed, with unstable "
					f"= NOT writing in log file for {str(ns.sim_kill_delay)} sec)"
				)
			else:
				print_stdout_forced(
					"  Equilibration run failed (simulation process terminated with error)"
				)
			eval_score, fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = [ns.worst_fit_score] * 5
			ns.gyr_cg, ns.gyr_cg_std, ns.sasa_cg, ns.sasa_cg_std = None, None, None, None
			ns.total_gmx_time += datetime.now().timestamp() - start_gmx_ts
	else:
		if mini_killed:
			print_stdout_forced(
				f"  Minimization run failed (unstable simulation was killed, with unstable "
				f"= NOT writing in log file for {str(ns.sim_kill_delay)} sec)"
			)
		else:
			print_stdout_forced(
				"  Minimization run failed (simulation process terminated with error)"
			)
		eval_score, fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals = [ns.worst_fit_score] * 5
		ns.gyr_cg, ns.gyr_cg_std, ns.sasa_cg, ns.sasa_cg_std = None, None, None, None
		ns.total_gmx_time += datetime.now().timestamp() - start_gmx_ts

	# exit current eval directory
	os.chdir('..')

	# store all log files
	if os.path.isfile(current_eval_dir+'/md.log'):
		shutil.copy(current_eval_dir+'/md.log', f'{config.log_files_all_evals_dirname}/md_sim_eval_step_{ns.nb_eval}.log')
	if os.path.isfile(current_eval_dir+'/equi.log'):
		shutil.copy(current_eval_dir+'/equi.log', f'{config.log_files_all_evals_dirname}/equi_sim_eval_step_{ns.nb_eval}.log')
	if os.path.isfile(current_eval_dir+'/mini.log'):
		shutil.copy(current_eval_dir+'/mini.log', f'{config.log_files_all_evals_dirname}/mini_sim_eval_step_{ns.nb_eval}.log')

	# update the best results distrib plot in execution directory
	if new_best_fit:
		shutil.copy(f'{config.distrib_plots_all_evals_dirname}/distributions_eval_step_{ns.nb_eval}.png', config.best_distrib_plots)

	# keep all sim files if user wants to
	if ns.keep_all_sims:
		shutil.copytree(current_eval_dir, config.sim_files_all_evals_dirname+'/'+current_eval_dir)

	# keep BI files (the very first guess of bonded parameters) only for figures
	# TODO: remove ?? this is redundant because we already produce a directory with output for the best current model
	if ns.nb_eval == 1:
		shutil.copytree(current_eval_dir, 'boltzmann_inv_CG_model')

	# store sim files for new best fit OR remove eval sim files
	if new_best_fit:
		if os.path.exists(config.best_fitted_model_dirname):
			shutil.rmtree(config.best_fitted_model_dirname)
		shutil.move(current_eval_dir, config.best_fitted_model_dirname)
	else:
		shutil.rmtree(current_eval_dir)

	# when simulation crashes, write the worst possible score considering all geoms
	if eval_score == ns.worst_fit_score:
		all_dist_pairwise = ''
		for _ in range(len(ns.cg_itp['constraint'])+len(ns.cg_itp['bond'])+len(ns.cg_itp['angle'])+len(ns.cg_itp['dihedral'])):
			all_dist_pairwise += str(config.sim_crash_EMD_indep_score)+' '
		all_dist_pairwise += '\n'
	else:
		print_stdout_forced('  Total mismatch score:', round(fit_score_total, 3), '(Bonds/Constraints:', fit_score_constraints_bonds, '-- Angles:', fit_score_angles, '-- Dihedrals:', str(fit_score_dihedrals)+')')
		if new_best_fit:
			print_stdout_forced('    --> Selected as new best bonded parametrization')
		# print_stdout_forced('  Opti context mismatch score:', round(eval_score, 3))
		print_stdout_forced(f'  Rg CG:   {round(ns.gyr_cg, 2)} nm   (Error abs. {round(abs(1 - ns.gyr_cg / ns.gyr_aa_mapped) * 100, 1)}% -- Reference Rg AA-mapped: {ns.gyr_aa_mapped} nm)')
		print_stdout_forced(f'  SASA CG: {ns.sasa_cg} nm2   (Error abs. {round(abs(1 - ns.sasa_cg / ns.sasa_aa_mapped) * 100, 1)}% -- Reference SASA AA-mapped: {ns.sasa_aa_mapped} nm2)')

	current_total_time = round((datetime.now().timestamp() - ns.start_opti_ts) / (60 * 60), 2)
	current_eval_time = datetime.now().timestamp() - start_eval_ts
	ns.total_eval_time += current_eval_time
	current_eval_time = round(current_eval_time / 60, 2)
	print_stdout_forced(f'  Iteration time: {current_eval_time} min')

	# write all pairwise distances between atom mapped and CG geoms to file for later global optimization perf plotting
	with open(config.opti_pairwise_distances_file, 'a') as fp:
		if 'dihedral' in ns.opti_cycle['geoms']:
			fp.write('1 '+all_dist_pairwise)
		else:
			fp.write('0 '+all_dist_pairwise)
	with open(config.opti_perf_recap_file, 'a') as fp:
		recap_line = ' '.join(list(map(str, (ns.opti_cycle['nb_cycle'], ns.nb_eval, fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, eval_score, ns.gyr_aa_mapped, ns.gyr_aa_mapped_std, ns.gyr_cg, ns.gyr_cg_std, ns.sasa_aa_mapped, ns.sasa_aa_mapped_std, ns.sasa_cg, ns.sasa_cg_std))))+' '
		for i in range(len(ns.cg_itp['constraint'])):
			recap_line += f"{ns.out_itp['constraint'][i]['value']} "
		for i in range(len(ns.cg_itp['bond'])):
			recap_line += f"{ns.out_itp['bond'][i]['value']} {ns.out_itp['bond'][i]['fct']} "
		for i in range(len(ns.cg_itp['angle'])):
			recap_line += f"{ns.out_itp['angle'][i]['value']} {ns.out_itp['angle'][i]['fct']} "
		for i in range(len(ns.cg_itp['dihedral'])):
			if ns.opti_cycle['nb_geoms']['dihedral'] == 0:
				recap_line += '0 0 '
			else:
				recap_line += f"{ns.out_itp['dihedral'][i]['value']} {ns.out_itp['dihedral'][i]['fct']} "
		recap_line += f'{current_eval_time} {current_total_time}'
		fp.write(recap_line+'\n')

	os.chdir('..')  # exit the execution directory

	return eval_score




