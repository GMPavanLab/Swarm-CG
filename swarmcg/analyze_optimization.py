# some numpy version have this ufunc warning at import + many packages call numpy and display annoying warnings
import warnings
warnings.filterwarnings("ignore")
import os, sys
from argparse import ArgumentParser, RawTextHelpFormatter, SUPPRESS
from shlex import quote as cmd_quote

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import swarmcg.shared.styling
from swarmcg import config
from swarmcg.shared import exceptions
from swarmcg.shared.styling import ANALYSE_DESCR
from swarmcg.shared.utils import forward_fill

warnings.resetwarnings()


def run(ns):

	# filter matplotlib warnings
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	warnings.filterwarnings("ignore", category=ImportWarning)

	# TODO: print some text to tell user if opti run finished or not -- then we can only look at the results files, not the running processes on the machine

	display_sim_crashes = False
	display_opti_cycles_sep = True
	plot_control_std = True
	opti_cycles_sep_color = 'black'
	color_scores = 'darkgreen'
	color_subscores = 'mediumseagreen'
	yrange_rg = [None, None]
	yrange_sasa = [None, None]
	all_gyr_aa_mapped_offset = 0.00  # rescaling offset

	plt.rcParams['axes.axisbelow'] = True

	# parameters
	read_offset = 15  # nb of trailing fields that have static lengths in the recap file (i.e. NOT dependent on number of bonds, angles, etc.)
	min_nb_cols = 9  # to be sure we have enough columns for opti process plots, even if number of bonds/angles/dihedrals is less than this

	# read scores for each geom at each fitness evaluation/simulation
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=UserWarning)
		iter_indep_scores = np.genfromtxt(ns.opti_dirname+'/'+config.opti_pairwise_distances_file, delimiter=' ')
	try:
		for i in range(1, iter_indep_scores.shape[1]):
			iter_indep_scores[:, i] = forward_fill(iter_indep_scores[:,i], config.sim_crash_EMD_indep_score)
	except IndexError:
		msg = (
			"The optimization recap file seems empty. Please wait for your optimization process\n"
			"to have performed a few iterations, or check for errors during execution."
		)
		raise exceptions.IncompleteOptimisationFile(msg)


	# process files and plot
	with open(ns.opti_dirname+'/'+config.opti_perf_recap_file, 'r') as fp:

		eval_lines = fp.read().split('\n')
		nb_evals = len(eval_lines)-7
		print("Found", nb_evals, "optimization steps")

		# to make sure the 2 opti recap files contain the same number of iterations (avoid buffering/writing problems so that this script can be executed anytime)
		iter_indep_scores = iter_indep_scores[:nb_evals,]

		# read header
		nb_constraints = int(eval_lines[0].split()[3])
		nb_bonds = int(eval_lines[1].split()[3])
		nb_angles = int(eval_lines[2].split()[3])
		nb_dihedrals = int(eval_lines[3].split()[3])

		# results storage structure
		parameters_vals = {
			'constraints': {'values': {}},
			'bonds': {'values': {}, 'force_ctr': {}},
			'angles': {'values': {}, 'force_ctr': {}},
			'dihedrals': {'values': {}, 'force_ctr': {}}
		}

		for i in range(nb_constraints):
			parameters_vals['constraints']['values'][i] = []
		for i in range(nb_bonds):
			parameters_vals['bonds']['values'][i] = []
			parameters_vals['bonds']['force_ctr'][i] = []
		for i in range(nb_angles):
			parameters_vals['angles']['values'][i] = []
			parameters_vals['angles']['force_ctr'][i] = []
		for i in range(nb_dihedrals):
			parameters_vals['dihedrals']['values'][i] = []
			parameters_vals['dihedrals']['force_ctr'][i] = []

		all_eval_scores, all_eval_times, all_total_times = [], [], []
		all_fit_score_total, all_fit_score_constraints_bonds, all_fit_score_angles, all_fit_score_dihedrals = np.array([]), np.array([]), np.array([]), np.array([])
		all_gyr_aa_mapped, all_gyr_aa_mapped_std, all_gyr_cg, all_gyr_cg_std = np.array([]), np.array([]), np.array([]), np.array([])
		all_sasa_aa_mapped, all_sasa_aa_mapped_std, all_sasa_cg, all_sasa_cg_std = np.array([]), np.array([]), np.array([]), np.array([])

		all_opti_cycles = []
		y_fct_range = {'bonds': [np.inf, 0], 'angles': [np.inf, 0], 'dihedrals': [np.inf, 0]}

		# read content
		for i in range(6, 6+nb_evals):  # nb of evaluations is taken from the independent scores to ignore a possible ongoing simulation, to be able to run this script during the opti process

			sp_eval_line = eval_lines[i].split()
			opti_cycle = int(sp_eval_line[0])
			all_opti_cycles.append(opti_cycle)
			fit_score_total, fit_score_constraints_bonds, fit_score_angles, fit_score_dihedrals, eval_score = list(map(float, sp_eval_line[2:read_offset-8]))
			try:
				gyr_aa_mapped, gyr_aa_mapped_std = float(sp_eval_line[read_offset-8]), float(sp_eval_line[read_offset-7])
			except ValueError:
				gyr_aa_mapped, gyr_aa_mapped_std = None, None
			try:
				gyr_cg, gyr_cg_std = float(sp_eval_line[read_offset-6]), float(sp_eval_line[read_offset-5])
			# this controls if simulation has crashed
			except ValueError:
				gyr_cg, gyr_cg_std = None, None
				eval_score, fit_score_total, fit_score_constraints_bonds = None, None, None
				fit_score_angles, fit_score_dihedrals = None, None

			try:
				sasa_aa_mapped, sasa_aa_mapped_std = float(sp_eval_line[read_offset-4]), float(sp_eval_line[read_offset-3])
			except ValueError:
				sasa_aa_mapped, sasa_aa_mapped_std = None, None
			try:
				sasa_cg, sasa_cg_std = float(sp_eval_line[read_offset-2]), float(sp_eval_line[read_offset-1])
			# this controls if simulation has crashed
			except ValueError:
				sasa_cg, sasa_cg_std = None, None

			eval_time = float(sp_eval_line[read_offset+nb_constraints+nb_bonds*2+nb_angles*2+nb_dihedrals*2])
			total_time = float(sp_eval_line[read_offset+nb_constraints+nb_bonds*2+nb_angles*2+nb_dihedrals*2+1])

			all_eval_scores = np.append(all_eval_scores, eval_score)
			all_eval_times = np.append(all_eval_times, eval_time)
			all_total_times = np.append(all_total_times, total_time)
			all_fit_score_total = np.append(all_fit_score_total, fit_score_total)
			all_fit_score_constraints_bonds = np.append(all_fit_score_constraints_bonds, fit_score_constraints_bonds)
			all_fit_score_angles = np.append(all_fit_score_angles, fit_score_angles)
			all_fit_score_dihedrals = np.append(all_fit_score_dihedrals, fit_score_dihedrals)
			all_gyr_aa_mapped = np.append(all_gyr_aa_mapped, gyr_aa_mapped)
			all_gyr_cg = np.append(all_gyr_cg, gyr_cg)
			all_sasa_aa_mapped = np.append(all_sasa_aa_mapped, sasa_aa_mapped)
			all_sasa_cg = np.append(all_sasa_cg, sasa_cg)
			all_gyr_aa_mapped_std = np.append(all_gyr_aa_mapped_std, gyr_aa_mapped_std)
			all_gyr_cg_std = np.append(all_gyr_cg_std, gyr_cg_std)
			all_sasa_aa_mapped_std = np.append(all_sasa_aa_mapped_std, sasa_aa_mapped_std)
			all_sasa_cg_std = np.append(all_sasa_cg_std, sasa_cg_std)

			# hide profiles when both value and force constant are 0
			for j in range(nb_constraints):
				parameters_vals['constraints']['values'][j].append(float(sp_eval_line[read_offset+j]))

			for j in range(nb_bonds):
				val, fct = float(sp_eval_line[read_offset+nb_constraints+j*2]), float(sp_eval_line[read_offset+nb_constraints+j*2+1])
				if val == 0 and fct == 0:
					val, fct = None, None
				else:
					if fct > y_fct_range['bonds'][1]:
						y_fct_range['bonds'][1] = fct
					if fct < y_fct_range['bonds'][0]:
						y_fct_range['bonds'][0] = fct
				parameters_vals['bonds']['values'][j].append(val)
				parameters_vals['bonds']['force_ctr'][j].append(fct)

			for j in range(nb_angles):
				val, fct = float(sp_eval_line[read_offset+nb_constraints+nb_bonds*2+j*2]), float(sp_eval_line[read_offset+nb_constraints+nb_bonds*2+j*2+1])
				if val == 0 and fct == 0:
					val, fct = None, None
				else:
					if fct > y_fct_range['angles'][1]:
						y_fct_range['angles'][1] = fct
					if fct < y_fct_range['angles'][0]:
						y_fct_range['angles'][0] = fct
				parameters_vals['angles']['values'][j].append(val)
				parameters_vals['angles']['force_ctr'][j].append(fct)

			for j in range(nb_dihedrals):
				val, fct = float(sp_eval_line[read_offset+nb_constraints+nb_bonds*2+nb_angles*2+j*2]), float(sp_eval_line[read_offset+nb_constraints+nb_bonds*2+nb_angles*2+j*2+1])
				if val == 0 and fct == 0:
					val, fct = None, None
				else:
					if fct > y_fct_range['dihedrals'][1]:
						y_fct_range['dihedrals'][1] = fct
					if fct < y_fct_range['dihedrals'][0]:
						y_fct_range['dihedrals'][0] = fct
				parameters_vals['dihedrals']['values'][j].append(val)
				parameters_vals['dihedrals']['force_ctr'][j].append(fct)

	# find separations between opti cycles
	opti_cycles_sep = []
	for i in range(1, len(all_opti_cycles)):
		if all_opti_cycles[i] != all_opti_cycles[i-1]:
			opti_cycles_sep.append(i+0.5)

	all_opti_cycles = np.array(all_opti_cycles)

	# select lowest bonded fitness score
	cyc_mask = np.where((all_opti_cycles > 0) & (all_fit_score_total != None))[0]
	id_best_all = np.where(all_fit_score_total == np.amin(all_fit_score_total[cyc_mask]))[0][0]

	print('Best bonded terms found at step', id_best_all+1, 'with estimated Rg', all_gyr_cg[id_best_all], 'nm and SASA', all_sasa_cg[id_best_all], 'nm2')

	print()
	print('  Rg CG: ', ' '+str(round(all_gyr_cg[id_best_all], 3)), 'nm   (Error abs.', str(round(abs(1-all_gyr_cg[id_best_all]/all_gyr_aa_mapped[id_best_all])*100, 1))+'% -- Reference Rg AA-mapped:', str(all_gyr_aa_mapped[id_best_all])+' nm)')
	print('  SASA CG:', round(all_sasa_cg[id_best_all], 2), 'nm2   (Error abs.', str(round(abs(1-all_sasa_cg[id_best_all]/all_sasa_aa_mapped[id_best_all])*100, 1))+'% -- Reference SASA AA-mapped:', str(all_sasa_aa_mapped[id_best_all])+' nm2)')

	# display indicator when simulation(s) crashed for any reason -- check for None gyr_cg to identify a simulation as crashed
	crashes_ids = np.where(all_gyr_cg == None)[0]+1

	all_eval_scores = forward_fill(all_eval_scores, None)
	all_fit_score_total = forward_fill(all_fit_score_total, None)
	all_fit_score_constraints_bonds = forward_fill(all_fit_score_constraints_bonds, None)
	all_fit_score_angles = forward_fill(all_fit_score_angles, None)
	all_fit_score_dihedrals = forward_fill(all_fit_score_dihedrals, None)
	all_gyr_aa_mapped = forward_fill(all_gyr_aa_mapped, None)
	all_gyr_aa_mapped_std = forward_fill(all_gyr_aa_mapped_std, None)
	all_gyr_cg = forward_fill(all_gyr_cg, None)
	all_gyr_cg_std = forward_fill(all_gyr_cg_std, None)
	all_sasa_aa_mapped = forward_fill(all_sasa_aa_mapped, None)
	all_sasa_aa_mapped_std = forward_fill(all_sasa_aa_mapped_std, None)
	all_sasa_cg = forward_fill(all_sasa_cg, None)
	all_sasa_cg_std = forward_fill(all_sasa_cg_std, None)

	for i in range(len(all_gyr_aa_mapped)):
		all_gyr_aa_mapped[i] += all_gyr_aa_mapped_offset

	larger_group = max(nb_constraints, nb_bonds, nb_angles, nb_dihedrals, min_nb_cols)
	nrow, nrows, ncols = 0, 9, larger_group
	nrows -= sum([nb_constraints == 0, nb_bonds == 0, nb_angles == 0, nb_dihedrals == 0]) * 2

	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4*ns.plot_scale, nrows*3*ns.plot_scale), squeeze=False)

	# all evaluations scores and stats
	x_evals = list(range(1, len(all_fit_score_total)+1))[:nb_evals]

	ax[0][0].set_title("All evaluations scores")
	ax[0][0].grid(zorder=0.5)
	ax[0][0].plot(x_evals, all_fit_score_total, color=color_scores)
	if display_sim_crashes:
		ax[0][0].scatter(crashes_ids, all_fit_score_total[crashes_ids-1], marker='x', color='black', zorder=2, label='sim crash')
		ax[0][0].legend(loc='best')
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][0].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)
	ax[0][0].plot(id_best_all+1, all_fit_score_total[id_best_all], marker='D', color='white', markerfacecolor='gold', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Selected model')
	ax[0][0].legend(loc='upper right')

	ax[0][1].set_title("All adaptives scores")
	ax[0][1].grid(zorder=0.5)
	ax[0][1].plot(x_evals, all_eval_scores, color=color_scores)
	if display_sim_crashes:
		ax[0][1].scatter(crashes_ids, all_eval_scores[crashes_ids-1], marker='x', color='black', zorder=2)
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][1].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)
	ax[0][1].plot(id_best_all+1, all_eval_scores[id_best_all], marker='D', color='white', markerfacecolor='gold', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Selected model')

	ax[0][2].set_title("Cstrs & bonds scores")
	ax[0][2].grid(zorder=0.5)
	ax[0][2].plot(x_evals, all_fit_score_constraints_bonds, color=color_scores)
	if display_sim_crashes:
		ax[0][2].scatter(crashes_ids, all_fit_score_constraints_bonds[crashes_ids-1], marker='x', color='black', zorder=2)
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][2].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)
	ax[0][2].plot(id_best_all+1, all_fit_score_constraints_bonds[id_best_all], marker='D', color='white', markerfacecolor='gold', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Selected model')

	ax[0][3].set_title("Angles scores")
	ax[0][3].grid(zorder=0.5)
	ax[0][3].plot(x_evals, all_fit_score_angles, color=color_scores)
	if display_sim_crashes:
		ax[0][3].scatter(crashes_ids, all_fit_score_angles[crashes_ids-1], marker='x', color='black', zorder=2)
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][3].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)
	ax[0][3].plot(id_best_all+1, all_fit_score_angles[id_best_all], marker='D', color='white', markerfacecolor='gold', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Selected model')

	ax[0][4].set_title("Dihedrals scores")
	ax[0][4].grid(zorder=0.5)
	ax[0][4].plot(x_evals, all_fit_score_dihedrals, color=color_scores)
	if display_sim_crashes:
		ax[0][4].scatter(crashes_ids, all_fit_score_dihedrals[crashes_ids-1], marker='x', color='black', zorder=2)
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][4].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)
	ax[0][4].plot(id_best_all+1, all_fit_score_dihedrals[id_best_all], marker='D', color='white', markerfacecolor='gold', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Selected model')

	ax[0][5].set_title("Radius of gyration")
	ax[0][5].grid(zorder=0.5)
	ax[0][5].plot(x_evals, all_gyr_aa_mapped, color=config.atom_color, label='AA-mapped', lw=2.5)
	if plot_control_std:
		ax[0][5].fill_between(x_evals, list(all_gyr_aa_mapped-all_gyr_aa_mapped_std), list(all_gyr_aa_mapped+all_gyr_aa_mapped_std), color=config.atom_color, alpha=0.1)
	ax[0][5].plot(x_evals, all_gyr_cg, color=config.cg_color, label='CG estimation')
	if plot_control_std:
		ax[0][5].fill_between(x_evals, list(all_gyr_cg-all_gyr_cg_std), list(all_gyr_cg+all_gyr_cg_std), color=config.cg_color, alpha=0.2)
	if display_sim_crashes:
		ax[0][5].scatter(crashes_ids, all_gyr_cg[crashes_ids-1], marker='x', color='black', zorder=2)
	ax[0][5].yaxis.set_major_locator(MaxNLocator(integer=True))
	ax[0][5].legend(loc='lower right')
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][5].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)
	ax[0][5].set_ylim(bottom=yrange_rg[0], top=yrange_rg[1])
	ax[0][5].plot(id_best_all+1, all_gyr_cg[id_best_all], marker='D', color='white', markerfacecolor='gold', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='') #label='Selected model')
	ax[0][5].legend(loc='lower right')

	ax[0][6].set_title("SASA")
	ax[0][6].grid(zorder=0.5)
	ax[0][6].plot(x_evals, all_sasa_aa_mapped, color=config.atom_color, label='AA-mapped', lw=2.5)
	if plot_control_std:
		ax[0][6].fill_between(x_evals, list(all_sasa_aa_mapped-all_sasa_aa_mapped_std), list(all_sasa_aa_mapped+all_sasa_aa_mapped_std), color=config.atom_color, alpha=0.1)
	ax[0][6].plot(x_evals, all_sasa_cg, color=config.cg_color, label='CG estimation')
	if plot_control_std:
		ax[0][6].fill_between(x_evals, list(all_sasa_cg-all_sasa_cg_std), list(all_sasa_cg+all_sasa_cg_std), color=config.cg_color, alpha=0.2)
	if display_sim_crashes:
		ax[0][6].scatter(crashes_ids, all_sasa_cg[crashes_ids-1], marker='x', color='black', zorder=2)
	ax[0][6].yaxis.set_major_locator(MaxNLocator(integer=True))
	ax[0][6].legend(loc='lower right')
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][6].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)
	ax[0][6].set_ylim(bottom=yrange_sasa[0], top=yrange_sasa[1])
	ax[0][6].plot(id_best_all+1, all_sasa_cg[id_best_all], marker='D', color='white', markerfacecolor='gold', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='') #label='Selected model')
	ax[0][6].legend(loc='lower right')

	ax[0][7].set_title("Total time (hours)")
	ax[0][7].grid(zorder=0.5)
	ax[0][7].plot(x_evals, all_total_times, color='purple')
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][7].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)
	ax[0][7].plot(id_best_all+1, all_total_times[id_best_all], marker='D', color='white', markerfacecolor='gold', markersize=10, markeredgewidth=1.5, markeredgecolor='black', label='Selected model')

	ax[0][8].set_title("All evaluation times (min)")
	ax[0][8].grid(zorder=0.5)
	ax[0][8].plot(x_evals, all_eval_times, color='mediumorchid')
	if display_opti_cycles_sep:
		for i in range(len(opti_cycles_sep)):
			ax[0][8].axvline(x=opti_cycles_sep[i], color=opti_cycles_sep_color)

	for i in range(9, ncols):
		ax[0][i].set_visible(False)

	plt_sidespace = 0.05  # ylim bottom and top margin as a percentage of y range of data
	x_min, x_max = 1-(nb_evals-1)*plt_sidespace, nb_evals+(nb_evals-1)*plt_sidespace

	# constraints
	if nb_constraints != 0:
		nrow += 2
		y_max = np.max(iter_indep_scores[:,1+nb_constraints])
		for i in range(ncols):
			if i < nb_constraints:
				# value
				ax[nrow-1][i].set_title('Constraint '+str(i+1)+' - Value')
				ax[nrow-1][i].grid(zorder=0.5)
				ax[nrow-1][i].plot(x_evals, parameters_vals['constraints']['values'][i])
				ax[nrow-1][i].set_xlim(x_min, x_max)
				if display_sim_crashes:
					ax[nrow-1][i].scatter(crashes_ids, np.array(parameters_vals['constraints']['values'][i])[crashes_ids-1], marker='x', color='black', zorder=2)
				if display_opti_cycles_sep:
					for j in range(len(opti_cycles_sep)):
						ax[nrow-1][i].axvline(x=opti_cycles_sep[j], color=opti_cycles_sep_color)

				# best models parameters
				if parameters_vals['constraints']['values'][i][id_best_all] != None:
					ax[nrow-1][i].plot(id_best_all+1, parameters_vals['constraints']['values'][i][id_best_all], marker='D', color='lightskyblue', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)

				# independant score
				ax[nrow][i].set_title('Constraint '+str(i+1)+' - Score')
				ax[nrow][i].grid(zorder=0.5)
				ax[nrow][i].plot(x_evals, iter_indep_scores[:,1+i], color=color_subscores)
				if display_sim_crashes:
					ax[nrow][i].scatter(crashes_ids, iter_indep_scores[:,1+i][crashes_ids-1], marker='x', color='black', zorder=2)
				ax[nrow][i].set_ylim(bottom=-y_max*plt_sidespace, top=y_max*(1+plt_sidespace))
				ax[nrow][i].set_xlim(x_min, x_max)
				if display_opti_cycles_sep:
					for j in range(len(opti_cycles_sep)):
						ax[nrow][i].axvline(x=opti_cycles_sep[j], color=opti_cycles_sep_color)

				# best models scores
				ax[nrow][i].plot(id_best_all+1, iter_indep_scores[id_best_all,1+i], marker='D', color='palegreen', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)
			else:
				ax[nrow-1][i].set_visible(False)
				ax[nrow][i].set_visible(False)

	# bonds
	if nb_bonds != 0:
		nrow += 2
		y_max = np.max(iter_indep_scores[:,1+nb_constraints:1+nb_constraints+nb_bonds])
		for i in range(ncols):
			if i < nb_bonds:
				# value and force constant
				ax[nrow-1][i].set_title('Bond '+str(i+1)+' - Value & Force constant')
				ax[nrow-1][i].grid(zorder=0.5)
				color = 'tab:blue'
				ax[nrow-1][i].plot(x_evals, parameters_vals['bonds']['values'][i], color=color)
				if display_sim_crashes:
					ax[nrow-1][i].scatter(crashes_ids, np.array(parameters_vals['bonds']['values'][i])[crashes_ids-1], marker='x', color='black', zorder=2)
				ax[nrow-1][i].tick_params(axis='y', labelcolor=color)
				ax[nrow-1][i].set_xlim(x_min, x_max)

				ax2 = ax[nrow-1][i].twinx()
				color = 'tab:red'
				ax2.plot(x_evals, parameters_vals['bonds']['force_ctr'][i], color=color)
				if display_sim_crashes:
					ax2.scatter(crashes_ids, np.array(parameters_vals['bonds']['force_ctr'][i])[crashes_ids-1], marker='x', color='black', zorder=2)
				if display_opti_cycles_sep:
					for j in range(len(opti_cycles_sep)):
						ax[nrow-1][i].axvline(x=opti_cycles_sep[j], color=opti_cycles_sep_color)
				ax2.set_ylim(bottom=y_fct_range['bonds'][0]-(y_fct_range['bonds'][1]-y_fct_range['bonds'][0])*plt_sidespace, top=y_fct_range['bonds'][1]+(y_fct_range['bonds'][1]-y_fct_range['bonds'][0])*plt_sidespace)
				if i == nb_bonds-1:
					ax2.tick_params(axis='y', labelcolor=color)
				else:
					ax2.yaxis.set_ticklabels([])

				# best models parameters
				if parameters_vals['bonds']['values'][i][id_best_all] != None:
					ax[nrow-1][i].plot(id_best_all+1, parameters_vals['bonds']['values'][i][id_best_all], marker='D', color='lightskyblue', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)
					ax2.plot(id_best_all+1, parameters_vals['bonds']['force_ctr'][i][id_best_all], marker='D', color='salmon', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)

				# independant score
				ax[nrow][i].set_title('Bond '+str(i+1)+' - Score')
				ax[nrow][i].grid(zorder=0.5)
				ax[nrow][i].plot(x_evals, iter_indep_scores[:,1+nb_constraints+i], color=color_subscores)
				if display_sim_crashes:
					ax[nrow][i].scatter(crashes_ids, iter_indep_scores[:,1+nb_constraints+i][crashes_ids-1], marker='x', color='black', zorder=2)
				ax[nrow][i].set_ylim(bottom=-y_max*plt_sidespace, top=y_max*(1+plt_sidespace))
				ax[nrow][i].set_xlim(x_min, x_max)
				if display_opti_cycles_sep:
					for j in range(len(opti_cycles_sep)):
						ax[nrow][i].axvline(x=opti_cycles_sep[j], color=opti_cycles_sep_color)

				# best models scores
				ax[nrow][i].plot(id_best_all+1, iter_indep_scores[id_best_all,1+nb_constraints+i], marker='D', color='palegreen', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)
			else:
				ax[nrow-1][i].set_visible(False)
				ax[nrow][i].set_visible(False)

	# angles
	if nb_angles != 0:
		nrow += 2
		y_max = np.max(iter_indep_scores[:,1+nb_constraints+nb_bonds:1+nb_constraints+nb_bonds+nb_angles])
		for i in range(ncols):
			if i < nb_angles:
				# value and force constant
				ax[nrow-1][i].set_title('Angle '+str(i+1)+' - Value & Force constant')
				ax[nrow-1][i].grid(zorder=0.5)
				color = 'tab:blue'
				ax[nrow-1][i].plot(x_evals, parameters_vals['angles']['values'][i], color=color)
				if display_sim_crashes:
					ax[nrow-1][i].scatter(crashes_ids, np.array(parameters_vals['angles']['values'][i])[crashes_ids-1], marker='x', color='black', zorder=2)
				ax[nrow-1][i].tick_params(axis='y', labelcolor=color)
				ax[nrow-1][i].set_xlim(x_min, x_max)
				ax2 = ax[nrow-1][i].twinx()
				color = 'tab:red'
				ax2.plot(x_evals, parameters_vals['angles']['force_ctr'][i], color=color)
				if display_sim_crashes:
					ax2.scatter(crashes_ids, np.array(parameters_vals['angles']['force_ctr'][i])[crashes_ids-1], marker='x', color='black', zorder=2)
				if display_opti_cycles_sep:
					for j in range(len(opti_cycles_sep)):
						ax[nrow-1][i].axvline(x=opti_cycles_sep[j], color=opti_cycles_sep_color)
				ax2.set_ylim(bottom=y_fct_range['angles'][0]-(y_fct_range['angles'][1]-y_fct_range['angles'][0])*plt_sidespace, top=y_fct_range['angles'][1]+(y_fct_range['angles'][1]-y_fct_range['angles'][0])*plt_sidespace)
				if i == nb_angles-1:
					ax2.tick_params(axis='y', labelcolor=color)
				else:
					ax2.yaxis.set_ticklabels([])

				# best models parameters
				if parameters_vals['angles']['values'][i][id_best_all] != None:
					ax[nrow-1][i].plot(id_best_all+1, parameters_vals['angles']['values'][i][id_best_all], marker='D', color='lightskyblue', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)
					ax2.plot(id_best_all+1, parameters_vals['angles']['force_ctr'][i][id_best_all], marker='D', color='salmon', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)

				# independant score
				ax[nrow][i].set_title('Angle '+str(i+1)+' - Score')
				ax[nrow][i].plot(x_evals, iter_indep_scores[:,1+nb_constraints+nb_bonds+i], color=color_subscores)
				if display_sim_crashes:
					ax[nrow][i].scatter(crashes_ids, iter_indep_scores[:,1+nb_constraints+nb_bonds+i][crashes_ids-1], marker='x', color='black', zorder=2)
				ax[nrow][i].grid(zorder=0.5)
				ax[nrow][i].set_ylim(bottom=-y_max*plt_sidespace, top=y_max*(1+plt_sidespace))
				ax[nrow][i].set_xlim(x_min, x_max)
				if display_opti_cycles_sep:
					for j in range(len(opti_cycles_sep)):
						ax[nrow][i].axvline(x=opti_cycles_sep[j], color=opti_cycles_sep_color)

				# best models scores
				ax[nrow][i].plot(id_best_all+1, iter_indep_scores[id_best_all,1+nb_constraints+nb_bonds+i], marker='D', color='palegreen', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)
			else:
				ax[nrow-1][i].set_visible(False)
				ax[nrow][i].set_visible(False)

	# dihedrals
	if nb_dihedrals != 0:
		nrow += 2
		y_max = np.max(iter_indep_scores[:,1+nb_constraints+nb_bonds+nb_angles:1+nb_constraints+nb_bonds+nb_angles+nb_dihedrals])
		for i in range(ncols):
			if i < nb_dihedrals:
				# value and force constant
				ax[nrow-1][i].set_title('Dihedral '+str(i+1)+' - Value & Force constant')
				ax[nrow-1][i].grid(zorder=0.5)
				color = 'tab:blue'
				ax[nrow-1][i].plot(x_evals, parameters_vals['dihedrals']['values'][i], color=color)
				if display_sim_crashes:
					ax[nrow-1][i].scatter(crashes_ids, np.array(parameters_vals['dihedrals']['values'][i])[crashes_ids-1], marker='x', color='black', zorder=2)
				ax[nrow-1][i].tick_params(axis='y', labelcolor=color)
				ax[nrow-1][i].set_xlim(x_min, x_max)
				ax2 = ax[nrow-1][i].twinx()
				color = 'tab:red'
				ax2.plot(x_evals, parameters_vals['dihedrals']['force_ctr'][i], color=color)
				if display_sim_crashes:
					ax2.scatter(crashes_ids, np.array(parameters_vals['dihedrals']['force_ctr'][i])[crashes_ids-1], marker='x', color='black', zorder=2)
				if display_opti_cycles_sep:
					for j in range(len(opti_cycles_sep)):
						ax[nrow-1][i].axvline(x=opti_cycles_sep[j], color=opti_cycles_sep_color)
				try:
					ax2.set_ylim(bottom=y_fct_range['dihedrals'][0]-(y_fct_range['dihedrals'][1]-y_fct_range['dihedrals'][0])*plt_sidespace, top=y_fct_range['dihedrals'][1]+(y_fct_range['dihedrals'][1]-y_fct_range['dihedrals'][0])*plt_sidespace)
				except ValueError:
					pass # TODO: modify this once we handle dihedrals properly
				if i == nb_dihedrals-1:
					ax2.tick_params(axis='y', labelcolor=color)
				else:
					ax2.yaxis.set_ticklabels([])

				# best models parameters
				if parameters_vals['dihedrals']['values'][i][id_best_all] != None:
					ax[nrow-1][i].plot(id_best_all+1, parameters_vals['dihedrals']['values'][i][id_best_all], marker='D', color='lightskyblue', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)
					ax2.plot(id_best_all+1, parameters_vals['dihedrals']['force_ctr'][i][id_best_all], marker='D', color='salmon', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)

				# independant score
				ax[nrow][i].set_title('Dihedral '+str(i+1)+' - Score')
				ax[nrow][i].plot(x_evals, iter_indep_scores[:,1+nb_constraints+nb_bonds+nb_angles+i], color=color_subscores)
				if display_sim_crashes:
					ax[nrow][i].scatter(crashes_ids, iter_indep_scores[:,1+nb_constraints+nb_bonds+nb_angles+i][crashes_ids-1], marker='x', color='black', zorder=2)
				ax[nrow][i].grid(zorder=0.5)
				ax[nrow][i].set_ylim(bottom=-y_max*plt_sidespace, top=y_max*(1+plt_sidespace))
				ax[nrow][i].set_xlim(x_min, x_max)
				if display_opti_cycles_sep:
					for j in range(len(opti_cycles_sep)):
						ax[nrow][i].axvline(x=opti_cycles_sep[j], color=opti_cycles_sep_color)

				# best models scores
				ax[nrow][i].plot(id_best_all+1, iter_indep_scores[id_best_all,1+nb_constraints+nb_bonds+nb_angles+i], marker='D', color='palegreen', markersize=10, markeredgewidth=1.5, markeredgecolor='black', zorder=3)
			else:
				ax[nrow-1][i].set_visible(False)
				ax[nrow][i].set_visible(False)

	plt.tight_layout()
	plt.savefig(ns.opti_dirname+'/'+ns.plot_filename)
	print()
	print('Wrote visual optimization summary file at location:\n ', os.path.normpath(ns.opti_dirname+'/'+ns.plot_filename))
	print()


def main():

	print(swarmcg.shared.styling.header_package(
		'                  Module: Optimization run analysis\n'))

	formatter = lambda prog: RawTextHelpFormatter(prog, width=135, max_help_position=52)
	args_parser = ArgumentParser(
		description=ANALYSE_DESCR,
		formatter_class=formatter,
		add_help=False,
		usage=SUPPRESS
	)

	args_header = swarmcg.shared.styling.sep_close + '\n|                                         ARGUMENTS                                           |\n' + swarmcg.shared.styling.sep_close
	bullet = ' '

	required_args = args_parser.add_argument_group(args_header + '\n\n' + bullet + 'INPUT/OUTPUT')
	required_args.add_argument('-opti_dir', dest='opti_dirname',
							   help='Directory created by module \'scg_optimize\' that contains all files\ngenerated during the optimization procedure',
							   type=str, metavar='')
	required_args.add_argument('-o', dest='plot_filename',
							   help='Filename for the output plot, produced in directory -opti_dir.\nExtension/format can be one of: eps, pdf, pgf, png, ps, raw, rgba,\nsvg, svgz',
							   type=str, default='opti_summary.png',
							   metavar='    (opti_summary.png)')

	optional_args = args_parser.add_argument_group(bullet + 'OTHERS')
	optional_args.add_argument('-plot_scale', dest='plot_scale', help='Scale factor of the plot',
							   type=float, default=1.0, metavar='        (1.0)')
	optional_args.add_argument('-h', '--help', help='Show this help message and exit',
							   action='help')

	# display help if script was called without arguments
	if len(sys.argv) == 1:
		args_parser.print_help()
		sys.exit()

	# arguments handling, display command line if help or no arguments provided
	ns = args_parser.parse_args()
	input_cmdline = ' '.join(map(cmd_quote, sys.argv))
	print('Working directory:', os.getcwd())
	print('Command line:', input_cmdline)
	print()
	print(swarmcg.shared.styling.sep_close)
	print(
		'| SUMMARIZING OPTIMIZATION PROCEDURE                                                          |')
	print(swarmcg.shared.styling.sep_close)
	print()

	run(ns)


if __name__ == "__main__":
	main()
