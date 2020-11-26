# general stuff
github_url = 'http://github.com/GMPavanLab/SwarmCG'
gmx_path = 'gmx'

# BI and FST-PSO OPTI, defaults
kB = 0.008314462
sim_temperature = 300  # Kelvin
bi_nb_bins = 50  # nb of bins to use for Boltzmann Inversion, will be doubled for dihedrals distributions binning during BI -- this has huge impact on the results of the BI and this value shall STAY AT 50 ! actually I did not try to modify much but this feels like dangerous atm
bonds_max_range = 15  # nm -- used to define grid for EMD calculations
# NOTE: increasing bonds_max_range increases computation time, but memory usage increases exponentially
# TODO: detect when a bond has longer values than bonds_max_range and suggest the user to raise the limit if he really
#       needs to, but I don't really see what kind of use case would require more than 5 nm (maybe super CG or elastic
#       net in proteins though)
bw_constraints = 0.002  # nm
bw_bonds = 0.01  # nm
bw_angles = 2.5  # degrees
bw_dihedrals = 2.5  # degrees
default_max_fct_bonds_bi = 17000
default_max_fct_bonds_opti = 18000
default_max_fct_angles_bi = 1200
default_max_fct_angles_opti_f1 = 1700
default_max_fct_angles_opti_f2 = 1700

default_abs_range_fct_dihedrals_bi_func_without_mult = 250
default_abs_range_fct_dihedrals_opti_func_without_mult = 1500
default_abs_range_fct_dihedrals_bi_func_with_mult = 3.5  # TODO: these limits for dihedrals are probably far too low but they guarantee we don't have all sims crashing at start
default_abs_range_fct_dihedrals_opti_func_with_mult = 15

bonds2angles_scoring_factor = 500  # multiplier applied to constraints/bonds EMD scores to retrieve angles/dihedrals mismatches that are comparable, for the opti scoring function
sim_crash_EMD_indep_score = 150  # when a simulation crashes or does not finish for any reason: EMD distance between 2 distributions, for 1 geom

# bonds scaling, default
bonds_scaling = 1.0  # ratio
min_bonds_length = 0.00  # nm
bonds_scaling_str = ''  # constraints and bonds ids + their required target AA-mapped distributions rescaled averages

# NOTE: in the benchmark of the paper we used bonds2_angles_scoring_factor = 500 and fct_guess_min_flat_diff_angles = 50,
#       fct_guess_min_flat_diff_dihedrals_without_mult = 0.50 and fct_guess_min_flat_diff_dihedrals_with_mult = 0.20
#       differences in the results should be very small with respect to the benchmark and otherwise an improvement

# building of the initial guesses for optimization, defaults
bond_dist_guess_variation = 0.025  # nm
angle_value_guess_variation = 10  # degrees
dihedral_value_guess_variation = 10  # degrees
fct_guess_min_flat_diff_bonds = 200  # flat minimum force constant variation that fct_guess_fact shall yield, used to find low and high boundaries for random generation of particles' force constants
fct_guess_min_flat_diff_angles = 100  # flat minimum force constant variation that fct_guess_fact shall yield, used to find low and high boundaries for random generation of particles' force constants
fct_guess_min_flat_diff_dihedrals_without_mult = 5  # flat minimum force constant variation that fct_guess_fact shall yield, used to find low and high boundaries for random generation of particles' force constants
fct_guess_min_flat_diff_dihedrals_with_mult = 1  # flat minimum force constant variation that fct_guess_fact shall yield, used to find low and high boundaries for random generation of particles' force constants

# gromacs functions that are properly treated at the moment
# if we find a function that is not handled, program will exit with an appropriate error message
# TODO: handle dihedral function 9 correctly so that different potentials can be stacked for the same beads
#       this is the primary purpose of function 9 !!
handled_functions = {
    'constraint': [1],  # tested and verified: 1
    'bond': [1],  # tested and verified: 1
    'angle': [1, 2],  # tested and verified: 1, 2
    'dihedral': [1, 2, 4],  # tested and verified: 1, 2, 4 -- ongoing: 9 (need to merge the 1+ dihedrals groups on plots)
    'virtual_sites2': [1],  # tested and verified: 1 -- ongoing: 2 (need GMX 2020)
    'virtual_sites3': [1, 2, 3, 4],  # tested and verified: 1, 2, 3, 4
    'virtual_sites4': [2],  # tested and verified: 2 -- irrelevant: 1
    'virtual_sitesn': [1, 2, 3]  # tested and verified: 1, 2, 3
}
dihedral_func_with_mult = [1, 4, 9]  # these functions use 3 parameters, the last one being multiplicity

# plots display parameters
use_hists = False  # hists are not implemented in a way that they will be displayed with left and right bold borders atm
line_alpha = 0.6  # line alpha for the density plots
fill_alpha = 0.30  # fill alpha for the density plots
cg_color = '#1f77b4'
atom_color = '#d62728'

# scripts inputs default filenames
metavar_aa_tpr = 'aa_topol.tpr'
metavar_aa_traj = 'aa_traj.xtc'
metavar_cg_map = 'cg_map.ndx'
metavar_cg_itp = 'cg_model.itp'
metavar_cg_tpr = 'cg_topol.tpr'
metavar_cg_traj = 'cg_traj.xtc'

# help descriptions for arguments
help_aa_tpr = 'Topology binary file of your reference AA simulation (TPR)'
help_aa_traj = 'Trajectory file of the reference AA simulation (XTC, TRR)\nPBC are handled internally if trajectory contains box dimensions'
help_cg_map = 'Mapping file of the atoms to CG beads (NDX-like file format)'
help_mapping_type = 'Center Of Mass (COM) or Center Of Geometry (COG), for\ninterpreting the mapping file'
help_verbose = 'Display more processing details & error traceback'
help_gmx_path = 'Your Gromacs alias/path'
help_bonds_scaling = 'Scaling factor for ALL AA-mapped bonds/constraints lengths\nOnly one of arguments -bonds_scaling, -bonds_scaling_str\nand -min_bonds_length can be provided'
help_min_bonds_length = 'Required minimum length of a bond or constraint between 2 CG\nbeads (distributions avg in nm), used both as:\n1. Threshold to identify ALL short AA-mapped bonds/constraints\n2. Target avg to rescale ALL those bonds/constraints'
help_bonds_scaling_str = 'String (use quotes) for providing SPECIFIC bonds/constraints\ngroups ids and their required lengths (nm, rescaled\ndistributions avg to use as target for optimization)\nEx: \'C1 0.23 B5 0.27\' will modify distributions of constraints\ngrp 1 and bonds grp 5 to averages 0.23 and 0.27 nm'
help_bonds2angles_scoring_factor = 'Weight of bonds vs. angles/dihedrals (constant C in the paper)\nAt 500, bonds mismatch 0.4 Å == angles/dihedrals mismatch 20°\nDecreasing would linearly increase the weight of bonds'
help_bw_constraints = 'Bandwidth for constraints distributions processing (nm)'
help_bw_bonds = 'Bandwidth for bonds distributions processing (nm)'
help_bw_angles = 'Bandwidth for angles distributions processing (degrees)'
help_bw_dihedrals = 'Bandwidth for dihedrals distributions processing (degrees)'
help_bonds_max_range = 'Max. range of grid for bonds/constraints distributions (nm)'
help_max_fct_bonds = 'Max. force constants for bonds function 1 (kJ.mol⁻¹.nm⁻²)'
help_max_fct_angles_f1 = 'Max. force ct. for angles function 1 (kJ.mol⁻¹.rad⁻²)'
help_max_fct_angles_f2 = 'Max. force ct. for angles function 2 (kJ.mol⁻¹)'
help_max_fct_dihedrals_with_mult = 'Max. force ct. for dihedrals functions 1, 4, 9 (abs. kJ.mol⁻¹)'
help_max_fct_dihedrals_without_mult = 'Max. force ct. for dihedrals function 2 (abs. kJ.mol⁻¹.rad⁻²)'

# optimization output filenames
input_sim_files_dirname = '.internal/input_CG_simulation_files'
iteration_sim_files_dirname = 'CG_sim_files'  # basename to be appended to with _NN
best_fitted_model_dirname = 'optimized_CG_model'
distrib_plots_all_evals_dirname = 'all_evals_distributions'
log_files_all_evals_dirname = 'all_evals_logs'
sim_files_all_evals_dirname = 'all_evals_all_sim_files'
opti_perf_recap_file = '.internal/opti_recap_evals_perfs_and_params.csv'
opti_pairwise_distances_file = '.internal/opti_recap_evals_pairwise_distribs_diffs.csv'
ref_distrib_plots = 'reference_AA_distributions.png'
best_distrib_plots = 'optimized_CG_model_distributions.png'







