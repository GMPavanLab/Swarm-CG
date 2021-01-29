from types import SimpleNamespace

from swarmcg import config


class BaseField(SimpleNamespace):

    @property
    def metavar(self):
        if getattr(self, "default") is not None:
            return f"({str(self.default)})".rjust(25, " ")
        else:
            return ""

    @property
    def args(self):
        return {**vars(self), "metavar": self.metavar}


# EXECUTION MODE
exec_mode = BaseField(
    dest="exec_mode",
    type=int,
    default=1,
    help="MODE 1: Tune both bonds/angles/dihedrals equilibrium values\n        and their force constants\nMODE 2: Tune only bonds/angles/dihedrals force constants\n        with FIXED equilibrium values from the prelim. CG ITP",
)
sim_type = BaseField(
    dest="sim_type",
    type=str,
    default="OPTIMAL",
    help="Simulation type setting",
)

# GROMACS SETTINGS
gmx = BaseField(
    dest="gmx_path",
    type=str,
    default="gmx",
    help="Your Gromacs alias/path",
)
nt = BaseField(
    dest="nb_threads",
    type=int,
    default=0,
    help="Nb of threads to use, forwarded to 'gmx mdrun -nt'",
)
mpi = BaseField(
    dest="mpi_tasks",
    type=int,
    default=0,
    help="Nb of mpi programs (X), triggers 'mpirun -np X gmx'",
)
gpu_id = BaseField(
    dest="gpu_id",
    type=str,
    default="",
    help="String (use quotes) space-separated list of GPU device IDs",
)
gmx_args_str = BaseField(
    dest="gmx_args_str",
    type=str,
    default="",
    help="String (use quotes) of arguments to forward to gmx mdrun\nIf provided, arguments -nt and -gpu_id are ignored",
)
mini_maxwarn = BaseField(
    dest="mini_maxwarn",
    type=int,
    default=1,
    help="Max. number of warnings to ignore, forwarded to gmx\ngrompp -maxwarn at each minimization step",
)
sim_kill_delay = BaseField(
    dest="sim_kill_delay",
    type=int,
    default=60,
    help="Time (s) after which to kill a simulation that has not been\nwriting into its log file, in case a simulation gets stuck",
)

# REFERENCE AA MODEL
aa_tpr = BaseField(
    dest="aa_tpr_filename",
    type=str,
    default="aa_topol.tpr",
    help="Topology binary file of your reference AA simulation (TPR)",
)
aa_traj = BaseField(
    dest="aa_traj_filename",
    type=str,
    default="aa_traj.xtc",
    help="Trajectory file of the reference AA simulation (XTC, TRR)\nPBC are handled internally if trajectory contains box dimensions",
)
cg_map = BaseField(
    dest="cg_map_filename",
    type=str,
    default="cg_map.ndx",
    help="Mapping file of the atoms to CG beads (NDX-like file format)",
)
mapping = BaseField(
    dest="mapping_type",
    type=str,
    default="COM",
    help="Center Of Mass (COM) or Center Of Geometry (COG), for\ninterpreting the mapping file",
)

# CG MODEL
cg_itp = BaseField(
    dest="cg_itp_filename",
    type=str,
    default="cg_model.itp",
    help="ITP file of the CG model to optimize",
)
user_params = BaseField(
    dest="user_input",
    default=False,
    help="If absent, only the BI is used as starting point for parametrization\nIf present, parameters in the input ITP files are considered",
    action="store_true"
)
cg_gro = BaseField(
    dest="gro_input_filename",
    type=str,
    default="start_conf.gro",
    help="Starting GRO file used for iterative simulation\nWill be minimized and relaxed before each MD run",
)
cg_top = BaseField(
    dest="top_input_filename",
    type=str,
    default="system.top",
    help="TOP file used for iterative simulation",
)
cg_tpr = BaseField(
    dest="cg_tpr_filename",
    type=str,
    default="cg_topol.tpr",
    help="TPR file of your CG simulation (omit for solo AA inspection)",
),
cg_traj = BaseField(
    dest="cg_traj_filename",
    type=str,
    default="cg_traj.xtc",
    help="XTC file of your CG trajectory (omit for solo AA inspection)",
),
cg_mdp_mini = BaseField(
    dest="mdp_minimization_filename",
    type=str,
    default="mini.mdp",
    help="MDP file used for minimization runs",
)
cg_mdp_equi = BaseField(
    dest="mdp_equi_filename",
    type=str,
    default="equi.mdp",
    help="MDP file used for equilibration runs",
)
cg_mdp_md = BaseField(
    dest="mdp_md_filename",
    type=str,
    default="md.mdp",
    help="MDP file used for the MD runs analyzed for optimization",
)

# FILES HANDLING
in_dir = BaseField(
    dest="input_folder",
    type=str,
    default="",
    help="Additional prefix path used to find argument-provided files\nIf ambiguous, files found without prefix are preferred",
)
out_dir = BaseField(
    dest="output_folder",
    type=str,
    default="",
    help="Directory where to store all outputs of this program\nDefault -out_dir is named after timestamp",
)
opti_dir=BaseField(
    dest="opti_dirname",
    type=str,
    help="Directory created by module 'scg_optimize' that contains all files\ngenerated during the optimization procedure",
),
o_an = BaseField(
    dest="plot_filename",
    type=str,
    default="opti_summary.png",
    help="Filename for the output plot, produced in directory -opti_dir.\nExtension/format can be one of: eps, pdf, pgf, png, ps, raw, rgba,\nsvg, svgz",
),
o_ev = BaseField(
    dest="plot_filename",
    type=str,
    default="distributions.png",
    help="Filename for the output plot (extension/format can be one of:\neps, pdf, pgf, png, ps, raw, rgba, svg, svgz)",
)

# CG MODEL FORCE CONSTANTS
max_fct_bonds_f1 = BaseField(
    dest="default_max_fct_bonds_opti",
    type=float,
    default=18000,
    help="Max. force constants for bonds function 1 (kJ.mol⁻¹.nm⁻²)",
)
max_fct_angles_f1 = BaseField(
    dest="default_max_fct_angles_opti_f1",
    type=float,
    default=1700,
    help="Max. force ct. for angles function 1 (kJ.mol⁻¹.rad⁻²)",
)
max_fct_angles_f2 = BaseField(
    dest="default_max_fct_angles_opti_f2",
    type=float,
    default=1700,
    help="Max. force ct. for angles function 2 (kJ.mol⁻¹)",
)
max_fct_dihedrals_f149 = BaseField(
    dest="default_abs_range_fct_dihedrals_opti_func_with_mult",
    type=float,
    default=15,
    help="Max. force ct. for dihedrals functions 1, 4, 9 (abs. kJ.mol⁻¹)",
)
max_fct_dihedrals_f2 = BaseField(
    dest="default_abs_range_fct_dihedrals_opti_func_without_mult",
    type=float,
    default=1500,
    help="Max. force ct. for dihedrals function 2 (abs. kJ.mol⁻¹.rad⁻²)",
)
# MODEL SCORING
cg_time_short = BaseField(
    dest="sim_duration_short",
    type=float,
    default=10.0,
    help="Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycles 1 and 2, this will modify MDP file for the MD runs",
)
cg_time_long = BaseField(
    dest="sim_duration_long",
    type=float,
    default=25.0,
    help="Simulation time (ns) of the MD runs analyzed for optimization\nIn opti. cycle 3, this will modify MDP file for the MD runs",
),
b2a_score_fact = BaseField(
    dest="bonds2angles_scoring_factor",
    type=float,
    default=500.0,  # multiplier applied to constraints/bonds EMD scores to retrieve angles/dihedrals mismatches that are comparable, for the opti scoring function
    help="Weight of bonds vs. angles/dihedrals (constant C in the paper)\nAt 500, bonds mismatch 0.4 Å == angles/dihedrals mismatch 20°\nDecreasing would linearly increase the weight of bonds",
),
bw_constraints = BaseField(
    dest="bw_constraints",
    type=float,
    default=0.002,  # nm
    help="Bandwidth for constraints distributions processing (nm)",
),
bw_bonds=BaseField(
    dest="bw_bonds",
    type=float,
    default=0.01,  # nm
    help="Bandwidth for bonds distributions processing (nm)",
),
bw_angles=BaseField(
    dest="bw_angles",
    type=float,
    default=2.5,  # degrees
    help="Bandwidth for angles distributions processing (degrees)",
),
bw_dihedrals=BaseField(
    dest="bw_dihedrals",
    type=float,
    default=2.5,  # degrees
    help="Bandwidth for dihedrals distributions processing (degrees)",
),
bonds_max_range=BaseField(
    dest="",
    type="",
    default=15,
    help="Max. range of grid for bonds/constraints distributions (nm)",
),
disable_x_scaling = BaseField(
    dest="row_x_scaling",
    default=True,
    help="Disable auto-scaling of X axis across each row of the plot",
    action="store_false",
),
disable_y_scaling = BaseField(
    dest="row_y_scaling",
    default=True,
    help="Disable auto-scaling of Y axis across each row of the plot",
    action="store_false",
),
bonds_max_range = BaseField(
    dest="bonded_max_range",
    type=float,
    default=15,  # nm 15 -- used to define grid for EMD calculations
    help="Max. range of grid for bonds/constraints distributions (nm)",
),


# MODEL SCALING
aa_rg_offset=BaseField(
    dest="aa_rg_offset",
    type=float,
    default=0.0,
    help="Radius of gyration offset (nm) to be applied to AA data\naccording to your potential bonds rescaling (for display only)",
)
bonds_scaling = BaseField(
    dest="bonds_scaling",
    type=float,
    default=config.bonds_scaling,
    help="Scaling factor for ALL AA-mapped bonds/constraints lengths\nOnly one of arguments -bonds_scaling, -bonds_scaling_str\nand -min_bonds_length can be provided",
)
bonds_scaling_str = BaseField(
    dest="bonds_scaling_str",
    type=str,
    default=config.bonds_scaling_str,
    # constraints and bonds ids + their required target AA-mapped distributions rescaled averages
    help="String (use quotes) for providing SPECIFIC bonds/constraints\ngroups ids and their required lengths (nm, rescaled\ndistributions avg to use as target for optimization)\nEx: \'C1 0.23 B5 0.27\' will modify distributions of constraints\ngrp 1 and bonds grp 5 to averages 0.23 and 0.27 nm",
)
min_bonds_length = BaseField(
    dest="min_bonds_length",
    type=float,
    default=config.min_bonds_length,
    help="Required minimum length of a bond or constraint between 2 CG\nbeads (distributions avg in nm), used both as:\n1. Threshold to identify ALL short AA-mapped bonds/constraints\n2. Target avg to rescale ALL those bonds/constraints",
)

# FIGURE DISPLAY
mismatch_ordering = BaseField(
    dest="mismatch_order",
    default=False,
    help="Enables ordering of bonds/angles/dihedrals by mismatch score\nbetween pairwise AA-mapped/CG distributions (can help diagnosis)",
    action="store_true",
),
ncols = BaseField(
    dest="ncols_max",
    type=int,
    default=0,
    help="Max. nb of columns displayed in figure",
    action="",
),  # TODO: make this a line return in plot instead of ignoring groups
plot_scale=BaseField(
    dest="plot_scale",
    type=float,
    default=1.0,
    help="Scale factor of the plot",
),

# OTHERS
temp = BaseField(
    dest="temp",
    type=float,
    default=300,  # Kelvin
    help="Temperature used to perform Boltzmann inversion (K)",
)
keep_all_sims = BaseField(
    dest="keep_all_sims",
    default=False,
    help="Store all gmx files for all simulations, may use disk space",
    action="store_true",
)
verbose = BaseField(
    dest="verbose",
    default=False,
    help="Display more processing details & error traceback",
    action="store_true"
)
help = BaseField(
    help="Show this help message and exit",
    action="help"
)
