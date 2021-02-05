import os, sys
from shlex import quote as cmd_quote

import numpy as np
import matplotlib

import swarmcg.shared.styling
import swarmcg.io as io
import swarmcg.scoring as scores
from swarmcg import swarmCG as scg
from swarmcg import config
from swarmcg.shared import catch_warnings, input_parameter_validation

matplotlib.use("AGG")  # use the Anti-Grain Geometry non-interactive backend suited for scripted PNG creation


@catch_warnings(np.VisibleDeprecationWarning)  # filter MDAnalysis + numpy deprecation stuff that is annoying
def run(ns):
    print()
    print(swarmcg.shared.styling.sep_close)
    print("| PRE-PROCESSING                                                                              |")
    print(swarmcg.shared.styling.sep_close)
    print()

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
    ns.mda_backend = "serial"  # actually serial is faster because MDA is not properly parallelized atm

    # TODO: this eventually will need to be taked out of this function when we can avoid adding new attributed to ns
    ns.mapping_type = ns.mapping_type.upper()
    input_parameter_validation(ns, config)

    # display parameters for function compare_models
    if not os.path.isfile(ns.cg_tpr_filename) or not os.path.isfile(ns.cg_traj_filename):
        # switch to atomistic mapping inspection exclusively (= do NOT plot the CG distributions)
        print("Could not find file(s) for either CG topology or trajectory")
        print("  Going for inspection of AA-mapped distributions exclusively")
        print()
        ns.atom_only = True
    else:
        ns.atom_only = False

    try:
        if not ns.plot_filename.split(".")[-1] in ["eps", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz"]:
            ns.plot_filename = ns.plot_filename + ".png"
    except IndexError as e:
        ns.plot_filename = ns.plot_filename + ".png"

    scores.create_bins_and_dist_matrices(ns)  # bins for EMD calculations
    scg.read_ndx_atoms2beads(ns)  # read mapping, get atoms accurences in beads
    scg.get_atoms_weights_in_beads(ns)  # get weights of atoms within beads

    ns.cg_itp = io.read_cg_itp_file(ns)  # load the ITP object and find out geoms grouping
    io.validate_cg_itp(ns.cg_itp)  # check ITP object is correct
    scg.process_scaling_str(ns)  # process the bonds scaling specified by user

    print()
    io.read_aa_traj(ns)  # create universe and read traj
    scg.load_aa_data(ns)  # read atoms attributes
    scg.make_aa_traj_whole_for_selected_mols(ns)

    # for each CG bead, create atom groups for trajectory geoms calculation using mass and atom weights across beads
    scg.get_beads_MDA_atomgroups(ns)

    print("\nMapping the trajectory from AA to CG representation")
    ns.aa2cg_universe = scg.initialize_cg_traj(ns.cg_itp)
    scg.map_aa2cg_traj(ns)
    print()

    scg.compare_models(ns, manual_mode=True, calc_sasa=False)


def main():
    args_parser = io.get_evaluate_args()

    # arguments handling, display command line if help or no arguments provided
    ns = args_parser.parse_args()
    input_cmdline = " ".join(map(cmd_quote, sys.argv))
    print("Working directory:", os.getcwd())
    print("Command line:", input_cmdline)

    run(ns)


if __name__ == "__main__":
    main()
