import numpy as np
import MDAnalysis as mda
from ..shared import exceptions

# All these functions for virtual sites definitions are explained
# in the GROMACS manual part 5.5.7 (page 379 in manual version 2020)
# Check also the bonded potentials table best viewed here:
# http://manual.gromacs.org/documentation/2020/reference-manual/topologies/topology-file-formats.html#tab-topfile2

# TODO: test all these functions


# Functions for virtual_sites2

# vs_2 func 1 -> Linear combination using 2 reference points
# weighted COG using a percentage in [0, 1]
# the weight is applied on the bead ID that comes first
def vs2_func_1(ns, traj, vs_def_beads_ids, vs_params):

    i, j = vs_def_beads_ids
    a = vs_params  # weight
    weights = np.array([1-a, a])

    for ts in ns.aa2cg_universe.trajectory:
        traj[ts.frame] = ns.aa2cg_universe.atoms[[i, j]].center(weights)


# vs_2 func 2 -> Linear combination using 2 reference points
# on the vector from i to j, at given distance (nm)
# NOTE: it seems this one exists only since GROMACS 2020
# TODO: check this one with a GMX 2020 installation
def vs2_func_2(ns, traj, vs_def_beads_ids, vs_params):

    i, j = vs_def_beads_ids
    a = vs_params  # nm
    a = a * 10  # retrieve amgstrom for MDA

    for ts in ns.aa2cg_universe.trajectory:
        pos_i = ns.aa2cg_universe.atoms[i].position
        pos_j = ns.aa2cg_universe.atoms[j].position
        r_ij = pos_j - pos_i
        traj[ts.frame] = pos_i + a * r_ij / mda.lib.mdamath.norm(r_ij)


# Functions for virtual_sites3

# vs_3 func 1 -> Linear combination using 3 reference points
# in the plane, using sum of vectors from i to j and from k to i
def vs3_func_1(ns, traj, vs_def_beads_ids, vs_params):

    i, j, k = vs_def_beads_ids
    a, b = vs_params  # nm, nm
    a, b = a * 10, b * 10  # retrieve amgstrom for MDA

    for ts in ns.aa2cg_universe.trajectory:
        pos_i = ns.aa2cg_universe.atoms[i].position
        pos_j = ns.aa2cg_universe.atoms[j].position
        pos_k = ns.aa2cg_universe.atoms[k].position
        r_ij = pos_j - pos_i
        r_ik = pos_k - pos_i
        traj[ts.frame] = pos_i + a * r_ij / mda.lib.mdamath.norm(r_ij) / 2 + b * r_ik / mda.lib.mdamath.norm(r_ik) / 2


# vs_3 func 2 -> Linear combination using 3 reference points
# in the plane, using WEIGHTS sum of vectors from j to i and from k to i + fixed distance
# I used their formula (hopefully) so the form differs from the explanation on line above, but it should be identical
def vs3_func_2(ns, traj, vs_def_beads_ids, vs_params):

    i, j, k = vs_def_beads_ids
    a, b = vs_params  # weight, nm
    b = b * 10  # retrieve amgstrom for MDA

    for ts in ns.aa2cg_universe.trajectory:
        pos_i = ns.aa2cg_universe.atoms[i].position
        pos_j = ns.aa2cg_universe.atoms[j].position
        pos_k = ns.aa2cg_universe.atoms[k].position
        r_ij = pos_j - pos_i
        r_jk = pos_k - pos_j
        comb_ijk = (1-a) * r_ij + a * r_jk
        traj[ts.frame] = pos_i + b * (comb_ijk / mda.lib.mdamath.norm(comb_ijk))


# vs_3 func 3 -> Linear combination using 3 reference points
# angle in the plane defined, at given distance of the 3rd point
def vs3_func_3(ns, traj, vs_def_beads_ids, vs_params):

    i, j, k = vs_def_beads_ids
    ang_deg, d = vs_params  # degrees, nm
    ang_rad = np.deg2rad(ang_deg)  # retrieve radians
    d = d * 10  # retrieve amgstrom for MDA

    for ts in ns.aa2cg_universe.trajectory:
        pos_i = ns.aa2cg_universe.atoms[i].position
        pos_j = ns.aa2cg_universe.atoms[j].position
        pos_k = ns.aa2cg_universe.atoms[k].position
        r_ij = pos_j - pos_i
        r_jk = pos_k - pos_j
        comb_ijk = r_jk - (np.dot(r_ij, r_jk) / np.dot(r_ij, r_ij)) * r_ij
        traj[ts.frame] = pos_i + d * np.cos(ang_rad) * (r_ij / mda.lib.mdamath.norm(r_ij)) + d * np.sin(ang_rad) * (comb_ijk / mda.lib.mdamath.norm(comb_ijk))


# vs_3 func 4 -> Linear combination using 3 reference points
# out of plane
def vs3_func_4(ns, traj, vs_def_beads_ids, vs_params):

    i, j, k = vs_def_beads_ids
    a, b, c = vs_params  # weight, weight, nm**(-1)
    c = c / 10  # retrieve amgstrom**(-1) for MDA

    for ts in ns.aa2cg_universe.trajectory:
        pos_i = ns.aa2cg_universe.atoms[i].position
        pos_j = ns.aa2cg_universe.atoms[j].position
        pos_k = ns.aa2cg_universe.atoms[k].position
        r_ij = pos_j - pos_i
        r_ik = pos_k - pos_i
        traj[ts.frame] = pos_i + a * r_ij + b * r_ik - c * (r_ij / mda.lib.mdamath.norm(r_ij) * r_ik / mda.lib.mdamath.norm(r_ik))


# Functions for virtual_sites4

# vs_4 func 2 -> Linear combination using 3 reference points
# NOTE: only function 2 is defined for vs_4 in GROMACS, because it replaces function 1
#       which still exists for retro compatibility but its usage must be avoided
def vs4_func_2(ns, traj, vs_def_beads_ids, vs_params):

    i, j, k, l = vs_def_beads_ids
    a, b, c = vs_params  # weight, weight, nm
    c = c * 10  # retrieve amgstrom for MDA

    for ts in ns.aa2cg_universe.trajectory:
        pos_i = ns.aa2cg_universe.atoms[i].position
        pos_j = ns.aa2cg_universe.atoms[j].position
        pos_k = ns.aa2cg_universe.atoms[k].position
        pos_l = ns.aa2cg_universe.atoms[l].position
        r_ij = pos_j - pos_i
        r_ik = pos_k - pos_i
        r_il = pos_l - pos_i
        r_ja = a * r_ik - r_ij
        r_jb = b * r_il - r_ij
        r_m = np.cross(r_ja, r_jb)
        traj[ts.frame] = pos_i - c * (r_m / mda.lib.mdamath.norm(r_m))


# Functions for virtual_sitesn

# vs_n func 1 -> Center of Geometry
def vsn_func_1(ns, traj, vs_def_beads_ids):

    for ts in ns.aa2cg_universe.trajectory:
        traj[ts.frame] = ns.aa2cg_universe.atoms[vs_def_beads_ids].center_of_geometry(pbc=None)


# vs_n func 2 -> Center of Mass
def vsn_func_2(ns, traj, vs_def_beads_ids, bead_id):

    # inform user if this VS definition uses beads (or VS) with mass 0,
    # because this is COM so 0 mass means a bead that was marked for defining the VS is in fact ignored
    zero_mass_beads_ids = []
    for bid in vs_def_beads_ids:
        if bid in ns.cg_itp['virtual_sitesn']:
            if ns.cg_itp['virtual_sitesn'][bid]['mass'] == 0:
                zero_mass_beads_ids.append(bid)
    if len(zero_mass_beads_ids) > 0:
        print('  WARNING: Virtual site ID {} uses function 2 for COM, but its definition contains IDs ' + ' '.join(zero_mass_beads_ids) + 'which have no mass'.format(bead_id + 1))

    for ts in ns.aa2cg_universe.trajectory:
        traj[ts.frame] = ns.aa2cg_universe.atoms[vs_def_beads_ids].center_of_mass(pbc=None)


# vs_n func 3 -> Center of Weights (each atom has a given weight, pairwise formatting: id1 w1 id2 w2 ..)
def vsn_func_3(ns, traj, vs_def_beads_ids, vs_params):

    masses_and_weights = np.array([ns.aa2cg_universe.atoms[vs_def_beads_ids[i]].mass * vs_params[i] for i in range(len(vs_def_beads_ids))])
    for ts in ns.aa2cg_universe.trajectory:
        traj[ts.frame] = ns.aa2cg_universe.atoms[vs_def_beads_ids].center(masses_and_weights)






