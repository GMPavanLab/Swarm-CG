import MDAnalysis as mda
import numpy as np

from swarmcg import config


def get_AA_bonds_distrib(ns, beads_ids, grp_type, grp_nb):
    """Calculate bonds distribution from AA trajectory.

    ns requires:
        aa2cg_universe
        mda_backend
        bw_constraints
        bw_bonds
        bonds_scaling
        bonds_scaling_specific
        bins_constraints
        bins_bonds

    ns creates:
        bonds_rescaling_performed
    """
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
        bond_values[len(beads_ids) * ts.frame:len(beads_ids) * (ts.frame + 1)] = frame_values / 10  # retrieved nm

    bond_avg_init = round(np.average(bond_values), 3)

    # NOTE: for rescaling we first take the average of the group, then we rescale
    #       this means if a bond group has a bimodal distribution, the rescale distribution is still bimodal

    # rescale all bonds length if argument -bonds_scaling is provided
    if ns.bonds_scaling != config.bonds_scaling:
        bond_values = [bond_length * ns.bonds_scaling for bond_length in bond_values]
        bond_avg_final = round(np.average(bond_values), 3)
        ns.bonds_rescaling_performed = True
        print('  Ref. AA-mapped distrib. rescaled to avg', bond_avg_final, 'nm for', grp_type, grp_nb + 1, '(initially',
              bond_avg_init, 'nm)')

    # or shift distributions for bonds that are too short for direct CG mapping (according to argument -min_bonds_length)
    elif bond_avg_init < ns.min_bonds_length:
        bond_rescale_factor = ns.min_bonds_length / bond_avg_init
        bond_values = [bond_length * bond_rescale_factor for bond_length in bond_values]
        bond_avg_final = round(np.average(bond_values), 3)
        ns.bonds_rescaling_performed = True
        print('  Ref. AA-mapped distrib. rescaled to avg', bond_avg_final, 'nm for', grp_type, grp_nb + 1, '(initially',
              bond_avg_init, 'nm)')

    # or if specific lengths were provided for constraints and/or bonds
    elif ns.bonds_scaling_specific is not None:

        if grp_type.startswith('constraint'):
            geom_id_full = f'C{grp_nb + 1}'
        elif grp_type.startswith('bond'):
            geom_id_full = f'B{grp_nb + 1}'
        else:
            # TODO: what should the code do here?
            pass

        if (geom_id_full.startswith('C') and geom_id_full in ns.bonds_scaling_specific) or (
                geom_id_full.startswith('B') and geom_id_full in ns.bonds_scaling_specific):
            bond_rescale_factor = ns.bonds_scaling_specific[geom_id_full] / bond_avg_init
            bond_values = [bond_length * bond_rescale_factor for bond_length in bond_values]
            bond_avg_final = round(np.average(bond_values), 3)
            ns.bonds_rescaling_performed = True
            print('  Ref. AA-mapped distrib. rescaled to avg', bond_avg_final, 'nm for', grp_type, grp_nb + 1,
                  '(initially', bond_avg_init, 'nm)')
        else:
            bond_avg_final = bond_avg_init

    else:
        bond_avg_final = bond_avg_init

    # or alternatively, do not rescale these bonds but add specific exclusion rules, OR JUST SUGGEST USER TO CHECK THIS
    # exclusions storage format: ns.cg_itp['exclusion'].append([int(bead_id)-1 for bead_id in sp_itp_line[0:2]])

    if grp_type.startswith('constraint'):
        bond_hist = np.histogram(bond_values, ns.bins_constraints, density=True)[
                        0] * ns.bw_constraints  # retrieve 1-sum densities
    elif grp_type.startswith('bond'):
        bond_hist = np.histogram(bond_values, ns.bins_bonds, density=True)[0] * ns.bw_bonds  # retrieve 1-sum densities
    else:
        # TODO: what should the code do here?
        pass

    return bond_avg_final, bond_hist, bond_values


def get_CG_bonds_distrib(ns, beads_ids, grp_type):
    """"Calculate bonds distribution from CG trajectory.

    ns requires:
        cg_universe
        bw_bonds
        bw_constraints
        bins_constraints
        bins_bonds
    """
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
        bond_hist = np.histogram(bond_values, ns.bins_constraints, density=True)[
                        0] * ns.bw_constraints  # retrieve 1-sum densities
    elif grp_type == 'bond':
        bond_hist = np.histogram(bond_values, ns.bins_bonds, density=True)[0] * ns.bw_bonds  # retrieve 1-sum densities
    else:
        # TODO: what should the code do here?
        pass

    return bond_avg, bond_hist, bond_values
