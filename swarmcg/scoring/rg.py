import numpy as np


def compute_Rg(ns, traj_type):
    """Compute average radius of gyration.

    ns requires:
        aa_universe
        aa2cg_universe
        cg_universe
        mda_backend

    ns creates:
        gyr_aa
        gyr_aa_std
        gyr_aa_mapped
        gyr_aa_mapped_std
        gyr_cg
        gyr_cg_std
    """
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