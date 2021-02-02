import MDAnalysis as mda
import numpy as np


def get_AA_angles_distrib(ns, beads_ids):
    """Calculate angles distribution from AA trajectory.

    ns requires:
        aa2cg_universe
        mda_backend
        bw_angles
    """
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


def get_CG_angles_distrib(ns, beads_ids):
    """Calculate angles using MDAnalysis.

    ns requires:
        cg_universe
        mda_backend
        bw_angles
    """
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