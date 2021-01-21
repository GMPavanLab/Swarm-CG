import pytest

from swarmcg.shared import exceptions
from swarmcg.shared.validation import _file_validation, _optimisation_input_validation


def test___file_validation(ns_opt):
    # All file are missing, here are added one by one so we verify the sequence
    # of errors triggered is correct
    # when:
    ns = ns_opt()

    # then:
    with pytest.raises(exceptions.MissingCoordinateFile):
        _file_validation(ns)

    # when:
    filename = "./G1_DATA/aa_topol.tpr"
    ns.aa_tpr_filename = filename

    # then:
    with pytest.raises(exceptions.MissingTrajectoryFile):
        _file_validation(ns)

    # when:
    filename = "./G1_DATA/aa_traj.xtc"
    ns.aa_traj_filename = filename

    # then:
    with pytest.raises(exceptions.MissingIndexFile):
        _file_validation(ns)

   # when:
    filename = "./G1_DATA/cg_map.ndx"
    ns.cg_map_filename = filename

    # then:
    with pytest.raises(exceptions.MissingItpFile):
        _file_validation(ns)

   # when:
    filename = "./G1_DATA/cg_model.itp"
    ns.cg_itp_filename = filename

    # then:
    _file_validation(ns)


def test__optimisation_input_validation(ns_opt):
    # when:
    ns = ns_opt()

    # then:
    _optimisation_input_validation(ns)

    # when
    ns = ns_opt(default_max_fct_bonds_opti=-1)

    # then:
    with pytest.raises(exceptions.InputArgumentError):
        _optimisation_input_validation(ns)

    # when
    ns = ns_opt(default_max_fct_angles_opti_f1=-1)

    # then:
    with pytest.raises(exceptions.InputArgumentError):
        _optimisation_input_validation(ns)

    # when
    ns = ns_opt(default_max_fct_angles_opti_f2=-1)

    # then:
    with pytest.raises(exceptions.InputArgumentError):
        _optimisation_input_validation(ns)

