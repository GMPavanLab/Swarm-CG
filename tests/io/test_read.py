import os

import pytest

from swarmcg.shared import exceptions
from swarmcg.io.read import read_itp, read_cg_itp_file, validate_cg_itp


required_itp_fields = ["real_beads_ids", "vs_beads_ids", "nb_bonds", "nb_angles",
                       "nb_dihedrals", "nb_constraints",
                       "moleculetype", "atoms", "constraint", "bond", "angle", "dihedral",
                       "virtual_sites2", "virtual_sites3", "virtual_sites4", "virtual_sitesn",
                       "exclusion"]


def check_ipt_dict(cg_itp):
    assert len(cg_itp["bond"]) == 23
    assert len(cg_itp["bond"]) == cg_itp["nb_bonds"]
    assert len(cg_itp["angle"]) == 6
    assert len(cg_itp["angle"]) == cg_itp["nb_angles"]
    assert len(cg_itp["dihedral"]) == 1
    assert len(cg_itp["dihedral"]) == cg_itp["nb_dihedrals"]
    assert len(cg_itp["constraint"]) == 0
    assert len(cg_itp["virtual_sites2"]) == 1


def test_read_itp():
    # when:
    filename = f"tests/data/test.itp"

    # then:
    _ = read_itp(filename)

    # when:
    filename = "missing"

    # then:
    with pytest.raises(FileNotFoundError):
        _ = read_itp(filename)


def test_read_cg_itp_file(ns_opt):
    # when:
    filename = f"tests/data/test.itp"
    ns = ns_opt(cg_itp_filename=filename)

    # then:
    result = read_cg_itp_file(ns)
    assert isinstance(result, dict)
    assert all([field in result for field in required_itp_fields])
    check_ipt_dict(result)

    # when:
    all_beads = []

    # then:
    with pytest.raises(exceptions.MissformattedFile):
        _ = validate_cg_itp(result, all_beads=all_beads)

    # when:
    all_beads = list(range(26))

    # then:
    _ = validate_cg_itp(result, all_beads=all_beads)


def test_read_cg_itp_file_basic(ns_opt):
    # when:
    filename = f"tests/data/cg_model.itp"
    ns = ns_opt(cg_itp_filename=filename)

    # then:
    result = read_cg_itp_file(ns)
    assert len(result["bond"]) == 4
    assert len(result["bond"]) == result["nb_bonds"]
    assert len(result["angle"]) == 5
    assert len(result["angle"]) == result["nb_angles"]
    assert result["nb_dihedrals"] == 0
    assert result["nb_constraints"] == 0
    assert all([field in result for field in required_itp_fields])