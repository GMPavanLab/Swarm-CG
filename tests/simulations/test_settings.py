import pytest

from swarmcg.simulations import get_settings


def test_get_settings_fail(ns_opt):
    # when:
    ns = ns_opt(sim_type="NO_VALID")
    print(ns.sim_type)

    # then:
    with pytest.raises(ValueError):
        _ = get_settings(ns)


def test_get_settings_optimal(ns_opt):
    # when:
    ns = ns_opt(sim_type="OPTIMAL", cg_itp={"nb_constraints": 2, "nb_bonds": 2, "nb_angles": 2, "nb_dihedrals": 2})

    # then:
    sim_types, opti_cycles, sim_cycles, particle_setter = get_settings(ns)

    # then:
    assert sim_cycles == [0, 1, 2]
    assert particle_setter([1, 2, 3, 4]) == 4
    assert opti_cycles == [["constraint", "bond", "angle"], ["angle", "dihedral"], ["constraint", "bond", "angle", "dihedral"]]


def test_get_settings_test(ns_opt):
    # when:
    ns = ns_opt(sim_type="TEST")

    # then:
    sim_types, opti_cycles, sim_cycles, particle_setter = get_settings(ns)

    # then:
    assert sim_cycles == [0, 1, 2]
    assert particle_setter([1, 2, 3, 4]) == 2
    assert particle_setter(list(range(50))) == 2
    assert opti_cycles == [["constraint", "bond", "angle"], ["dihedral"], ["constraint", "bond", "angle", "dihedral"]]

