import pytest

from swarmcg.io.input import BaseInput, OptInput


class TestBaseInput:

    def test_init(self, ns_opt):
        # given:
        ns = ns_opt()

        # then:
        _ = BaseInput(**vars(ns))

    def test__get_basename(self, ns_opt):
        # given:
        ns = ns_opt(mdp_equi_filename=__file__)

        # when:
        base_input = BaseInput(**vars(ns))

        # then:
        assert "test_input.py" == base_input._get_basename("mdp_equi_filename")
        assert "md.mdp" == base_input._get_basename("mdp_md_filename")

    def test__file_path_exists(self, ns_opt):
        assert BaseInput._file_path_exists("./tests/data/aa_topol.tpr")
        assert not BaseInput._file_path_exists("non_existing_file.txt")


class TestOptInput:

    def test_init(self, ns_opt):
        # given:
        ns = ns_opt()

        # then:
        _ = OptInput(**vars(ns))

    def test_attributes(self, ns_opt):
        # given:
        ns = ns_opt()

        # when:
        opt_input = OptInput(**vars(ns))

        # then:
        assert opt_input.cg_itp_basename == "cg_model.itp"

    def test_input_files(self, ns_opt):
        # given:
        ns = ns_opt()

        # thwn:
        expected = ["./tests/data/aa_topol.tpr", "./tests/data/aa_traj.xtc", "./tests/data/cg_map.ndx"]
        assert expected == OptInput(**vars(ns)).simulation_filenames(False)[:3]

        # given:
        ns = ns_opt(mdp_equi_filename="non_existing_file.txt")

        # when:
        with pytest.raises(FileNotFoundError):
            _ = OptInput(**vars(ns)).simulation_filenames()

