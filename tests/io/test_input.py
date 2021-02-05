from swarmcg.io.input import BaseInput, OptInput


class TestBaseInput:

    def test_init(self, ns_opt):
        # given:
        ns = ns_opt()

        # then:
        _ = BaseInput(namespace=ns)

    def test_vars(self, ns_opt):
        # given:
        ns = ns_opt()

        # when:
        base_input = BaseInput(namespace=ns)

        # then:
        assert "aa_tpr_filename" in base_input.vars

    def test__get_basename(self, ns_opt):
        # given:
        ns = ns_opt(mdp_equi_filename=__file__)

        # when:
        base_input = BaseInput(namespace=ns)

        # then:
        assert "test_input.py" == base_input._get_basename("mdp_equi_filename")
        assert "md.mdp" == base_input._get_basename("mdp_md_filename")

    def test__file_path_exists(self, ns_opt):
        # given:
        ns = ns_opt(mdp_equi_filename="non_existing_file.txt")

        # when:
        base_input = BaseInput(namespace=ns)

        # then:
        assert not base_input._file_path_exists("mdp_equi_filename")
        assert base_input._file_path_exists("mdp_md_filename")


class TestOptInput:

    def test_init(self, ns_opt):
        # given:
        ns = ns_opt()

        # then:
        _ = OptInput(namespace=ns)