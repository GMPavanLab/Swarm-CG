from swarmcg.io.input import BaseInput


class TestBaseInput:

    def test_init(self, ns_opt):
        # given:
        ns = ns_opt()

        # then:
        _ = BaseInput(namespace=ns)

    def test__get_basename(self, ns_opt):
        # given:
        ns = ns_opt(mdp_equi_filename=__file__)

        # when:
        baseinput = BaseInput(namespace=ns)

        # then:
        assert "test_input.py" == baseinput._get_basename("mdp_equi_filename")
        assert "md.mdp" == baseinput._get_basename("mdp_md_filename")

    def test__file_path_exists(self, ns_opt):
        # given:
        ns = ns_opt(mdp_equi_filename="non_existing_file.txt")

        # when:
        baseinput = BaseInput(namespace=ns)

        # then:
        assert not baseinput._file_path_exists("mdp_equi_filename")
        assert baseinput._file_path_exists("mdp_md_filename")