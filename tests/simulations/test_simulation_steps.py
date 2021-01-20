import pytest

from swarmcg.shared import exceptions
from swarmcg.simulations.simulation_steps import Minimisation, Equilibration, Production


class TestMinimisation:

    filename = "./tests/data/mini.mdp"

    def test_init(self):
        Minimisation(TestMinimisation.filename)

    def test_init_file_not_found(self):
        # given:
        filename = "not_existing_file.mdp"

        # then:
        with pytest.raises(exceptions.MissingMdpFile):
            Minimisation(filename)

    def test_to_string(self):
        # given
        mini = Minimisation(TestMinimisation.filename)

        # then:
        assert isinstance(mini.to_string(), str)

    def test_read_mdp(self):
        assert isinstance(Minimisation.read_mdp(TestMinimisation.filename), dict)


class TestEquilibration:

    filename = "./tests/data/equi.mdp"

    def test_init(self):
        Equilibration(TestEquilibration.filename)

    def test_init_file_not_found(self):
        # given:
        filename = "not_existing_file.mdp"

        # then:
        with pytest.raises(exceptions.MissingMdpFile):
            Equilibration(filename)

    def test_to_string(self):
        # given:
        equi = Equilibration(TestEquilibration.filename)

        # then:
        assert isinstance(equi.to_string(), str)

    def test_read_mdp(self):
        assert isinstance(Equilibration.read_mdp(TestEquilibration.filename), dict)


class TestProduction:

    filename = "./tests/data/md.mdp"

    def test_init(self):
        Production(TestProduction.filename)

    def test_init_file_not_found(self):
        # given:
        filename = "not_existing_file.mdp"

        # then:
        with pytest.raises(exceptions.MissingMdpFile):
            Production(filename)

    def test_to_string(self):
        prod = Production(TestProduction.filename)

        # then:
        assert isinstance(prod.to_string(), str)

    def test_read_mdp(self):
        assert isinstance(Production.read_mdp(TestProduction.filename), dict)