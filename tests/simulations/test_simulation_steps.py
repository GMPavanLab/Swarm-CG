import os

import pytest

from swarmcg.shared import exceptions
from swarmcg.simulations.simulation_steps import Minimisation, Equilibration, Production


class BaseTest:

    def cleanup(self, delete_files):
        if isinstance(delete_files, str):
            delete_files = [delete_files]
        for f in delete_files:
            os.remove(f)


class TestMinimisation(BaseTest):

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

    def test_modify_mdp(self):
        # given:
        mini = Minimisation(TestEquilibration.filename)

        # then:
        assert mini.sim_setup["nsteps"] == 10000

        # when:
        mini.modify_mdp(sim_time=10)

        # then:
        assert mini.sim_setup["nsteps"] == 10000

        # when:
        mini.to_file(".")

        # then:
        mini = Minimisation("./equi.mdp")
        assert mini.sim_setup["nsteps"] == 10000
        self.cleanup("./equi.mdp")


class TestEquilibration(BaseTest):

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

    def test_modify_mdp(self):
        # given:
        equi = Equilibration(TestEquilibration.filename)

        # then:
        assert equi.sim_setup["nsteps"] == 10000

        # when:
        equi.modify_mdp(sim_time=10)

        # then:
        assert equi.sim_setup["nsteps"] == 10000

        # when:
        equi.to_file(".")

        # then:
        equi = Production("./equi.mdp")
        assert equi.sim_setup["nsteps"] == 10000
        self.cleanup("./equi.mdp")


class TestProduction(BaseTest):

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

    def test_modify_mdp(self):
        # given:
        prod = Production(TestProduction.filename)

        # then:
        assert prod.sim_setup["nsteps"] == 15000

        # when:
        prod.modify_mdp(sim_time=10)

        # then:
        assert prod.sim_setup["nsteps"] == 10 * 1000 / 0.02

        # when:
        prod.to_file(".")

        # then:
        prod = Production("./md.mdp")
        assert prod.sim_setup["nsteps"] == 10 * 1000 / 0.02
        self.cleanup("./md.mdp")



