from swarmcg.simulations.runner import generate_steps, SimulationStep


def test_generate_steps(ns_opt):
    # given:
    sim_steps = generate_steps(ns_opt(gro_input_basename="./test/data/start_conf.gro"))

    # when:
    sim1 = next(sim_steps)

    # then:
    assert sim1.step_name == "minimisation"

    # when:
    sim2 = next(sim_steps)

    # then:
    assert sim2.step_name == "equilibration"

    # when:
    sim3 = next(sim_steps)

    # then:
    assert sim3.step_name == "production"


class TestSimulationStep:

    def test_init(self, simstep_mini):
        SimulationStep(simstep_mini)

    def test__prepare_cmd_mini(self, simstep_mini):
        # given:
        step = SimulationStep(simstep_mini)

        # when:
        command = step._prepare_cmd()

        # then:
        expected = "gmx grompp -c tests/data/start_conf.gro -f tests/data/mini.mdp -p tests/data/system.top -o mini -maxwarn 0"
        assert command == expected

    def test__prepare_cmd_equi(self, simstep_equi):
        # given:
        step = SimulationStep(simstep_equi)

        # when:
        command = step._prepare_cmd()

        # then:
        expected = "gmx grompp -c tests/data/mini.gro -f tests/data/equi.mdp -p tests/data/system.top -o equi -maxwarn 0"
        assert command == expected

    def test__prepare_cmd_md(self, simstep_md):
        # given:
        step = SimulationStep(simstep_md)

        # when:
        command = step._prepare_cmd()

        # then:
        expected = "gmx grompp -c tests/data/equi.gro -f tests/data/md.mdp -p tests/data/system.top -o md -maxwarn 0"
        assert command == expected

    def test__run_cmd(self, simstep_equi):
        # given:
        step = SimulationStep(simstep_equi)

        # when:
        command = step._run_cmd()

        # then:
        expected = "gmx mdrun -deffnm equi"
        assert command == expected

        # when:
        step.sim_setup["nb_threads"] = 1
        command = step._run_cmd()

        # then:
        expected = "gmx mdrun -deffnm equi -nt 1"
        print(step.sim_setup, command)
        assert command == expected

        # when:
        step.sim_setup["mpi_tasks"] = 1
        command = step._run_cmd(mpi=True)

        # then:
        expected = "mpirun -np 1 gmx mdrun -deffnm equi -nt 1"
        print(step.sim_setup, command)
        assert command == expected