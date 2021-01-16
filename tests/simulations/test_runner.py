import swarmcg.simulations.runner as sim


def test_generate_steps(ns_opt):
    # given:
    sim_steps = sim.generate_steps(ns_opt)

    # when:
    sim1 = next(sim_steps)

    # then:
    assert isinstance(sim1, "Minimisation")
