from swarmcg.io.job_args import defaults


def test_defaults():
    assert defaults.bw_bonds.default == 0.01
    assert not defaults.mismatch_ordering.default
    assert defaults.plot_scale.default == 1.0
    assert defaults.plot_scale.metavar == "(1.0)".rjust(25, " ")
    assert defaults.opti_dir.metavar == ""
    assert defaults.user_params.action == "store_true"
    assert "metavar" not in defaults.user_params.args
