# SwarmCG

SwarmCG is designed for the optimization of bonded parameters of a coarse-grained (CG) molecular model, with respect to a reference all-atom (AA) trajectory. The package contains 3 routines for:

1. Model optimization
2. Model evaluation
3. Summarizing an optimization procedure visually

#### Installation and libraries

```
$ pip install numpy scipy matplotlib MDAnalysis pyemd networkx fst-pso # TODO: remove networkx
```

And currently, download the few files in the github directory.

#### Optimize bonded parameters of a CG model, according to a reference AA trajectory

For optimizing bonded parameters of a model according to a reference AA trajectory, using the example data of PAMAM G1:

```
$ ./optimize_model.py -input_dir G1_CG_SIM/ -gmx gmx_2018.6_p
```

Which will use all default filenames of the software and is *exactly identical* to this command:

```
$ ./optimize_model.py -aa_tpr G1_CG_SIM/aa_topol.tpr -aa_traj G1_CG_SIM/aa_traj.xtc -cg_map G1_CG_SIM/cg_map.ndx -cg_itp G1_CG_SIM/cg_model.itp -cg_gro G1_CG_SIM/start_conf.gro -cg_top G1_CG_SIM/system.top -cg_mdp_mini G1_CG_SIM/mini.mdp -cg_mdp_pre_md G1_CG_SIM/pre-md.mdp -cg_mdp_md G1_CG_SIM/md.mdp -gmx gmx_2018.6_p
```

Preparing files in a directory that can be fed as argument to SwarmCG is way easier and recommended.

#### Evaluate bonded parametrization of a CG model, with respect to a reference AA trajectory

```
./evaluate_model.py -aa_tpr G1_CG_SIM/aa_topol.tpr -aa_traj G1_CG_SIM/aa_traj.xtc -cg_map G1_CG_SIM/cg_map.ndx -cg_itp G1_CG_SIM/cg_model.itp -cg_tpr G1_OPTI_mode1_200ns_valid/longer_run.tpr -cg_traj G1_OPTI_mode1_200ns_valid/longer_run.xtc
```

This is particularly useful to first assess the need to run an optimization procedure (assuming one has an initial CG model). It is also suited to the assesssment of geometrical changes trigger by a modification of CG particles types (non-bonded parameters).
